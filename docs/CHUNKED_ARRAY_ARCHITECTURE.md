# Chunked Array Architecture for PoCA Images

## Current dependency boundary (2026-10-02)

The earlier direct TensorStore integration is superseded. TensorStore now belongs exclusively to the independent [poca_zarr_backend](../../poca_zarr_backend/README.md) C++17 shared library. Normal PoCA CMake must never configure TensorStore or add either source project. `POCA_ENABLE_ZARR=ON` links only the installed C ABI through `POCA_ZARR_BACKEND_ROOT`; an optional `POCA_ZARR_BACKEND_DEBUG_ROOT` selects a Debug backend (with automatic `<release-root>_debug` discovery and Release fallback), while OFF needs no backend. OME interpretation stays in PoCA, and opaque handles plus caller-owned buffers isolate allocation and runtime ownership. Future labels/features/points/meshes should use generic N-D backend operations without exposing TensorStore types. Historical proposals below describe the broader storage direction; their dependency boundary is governed by this update.

## Why this change is needed

Today, PoCA images are still fundamentally "fully materialized arrays":

- `Image<T>` stores voxel values through the `"intensity"` feature created as `MyData(new Histogram<T>())` in `poca/src/poca_core/General/Image.hpp`.
- `Image<T>::pixels()` returns `dynamic_cast<Histogram<T>*>(getOriginalHistogram("intensity"))->getValues()`, so the image payload is the histogram payload.
- `Histogram<T>` owns a full `std::vector<T> m_values` in `poca/src/poca_core/General/Histogram.hpp`.
- `MyData` assumes the underlying feature can hand back a `std::vector<T>&`, which keeps the "all values resident in RAM" assumption alive.

The new pyramid cache helps GPU usage, but it does not address the main CPU bottleneck because level 0 still lives as one huge contiguous vector.

The `agave` code shows the right direction:

- `FileReaderZarr` already separates metadata discovery from data access.
- `supportChunkedLoading()` explicitly distinguishes chunk-capable backends from monolithic ones.
- TensorStore is used as a backend cache and random-access reader rather than eagerly loading everything.
- TIFF and Zarr both expose a common `loadDimensions` / `loadMultiscaleDims` view of the dataset.

That separation should become the basis of image handling in PoCA.

## Main design idea

The architectural change is:

1. Stop treating image voxels as a histogram-owned `std::vector<T>`.
2. Introduce a storage abstraction for N-D scalar arrays with chunked reads.
3. Make histograms/statistics operate on an array source, not only on a resident vector.
4. Keep a small RAM cache for chunks and derived pyramid levels.

In short: PoCA images should become "views over a voxel store", not "vectors wrapped in a histogram".

## Scientific invariants to preserve

This point is critical for PoCA and is different from a pure viewer such as Agave.

Chunking must change **storage and access**, not the scientific meaning of the dataset.

The following invariants should hold:

- the canonical image is always the **full-resolution** dataset
- rendering may use chunks, cached bricks, or lower pyramid levels
- the **original histogram** must be computed from the full-resolution dataset
- image analysis operations such as wavelet processing, segmentation, thresholding, labeling, and quantification must run against the full-resolution dataset
- display caches and multiscale levels are acceleration structures only; they must never silently become the source for analysis or for the original histogram

So PoCA should distinguish very clearly between:

- **source data**: full-resolution scientific truth
- **display data**: transient chunked/multiscale cache for interaction
- **derived metadata**: histogram/statistics computed from the full-resolution source
- **analysis workspace**: temporary buffers or streamed passes created by algorithms when needed

This distinction should be explicit in the architecture because it protects scientific correctness while still solving the RAM problem.

## Proposed components

### 1. `ArraySource` / `ChunkedArraySource`

Add a new core interface, for example in `poca_core/Interfaces/ArraySourceInterface.hpp`:

- element type
- shape `(x, y, z[, c, t])`
- chunk shape
- `readRegion(...)`
- `readChunk(...)`
- `prefetch(...)` optional
- `memoryFootprint()` for the in-RAM cache, not total dataset size
- metadata access: physical voxel size, units, number of pyramid levels

Two first implementations would be enough:

- `DenseArraySource<T>`: wraps a `std::vector<T>` for legacy data and computed outputs
- `TensorStoreArraySource<T>`: wraps Zarr and later chunked TIFF / tiled TIFF

This is the core abstraction that makes the solution generalizable to other `BasicComponent`s without forcing everything to become chunked immediately.

### 2. `ImageDataSource` owned by `Image<T>`

Refactor `Image<T>` so it owns image storage directly instead of hiding it inside `Histogram<T>`.

Suggested shape:

- `std::shared_ptr<ArraySourceInterface> m_storage;`
- `std::shared_ptr<ArrayStatisticsProvider> m_statsProvider;`
- a small chunk cache / tile cache
- optional derived pyramid cache keyed by level and downsample mode

Then:

- `Image<T>::pixels()` should become a legacy-only method, available only when storage is dense or when an explicit materialization is requested.
- new methods should be added instead:
  - `readVoxelBlock(...)`
  - `readPlane(z)`
  - `readForRendering(level, region)`
  - `materializeIfNeeded()` only for algorithms that truly require it

This is the most important ownership change: `Image<T>` becomes the owner of voxels, while histogram becomes a consumer of voxels.

### 3. `Histogram` split into two roles

Right now `Histogram<T>` mixes:

- data ownership
- statistics
- bin computation
- typed value access

That is the coupling to break.

I would split it into:

- `ValueDistribution` or `HistogramData`
  - owns `bins`, `ts`, `currentMin`, `currentMax`, `stats`
  - does **not** own the full raw values
- `HistogramBuilder`
  - computes the distribution from either:
    - a dense vector
    - an `ArraySourceInterface`
    - a generic value stream / iterator

For compatibility, `Histogram<T>` can remain temporarily, but internally it should delegate to a builder and only keep `m_values` for dense mode.

This lets images compute histograms by streaming chunks:

- iterate chunk by chunk
- update min/max/mean/variance
- fill bins in a second pass, or in one pass if bounds are known

For large images, this gives bounded memory usage.

Most importantly for PoCA:

- the **original image histogram** is built from the **full-resolution** source
- once computed, the histogram object behaves like the existing PoCA histogram object
- changing display chunks, visible region, or display level must not alter that original histogram
- optional per-level or per-region histograms may exist for visualization, but they must be clearly separate from the canonical histogram used by analysis and standard image controls

### 4. `FeatureStorage` for `BasicComponent`

If you want a path that can extend beyond images, add a thin abstraction above `MyData`.

Today `MyData` is essentially:

- one raw histogram
- one optional log histogram

I would replace or extend it with something like:

- `FeatureStorageInterface`
  - `distribution()`
  - `selectionAwareDistribution()`
  - `denseValues<T>()` optional
  - `streamValues(...)`
  - `materializationState()`

Then:

- geometry features can continue using dense storage
- image intensity can use chunked storage
- future huge per-cell or per-object attributes could also move to chunked storage if needed

This makes chunking generalizable without forcing a full rewrite of every `BasicComponent`.

## Reader architecture

The `agave` readers already suggest a clean separation:

- metadata reading
- multiscale discovery
- data transport

PoCA should adopt that pattern in its own loader side.

### Unified reader contract

Create a reader contract that returns:

- dataset metadata
- a storage object
- available multiscales
- backend capabilities

For example:

- `ImageDatasetDescriptor`
  - dimensions
  - dtype
  - physical sizes
  - channels
  - multiscale levels
  - preferred chunk sizes
- `std::shared_ptr<ArraySourceInterface>`

Instead of "reader returns fully loaded pixels", the reader should return "reader returns a readable dataset".

## TensorStore and multiresolution

TensorStore is very useful here, but it is important to be precise about what problem it solves.

TensorStore is a strong solution for:

- chunked array abstraction
- cached block access
- backend-independent region reads
- Zarr access in particular

TensorStore does **not** by itself solve multiresolution.

If a dataset already contains multiresolution levels on disk, TensorStore can expose them nicely.
If a dataset only contains a full-resolution TIFF, TensorStore will not invent the pyramid for PoCA.

So the architecture should separate two concerns:

- **chunked access backend**
  - TensorStore is a strong candidate
- **multiresolution policy**
  - PoCA must define when pyramids exist, how they are built, where they are stored, and which object types require them

For that reason, the recommended design is:

- keep PoCA interfaces such as `ArraySourceInterface` / `ImageStorageInterface`
- use TensorStore behind a `ZarrArraySource`
- optionally use TensorStore behind TIFF-derived sidecars too
- keep multiresolution policy owned by PoCA

### Zarr

For Zarr, PoCA can directly benefit from the existing `agave` approach:

- TensorStore stays the backend
- each multiscale path becomes one `TensorStoreArraySource`
- chunk shape comes from the store metadata
- the backend cache remains in TensorStore plus a PoCA-side render cache

This should be the reference implementation.

### TIFF

TIFF should support two modes:

- `DenseArraySource<T>` for simple TIFFs or small files
- `TiffArraySource<T>` for large stripped/tiled TIFFs

`agave/FileReaderTIFF.cpp` already reads plane by plane, which is a good start. Even without true TIFF chunk metadata, a useful first step is:

- treat each Z plane as the unit of streaming
- optionally subdivide into Y-strips when `ROWSPERSTRIP` is available
- expose those as chunks to the rest of PoCA

This gives you bounded memory immediately, even before a more advanced tiled TIFF backend exists.

For TIFF specifically, there are really two different architectural choices:

- **direct TIFF source**
  - PoCA reads TIFF planes/strips/tiles lazily
  - good for compatibility and single-image workflows
- **TIFF-derived sidecar source**
  - PoCA converts TIFF into a chunked multiresolution representation
  - best for large multi-image rendering workloads

For heavy multi-image visualization, the second option is likely the better long-term design.

## Source vs sidecar model

For PoCA, I recommend distinguishing:

- **canonical source**
  - the original TIFF or Zarr dataset
  - the source used for full-resolution analysis and canonical histogram computation
- **display sidecar**
  - a chunked multiresolution representation optimized for rendering
  - may be absent for simple workflows
  - may be required for large multi-image workflows

This means:

- analysis stays scientifically attached to the original full-resolution source
- rendering may prefer the sidecar whenever it exists
- histograms remain derived from the canonical full-resolution source unless a display-specific helper histogram is explicitly requested

For TIFF, the sidecar could be:

- a Zarr/TensorStore-backed multiresolution dataset
- a PoCA-managed chunked pyramid representation
- another chunked persistent format if later needed

The important point is not the exact file format yet.
The important point is that PoCA should be allowed to build and reuse a persistent multiresolution companion for TIFF when rendering performance matters.

## `MyObject` vs `MyMultipleObject`

This is a very good split for PoCA.

The current classes already support two different usage patterns:

- [MyObject.hpp](D:/Git/poca_codex/poca/src/poca/Objects/MyObject/MyObject.hpp#L46)
- [MyMultipleObject.hpp](D:/Git/poca_codex/poca/src/poca_core/Objects/MyMultipleObject.hpp#L45)

They should not necessarily impose the same image-storage policy.

### `MyObject` policy

For `MyObject`, the priority is compatibility and scientific correctness.

Recommended behavior:

- allow direct TIFF and Zarr opening without mandatory preprocessing
- support chunked access when possible
- do not require a sidecar pyramid
- allow explicit temporary dense materialization for analysis paths that still need it
- optionally build a sidecar later if the user requests performance-oriented rendering

So `MyObject` can remain the "simple/open directly" path.

### `MyMultipleObject` policy

For `MyMultipleObject`, the priority is scalable composite rendering.

This is exactly the case where:

- hundreds of images may be visible at once
- many images may overlap spatially
- loading full-resolution full-extent arrays for every visible image is not acceptable

Recommended behavior:

- require or strongly encourage a display sidecar with chunked multiresolution levels
- if a sidecar is missing, offer to build it lazily
- render from the sidecar, not from the canonical full-resolution source
- keep the canonical source only for analysis and full-resolution histogram/statistics

This gives a much cleaner performance model for `MyMultipleObject`.

## Recommended TIFF strategy

For TIFF-backed data, I would recommend the following:

### Single image

- open TIFF directly as canonical source
- optionally use streamed chunked reads
- no mandatory sidecar

### Multi-image or `MyMultipleObject`

- canonical source remains TIFF
- PoCA builds or reuses a persistent chunked multiresolution sidecar
- rendering uses the sidecar
- analysis uses the canonical TIFF source

This avoids forcing every user to preprocess every TIFF, while still giving PoCA a scalable path for the heavy multi-image case.

## Sidecar lifecycle

For a TIFF-backed display sidecar, PoCA should decide:

- where the sidecar is stored
- how validity is checked
- when it is rebuilt

Suggested policy:

- sidecar stored next to the TIFF or in a dedicated PoCA cache directory
- metadata contains:
  - original file path
  - size / timestamp / checksum
  - chunk shape
  - pyramid levels
  - scalar type
  - spacing / units
- sidecar is reused if metadata still matches the TIFF
- sidecar is rebuilt if TIFF metadata or content has changed

This is especially important if `MyMultipleObject` is expected to reopen the same large image sets repeatedly.

## Rendering path

The rendering path should become demand-driven:

1. Camera / zoom determines desired resolution and visible region.
2. `Image<T>` picks the best source level:
   - native level for close zoom
   - Zarr multiscale level if available
   - otherwise PoCA-generated pyramid level
3. Only the needed chunks are loaded.
4. GPU upload uses a bounded staging buffer and an LRU texture cache.

That means the current pyramid cache should move conceptually from:

- "cache derived full downsampled volumes"

to:

- "cache derived chunks or bricks per level"

For Zarr multiscale datasets, PoCA should prefer the on-disk multiscale over generating a new one in RAM.

Important: rendering data is not analysis data.

The renderer may use:

- lower-resolution multiscales
- visible-region chunk fetches
- GPU-side texture caches

but these must remain isolated from:

- the full-resolution source used by algorithms
- the original histogram/statistics attached to the image

## Statistics and histogram strategy

For large arrays, histogram/statistics should become explicitly lazy.

Recommended behavior:

- On image open:
  - read metadata only
  - do not compute the full histogram immediately unless needed by the UI
- On first histogram display:
  - compute statistics by streaming chunks
  - cache the result
- On contrast/window changes:
  - reuse cached bins if possible
- On selection/filter changes:
  - if selection is voxel-based and huge, use sampled or chunk-streamed recomputation

Useful additions:

- coarse sampled histogram for instant UI response
- background refinement to full histogram
- cached per-level histograms for pyramid levels

This would make the histogram UI much more scalable than the current eager model.

For PoCA, I would recommend distinguishing two histogram concepts:

- **canonical histogram**
  - computed from the full-resolution dataset
  - cached after computation
  - used by normal image analysis and canonical UI state
- **display helper histogram** optional
  - computed from current chunks, current level, or a sample
  - used only for responsiveness if needed
  - must never overwrite or replace the canonical histogram silently

In many cases, PoCA may not even need the second type. The important rule is that the histogram attached to the image in the current PoCA sense should remain the full-resolution one.

This gives the behavior you want:

- the histogram does not require the full image to stay in memory forever
- but the histogram still represents the full-resolution dataset, not the currently displayed chunks

## Compatibility strategy

A full rewrite would be risky. I would stage it as follows.

### Phase 1: Introduce storage abstraction without breaking existing components

- Add `ArraySourceInterface`
- Add `DenseArraySource<T>`
- Add a new histogram builder that can consume either dense vectors or streamed chunks
- Keep existing `Histogram<T>` and `MyData` working for non-image components

### Phase 2: Refactor `Image<T>`

- Move voxel ownership out of `Histogram<T>`
- Let `Image<T>` own either `DenseArraySource<T>` or `ChunkedArraySource<T>`
- Keep `pixels()` only as a compatibility bridge for dense images
- Add region/chunk read APIs

### Phase 3: Add chunked backends

- `TensorStoreArraySource<T>` for Zarr
- `TiffArraySource<T>` for streamed TIFF access
- optional memory-mapped dense source for large local raw-like datasets

### Phase 4: Update consumers

Prioritize the places that currently force materialization:

- `BasicOperationsImage.cu` uses `casted->pixels()` and uploads whole vectors
- thresholding and feature extraction on label images assume full host vectors
- rendering path assumes pointer-contiguous image memory

For these, add chunk-aware variants:

- whole-volume upload only when the volume is small enough
- block-wise GPU processing for large images
- explicit "materialize" fallback for legacy tools

For analysis operations, there should be two execution paths:

- **streamed full-resolution path**
  - preferred for algorithms that can operate chunk by chunk with halo management
  - examples: some filtering passes, some statistics, some reductions
- **explicit materialization path**
  - for algorithms that truly need dense global access
  - materialize a temporary full-resolution buffer only for the duration of the operation
  - free it immediately after the result has been produced

This is a better fit for PoCA than a viewer-oriented design because it preserves full-resolution analysis while still removing the requirement that the raw image must remain permanently resident in RAM.

## Implications for image analysis

PoCA does much more than display images, so chunking should support analysis rather than compete with it.

That means each image algorithm should eventually declare which access pattern it needs:

- full scan without global random access
- neighborhood/halo access
- full dense random access
- label-wise global reductions

Then PoCA can choose the cheapest execution strategy:

- stream chunks directly from the source
- stream chunks with overlap
- materialize a temporary dense buffer

Examples:

- histogram/statistics: stream full-resolution chunks
- simple thresholding: stream full-resolution chunks
- convolution/wavelet with bounded support: stream full-resolution chunks plus halo
- connected components / some segmentations: possibly materialize or use blockwise algorithms with merge steps
- operations that currently upload the whole image to CUDA: keep as-is first, but make the dense upload an explicit temporary workspace rather than permanent image ownership

This preserves correctness while still making memory usage proportional to the active algorithm, not to every opened image.

### Phase 5: Optional generalization to other `BasicComponent`s

Once images are stable:

- evaluate whether object features ever need chunked storage
- if yes, migrate `MyData` to a generic `FeatureStorageInterface`
- if not, keep images as the special large-array case

This keeps the generalization opportunity without forcing complexity too early.

## Concrete class sketch

One possible PoCA-oriented sketch:

```cpp
class ArraySourceInterface {
public:
  virtual ~ArraySourceInterface() = default;
  virtual DataType dataType() const = 0;
  virtual std::vector<uint64_t> shape() const = 0;
  virtual std::vector<uint32_t> chunkShape() const = 0;
  virtual bool isChunked() const = 0;
  virtual bool hasMultiscale() const = 0;
  virtual size_t numLevels() const = 0;
  virtual void readRegion(uint32_t level, const Region3D&, void* dst) const = 0;
  virtual ArrayStatistics computeStats() const = 0;
  virtual HistogramSummary computeHistogram(uint32_t bins, const HistogramBounds&) const = 0;
};

template<class T>
class DenseArraySource : public ArraySourceInterface {
  std::vector<T> m_values;
};

template<class T>
class ZarrArraySource : public ArraySourceInterface {
  std::shared_ptr<const poca::zarr::ZarrImageStorage> m_storage;
  std::vector<MultiscaleDims> m_levels;
};

template<class T>
class Image : public ImageInterface {
  std::shared_ptr<ArraySourceInterface> m_storage;
  mutable ImageChunkCache m_cache;
  mutable PyramidCache m_pyramid;
};
```

The key point is not the exact class names. The important part is: raw values are accessed through a source, not by reaching into a histogram vector.

## Recommended first implementation in PoCA

If I were implementing this incrementally in this repository, I would start here:

1. Add `ArraySourceInterface` and `DenseArraySource<T>`.
2. Add a streamed histogram/statistics builder.
3. Refactor `Image<T>` so `"intensity"` histogram is derived metadata, not the owner of voxels.
4. Create `ZarrArraySource<T>` using the `agave` TensorStore logic.
5. Create `TiffArraySource<T>` that streams plane-by-plane first, strip-by-strip later.
6. Update rendering to request regions/levels from the image source instead of calling `pixels()`.

That sequence gives you value early while minimizing the number of files touched at once.

## Recommendation on generalization

I do think the idea is generalizable to other `BasicComponent`s, but I would not force full generalization in the first pass.

My recommendation is:

- make the storage abstraction generic
- make the image migration first-class
- keep current dense `MyData` behavior for point-cloud/object features
- only generalize further when a second real use case appears

That avoids over-design while still preventing the image solution from becoming another special-case dead end.

## Bottom line

The architectural bug is not just "TIFF loading is eager". The deeper issue is that PoCA currently models image voxels as histogram-owned dense values.

The fix is to introduce a chunk-readable array source and make:

- readers return storage-backed full-resolution datasets
- images own the full-resolution source
- histograms are derived lazily from that full-resolution source
- rendering uses chunked or multiscale caches that are separate from the canonical source
- analysis operates on the full-resolution source, either by streaming or by explicit temporary materialization

If you want, the next useful step would be to turn this proposal into a concrete refactor plan with target classes/files in `poca_core`, starting from `Image.hpp`, `Histogram.hpp`, `MyData.*`, and the image-processing CUDA entry points.

## Concrete refactor map

This section proposes a practical implementation order in the current PoCA codebase.

The goal is to:

- preserve the existing histogram-facing API for most of PoCA
- move image voxel ownership out of `Histogram<T>`
- allow display to use chunked data
- keep analysis attached to the full-resolution source

### Step 0: Freeze the semantic contract

Before code changes, the following contract should be treated as fixed:

- `getOriginalHistogram("intensity")` for images means full-resolution histogram
- `pixels()` is no longer the canonical image API; it becomes a dense compatibility API
- display level / display chunks must not alter full-resolution histogram state
- image algorithms must explicitly choose between streamed full-resolution access and temporary dense materialization

This semantic contract will guide the refactor and prevent accidental viewer-style shortcuts.

### Step 1: Add new storage interfaces in `poca_core`

Add new interfaces and small utility types under `poca/src/poca_core/Interfaces` and `poca/src/poca_core/General`.

Suggested new files:

- `poca/src/poca_core/Interfaces/ArraySourceInterface.hpp`
- `poca/src/poca_core/Interfaces/ImageStorageInterface.hpp`
- `poca/src/poca_core/General/Region3D.hpp`
- `poca/src/poca_core/General/ChunkKey.hpp`
- `poca/src/poca_core/General/DenseArraySource.hpp`

Minimum responsibilities:

- shape, dtype, chunk shape
- `readRegion(level, region, dst)`
- `readPlane(z, dst)` convenience helper
- `isChunked()`
- `hasMultiscale()`
- metadata access for spacing and units

This step should add code only, without changing current image behavior yet.

### Step 2: Add streamed statistics/histogram builders

Add a new builder layer without breaking `Histogram<T>`.

Suggested new files:

- `poca/src/poca_core/General/HistogramSummary.hpp`
- `poca/src/poca_core/General/HistogramBuilder.hpp`
- `poca/src/poca_core/General/ArrayStatisticsBuilder.hpp`

Responsibilities:

- compute min/max/mean/stddev by streaming chunks
- compute bins from full-resolution data without storing all values
- produce a result object that can populate the existing histogram-facing structures

At this stage, the existing `Histogram<T>` can still exist, but it should stop being the only way to create histogram state.

### Step 3: Introduce image-owned storage in `ImageInterface`

The first real API change should happen in:

- [ImageInterface.hpp](D:/Git/poca_codex/poca/src/poca_core/Interfaces/ImageInterface.hpp#L38)
- [Image.hpp](D:/Git/poca_codex/poca/src/poca_core/General/Image.hpp#L154)

Add new virtual methods to `ImageInterface`, for example:

- `virtual bool hasDensePixels() const = 0;`
- `virtual bool canMaterializePixels() const = 0;`
- `virtual void readFullResolutionRegion(...) const = 0;`
- `virtual void materializePixelsIfNeeded() = 0;`
- `virtual bool isChunkBacked() const = 0;`

Then refactor `Image<T>` to own:

- `std::shared_ptr<ArraySourceInterface> m_source;`
- optional dense cache `std::vector<T> m_materializedPixels;`
- display chunk/pyramid cache
- cached canonical histogram/statistics

The key ownership move is:

- voxels live in `m_source`
- histogram becomes metadata derived from `m_source`

### Step 4: Decouple `"intensity"` from raw voxel ownership

This is the highest-value architectural change.

Current coupling points:

- [Image.hpp](D:/Git/poca_codex/poca/src/poca_core/General/Image.hpp#L156)
- [Image.hpp](D:/Git/poca_codex/poca/src/poca_core/General/Image.hpp#L544)
- [Histogram.hpp](D:/Git/poca_codex/poca/src/poca_core/General/Histogram.hpp#L125)
- [MyData.hpp](D:/Git/poca_codex/poca/src/poca_core/General/MyData.hpp#L54)

Refactor idea:

- keep `MyData` for dense scalar features
- introduce a new image-specific metadata holder, or extend `MyData` carefully, so `"intensity"` can be backed by:
  - a canonical histogram object
  - a source descriptor
  - optional dense materialization state

Two viable options:

- conservative option:
  - keep `MyData` unchanged for all non-image features
  - add a dedicated image member for canonical histogram and do not route image intensity through `MyData` ownership anymore
- more general option:
  - evolve `MyData` into `FeatureStorage`
  - allow it to contain either dense values or streamed-source metadata

For the first implementation, I recommend the conservative option. It is much less risky.

### Step 5: Add the first backends

The first two concrete backends should be:

- `DenseArraySource<T>`
- `ZarrArraySource<T>`

Suggested new files:

- `poca/src/poca_core/General/ZarrArraySource.hpp`
- `poca/src/poca_core/General/ZarrArraySource.cpp`
- `poca/src/poca_core/General/TiffArraySource.hpp`
- `poca/src/poca_core/General/TiffArraySource.cpp`

The implementation can reuse the `agave` logic conceptually:

- [FileReaderZarr.h](D:/Git/poca_codex/agave/FileReaderZarr.h#L17) for capability split
- [FileReaderZarr.cpp](D:/Git/poca_codex/agave/FileReaderZarr.cpp#L457) for TensorStore-backed reads
- [FileReaderTIFF.cpp](D:/Git/poca_codex/agave/FileReaderTIFF.cpp#L723) for plane-wise TIFF reading

Backend priorities:

- Zarr:
  - full multiscale awareness
  - true chunk reads via TensorStore
- TIFF:
  - first version can expose planes as chunks
  - second version can expose strips/tiles as chunks

### Step 6: Add a loader layer that returns sources instead of dense pixels

Right now the reader pattern in `agave` still ends by materializing the volume.

For PoCA, create a new image loading layer that returns:

- metadata
- a source object
- optional precomputed multiscale descriptors

Suggested new files:

- `poca/src/poca_core/General/ImageDatasetDescriptor.hpp`
- `poca/src/poca_core/General/ImageSourceFactory.hpp`
- `poca/src/poca_core/General/ImageSourceFactory.cpp`

This layer should decide:

- use `DenseArraySource<T>` for small images
- use `TiffArraySource<T>` for large TIFFs
- use `ZarrArraySource<T>` for Zarr datasets

### Step 7: Make histogram creation lazy and canonical

Once `Image<T>` owns a source:

- `finalizeImage()` should no longer imply "all pixels are in RAM"
- it should initialize image metadata and mark canonical histogram as not-yet-built
- on first `getOriginalHistogram("intensity")`, PoCA should stream the full-resolution source, compute the canonical histogram, and cache it

This preserves the current user-facing meaning while removing permanent RAM residency.

### Step 8: Keep `pixels()` as a compatibility bridge

There are too many existing call sites to delete `pixels()` immediately.

For the first migration, keep:

- `pixels()`
- `data()`
- `getImagePtr()`

but redefine them as:

- valid only when dense storage exists
- or they trigger explicit materialization of a temporary dense full-resolution buffer

This is the bridge that lets PoCA continue working while the algorithms are migrated one by one.

### Step 9: Migrate the first algorithm cluster

The first files to update after image ownership are:

- [BasicOperationsImage.cu](D:/Git/poca_codex/poca/src/poca_core/Cuda/BasicOperationsImage.cu#L369)
- [Image.hpp](D:/Git/poca_codex/poca/src/poca_core/General/Image.hpp#L380)
- [MainWindow.cpp](D:/Git/poca_codex/poca/src/poca/Widgets/MainWindow.cpp)

Why these first:

- they are image-specific
- they already assume dense full-res vectors
- they are the main pressure point for image analysis semantics

Suggested migration order inside `BasicOperationsImage.cu`:

1. keep current behavior via `materializePixelsIfNeeded()`
2. make that materialization explicit in function names/comments
3. add streamed full-resolution variants for operations that do not need global dense access

### Step 10: Add algorithm access categories

Each image algorithm should eventually declare its access type.

Suggested categories:

- `StreamingFullResolution`
- `StreamingWithHalo`
- `DenseFullResolutionTemporary`
- `GlobalLabelReduction`

This can be represented by helper functions or a small enum rather than a big framework at first.

Examples:

- histogram/statistics: `StreamingFullResolution`
- thresholding: `StreamingFullResolution`
- normalization: `StreamingFullResolution`
- local filtering/wavelets with bounded kernels: `StreamingWithHalo`
- connected components / some segmentations: `DenseFullResolutionTemporary` first, later optimized
- label volume counting: `GlobalLabelReduction`

This classification will tell you where to invest engineering effort first.

## What can stay unchanged initially

A lot of PoCA can remain unchanged in the first pass.

These areas should mostly continue to work if the histogram public API is preserved:

- histogram widgets and generic histogram commands in [BasicComponent.cpp](D:/Git/poca_codex/poca/src/poca_core/General/BasicComponent.cpp#L140)
- non-image `BasicComponent`s that use dense scalar features
- display code that only needs `HistogramInterface`
- geometry/object workflows that rely on dense float feature vectors

This is why preserving the histogram-facing API is so valuable.

## What needs a new access API first

These areas are likely the earliest required changes:

### 1. Image ownership and initialization

- [Image.hpp](D:/Git/poca_codex/poca/src/poca_core/General/Image.hpp#L154)
- [ImageInterface.hpp](D:/Git/poca_codex/poca/src/poca_core/Interfaces/ImageInterface.hpp#L38)

Because this is where full-res source ownership must move.

### 2. Dense image CUDA entry points

- [BasicOperationsImage.cu](D:/Git/poca_codex/poca/src/poca_core/Cuda/BasicOperationsImage.cu#L369)

Because these currently pull the full voxel vector unconditionally.

### 3. Pyramid cache

- [Image.hpp](D:/Git/poca_codex/poca/src/poca_core/General/Image.hpp#L380)

Because the current pyramid code copies level 0 from `pixels()`, which defeats the new ownership model.

### 4. Any direct `pixels()` call in UI helpers

- [MainWindow.cpp](D:/Git/poca_codex/poca/src/poca/Widgets/MainWindow.cpp#L2343)
- [MainWindow.cpp](D:/Git/poca_codex/poca/src/poca/Widgets/MainWindow.cpp#L2725)

These should either use display-oriented region reads or explicit temporary dense materialization.

## Recommended first milestone

The first milestone should be intentionally narrow:

1. add `ArraySourceInterface`
2. make `Image<T>` own a source
3. compute canonical full-resolution histogram lazily from the source
4. keep `pixels()` as explicit temporary materialization
5. leave most algorithms unchanged for now

If this milestone works, PoCA will already gain:

- much lower idle RAM usage for opened large images
- correct full-resolution histogram semantics
- a path to chunked rendering
- a safe compatibility bridge for analysis code

## Recommended second milestone

After that, target the highest-cost analysis operations:

1. migrate rendering to chunk/region reads
2. migrate simple image scans to streamed full-resolution reads
3. keep difficult algorithms dense-temporary at first
4. optimize those later only if they are proven bottlenecks

This will keep the implementation grounded in PoCA’s real needs rather than over-optimizing around a viewer model.

## Revised backend plan

Taking the TensorStore and object-type discussion into account, the recommended backend plan is:

### Zarr

- canonical source: Zarr
- chunk backend: TensorStore
- multiresolution: use on-disk multiscales when present

### TIFF in `MyObject`

- canonical source: TIFF
- chunk backend: PoCA streamed TIFF source first
- multiresolution: optional, not required

### TIFF in `MyMultipleObject`

- canonical source: TIFF
- display backend: persistent chunked multiresolution sidecar
- sidecar backend: preferably TensorStore-compatible if practical
- multiresolution: required or strongly enforced

This is a better fit for PoCA than trying to force one identical strategy on every image workflow.

## Phase 3 scalar export implementation (2026-10-02)

Local scalar RAW images can be exported with the recordable saveOmeZarr command to OME-NGFF 0.5 / Zarr v3 stores. Depth 1 writes YX; volumes write ZYX, with uint8/uint16/uint32/int32/float32 values. Resident scientific level-zero pixels are read directly in chunks; unloaded sources use ImageInterface full-resolution regions. Rendering LOD never selects output dataset zero.

Native-level metadata now carries optional actual calibration, and storage-backed images expose native region reads. Exports preserve calibrated native levels. Generated pyramids stream the preceding written array through the existing PoCA Average downsampler and use actual dimension ratios for physical scale. No second full-resolution output volume, independent image class or analysis LOD change is introduced. Signed integer Average accumulation is corrected in the shared downsampler.

The Zarr plugin owns OME metadata and staged filesystem publication. Its generic write adapter uses the existing standalone backend C ABI; no backend API additions, TensorStore integration in PoCA, or Debug/Release import changes are required. Labels, geometry/features, multichannel/time, collections and remote writing remain future work.

Source implementation/review only; build and runtime validation remain UNCONFIRMED. See [Phase 3 export report](../../poca_extra/src/poca_loaderZarrFile/EXPORT_README.md) for command parameters, safety, source tests and full file inventory.
## Phase 4 OME-NGFF labels (2026-10-02)

Associated labels remain ordinary integer Image<T>(LABEL) entries inside ImagesList with source-entry indices. The existing saveOmeZarr command gathers associated entries through the list or image owner, and File/Export dispatches that recorded command. Standard labels groups and image-label metadata are plugin-owned; the generic backend ABI and dependency isolation are unchanged.

Label export shares Phase 3 staging/publication and streams level zero, matching calibrated native levels or bounded physical nearest sampling into the source image's actual target pyramid. Scientific level zero remains independent of display LOD. Sparse explicit colors are retained without a new categorical rendering subsystem; labels lacking object feature tables use scalar palette display and nearest filtering. Imported labels start hidden to avoid materializing single-level labels when opening the source.

Source implementation complete / manual validation required. No configure/generate/build/link/install, application executable, Python script or test was run. See [Phase 4 report](../../poca_extra/src/poca_loaderZarrFile/LABELS_README.md) for source inventory, test coverage and limitations.
