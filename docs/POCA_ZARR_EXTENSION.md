# PoCA quantitative data alongside OME-NGFF

Current reopening policy: see **Reusing persisted features, normals and mesh certification (2026-10-08)** at the end. Earlier dated sections describe previous milestones.

## Command characteristics, load report and statistics repair (2026-10-06)

Source-only change; no configure, build, application, Python/helper or test execution. Runtime behavior was not verified.

### Statistics regression

The vector constructor means five **precomputed statistics**, not raw samples. Its guard incorrectly tested the default-empty destination `m_data`, so even a valid five-element input failed. The reused-Voronoi path is `organoComputeSample -> organoPrepareVoronoi` (existing cut with topology requested) `-> DetectionSet(coordinates) -> BasicComponent::setData -> MyData/Histogram::setHistogram -> generateArrayStatistics -> computeStats -> vector<float>(5) -> ArrayStatistics`. Publication through `generateDataWithLog` reaches the same path. Current callers already compute the five entries correctly; no raw-feature caller needed changing.

`ArrayStatistics` now holds `std::array<float, STATS_NB_PARAMS>`, initializes all five default/scalar entries by construction, validates `_vals.size()` and copies valid precomputed statistics. The guard stays strict, the public API stays stable, and copying/moving cannot produce empty internal storage. Numerical algorithms and scientific calculations are unchanged.

### Load diagnostics

`PocaZarrLoadReport` owns steady-clock stages/dataset timing, output, counts and generic characteristic status. `PocaZarrDatasetLoader` passes one report through image, quantitative component and feature restoration. It reports root/child manifests, multiple reconstruction, ImagesList/RAW/LABEL, ObjectLists/points/meshes, persisted features, assembly and command restoration. Component phases print start/completion; no per-chunk output is added. Each child prints count deltas and elapsed reconstruction time. One summary follows completed root reconstruction, before Engine registration and GUI refresh.

Illustrative output; timings below are not measurements:

```text
[PoCA][Zarr][Load] Dataset 1/19: sample_0
[PoCA][Zarr][Load]   Reconstructing ObjectListMesh
[PoCA][Zarr][Load]   ObjectListMesh restored in 0.018 s
[PoCA][Zarr][Load]   Components: ImagesList=1 RAW images=2 LABEL images=1 ObjectLists=1 ObjectListMesh=2 Mesh objects=73 Persisted feature arrays=37
[PoCA][Zarr][Load]   Dataset reconstruction restored in 0.066 s
...
[PoCA][Zarr][Load]   OrganoGraph: restored (19 samples, 55 base features, 4 embedding coordinates), 0.041 s
[PoCA][Zarr][Load] ===== Load summary: MyMultipleObject =====
[PoCA][Zarr][Load] Datasets reconstructed: 19
[PoCA][Zarr][Load] CGAL mesh collections materialized: 0
```

Counts inspect `hasPixels`, `valuesUnloaded`, `nbObjects` and `meshesMaterialized`; they never call pixel/value/CGAL accessors. Persisted feature-array counts come from restored descriptors; feature records also include intensity and derived/native display features. Lazy-state counts describe components at reconstruction completion, not lifetime chunk traffic or temporary sampling. Nested timings are inclusive and must not be summed. The 2026-10-08 diagnostic extension below adds direct-child/self accounting and detailed cost groups, superseding the former inclusive-only report.

### Ownership and layout

`Command` gains storage-neutral `saveState/restoreState/characteristicName/discardState` hooks and `CommandStateStorage` double-array access. Default state delegates to existing `saveCommands/loadParameters`; restoration never executes a processing command. `PocaZarrCommandState` matches owner indices and exact owner/command names to installed commands, and implements array access with existing ZarrArrayWriter/ZarrArrayAccess. Neither core nor OrganoGraph sees TensorStore/backend types.

The optional manifest `characteristics` field changes neither dataset version nor backend ABI. Each object saves only its own commands/components; child objects use their existing individual serializer. Typed ordinal physical keys do not derive from logical names. Export remains under the existing staging transaction.

```text
collection.ome.zarr/
  objects/o000000/...                       # child geometry and MyData arrays
  poca/zarr.json                           # manifest + characteristic metadata
  poca/characteristics/q000000/zarr.json    # generic command group
  poca/characteristics/q000000/a000000/...  # float64 [successful samples, 55]
  poca/characteristics/q000000/a000001/...  # optional [samples, embedding coordinates]
  poca/characteristics/q000000/a000002/...  # optional flattened spatial edges
  poca/characteristics/q000000/a000003/...  # optional pair-correlation values
  poca/characteristics/q000000/a000004/...  # optional Ripley deviations
```

Numbers are illustrative; descriptors carry explicit paths. Numeric arrays use row chunks; curve offsets/counts remain small per-sample metadata. Float64 preserves float features, double curves and undefined NaNs.

Collection-owned OrganoGraph state includes all 55 base slots, PCA/UMAP coordinates, curves, mapping, configuration (including small embedding-method metadata), successful source indices, signatures, selected-feature sets and nucleus-feature availability, plus plot display/identification. Hierarchy remains dataset-owned and is checked against the saved signature.

Computed/custom nucleus outputs and cell-derived outputs published onto nucleus `MyData` belong to child components. Cell outputs additionally published onto the Voronoi mesh remain in its normal storage. The root stores references/names/availability, never second copies of these vectors or standalone archive `nuclei/cells` arrays. Reference validation checks existence and feature lengths using metadata; components retain normal ownership/lifetimes.

### Restoration and compatibility

Engine assembly installs plugin commands before returning the object. After components/features and hierarchy are restored, the generic adapter calls `OrganoGraphCommand::restoreState`. `OrganoGraphPersistence` reuses existing domain structures and spatial-curve validation; it checks identities, shapes, offsets, feature references, embeddings and view state before publishing the normal command-owned graph. Its explicit already-restored path skips feature publication, full fingerprint recalculation and duplicate debug snapshots. It never runs scientific preflight, morphology, intensities, Voronoi, spatial computation or dimensionality reduction, and does not replace child `MyData`.

MainWindow subsequently attaches observers and sends existing `LoadObjCharacteristicsAllWidgets`. OrganoGraphWidget reads its normal command and repopulates selectors, categories, embeddings and plots. Its only added refresh behavior restores stable source/nucleus identification; it has no Zarr parsing or storage dependency. Drawing a selected plot may subsequently read needed child features as usual.

Malformed optional command state produces a clear failure/timing and leaves OrganoGraph unavailable while retaining core data. Missing plugins are reported; bad allocation remains fatal. Legacy 0.2 manifests without characteristics and 0.1/external image paths remain supported. Selected-child export omits the parent analysis. Standalone version-1/version-2 parsing and streamed format remain intact; explicit standalone operations retain existing full fingerprint checks. Debug streamed comparison uses the legacy snapshot when present; lazy restoration intentionally does not create that duplicate archive.

Limits: structural validation retains historical fingerprints without rehashing content, so matching names/counts do not detect same-shape edits. Only currently accepted PCA/UMAP coordinate names are supported. Root numeric results are resident, rather than lazy. Legacy graph/data plot identification is retained in command state but not automatically highlighted on generic refresh because it depends on the original view. Lazy-state counters are snapshots, not read-traffic instrumentation.

### Source coverage (not executed)

Loader fixtures use existing `POCA_ZARR_EXPORT_SOURCE_TESTS`. New opt-in `ORGANO_PERSISTENCE_SOURCE_TESTS` registers OrganoGraph fixtures; when Zarr is enabled its manual integration branch requires installed LoaderZarr/OrganoGraph plugins, exports through the existing command and reopens through Engine without a plugin link dependency. Widget fixtures require normal Qt application context.

| Requested cases | Source coverage |
| --- | --- |
| 1–6 | `LoadCharacteristicFixture::statistics`: defaults/scalars/precomputed five values, rejected invalid vector, seven raw samples, MyData/Histogram and reused-Voronoi seed DetectionSet path; strict guard retained |
| 7–12 | Report/plain/three-child fixtures: counts, deterministic phase accumulation, dataset timing, storage-backed images and no CGAL conversion |
| 13–14, 23, 29, 31 | Root real-array manifest; plugin integration root export, no root nucleus/cell copies, selected-child omission/no-analysis reopen, erased-characteristics legacy manifest |
| 15–22, 26 | Codec/command: partial reordered source indices, mapping/config/signatures, base NaNs, PCA/UMAP, double curves/NaNs, selected features/availability and normal graph ownership |
| 24–25 | Explicit restored path, unchanged MyData pointer/read probes, reopened lazy features/meshes; no scientific dispatch/publication |
| 27–28 | Generic observer and actual widget probe: graph, selectors, hierarchy, embeddings, nucleus names, stable identification; no Zarr widget code |
| 30 | Standalone streamed save/load fixture with referenced nucleus features; existing formats retained |
| Corruption | Duplicate indices, missing feature/component, wrong identity/offsets, malformed plot/configuration, generic array-shape failure isolation and missing plugin reporting |

Static source audits cover declarations, includes/registration, ownership/refresh order, backend boundaries, unchanged scientific paths, delimiters, whitespace, UTF-8/BOM and CRLF. Fixtures describe intended coverage; they are not evidence of runtime success.

## Persistence-aware components (2026-10-06)

This section supersedes the eager mesh reconstruction and histogram sampling descriptions in earlier milestones below. These changes are source-only; runtime behavior was not verified.

Persisted scientific data, display representation and heavy analysis representation have separate lifetimes. A saved feature is restored as authoritative data; only absent state follows the existing compatibility computation. BasicComponent previously owned features/selection rather than scientific calculation; it now centralizes adoption of normal MyData ownership and current-feature/selection metadata, without backend knowledge. DetectionSet retains its coordinate readers and first-query KD-tree boundary.

ObjectListMesh now owns backend-independent IndexedMeshGeometry: resident double vertices, global uint64 triangle indices and object offsets. Its explicit persistence constructor creates existing float display arrays, vertex/face normals, bounds, centroids and triangle-corner z directly from those arrays. It does not construct Surface_mesh, repair topology, measure volumes or run PCA. Normal constructors retain their existing scientific processing. Direct smooth normals reuse the installed CGAL most-visible-normal solver with indexed incidence; CGAL 5.3 and 6.0.1 header signatures were inspected. No Surface_mesh instance is needed for that solver.

Immutable indexed backing is shared by lazy copies. Const getMeshes builds and caches all analysis meshes once, preserving double coordinates, face winding/start corner and required normal properties; a failed conversion publishes nothing. First conversion is mutex-protected. Mutable getMeshes permanently invalidates indexed authority because a returned reference can mutate again after an export. Subsequent indexedGeometry access snapshots current CGAL geometry. As with existing APIs, mutation must be serialized against display/copy/export, and existing processing commands own rebuilding display/features after edits. Copy assignment is explicitly unavailable; copy() and the copy constructor retain safe feature ownership and do not force conversion.

nbObjects uses indexed offsets before conversion. Eight count/existence callers in Organograph/VMAS now use it; feature availability also uses histogram metadata count. Actual CGAL analysis remains on getMeshes. The ordinary and multiple-object renderer consume existing display arrays. Loading or switching children preserves hierarchy, names, order, transforms and current-child semantics without building their CGAL caches. Legacy binary mesh export and genuine processing/geometry-copy operations still use the CGAL interface.

Zarr export uses indexedGeometry once: unchanged persisted meshes reuse the exact immutable backing without CGAL; normal or writable meshes serialize a fresh snapshot of current topology/coordinates. Existing geometry schema, root 0.1/0.2 dispatch, commands/macros, transaction and backend ABI remain unchanged. Object features stay storage-backed; no scientific measurements are recomputed. Triangle-z retains its separate corner domain, noninteractive policy and saved histogram/display state.

Histogram now snapshots/restores count, bins, statistics/provenance, bounded statistics sample and display state without reading feature values. New feature metadata optionally embeds this state. Old feature metadata retains bounded compatibility sampling. PocaZarrImagePersistence groups image feature serialization/restoration in the existing feature files; optional image-group poca_image metadata works for RAW, direct/orphan LABEL and associated LABEL images. Image::initializeFromPersistence restores intensity state without native-level sampling; values remain in the OME pixel array, with no duplicate intensity array. Independent float32 features have explicit individual counts and existing lazy readers. Saved label/volume features survive addFeatureLabels; missing features use available legacy volume data. Old images without this metadata retain their native-sample/display-window path. Feature-less legacy meshes receive an ordinal id display feature; absent quantitative measurements are not invented.

One cohesive geometry implementation file was added and registered in poca_geometry/CMakeLists.txt. BasicComponent, Histogram/HistogramInterface, Image, DetectionSet and ObjectListMesh were extended; IndexedMeshGeometry belongs to ObjectListMesh, and PocaZarrImagePersistence owns the backend-specific image workflow. No new dependency, GUI workflow or image-streaming framework was introduced.

### Source fixtures and audit coverage

The existing optional fixtures were extended, not executed:

| Requested coverage | Source evidence |
| --- | --- |
| 1–4, 12: lazy load/count, direct triangles/normals | PocaZarrMeshTests and DatasetContainerFixture::lazyMesh |
| 5: triangle-z display/state/export without CGAL | PocaZarrMeshTests derived mesh round trip |
| 6–7: authoritative lazy measurements, no recomputation | Open-triangle volumes 123.25/456.75; tetra volume 999; throw-on-read histogram restoration in PocaZarrLazyTests |
| 8–10: first/repeated CGAL cache, exact coordinates/topology | Const cache address reuse, float64 vertices and uint64 face/start-corner comparisons |
| 11, 17: real processing and normal creation | Subdivision fixture; ordinary diagnostic constructor area feature |
| 13: count-only callers | Source search/audit of Organograph/VMAS and feature availability |
| 14–16: direct export, round trip, modified/retained mutable geometry | Unmaterialized export and two exports through a retained writable reference |
| 18: multiple children/current switching | Existing 1/4/133-child fixture with lazy checks per child and loaded-dataset re-export |
| 19–20: image and LABEL restoration | Saved independent measurements, mean 1.5 vs constant-2 pixels, label/volume pointer/state preservation |
| 21: legacy compatibility | Missing mesh features; missing image poca_image metadata; supplied legacy LABEL volumes |
| 22: DetectionSet unchanged laziness | Existing no-coordinate/no-KD load/copy/default selection and first-query cache checks |
| Ownership and malformed geometry | Clone after source destruction; invalid offsets/cross-object indices, disconnected fans, directed-edge conflicts and polygon rejection |

Static inspection verifies loader/display/export boundaries, counts, authoritative features, lazy image/KD paths, ownership, source registration and unchanged backend/streaming/hierarchy boundaries. Whitespace, strict UTF-8, consistent CRLF and lexical source checks are recorded in CONTINUITY.md. These checks establish no compilation or runtime result.

Remaining limits: indexed geometry and display arrays are resident (not out-of-core); a fresh snapshot adds temporary memory for normal/writable export. Independent persisted features remain float32, matching the existing quantitative writer. Internal CGAL normal-solver calls require review when changing CGAL versions. The raw mutable mesh API retains its existing external synchronization/display-rebuild responsibility; stale triangle-z is explicitly rejected during export. Exact persistence restores measured histogram state, but explicit later rebin/log/filter/materialization operations retain existing Histogram semantics. Resident-memory reporting remains a lower bound and may count shared indexed backing per component.

## Compact storage keys (2026-10-06)

Logical names and physical keys are separate. ZarrStorageKeys in the existing ZarrSafeNames.hpp/.cpp owns deterministic typed ordinal keys: objects o000005, image components c000004, images i000002, associated labels l000003, points p000001, meshes m000002 and features f000017. Six decimal digits are a minimum width; larger indices retain every digit. Keys never accept logical names. Duplicate names cannot collide through this policy.

Exact names, order, associations, transforms, ObjectLists entry/plugin/CommandInfo data and feature names remain in the existing manifests and NGFF/array metadata. Schema 0.1/0.2 already has explicit path fields; no version bump or loader changes are required. Loaders open recorded paths and restore recorded names for legacy safe-name stores and new compact stores. Ordinary OME image stores and their legacy label paths remain supported. Standard OME labels declarations reference the actual compact subgroups, with source.image="../../" unchanged; orphans remain independent.

Audit covered child objects, image components/ImagesList, RAW and associated/orphan/direct LABELs, points, ObjectLists/direct meshes and scalar features. Remaining positions, vertices, faces, offsets, rendering axes, pyramid levels and chunk hierarchies use fixed semantic/numeric segments. Source/plugin/object/component/image/label/feature strings never determine internal writer keys. Users must not treat subgroup names as display names.

Standalone, selected-child and full multiple export reuse the same object serializer. One root transaction and staging UUID remain; scientific algorithms, ownership, GUI/CommandInfo dispatch and backend ABI are unchanged. A user-selected extremely deep destination can still exceed filesystem limits; the existing contextual storage error is retained without renaming it. Image errors now include the exact logical name, and dataset writes retain child/component/image indices and names.

Optional source fixtures were updated, not executed:

| Requested cases | Source coverage |
| --- | --- |
| 1-11, 17-18 long/duplicate simultaneous names | DatasetContainerFixture::longObject/checkLongNames: 420-character Windows source-path stems, duplicate and nearly duplicate images, associated/orphan/direct labels, long points/ObjectLists/meshes/features/plugin/command strings |
| 12, 14 name/path symmetry | checkCompactPaths/checkLongNames: stored paths, NGFF/array exact names, lazy loaded features, round-trip re-export and actual chunk-path bounds (fixture relative paths <=128; segments <=24) |
| 13 legacy compatibility | rewriteLegacyObjectPaths/checkLegacyPaths: old safe-name images/labels/quantitative/features and duplicate children; full 0.2 reconstruction and 0.1 quantitative loading |
| 15-16 shared selected/full policy | checkLongNames: selected command equals standalone serializer; every full multiple child has ordered o-prefixed paths |
| 19-20 rollback/diagnostics | checkRollback: late failing child, previous destination/staging cleanup and exact long logical child/component/image plus level diagnostics |
| Policy edge cases | PocaZarrSchemaTests: all seven prefixes, deterministic keys, >999999 and maximum-size_t ordinals; OmeZarrLabelsTests: compact declarations |

Modified files: ZarrSafeNames.hpp/.cpp; PocaZarrDatasetExporter.cpp; PocaZarrExtensionExport.cpp; PocaZarrFeatures.cpp; OmeZarrLabelsMetadata.cpp; OmeZarrExport.cpp; PocaZarrSchemaTests.cpp; OmeZarrLabelsTests.cpp; PocaZarrDatasetContainerTests.cpp; LABELS_README.md; EXPORT_README.md; this document; CONTINUITY.md. No files added. Existing optional test registration covers all changed fixtures.

Static review only: runtime, compilation and test results remain UNCONFIRMED.

## Phase 6.1 — complete dataset containers (2026-10-06)

This section supersedes the image-only scope statements in the historical Phase-6 sections below. Source implementation and static source audit only; compilation, Qt/backend/CGAL behavior and runtime round trips are UNCONFIRMED.

### Actions and recordable commands

| Active entity | Export image as OME-Zarr... | Export dataset as OME-Zarr... | Export selected dataset as OME-Zarr... |
| --- | --- | --- | --- |
| MyObject | Existing action; enabled for the current RAW | Complete active MyObject | Hidden |
| MyMultipleObject | Existing action; enabled for the current child's current RAW | Complete outer object and ALL children | Visible; currentObject() as a standalone MyObject |
| No selected child | Disabled image action | Full export still traverses every available child | Disabled and safely rejected |

The image action retains `saveOmeZarr`, one current RAW, explicitly associated labels, and the existing optional Phase-6 quantitative extension. Dataset actions record `saveDatasetOmeZarr` or `saveSelectedDatasetOmeZarr` through LoaderZarr's `actionNeeded` and normal CommandInfo dispatch. MDI activation, child selection and File-menu opening update action state through PluginInterface::updateActions; no polling timer is used. The active camera's entity is the target, never all open windows.

Both dataset commands accept outputPath, overwrite=false, multiscale=true, componentChunkRows=65536 and chunkX/Y/Z=256/256/4. The command specification includes executeOnObjectOnly=true. MyMultipleObject also recognizes both commands as owner operations for direct/replayed CommandInfo without that parameter, so child command forwarding cannot produce repeated partial exports. Complete export cannot turn off labels or quantitative data; direct options requesting partial export are rejected.

Default GUI filenames use the existing ZarrSafeNames and suffix normalization: the active object name, multiple-object name, or selected child's name followed by .ome.zarr. No internal ID is added. Empty names use the sanitizer's explicit dataset name policy. Errors include exactly "No current dataset to export.", "No datasets are available in the current multiple dataset.", and "No dataset is currently selected." where applicable. Dataset actions do not use the image-only intensity-selection error.

### Class responsibilities and shared serialization

| Class | Responsibility |
| --- | --- |
| PocaZarrDatasetManifest (added) | Captures and restores names/transforms/hierarchy; owns ordered component/image/child descriptors, quantitative references, validation, JSON and persistence |
| PocaZarrDatasetExporter (added) | Resolves full/selected targets; preflights all scientific content; writes one MyObject, all multiple children and every ImagesList entry; delegates payload writers; owns one staging transaction |
| PocaZarrDatasetLoader (added) | Loads declared entries and explicit associations, reuses scalar/point/mesh reconstruction, assembles ordered owned children/aggregate components, restores state, registers only the completed root |
| OmeZarrExportCommand (extended) | Existing image command plus dataset command specs/dispatch and owner-aware copies |
| LoaderZarr (extended) | Three actions, dynamic visibility/enabling, filename/dialog/error collection, command creation and version dispatch |
| Engine (extended) | Unregistered owning assembly for MyObject and MyMultipleObject; one final root registration; components/plugins installed once |
| ImagesList (extended) | Reads explicit source indices and accepts owned unassociated images without selection-based inference |
| MyMultipleObject (extended) | Persistence constructor can bypass grid recomputation; dataset commands execute once at the owner |
| Command / CommandableObject (extended) | copyFor(owner) lets export commands bind the new owner during base construction; the default preserves other commands' existing copy behavior |
| PluginInterface (extended in both distributions) | Default updateActions hook for event-driven menu state |

One shared header declares the three dataset classes; three substantial implementation files separate metadata, export and reconstruction. No class wraps one trivial operation. The existing PocaZarrManifest.cpp now also supplies the manifest class's shared quantitative validator, preserving the v0.1 reader entry point. Existing scalar/pyramid/label/template writers remain in their current files; the new group-writing and label-validation entry points are adapters needed to reuse those implementations. Tiny local matrix conversions and existing safe-name/arithmetic helpers remain stateless.

Exactly one `exportObject` implementation serializes a standalone object, a selected child and every full multiple-object child. Full export never chooses currentObject as its target; only selected export does. Selected export creates kind=object and no one-child wrapper.

### Scientific hierarchy and schema 0.2

The outer store is a **PoCA extension stored in Zarr**, containing standards-compliant NGFF 0.5 image groups where applicable. It is not an official OME-NGFF multiple-dataset standard. Generic OME viewers may need to open an individual contained image group; they need not understand the outer container.

Every root and child has a Zarr v3 group. Root attributes.poca_dataset identifies version=0.2 and manifest=poca. The authoritative manifest is attributes.poca in poca/zarr.json:

- version=0.2, kind=object or multiple_object, exact name and transform.
- components is ordered and has contiguous component_index, exact name and kind: images_list, image, points, mesh_collection or object_lists.
- An images_list descriptor has images with contiguous index, exact entry name, type=raw/label, relative path and source=null or an explicit same-list RAW index.
- An object_lists descriptor also has entry_count. quantitative retains the unchanged Phase-6 component payload references, including container ID/name, ordered entry index/name, plugin and normalized CommandInfo.
- A multiple_object additionally has ordered objects: index, exact name, unique relative path and kind=object; hierarchy nodes with label, level_name, parent, children, object indices and string metadata; grid_boxes; and an optional current_object preference.
- Parent hierarchy nodes precede their children, matching the current addHierarchyNode API. Duplicate visible object/image names are valid; unique indices and deterministic safe paths identify entries.

Conceptual compact layout (keys depend on serialized ordinals; logical names remain metadata):

```text
dataset.ome.zarr/
  zarr.json                         PoCA container; no dummy root image
  poca/zarr.json                    0.2 manifest
  images/c000002/
    i000000/                       independent NGFF RAW group (logical actin)
      0/ ...
      labels/l000000/              explicitly associated NGFF LABEL (logical mask)
      labels/l000001/              second label for the same RAW (logical mask)
    i000002/                       another independent RAW (logical actin)
    i000003/                       scalar LABEL; no image-label.source
  poca/points/...                  existing Phase-6 positions/features
  poca/meshes/...                  existing indexed geometry/object features

multiple.ome.zarr/
  zarr.json
  poca/zarr.json                    kind=multiple_object, ordered child descriptors
  objects/o000000/                 complete kind=object dataset
  objects/o000001/                 another child with the same exact visible name
  objects/...                     all remaining children
```

All ImagesList entries are considered independently of current selection: every RAW, all explicitly associated labels, and every orphan. Associated labels use the existing Phase-4 writer beneath their declared RAW. Orphans use that same scalar label/pyramid implementation at an independent group and remove the source member; type=label and source=null restore LABEL semantics and remain unassociated. Owned insertion avoids addImage's existing current-entry inference; associations are restored after all entries exist, including labels that precede their RAW.

Points-only, mesh-only, LABEL-only and mixed datasets need no RAW or dummy image. DetectionSets and direct top-level ObjectListMesh components need no image association. ObjectLists retains its container, exact entry order/names/plugins/commands; it is not flattened. Top-level component order is independent of path sorting. Every supported component is written once, including supported aggregate components owned directly by MyMultipleObject.

The existing positions, indexed double vertices/u64 faces/offsets, coordinate metadata, nbSlices, bounding boxes, float32 feature arrays, lazy feature sampling and lazy KD-tree paths are reused. No new source-image associations or scientific units are guessed.

### Transforms, metadata and ownership

Source inspection found MyMultipleObject's ordered raw-pointer child vector, unbounded nbColors count, unchecked currentObject access, an owning destructor, hierarchy metadata, selected-object indices and batch-rendering/grid state. Its constructor normally recomputes grid placement. Its interface accepts arbitrary MyObjectInterface pointers, so nested multiple containers are technically possible but are explicitly unsupported by this schema and rejected with child context.

For each object, model and rotation are persisted as 16 row-major float32 values (JSON numeric scalars), and translation as three float32 values. Access/reconstruction converts GLM's column-indexed matrix to this documented representation. The authoritative model matrix, including any scale, is restored directly; rotation and PoCA's stored translation are also restored so subsequent gizmo edits retain the saved decomposition. Nothing is baked into pixels, vertices or localizations. Saved grid boxes and hierarchy node metadata are preserved. The multiple constructor bypasses placement recalculation during reconstruction.

The current-child index is an optional UI preference. Valid indices are restored; malformed/out-of-range preferences are ignored without discarding scientific children. Multi-selection, display visibility/category/colors, batch-rendering switches, camera and command history are not captured as generic dataset state. Label color metadata and already supported mesh feature/display metadata retain their existing payload behavior.

Loader reconstruction uses local unique_ptr owners throughout. Children are assembled without registration, moved into MyMultipleObject once, and deleted by its existing destructor. Aggregate/child component commands and root commands are installed once, after the relevant components exist. Only the completely restored root is registered in Engine. Failure releases all partially reconstructed owners without registering children or an incomplete root. The new command-copy binding avoids exporting through an original owner pointer after MyObject/ImagesList/Image copies.

### Preflight, transaction and compatibility

Complete export preflights the outer object and every child, all top-level components, all ImagesList entries, ObjectLists entry types, quantitative feature types/readability/display metadata, and ROIs stored outside components. Unsupported state is aggregated with object name, child index/name, component or entry name and reason. Skeleton, TrackSet, Voronoi/Delaunay, plugin-private unsupported components, ObjectListPolygon and ROI/annotation state must not be silently omitted. Empty ImagesList/ObjectLists cannot currently be reconstructed and are explicitly rejected. Null/repeated children or aliased owned entries are rejected.

Unsupported-content preflight finishes before creating staging. Pixel/feature reads, geometry validation and metadata/filesystem failures during serialization still fail the entire transaction. Exactly one OmeZarrExportStore stages the full hierarchy; subgroup/child writers do not publish. Only after all children and manifests succeed does the existing rename/backup publication run. A late child failure cleans staging and preserves the old destination. Existing overwrite/link checks and rollback/backup retention remain. The operation remains a synchronous local-filesystem transaction, without a new cross-process or crash-durability guarantee; existing same-store source-handle limitations remain.

LoaderZarr recognizes 0.2 before image-centric loading, restores object or multiple_object without requiring a root image, and rejects malformed supported manifests. Version 0.1 remains readable through its shared quantitative validation and old root-image path. Stores without a PoCA manifest continue through the external normal OME-Zarr image/labels loader. The image export still writes the existing 0.1 extension and retains its established unsupported-component skip policy; the new complete export has the stricter policy described above.

No backend ABI/TensorStore wrapper changes, dependency installations, source changes in poca_zarr_backend or C:/tsb/C:/tsbd, camera changes, image scaling/calibration changes, navigation/detail streaming changes, GPU-budget changes or unrelated refactors were made. The PoCA plugin interface gains a default action hook; the eventual user-controlled build must include the updated interface consistently across core and plugins.

### Source fixtures and requested coverage

PocaZarrDatasetContainerTests.cpp contains one cohesive DatasetContainerFixture, two necessary synthetic unsupported/null-selection fixtures, and one TestRegistry adapter. It is registered as an optional manual Tests/Images action under the existing POCA_ZARR_EXPORT_SOURCE_TESTS flag (OFF by default). These sources were written and statically reviewed only, never executed.

| Requested cases | Source evidence for later execution |
| --- | --- |
| 1–8 GUI | checkGui: three actual QActions, both switch directions, invalid/null child, image action retained; MainWindow activation/menu/selection hooks audited |
| 9–17 every image entry | checkObject: duplicate RAW names, two labels on one RAW, second RAW's label, orphan source=null, exact order/name/type/source; checkImageOnlyVariants adds LABEL-only and LABEL-before-RAW |
| 18–22 quantitative components | checkObject/checkImageOnlyVariants: points with/without images, ordered ObjectLists metadata, direct mesh, direct scalar component, lazy open/re-export; existing point/mesh fixtures retain scientific payload coverage |
| 23–33 multiple datasets | checkMultiple: 1, 4 and 133 children, all children vs last current child, exact duplicate names, paths/order, model/rotation/translation, grid boxes, hierarchy metadata and current preference; empty case in checkPreflight |
| 34–37 selected | checkSelected: command-selected/direct manifest equality, standalone kind=object with no wrapper, changing selection, full direct command owner scope |
| 38–42 mixed children | checkMultiple cycles RAW+LABEL+points+ObjectLists, points-only, mesh-only, orphan LABEL+mesh; aggregate outer points also round-trip |
| 43–46 transaction | checkRollback: final child pixel-reader failure after earlier children wrote staging, old manifest retained, no published objects, staging/backup absent |
| 47–51 unsupported | checkPreflight: unsupported normal component, nested multiple rejection, two different child issues, name/index/reason aggregation, external ROI marker, no destination created; loader/manifest strict supported-kind checks audited |
| 52–56 compatibility | Existing ordinary and v0.1 whole-object/image fixtures remain registered; checkObject/checkMultiple/checkRegistration exercise new 0.2 dispatch; old image command/dialog/labels fixtures retained |
| Additional guards | checkManifestRejection: path alias/traversal, hierarchy cycles, optional invalid current preference, fabricated orphan source; checkCopiedOwner: original deleted before copied object's command runs |

Static source audit covers the 34 requested structural checks: retained image action; correct normal/multiple visibility; all-child traversal; selected-only currentObject; one shared object serializer; every image/orphan/RAW; independent point/direct-mesh components; explicit ObjectLists/multiple hierarchy and ordering; safe duplicate names; persisted transforms; no inferred associations; full unsupported-content preflight; one transaction and rollback; v0.1/external dispatch; unchanged backend/rendering/calibration boundaries; and three substantive OO responsibilities rather than tiny utility files. Final static checks pass 39 UTF-8/consistent-CRLF files, 34 balanced lexical source streams, 78 existing plugin source-list paths, 26 new-source local includes, new-source whitespace and git diff --check. The documented 5-added/34-modified inventory exactly matches the working tree after excluding externally edited AGENTS.md. These are source findings, not runtime assertions.

### Known limitations and later manual validation

Nested MyMultipleObject, empty component lists and unsupported scientific/ROI components are explicit complete-export errors. Source callbacks, objects and destination trees must remain stable for synchronous export. The existing lazy sampling, in-memory mesh reconstruction, scalar dtype restrictions, same-store source handles and local publication limitations remain. Build/link/API/Qt/backend/CGAL behavior remains UNCONFIRMED.

Manual checklist for a separately supplied build and user-controlled validation:

1. Switch normal → multiple → normal MDI windows; confirm both dataset actions update and the old image action remains present.
2. Select different children and RAW/LABEL entries; confirm image applicability and selected-dataset enablement.
3. Validate empty entity/empty multiple/null current-child messages without unchecked vector access.
4. Export a normal object with two RAWs, multiple associated labels and an orphan; open each RAW group in an NGFF-aware viewer.
5. Reopen in PoCA; compare every entry's exact name, type, order and explicit/orphan association, including LABEL-before-RAW.
6. Round-trip LABEL-only, points-only, mesh-only and LABEL+mesh datasets without a dummy RAW.
7. Compare ObjectLists entry order/names/plugin/normalized commands and direct mesh presence.
8. Inspect lazy feature/index state after load and after unchanged re-export; use existing scientific point/mesh fixtures to compare values/topology.
9. Full-export one, several and 133 children with duplicate names and a selected child near the end; count all ordered child datasets.
10. Compare saved/reopened child and outer matrices, rotation/translation, hierarchy labels/membership/metadata and grid boxes; then make a gizmo edit.
11. Export selected child twice after changing selection; compare each standalone store against direct export of that child.
12. Replay both recordable dataset commands; confirm owner execution once and copied objects export their own content.
13. Introduce unsupported components and ROI state in separate children; check all diagnostics and no staging/destination write.
14. Inject a late child read failure over an existing destination; verify previous data, no partial children and staging cleanup.
15. Load ordinary external OME-Zarr, old 0.1 stores and both 0.2 container kinds through Engine; check one root registration and no duplicate commands. Run the optional source fixtures only when separately authorized.

### Phase 6.1 complete file inventory

The externally edited AGENTS.md is preserved and excluded from the implementation inventory below.


Added (5 files):

- `poca_extra/src/poca_loaderZarrFile/PocaZarrDataset.hpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrDatasetContainerTests.cpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrDatasetExporter.cpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrDatasetLoader.cpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrDatasetManifest.cpp`

Modified (34 files):

- `CONTINUITY.md`
- `poca_extra/include/PluginInterface.hpp`
- `poca_extra/src/poca_loaderZarrFile/CMakeLists.txt`
- `poca_extra/src/poca_loaderZarrFile/EXPORT_README.md`
- `poca_extra/src/poca_loaderZarrFile/LoaderZarr.cpp`
- `poca_extra/src/poca_loaderZarrFile/LoaderZarr.hpp`
- `poca_extra/src/poca_loaderZarrFile/LoaderZarrGui.cpp`
- `poca_extra/src/poca_loaderZarrFile/LoaderZarrObject.cpp`
- `poca_extra/src/poca_loaderZarrFile/OmeZarrExport.cpp`
- `poca_extra/src/poca_loaderZarrFile/OmeZarrExport.hpp`
- `poca_extra/src/poca_loaderZarrFile/OmeZarrExportCommand.cpp`
- `poca_extra/src/poca_loaderZarrFile/OmeZarrExportCommand.hpp`
- `poca_extra/src/poca_loaderZarrFile/OmeZarrExportGui.cpp`
- `poca_extra/src/poca_loaderZarrFile/OmeZarrExportGui.hpp`
- `poca_extra/src/poca_loaderZarrFile/OmeZarrExportMetadataTests.cpp`
- `poca_extra/src/poca_loaderZarrFile/OmeZarrExportTests.hpp`
- `poca_extra/src/poca_loaderZarrFile/OmeZarrLabelsExport.cpp`
- `poca_extra/src/poca_loaderZarrFile/OmeZarrLabelsExport.hpp`
- `poca_extra/src/poca_loaderZarrFile/OmeZarrLabelsLoad.cpp`
- `poca_extra/src/poca_loaderZarrFile/OmeZarrLabelsLoad.hpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrExtensionExport.cpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrManifest.cpp`
- `poca_extra/src/poca_loaderZarrFile/README.md`
- `poca/docs/POCA_ZARR_EXTENSION.md`
- `poca/include/PluginInterface.hpp`
- `poca/src/poca_core/General/Command.hpp`
- `poca/src/poca_core/General/CommandableObject.cpp`
- `poca/src/poca_core/General/Engine.hpp`
- `poca/src/poca_core/General/EngineObjectAssembly.cpp`
- `poca/src/poca_core/General/ImagesList.cpp`
- `poca/src/poca_core/General/ImagesList.hpp`
- `poca/src/poca_core/Objects/MyMultipleObject.cpp`
- `poca/src/poca_core/Objects/MyMultipleObject.hpp`
- `poca/src/poca/Widgets/MainWindow.cpp`

Suggested commit: `feat(zarr): export complete MyObject and MyMultipleObject datasets`.

No CMake configure/generate, compilation, linking, installation, application executable, Python script, benchmark, or test was run.

## Historical Phase-6 image-oriented foundation (version 0.1)

2026-10-05 — Phase 6 source implementation. This is a **PoCA extension stored alongside OME-NGFF**, not an OME-NGFF point-cloud or mesh standard. Compilation and runtime behavior are UNCONFIRMED. The source fixtures below are registered for later manual execution and have not been run.

## Scope and entry points

The existing `saveOmeZarr` command exports one selected scalar RAW image, its explicitly associated labels, and supported quantitative components of the owning PoCA object into one local `.ome.zarr` store. DetectionSets, ObjectLists entries with concrete ObjectListMesh geometry, and direct ObjectListMesh components are supported. Images are not duplicated under `poca/`.

`includeLabels=true` and `includePocaData=true` are independent defaults. `componentChunkRows=65536` controls quantitative row chunks independently of image `chunkX/Y/Z`; it must be a positive uint32. The existing File -> Export -> Export image as OME-Zarr... action records these defaults through the existing command dispatch. No additional export action or advanced dialog was introduced.

The direct API is `saveOmeZarr(image, path, options, labels = {}, object = nullptr)`. An explicit owner is required to include quantitative components through this API. The command resolves ownership with Engine and passes that owner to orchestration. Omitting the owner keeps the existing standalone-image API valid.

## Version and hierarchy

`poca/zarr.json` is a normal Zarr v3 group:

```json
{
  "zarr_format": 3,
  "node_type": "group",
  "attributes": {
    "poca": {
      "version": "0.1",
      "components": [
        {"kind": "points", "path": "points/DetectionSet", "name": "DetectionSet", "component_index": 1}
      ]
    }
  }
}
```

```text
dataset.ome.zarr/
  zarr.json                    # unchanged OME-NGFF 0.5 image group
  0/, 1/, ...                  # scientific image and pyramid
  labels/<safe-label>/...      # standard associated OME labels
  poca/
    zarr.json                  # version and component manifest
    points/<safe-component>/
      zarr.json
      positions/
      features/<safe-feature>/
    meshes/<safe-component>/
      zarr.json
      vertices/, faces/
      vertex_offsets/, face_offsets/
      object_features/<safe-feature>/
      render_axes/             # optional existing rendering cache
```

Every extension container is an ordinary v3 group. Generic array metadata includes shape, data_type, dimension_names, zero fill, regular chunks, default slash chunk-key encoding, attributes and a little-endian bytes codec. Image metadata and its XYZ mapping retain the original image-specific path. Generic `ZarrArrayAccess::readRaw` accepts positive regions of arbitrary nonzero rank and returns packed C-order data through the existing C ABI.

The manifest uses semantic `points` and `mesh_collection` kinds, exact original component names and safe relative paths. Supported top-level component indices are unique; gaps from skipped components are accepted. Unsupported extension versions fail explicitly and cannot trigger image-only fallback. Within version 0.1, unknown kinds with valid basic names/paths are skipped with a diagnostic; unknown auxiliary fields are ignored. Malformed supported components fail the whole load. No older-version conversion is implied.

## Quantitative arrays

| Array | Shape | Dtype | Dimension names | Row chunks |
| --- | --- | --- | --- | --- |
| Point positions | `[N,D]`, D=2 or 3 | float32 | point, coordinate | `[min(N,rows),D]` |
| Point feature | `[N]` | float32 | point | `[min(N,rows)]` |
| Mesh vertices | `[V,3]` | float64 | vertex, coordinate | `[min(V,rows),3]` |
| Mesh faces | `[F,3]` | uint64 | face, corner | `[min(F,rows),3]` |
| Vertex/face offsets | `[O+1]` | uint64 | object_boundary | `[min(O+1,rows)]` |
| Mesh object feature | `[O]` | float32 | object | `[min(O,rows)]` |
| Optional render axes | `[O,3,3]` | float32 | object, axis, coordinate | `[min(O,rows),3,3]` |

Zero-face arrays are permitted and use chunk extents of at least one; zero-sized regions are never read. Components must have positive point/object counts; each mesh object must contain vertices. Count, offset, chunk element and byte/address arithmetic is checked. Current resident PoCA indices constrain point/object/vertex counts to uint32 and mesh triangle corners to uint32; uint64 storage does not remove those application limits.

Both geometry kinds declare `space="poca-world"`. Numeric values are preserved without inventing units, image transforms or parent/source associations. An explicit trustworthy future unit/source association can be auxiliary metadata; this implementation emits none for generic geometry. The existing image calibration and label associations remain separate.

## Point metadata and loading

The component group contains `attributes.poca` with kind, name, space, count, dimension, nbSlices, bounding_box, positions, coordinate_names, coordinate_display, current_feature and features. Bounding boxes use `[xmin,ymin,zmin,xmax,ymax,zmax]`. A valid existing 2D Z range is retained without inventing a third coordinate.

Coordinate names are exactly x/y or x/y/z. They are stored only in the positions matrix and recreated as existing PoCA MyData feature names on load. Coordinate reads select `[start,coordinateIndex]` with shape `[count,1]`; every returned coordinate is checked for finiteness and membership in the persisted bounding box. Export validates coordinates and feature lengths while streaming. Arbitrary additional float features are discovered from BasicComponent::getData rather than a fixed registry.

Opening validates metadata and obtains bounded coordinate/feature samples. It uses persisted count, dimension, bounding box and nbSlices; no full coordinate scan or spatial index is required. Selection starts all true. Display may subsequently require coordinate arrays, independently of spatial indexing.

Storage-backed DetectionSet::getKdTree and getKdTreeCloud call ensureSpatialIndex. The first request materializes only x/y/z, creates the cloud and builds one owned KD-tree under its mutex; later requests reuse it. Existing eager constructors keep their eager behavior. Storage-backed copies clone mutable feature/histogram state and callbacks, retain backing handles, and start without a tree. The destructor deletes its owned tree.

## Feature storage, sampling and display

Supported scientific feature data is explicitly Histogram<float>; other scalar types fail with component/feature/type context instead of lossy conversion. Each feature manifest has exact name, safe relative path and display metadata. The array also stores the exact original name in attributes. Load validates rank 1, exact component length, float32, dimension name, name agreement, nonempty/unique exact names, reserved coordinates, case-insensitive path uniqueness and safe containment.

Display state stores finite min/max, currentMin/currentMax, interaction, scaleLUT and requested log preference. The selected current_feature is preserved and validated. Complete histogram bins are not persisted as scientific values.

At most eight distributed contiguous windows and 32768 values per feature are sampled. Finite samples drive approximate bins/statistics; full scientific float32 values, including signed zero and nonfinite values in extra features, are not replaced by samples. Every lazy feature owns its read-only Zarr handle through value-captured shared ownership. No callback points at the loader or temporary metadata.

Histogram<T> has a generic typed region-reader seam in core, with no Zarr dependency. Resident values take priority over callbacks so resident edits export correctly. Untouched lazy features and coordinate columns stream to another store in bounded chunks without full materialization. Full reads allocate one requested feature, read bounded regions into a temporary vector, and publish it only after success.

An unloaded feature whose filter covers the full known range, or has interaction disabled, contributes no restriction and does not materialize during default selection. Restrictive filtering, explicit data access, CSV/algorithm requests and log transforms may materialize the requested feature. Existing resident selection semantics remain.

Log capability is advertised without eagerly creating log arrays. Saved log preference survives untouched re-export, but the reopened histogram initially displays linearly; an explicit log toggle materializes and computes the log histogram. Active log bounds are serialized in the original value domain. This deliberate UI limitation avoids accidental full-array loading at open. Existing log-domain behavior for nonpositive values remains; invalid/nonfinite display state is rejected.

## Indexed meshes and authoritative measurements

Export enumerates actual Surface_mesh_3_double vertices in iterator order, keeps a deterministic per-object descriptor-to-global-index map, and writes bounded vertex/face buffers. It does not export only expanded rendering triangles. One first pass computes offsets and verifies triangular topology. No second permanent flattened O(V+F) topology buffer is required.

Offsets start at zero, are nondecreasing, have O+1 entries and end exactly at V/F. Faces have three distinct global vertex indices; every index must be below V and inside that object's vertex interval. Load validates all stored face indices before constructing CGAL topology, then remaps to local descriptors. add_face failure is an error. The saved first corner and cyclic orientation are preserved for re-export.

The persistence constructor adopts indexed meshes without repair, stitching, orientation correction, triangulation, PCA measurement calculation or remeshing. Double coordinates remain authoritative in CGAL; float rendering arrays, normals, bounding boxes and centroids are rebuilt for existing APIs. The use_vertex_normals display choice is preserved. Rendering coordinates must be finite and representable by float32. Scientific feature arrays are adopted lazily, with safe ownership; no constructor-generated measurements replace persisted values.

`render_axes` stores existing finite per-object axis directions only as an auxiliary rendering cache. It preserves ellipsoid overlays without recomputing measurements. It is null/absent as an array when the source has no cache; ellipsoid measurements require a matching cache. It is not a general per-vertex/per-face feature registry.

Existing ObjectListMesh `z` may be triangle-corner display data rather than O object values. Export accepts this special case only when noninteractive, length exactly 3F and every value agrees with rendered geometry. It records `derived_triangle_z` display metadata and regenerates that display array on reopen; it never mislabels it as an object feature. A genuine `[O]` z feature uses normal feature storage. Other mismatched feature lengths fail.

Mesh geometry, offsets, rendering arrays and caches are resident after reopen. Per-object scientific features remain lazy until consumers request them. Rendering an ellipsoid can request major/minor/minor2; rendering the current coloring feature can request that feature. This is not out-of-core mesh rendering.

## ObjectLists and whole-object integration

Each container-derived manifest entry adds `container: {id,name,entry_index,entry_name,plugin,command}`. The id groups supported entries; component_index preserves supported top-level order and entry_index preserves supported entry order. Exact user-visible strings, including empty entry/plugin strings, are retained. CommandInfo::toNormalizedJson writes the existing name/params/recordable representation; load validates and restores it with fromJson. Full command history is not stored.

LoaderZarr::loadObject returns nullptr when poca/zarr.json is absent, preserving the ordinary loadData route. When present, it validates the version/manifest, reuses one shared image-and-label helper, builds points/meshes and restores ObjectLists. Supported load failures propagate explicitly rather than returning an incomplete image-only object.

The new owning Engine::createObject overload accepts all components, uses existing addComponentToObject for each container and its children, installs object commands once, and registers exactly one dataset only after assembly succeeds. Components are added in reverse because MyObject inserts at the front. Unique names and unique_ptr ownership prevent ambiguous merges and failure leaks. Existing raw-pointer APIs remain available.

## Transaction, security and memory

Image arrays, labels and poca arrays are all written inside the same existing sibling staging directory. Quantitative destination handles close before extension metadata; poca manifest is validated before existing root publication. Supported component errors abort export and invoke existing staging cleanup/backup rollback. Unsupported components emit one diagnostic and are skipped. No partial supported component is published.

The shared Phase-4 sanitizer retains label behavior: separators/invalid/control characters are sanitized, UTF-8 bytes escaped, Windows device names protected, bases bounded and case-insensitive collisions resolved with deterministic suffixes. Exact display names remain metadata. Extension paths are safe ASCII relative children, cannot be absolute/drive/backslash/traversal/self references, must remain canonically contained, and refuse symlink/reparse surprises.

BasicComponent reports resident histogram/sample/selection storage rather than claiming unloaded feature lengths as RAM. Storage-backed DetectionSet adds resident coordinates/cloud/tree; ObjectListMesh reports measured resident rendering arrays/caches and features. Reports saturate the existing unsigned-int API. CGAL property maps/allocator overhead, map/string/callback overhead and backend internal scratch are not fully measured; these are lower bounds, not a memory-budget guarantee. Source/destination and callbacks must remain stable during synchronous export and copy operations.

## Unsupported scope and known limitations

Skeleton, TrackSet, ROIs, annotations, Delaunay, Voronoi, arbitrary ObjectListPolygon, graph topology, Organograph hierarchy, complete command history/provenance, generic per-vertex/per-face feature registries, remote/cloud edits and out-of-core mesh rendering are not persisted. Unknown component kinds are skipped; unsupported feature types inside supported components abort export.

Image scope stays one selected scalar RAW image plus associated labels, not every RAW image in an ImagesList. Existing restrictions on multichannel/time/plate stores, nonfinite image display metadata, same-store replacement handles and local synchronous transaction crash durability remain. Opening histograms consumes O(features * bounded sample) memory; selection costs O(N) bits; a first KD request costs O(N) coordinates/tree. Explicit large componentChunkRows can increase export scratch memory. All Qt/backend/CGAL/filesystem runtime behavior remains UNCONFIRMED.

## Source fixtures and requested coverage

`POCA_ZARR_EXPORT_SOURCE_TESTS` remains OFF by default. Existing manual Tests/Images registration adds a combined schema/array/lazy/geometry/dataset action and an Engine whole-object action. No source fixture was executed. The following matrix maps every requested case; structural properties also require the stated source audit, not an assertion of runtime proof.

| Requested cases | Source fixtures / static evidence |
| --- | --- |
| 1–9 schema/version/manifest/names/paths | PocaZarrSchemaTests; shared sanitizer + manifest audit |
| 10–15 generic rank/bytes/bounds/dtypes | PocaZarrArrayTests; readRaw interval/arithmetic review; rank-3 axes fixture |
| 16–22 point export | PocaZarrPointsTests: 2D/3D, no duplicate coordinates, arbitrary/constant features, bbox/slices, large chunked reader |
| 23–25 metadata discovery/one-feature access | PocaZarrLazyTests + PointsTests + DatasetTests |
| 26–28 lazy x/y/z columns | PocaZarrPointsTests selected-column reads for both dimensions |
| 29–34 lazy KD/default/restrictive/log | PocaZarrLazyTests; storage constructor and getter audit |
| 35–40 point round trip | PocaZarrPointsTests float bit comparisons, names/count/bbox/slices |
| 41–47 mesh export | PocaZarrMeshTests: two indexed meshes, float64/u64, offsets, globals, nontriangle rejection, object features |
| 48 bounded topology output | Mesh export source audit: two bounded buffers + per-object descriptor map + O(O) offsets |
| 49–57 mesh load | PocaZarrMeshTests: real malformed offsets/faces, cross-object/out-of-range rejection, exact coordinates/connectivity, lazy authoritative features; constructor audit |
| 58–62 ObjectLists | PocaZarrDatasetTests: two entries, order/exact names/plugins/normalized CommandInfo |
| 63–67 transaction | PocaZarrDatasetTests image+label+poca, unsupported skip, failure/old destination/cleanup, OME root and independent options |
| 68–72 whole-object path | PocaZarrDatasetTests ordinary loadObject null + old loadData, actual Engine dispatch with all supported components |
| 73–74 commands/registration | Dataset whole-object fixture command type uniqueness/owner/dataset count; assembly source audit proves one installation path per component/container child |
| 75–76 lazy re-export | PocaZarrPointsTests + MeshTests + DatasetTests: chunk copy and unloaded state; feature-export source audit |

Additional PocaZarrFeatureTests cover rank/length/dtype/domain/original-name rejection, empty names, case collisions, reversed display metadata and scientific nonfinite/signed-zero float32 bits. Mesh fixtures also cover preserved rendering axes and the derived triangle-z recipe. Existing Phase-3/4 image/label fixtures remain registered.

## Static audit

Reviewed all 28 requested audit areas at source level: ordinary image/label route, separate generic image/raw reads, backend dependency boundary, one staging transaction, lazy histogram/KD/copy/selection paths, indexed double/u64 topology without repair, authoritative features/entry metadata, unsupported skips/no guessed provenance, scope, checked arithmetic, ownership and Windows safe names. Final audit passes 66 UTF-8/no-BOM CRLF files, 59 balanced lexical source streams and 230 existing source-list paths, local quoted includes, new-file whitespace, quoting and git diff --check. The inventory matches 32 added and 34 modified files; results are also recorded in CONTINUITY.md. This cannot establish compilation or runtime correctness.

## Manual validation after the user compiles

1. Open an image-only OME-Zarr; verify the old image-only route and image display.
2. Open image + labels; verify label IDs, colors, source association and existing rendering.
3. Export a selected RAW image with a DetectionSet, includePocaData enabled; inspect image and poca manifests.
4. Close/reopen and display detections; compare point count, 2D/3D coordinates, bbox and nbSlices.
5. Compare feature names/current feature/histograms and display bounds; expect approximate bins initially.
6. Access one extra feature; verify that unrelated features stay unloaded. Check a restrictive filter and explicit log toggle.
7. Run a nearest-neighbor/KD operation; verify coordinate loading, one index build and untouched extra features.
8. Export image + one ObjectListMesh; inspect float64 vertices, uint64 faces/offsets and object feature manifests.
9. Reopen and visually compare mesh geometry, normals and ellipsoid overlays; compare exact coordinates/connectivity numerically.
10. Compare all supported object feature values, including added measurements and any derived triangle-z coloring.
11. Round-trip multiple mesh entries in ObjectLists; verify order, entry names, plugins and normalized commands.
12. Re-export an untouched reopened extended store; verify identical supported values and that feature vectors remain unloaded.
13. Use a very large point/feature set; inspect bounded samples, selection RAM, one-feature reads and first KD memory separately.
14. Inject a supported-component/read failure during overwrite; verify the old destination remains and staging/backup cleanup or exact recovery diagnostics.
15. Open the result in an external OME reader; verify image and labels remain understandable while poca/ is ignored.

## Complete Phase-6 changed-file inventory

This inventory separates new files from modified files. CMake files were edited only to register sources, never executed.

### Added

- `poca/docs/POCA_ZARR_EXTENSION.md`
- `poca/src/poca_core/General/EngineObjectAssembly.cpp`
- `poca/src/poca_geometry/Geometry/DetectionSetStorage.cpp`
- `poca/src/poca_geometry/Geometry/ObjectListMeshPersistence.cpp`
- `poca_extra/src/poca_loaderZarrFile/LoaderZarrObject.cpp`
- `poca_extra/src/poca_loaderZarrFile/OmeZarrObjectLoad.cpp`
- `poca_extra/src/poca_loaderZarrFile/OmeZarrObjectLoad.hpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrArrayTests.cpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrDatasetTests.cpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrExtension.hpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrExtensionExport.cpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrExtensionLoad.cpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrFeatureDisplay.cpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrFeatureTests.cpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrFeatures.cpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrFeatures.hpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrLazyTests.cpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrManifest.cpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrMeshDisplay.cpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrMeshExport.cpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrMeshLoad.cpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrMeshTests.cpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrMeshes.hpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrPoints.cpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrPoints.hpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrPointsTests.cpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrSchema.cpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrSchema.hpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrSchemaTests.cpp`
- `poca_extra/src/poca_loaderZarrFile/PocaZarrTestSupport.hpp`
- `poca_extra/src/poca_loaderZarrFile/ZarrSafeNames.cpp`
- `poca_extra/src/poca_loaderZarrFile/ZarrSafeNames.hpp`

### Modified

- `CONTINUITY.md`
- `poca/src/poca_core/CMakeLists.txt`
- `poca/src/poca_core/General/BasicComponent.cpp`
- `poca/src/poca_core/General/BasicComponent.hpp`
- `poca/src/poca_core/General/BasicComponentList.hpp`
- `poca/src/poca_core/General/Engine.hpp`
- `poca/src/poca_core/General/Histogram.hpp`
- `poca/src/poca_core/General/ImagesList.cpp`
- `poca/src/poca_core/General/ImagesList.hpp`
- `poca/src/poca_core/General/MyData.cpp`
- `poca/src/poca_core/General/MyData.hpp`
- `poca/src/poca_core/Interfaces/HistogramInterface.hpp`
- `poca/src/poca_geometry/CMakeLists.txt`
- `poca/src/poca_geometry/Geometry/DetectionSet.cpp`
- `poca/src/poca_geometry/Geometry/DetectionSet.hpp`
- `poca/src/poca_geometry/Geometry/ObjectListMesh.hpp`
- `poca/src/poca_geometry/Geometry/ObjectLists.cpp`
- `poca/src/poca_geometry/Geometry/ObjectLists.hpp`
- `poca_extra/src/poca_loaderZarrFile/CMakeLists.txt`
- `poca_extra/src/poca_loaderZarrFile/EXPORT_README.md`
- `poca_extra/src/poca_loaderZarrFile/LoaderZarr.cpp`
- `poca_extra/src/poca_loaderZarrFile/LoaderZarr.hpp`
- `poca_extra/src/poca_loaderZarrFile/OmeZarrExport.cpp`
- `poca_extra/src/poca_loaderZarrFile/OmeZarrExport.hpp`
- `poca_extra/src/poca_loaderZarrFile/OmeZarrExportCommand.cpp`
- `poca_extra/src/poca_loaderZarrFile/OmeZarrExportGui.cpp`
- `poca_extra/src/poca_loaderZarrFile/OmeZarrExportGuiTests.cpp`
- `poca_extra/src/poca_loaderZarrFile/OmeZarrExportMetadataTests.cpp`
- `poca_extra/src/poca_loaderZarrFile/OmeZarrExportTests.hpp`
- `poca_extra/src/poca_loaderZarrFile/OmeZarrLabelsLoad.cpp`
- `poca_extra/src/poca_loaderZarrFile/OmeZarrLabelsMetadata.cpp`
- `poca_extra/src/poca_loaderZarrFile/README.md`
- `poca_extra/src/poca_loaderZarrFile/ZarrArrayAccess.cpp`
- `poca_extra/src/poca_loaderZarrFile/ZarrArrayAccess.hpp`

Suggested commit: `feat(zarr): persist PoCA points meshes and lazy features`.

No CMake configure/generate, compilation, linking, installation, application executable, Python script, benchmark, or test was run.

## 2026-10-07: finite feature statistics and undefined-value diagnostics

This source-only update supersedes the earlier non-finite-sample rejection policy. It does not change scientific feature equations, neighborhood sizes, geometry criteria, thresholds, floating array payloads, the dataset version, or the Zarr backend ABI.

### Statistics and display

The CPU statistics body was empty, floating CUDA reduction/sort accepted non-finite input, and histogram bin conversion used non-finite values. The CUDA reduction identity initialized its maximum with `numeric_limits<T>::min()`, which is positive for float. These paths could leave reversed or unusable bounds. `eraseBounds()` also used positive `FLT_MIN` for its lower bound.

`ArrayStatistics` retains its fixed `std::array<float, STATS_NB_PARAMS>` and strict five-value vector constructor. Floating reductions now collect finite values into a separate CPU work buffer, calculate mean/population standard deviation with double Welford accumulation, sort only that buffer for median/extrema, and leave the scientific array unchanged. Resident upper-middle even medians and storage averaged-middle even medians retain their respective prior conventions. Integer device routing and the existing CPU integer -1 sentinel comparison remain in place. Negative floating -1 is a scientific value and is counted.

No finite samples means five NaN statistics. `nbElements()` remains the original scientific row count; a native storage sample retains its original sampled count, including non-finite samples. Floating bins omit NaN and both infinities. A resident or sampled feature with no measured finite values and no supplied display interval exposes NaN minimum, maximum, current minimum and current maximum, empty bins/positions, zero step/maxY, and no effective histogram interaction. Filtering such a feature contributes nothing to component selection. Histogram widgets show an unavailable message and disable bounds/LUT/log controls. Rendering hides an unavailable component and maps non-finite values only in temporary shader feature buffers to the existing hidden-value sentinel.

An explicit finite display interval with unavailable measured statistics remains valid for metadata-only images. It does not manufacture statistics. Existing finite constant-feature bounds remain exact for quantitative features; the established storage-image constant interval expansion remains. Explicit recomputation rebuilds statistics and bins, and updates provenance; loading valid saved statistics does not recompute them.

### Persistence and validation

Display and histogram metadata now include `displayAvailable`. Missing flags in older finite metadata mean available. An unavailable display requires four JSON null bounds, five null statistics, empty bins/positions, zero step/maxY, and matching availability. JSON null encodes unavailable measured statistics. Native floating display samples encode NaN as null and infinities as "+inf"/"-inf"; this metadata conversion never touches scientific Zarr arrays.

Validation still rejects reversed statistical bounds, reversed display extents, reversed current bounds, non-finite finite-display bounds, partial/unexpected non-finite statistics, negative standard deviation/bins, malformed bins/count/provenance, and inconsistent histogram/display extents or unavailable states. Failures include persisted statistics and actual min/max/currentMin/currentMax values. Saved histogram restoration is metadata-only with callbacks installed afterward. Legacy display-only stores retain their bounded compatibility sample. Sample unavailability describes the sample, not a claim that every unseen storage value is NaN.

Long complete-export errors use `OmeZarrDiagnosticDialog` in the existing export GUI module: a normal resizable 900 x 600 QDialog, read-only no-wrap QPlainTextEdit with scrollbars and selection, complete-text Copy, and Close. Small errors keep QMessageBox. Export logic remains in commands and preflight. The complete preflight report prints the root dataset name once and groups child index/name, component/ObjectLists entry, feature, and detailed issue.

### OrganoGraph report

`OrganoFeatureDiagnostics` owns one collector per computed sample. Producers aggregate a feature/cause count under a mutex and retain at most five lowest example row indices per cause. No per-NaN string/index archive is created. Result objects and the collection share the collector, so the final report retains computed features even when publication of a sample fails.

Current and legacy computation constructors print the report at completion. Loading saved graph results does not rerun diagnostics. Requested canonical output rows are counted once even when a value is published to both nuclei ObjectLists and DetectionSet or projected to retained cells. Categories identify existing nucleus features, nucleus/cell outputs and their mapped components, organoid-level results, and spatial/global curves. Dataset identities retain original source indices. The summary includes NaN/total/percentage, per-cause counts/example indices, ALL VALUES UNDEFINED highlights, affected datasets/features, and global counts across all computed datasets for each feature. The no-undefined case is a single line. Infinities are counted separately and excluded from display statistics.

Recorded source branches include insufficient neighborhood ellipsoid points, a non-full-dimensional ellipsoid, missing ranked neighbors, no other radius neighbor, non-positive accessible volume, covariance solver failure/non-positive eigenvalue, nuclei outside the valid organoid, insufficient finite reference replicates or zero reference variance, undefined pair-shell scores/domain agreement, empty mesh intensity sampling regions, zero intensity mean, non-finite covariance-root/bounding-box arithmetic, zero normalization denominators, and absent finite whole-organoid summary inputs. Existing warnings and dataset failures remain. Voronoi core geometry keeps its existing zero-valued degenerate cases; this task does not turn them into NaNs. Non-finite intensity voxels still trigger the existing scientific-computation failure rather than silently changing its sampling policy.

Existing source nucleus features have no producing computation in this run. Their NaNs are counted using bounded region reads (65,536 floats) and explicitly report the upstream scientific cause as UNCONFIRMED. Unexpected count/cause mismatches are also printed as UNCONFIRMED. A feature computation that fails before returning a result continues to be described by the existing dataset failure report; the NaN report covers completed sample results, including publication failures.

### Source fixtures: requested case coverage (never executed)

The existing optional manual source-test switches stay OFF by default. `POCA_ZARR_EXPORT_SOURCE_TESTS` registers `PocaZarrNaNTests.cpp` through the existing export source-test menu. `ORGANO_PERSISTENCE_SOURCE_TESTS` registers `OrganoFeatureDiagnosticsTests.cpp` through the existing TestRegistry. `OmeZarrExportGuiTests.cpp` extends the existing GUI fixture.

| Requested cases | Source fixture |
| --- | --- |
| 1, 2, 3: positive, negative, mixed sign | NaNFeatureFixture::statistics |
| 4, 5, 6: one NaN, many NaNs, all NaNs | NaNFeatureFixture::statistics |
| 7, 8, 9: first/last NaN and negative-only extrema | NaNFeatureFixture::statistics |
| 10: finite-only mean/median/population stddev | NaNFeatureFixture::statistics, including [1,2,NaN,5] explicit expectations |
| 11, 12, 13: all five slots, scalar constructor, strict vector constructor | NaNFeatureFixture::statistics |
| 14, 15, 16, 17: ordered finite/mixed and explicit unavailable bounds | NaNFeatureFixture::statistics / display / storage |
| 18: valid unavailable metadata and complete preflight | NaNFeatureFixture::display / preflight |
| 19, 20: reject reversed bounds with actual values | NaNFeatureFixture::display / preflight |
| 21, 22, 23: mixed/all-NaN float Zarr roundtrip and lazy coherent restore | NaNFeatureFixture::roundTrip / storage |
| 24: integer semantics, including unsigned maximum not a -1 sentinel | NaNFeatureFixture::statistics |
| 25, 26: omit finite output, identify one NaN/dataset/feature/count | OrganoDiagnosticsFixture::aggregation |
| 27, 28, 29: separate causes, all-NaN highlight, multiple dataset totals | OrganoDiagnosticsFixture::aggregation |
| 30: scientific values unchanged | OrganoDiagnosticsFixture::aggregation; NaNFeatureFixture::statistics / roundTrip |
| 31: actual neighborhood/covariance/reference producing branches | OrganoDiagnosticsFixture::producingPaths |
| 32: exact concise no-undefined line | OrganoDiagnosticsFixture::noUndefined |
| 33, 34, 35: resizing, scroll/select, complete Copy | checkOmeZarrExportGuiSource |
| 36: root name once with child/feature context | NaNFeatureFixture::preflight |

Additional source cases cover infinities, bounded/parallel-arrival-independent examples, sample counts, uint32 samples beyond exact float integers, metadata-only finite display with unavailable measured statistics, inconsistent null bounds, statistical reversal, histogram/display mismatch, and negative bins.

### Deliberate limits and verification

Floating resident median computation uses an O(finite rows) double work buffer and sort on CPU, even with CUDA available. Native storage samples stay bounded; valid saved metadata stays lazy. Stored scientific NaN values are preserved by the unchanged float-array writer; payload bit preservation is asserted by source fixtures, not established by execution. Statistics/display fields remain float, so values/statistics outside that representation cannot be made finite by this change. No effort is made to repair already corrupted saved bounds or retrospectively invent historical scientific causes.

Static checks inspect declarations/call sites, command routing, ownership, source registration, scientific expressions, diff whitespace, UTF-8/BOM and CRLF. No CMake configure/generate, compilation, linking, installation, application/backend execution, Python/helper executable, test, or benchmark was run. Runtime behavior was not verified.

## 2026-10-08: diagnostic-only load-cost profiling

This update extends the existing PocaZarrLoadReport. No loading optimization, parallelization, extra data cache, feature batching, validation removal, scientific change, additional laziness, persistence-format change, TensorStore setting change, or backend ABI change was implemented. No new files or independent profiler classes were added.

### Current call graph and ownership

LoaderZarr -> PocaZarrDatasetLoader::load -> reconstruct -> root manifest -> sequential loadMultipleObject children (child manifest -> loadObject) -> loadComponents -> loadPocaZarrExtension -> loadPocaZarrMeshes/loadPocaZarrPoints -> loadPocaZarrFeatures -> pocaZarrOpenArray -> ZarrArrayAccess::open -> unchanged C backend. Images use loadImagesList/loadImage -> ZarrImageStorage::open -> ZarrArrayAccess::open -> createStorageBackedZarrImage. Component reconstruction precedes Engine::assembleObject, saved state and PocaZarrCommandState::restore. The multiple-object root is assembled/restored after its children. The summary precedes the existing single root Engine registration and GUI refresh.

Existing unique ownership transfers, immutable indexed geometry and shared backend-array ownership are retained. Lazy callbacks capture their existing arrays/readers by value; they never retain a load-report pointer. The core ObjectListMesh constructor optionally fills a six-entry PersistedConstructionTiming result owned by its caller. It has no dependency on the Zarr plugin. The core-dependent PocaZarrLoadReport::countComponent definition sits in the existing dataset loader so the reporter and array wrapper remain usable by independent adapter mocks without Qt/PoCA geometry dependencies.

### Actual measured boundaries

| Path | Boundaries observed without changing operations |
| --- | --- |
| ObjectListMesh metadata | Existing group JSON parse, mesh properties/counts/path declarations; separate derived triangle-z metadata/display restoration |
| Indexed geometry setup | Existing vertices/faces/vertex-offset/face-offset opens and shape/type validation |
| Indexed geometry values | Existing offset reads and resident vertices/faces reads, allocation, copies and packing; actual generic read time is nested separately |
| Geometry validation | Existing loader offset validation; existing full IndexedMeshGeometry::validate topology/index/range/fan validation in the constructor |
| Mesh display | Existing triangle flattening/ranges and MyArray initialization; existing combined face/vertex normal generation; existing xs/ys/zs, locs/ranges, bounds/centroids and axes installation; render-axis open/read/validation has its own phase |
| Mesh triangle-z | Existing z-vector, histogram initialization/state adoption and MyData creation, separate from persisted object-feature array loading |
| Mesh construction/adoption | Constructor total, measured internal phases, existing legacy id feature creation, feature/selection adoption and immutable backing installation |
| DetectionSet | Existing component metadata, spatial bounding-box metadata, positions array setup, coordinate Histogram/MyData setup, persisted-feature restoration and constructor/feature validation/selection/bookkeeping |
| Persisted features | Existing feature-group/manifest checks; path/name locating and array JSON reads; generic array initialization; display/statistics parse/validate/restore; storage-backed Histogram/MyData and reader setup; legacy bounded samples; full feature/coordinate materialization callback only if normally invoked |
| Images | Existing image metadata parse, reader/native-array initialization, feature/display state, generic image read time, level-0 region reads and full level-0 loading callback if normally invoked |

The constructor's existing per-object vertex and face loops remain interleaved and in the same order. Two clock checkpoints per object separate auxiliary work from triangle work without restructuring either algorithm. Other constructor checkpoints are per phase. When no report is supplied, the optional core clock checkpoints do no work.

### Accounting, counters and output

Every synchronous scope uses steady_clock. Inclusive time is the scope duration; children is the sum of immediate child durations; self/unclassified is max(0, inclusive - children). Grandchildren are not subtracted again. The loader imports the six disjoint constructor intervals as measured children of Mesh construction. That existing constructor path does not read lazy scientific feature arrays. Unaccounted reconstruction time sums self at structural boundaries (reconstruction/aggregate/component/feature/mesh-constructor/image containers) plus time outside the outer reconstruction scope. Coarse explicitly measured leaf phases are accounted work, even when their internal allocation/validation details are not further split. Summary output itself is excluded from the captured total.

Per-component output keeps completion plus three concise lines: identity/cost categories, structural counts/open/read counters, and feature setup costs. Detailed final groups show ObjectListMesh and DetectionSet feature subtotals independently, as well as global feature and generic array costs. Top five meshes and top five DetectionSets retain child index/name and persisted component path/name, total seconds, object/detection count, vertex/face count, and actual restored feature-record count. The rankings store only five entries per kind. There is no per-feature timing log or retained per-feature timing record.

Counts use already parsed counts, existing vector lengths, successful PoCA array initializations and successful region reads. Mesh bytes cover resident vertex/face/offset payloads; render axes remain separately timed and are included in generic array/read bytes. Numeric array identities distinguish temporary readers even when allocator addresses are reused. Distinct arrays-read sets contain only arrays that were actually read, not every opened lazy feature array. Failed initialization attempts remain timed and separately counted; failed reads do not increment successful-read bytes/counters.

Feature records visited are persisted manifest entries. Feature arrays initialized are ordinary persisted feature arrays; native coordinate arrays are reported separately. Component/ranking feature counts include generated/native records such as triangle-z and coordinates. Feature records/storage-backed/materialized counts remain completion snapshots across all component MyData, including image intensity and native/derived records. Arrays with values read is distinct from full materialization: legacy samples and region reads count as reads while the histogram can remain storage-backed. Feature/coordinate materialization callbacks and image level-0 full-load/region-read timers observe existing calls only. Display/statistics setup totals include reuse by coordinate features; statistics metadata reads count actual parser invocations (including repeated existing validation), not filesystem reads. Embedded feature-manifest JSON parsing is part of the enclosing component metadata phase; the feature group/manifest phase covers subsequent existing checks/group reads.

The lazy-state summary inspects hasPixels, valuesUnloaded, nbObjects, meshesMaterialized and hasSpatialIndex only. It never calls getMeshes, getKdTree, getOriginalData or pixel getters. Normal persisted DetectionSet initialization does not construct a KD-tree, so zero observed indices prints 0.000 s. If an installed command unexpectedly constructs an index, its flag is reported and the tree construction is explicitly UNCLASSIFIED rather than assigned a fabricated duration. Report binding is scoped/thread_local, restored on exit, and unavailable after reconstruction; later lazy reads cannot update an expired reporter.

The build label is Debug for _DEBUG or absence of NDEBUG, otherwise Release, following the existing image diagnostic convention. One cache note states that filesystem caching affects timings; no cache state is probed or changed.

### Static pre-audit for a later threading phase

Independent child paths/manifests, geometry vectors, feature maps and read-only array handles have local ownership. Component metadata/array setup, resident geometry reads/validation and CPU display preparation are candidates for a future worker stage, after auditing their constructor and image-sampling dependencies. Whole-child loadObject is not currently established as worker-safe.

- MyObject.cpp increments the header-static poca::core::NbObjects counter from Misc.h without synchronization. Creating MyObjects concurrently would race on that counter in the translation unit.
- Engine::instance reads/writes application properties and a singleton pointer. assembleObject installs display/component/object commands through shared plugin instances. registerObject mutates m_datasets and m_currentDataset; children are currently unregistered, and only the root is registered.
- PluginList::addCommands iterates shared Qt plugin instances with no synchronization. Command constructors may read Engine global parameters; a representative DetectionSet display constructor does so. The Engine/plugin registry, parameters, mediator, macro recorder, Python interpreter and OpenGL helper singletons have no worker-safety guarantee from this audit. Command-state restore delegates into installed plugins, so each participating plugin must be qualified separately.
- GUI/widget creation, notifications, dataset registration/publication and all Qt/OpenGL resource work must remain on the GUI/main thread with the required GL context. Pending plugin/command installation and object creation should also stay there until separately qualified; QObject thread affinity and command lifetime must be respected.
- ZarrImageStorage's existing Release readRegion diagnostic path consults Engine verbose settings; therefore even compatibility image sampling cannot simply be assumed independent of shared application state.
- PocaZarrLoadReport maps, active scope, dataset identity, output and rankings are unsynchronized. thread_local observation binds a reporter; it does not make sharing one report safe. A future worker design would need independent reports and controlled merging/output. The diagnostic array identity counter alone is atomic.
- Backend API version 1 explicitly documents synchronous operations, operation-local read state and no global error state; close must not overlap an operation. BackendOpen uses independent local handles/specs; BackendIO uses operation-local transforms/decode buffers and const handle metadata. This supports concurrent independent reads according to current API usage, with retained handles and separate output buffers. It is a static assessment, not a runtime guarantee for the installed TensorStore/backend binary. Overlapping writes/external metadata mutation remain unsupported without caller coordination.
- No additional mutable global cache was identified in the inspected open/read implementation; Palette::getStaticLutPtr creates an owned palette. This is not an exhaustive audit of every installed plugin's caches.

No threading or synchronization changes were implemented.

### Source fixtures and static checks

The existing POCA_ZARR_EXPORT_SOURCE_TESTS flag stays OFF by default. A separate manual TestRegistry action calls checkPocaZarrLoadReportSource in the existing PocaZarrLazyTests.cpp; no new framework/file is needed. The independent backend adapter mock target now includes the core-independent report implementation. Its CMake source list was edited, never configured or built.

| Requested case | Source coverage, not executed |
| --- | --- |
| 1: nested total/subphase/self | Exact recordTiming reducer arithmetic plus actual parent/child/grandchild/sibling scopes; idempotent stop and nested reporter restoration |
| 2: counter aggregation | Multiple component snapshots and component-type feature totals in checkPocaZarrLoadReportSource |
| 3: mesh subphases | Synthetic scope recording and real mesh round-trip instrumentation/counts in PocaZarrMeshTests.cpp |
| 4: DetectionSet subphases | Synthetic scopes and positions/feature setup counts in PocaZarrPointsTests.cpp |
| 5: open/read distinction | Synthetic unique-array/region counts and adapter mock real opens/readRaw/post-load read checks |
| 6: lazy feature counting | Throwing lazy readers with saved state; flags and counts observed without reads |
| 7: CGAL stays lazy | Real mesh snapshot/round-trip diagnostics assert zero materialized CGAL collections |
| 8: KD-tree stays lazy | Storage-backed snapshots assert zero; an explicitly indexed manual fixture is observed as one |
| 9: bounded top-N | Unsorted synthetic costs assert descending top five and structural snapshots |
| 10: build configuration | Compile-time conditional expected Debug/Release label in the source fixture |

Existing characteristic source-test output assertions were updated for the detailed lazy-state line. No fixture was run. Static review checks scopes/definitions/call sites, optional constructor compatibility, ownership, source registration, unchanged loops/backend ABI/scientific payloads, quoting/whitespace and UTF-8/BOM/CRLF.

### Remaining timing limits

PoCA array initialization aggregates backend open plus rank/shape/type/name queries; it cannot distinguish individual TensorStore metadata/file/cache/decode operations or physical disk bytes. Generic read bytes are logical requested payload bytes, not chunk traffic. Geometry-value phases include allocation/packing, normal generation stays one combined face/vertex phase, auxiliary geometry stays a combined bounds/centroids/coordinate/range phase, and axes include their existing validation. Completion flags do not record transient materialization followed by release, and the LoaderZarr initial format/version probe, post-summary GUI/Engine registration and later lazy reads are outside this reconstruction profiler. Failed root reconstruction retains existing exception behavior and has no success summary; completed failure scopes still record durations. Debug per-image native-region logging remains the pre-existing behavior and can affect comparisons. Profiler overhead itself has not been benchmarked.

### Files modified (no new files)

- poca/src/poca_geometry/Geometry/ObjectListMesh.hpp
- poca/src/poca_geometry/Geometry/ObjectListMeshPersistence.cpp
- poca_extra/src/poca_loaderZarrFile/PocaZarrLoadReport.hpp
- poca_extra/src/poca_loaderZarrFile/PocaZarrLoadReport.cpp
- poca_extra/src/poca_loaderZarrFile/PocaZarrDatasetLoader.cpp
- poca_extra/src/poca_loaderZarrFile/PocaZarrExtensionLoad.cpp
- poca_extra/src/poca_loaderZarrFile/PocaZarrMeshLoad.cpp
- poca_extra/src/poca_loaderZarrFile/PocaZarrPoints.cpp
- poca_extra/src/poca_loaderZarrFile/PocaZarrFeatures.hpp
- poca_extra/src/poca_loaderZarrFile/PocaZarrFeatures.cpp
- poca_extra/src/poca_loaderZarrFile/PocaZarrFeatureDisplay.cpp
- poca_extra/src/poca_loaderZarrFile/ZarrArrayAccess.cpp
- poca_extra/src/poca_loaderZarrFile/ZarrImageFactory.hpp
- poca_extra/src/poca_loaderZarrFile/ZarrImageStorage.cpp
- poca_extra/src/poca_loaderZarrFile/PocaZarrLazyTests.cpp
- poca_extra/src/poca_loaderZarrFile/PocaZarrMeshTests.cpp
- poca_extra/src/poca_loaderZarrFile/PocaZarrPointsTests.cpp
- poca_extra/src/poca_loaderZarrFile/PocaZarrCharacteristicTests.cpp
- poca_extra/src/poca_loaderZarrFile/OmeZarrExportMetadataTests.cpp
- poca_extra/src/poca_loaderZarrFile/OmeZarrExportTests.hpp
- poca_extra/src/poca_loaderZarrFile/backend_tests/ZarrArrayAccessMockTests.cpp
- poca_extra/src/poca_loaderZarrFile/backend_tests/CMakeLists.txt
- poca/docs/POCA_ZARR_EXTENSION.md
- CONTINUITY.md

No CMake configure/generate, compilation, linking, installation, PoCA/backend execution, Python/helper executable, benchmark or test was run. Runtime behavior was not verified.

Suggested commit: feat(zarr): add detailed diagnostic load-cost profiling.

## Reusing persisted features, normals and mesh certification (2026-10-08)

Source-only implementation. No configure/generate, compilation, application/backend execution, Python/helper executable, test or benchmark was run. Runtime behavior was not verified. This section supersedes the preceding descriptions of eager feature-array initialization and unconditional indexed topology validation/normal generation.

### Lightweight features and first opening

Previously `loadPocaZarrFeatures` called `pocaZarrOpenArray` and read every feature's `zarr.json` before installing a value callback. The callback was lazy about values, but dtype/shape/dimension/name verification opened every physical array during reconstruction. The small manifest additions below remove that dependency for new stores:

```json
{
  "name": "measurement/name",
  "path": "object_features/f000001",
  "array_descriptor": {
    "version": 1,
    "dtype": "float32",
    "shape": [73],
    "dimension_names": ["object"]
  },
  "display": "existing complete display/histogram metadata"
}
```

The physical ordinal key above is illustrative; `path` remains authoritative. Point features use `point`, object features use `object`, image extras use `element`; extras with independent lengths retain their existing per-record `count`. Logical feature names, lengths, original histogram statistics, bins, interaction, LUT and deferred log choice are restored into normal `Histogram<float>`/`MyData` immediately. JSON contains descriptors and existing bounded display metadata, never a second scientific value vector.

`ZarrArrayAccess::deferred` extends the existing array boundary, with metadata-only getters, an `opened()` flag and an owned factory. It makes no backend call and reads no array JSON. First `readRaw`/`read` opens through the normal adapter, verifies physical shape/type/element width/dimension names/original logical name, then publishes one retained read-only handle with `call_once`. Failed opening publishes nothing and propagates the contextual failure; a later explicit read can retry. Histogram region/full materialization checks remain unchanged. Copies share the descriptor/handle through their existing value callbacks but own independent histogram/display state. No callback retains a report pointer or Qt object.

Manifest name/path uniqueness, reserved names, path containment/link checks, descriptor contract and saved histogram consistency still run during restoration. Physical existence/metadata/chunk corruption can now be detected later at first actual access. Selecting a feature for a workflow that actually requests values opens it; listing/counting and reading saved statistics do not. A new descriptor with legacy display metadata lacking the complete saved histogram still needs the existing bounded compatibility sample, so such a partial payload can open/read during restoration.

Missing `array_descriptor` retains the old eager metadata/name check. Complete old histogram state remains value-lazy; old display-only state uses the existing finite bounded sample. Present malformed/unsupported descriptors reject rather than silently switching to a different interpretation. Missing physical storage on the descriptor path is a first-read error. Coordinate/image readers and generic command-state arrays retain their previous opening policy. No backend C ABI change was made; only PoCA wrapper signatures gained an optional explicit feature-role argument so post-load opens retain diagnostic identity.

### Normal arrays and adoption

Mesh groups add ordinary Zarr numeric arrays:

| Relative array | Shape | Dtype | Dimensions | Meaning |
| --- | --- | --- | --- | --- |
| `vertex_normals` | `[vertex_count, 3]` | `float32` | `vertex, coordinate` | One cached display normal per indexed vertex, in persisted vertex order |
| `face_normals` | `[face_count, 3]` | `float32` | `face, coordinate` | One cached display normal per triangle, in persisted face order |

```json
"normals": {"version": 1, "vertex": "vertex_normals", "face": "face_normals"}
```

Normals are not placed in JSON and are not expanded into three copies per triangle. Float32 is the existing `Vec3mf` cached/rendering representation. Export copies the immutable backing's cached vertex/face buffers, or reads the corresponding existing CGAL normal properties for unchanged resident meshes. The two arrays use normal row-chunk writers and explicit component packing, without assuming `Vec3mf` binary layout. Untouched indexed export invokes neither normal generation nor CGAL materialization.

The loader opens/reads these required numeric arrays eagerly and passes `PersistedNormals` into the existing persistence constructor. Counts, shapes, domains and finite components are checked; valid buffers are moved into normal mesh ownership without normalization or recomputation. Existing `use_vertex_normals` selection still selects vertex versus face normals and expands them through the same rendering code/equations.

An absent normal descriptor recomputes both buffers using the existing indexed/CGAL most-visible-normal equations. Present malformed paths/version, missing referenced arrays, wrong sizes/type/domain or nonfinite components reject the component; corruption is not hidden by a recomputation fallback. A mutable CGAL reference invalidates both indexed normal buffers. Re-export then regenerates normals for the fresh geometry snapshot instead of trusting possibly stale CGAL normal properties. A copy of such a mutable source inherits invalidation. No renderer redesign or scientific feature/geometry value changes were introduced.

### Validation model and persisted contract

`ObjectListMesh::MeshValidationLevel` has `Unknown`, `IndexedValidated` and `CgalValidated`. The state belongs to core `IndexedMeshGeometry`, rather than a Zarr-only flag. Resident CGAL collections track complete per-object inspection results; the collection is CGAL-certified only when every object passed and no mutable reference has escaped.

```json
"topology_validation": {
  "level": "indexed",
  "validator_version": 1,
  "contract": "poca-indexed-topology"
}
```

The stronger form uses `level: "cgal"` and `contract: "poca-cgal-meshrepair"`, also version 1. These fields are PoCA-specific extension metadata, not OME-NGFF standard fields. They are optional additions to the existing payload/dataset versions; no migration or backend format change is required. Missing, incomplete, unknown-level, unknown-contract or incompatible-version certification becomes `Unknown` and runs current indexed validation.

**Indexed validation is not CGAL validation.** `validate()` preserves the existing index/directed-edge/orientation-conflict/connected-vertex-fan checks using maps/sets, after linear storage checks, and records `IndexedValidated` only after success. It does not certify closedness, global connectedness, degeneracy, self-intersections, outward orientation or positive bounded volume. `fromMeshes` normally records indexed validation of the exact converted snapshot; when the owner supplies already-established certification for the exact unchanged inputs, conversion preserves it after linear safety checks. Mere conversion does not certify CGAL semantics.

`validateCgal()` explicitly runs indexed validation, materializes a temporary collection and requires `MeshRepair::isStrictlyValid(MeshRepair::inspect(mesh))` for every object before recording `CgalValidated`. The strict version-1 contract requires nonempty finite valid triangular polygon meshes, no degenerate triangles, closed/no-border topology, one connected component per object, no self-intersections/nonmanifold vertices, finite vertex/face normals, completed outward orientation/bounded-volume checks and finite positive volume, with no inspection errors. Existing CGAL/MeshRepair equations, thresholds, repair choices and acceptance checks are unchanged.

The existing object mesh-quality command now uses `ObjectListMesh::inspectMesh` and const geometry access: its full inspection establishes collection certification once every object passes. The repair command retains the complete successful before/after inspection for each exact accepted output, passes those inspections to the existing mesh constructor, and records the stronger state only when construction does no orientation/repair/remeshing. Rejected or partial results do not grant CGAL certification. This reuses completed repair inspections rather than repeating the heavy contract at persistence time. Inspections of rescaled/transformed temporary meshes elsewhere do not certify the original geometry.

### Certified loading and remaining linear checks

Required geometry array shape/dtype/domain and object/render index limits are checked by the existing loader. Offsets still have O+1 entries, start at zero, are non-decreasing and end at the declared totals. The constructor always runs `validateStorage`: nonempty per-object vertex intervals, valid face intervals, finite float-renderable coordinates, each face index inside its object's vertex interval (therefore below the total vertex count), and distinct triangle indices. Normal array shape/count/finite checks remain separate. These passes allocate no adjacency structures, maps, sets or fan graphs.

Compatible indexed or CGAL certification skips the expensive indexed topology pass. Restoring the CGAL label does not construct a `Surface_mesh` or re-run CGAL. Unknown certification runs the unchanged full indexed contract and records its current level. The optimized path remains resident required geometry/display data, lazy analysis CGAL cache, and unopened scientific feature arrays. Existing bounds, centroids, triangles, rendering axes, derived triangle-z and normal MyData ownership are retained.

### Mutation and certification lifetime audit

| Actual mutation boundary | Certification / normal policy |
| --- | --- |
| `IndexedMeshGeometry::setVertices`, `setFaces`, `setVertexOffsets`, `setFaceOffsets` | Private arrays replaced by value; certification becomes `Unknown`. No mutable array getter is exposed. New object adoption recomputes normals unless explicitly supplied valid persisted buffers. |
| Nonconst `ObjectListMesh::getMeshes` | Materializes if needed, drops indexed backing and normal buffers, clears inspections, permanently marks escaped mutable authority for this instance. |
| `remesh`, `subdivide` | Already enter through mutable `getMeshes`, so topology changes invalidate reusable certification and persisted normals before the operation. |
| External smoothing/scaling/transforms/repair via `getMeshes` | Same conservative invalidation, including retained references mutated again after export. Fresh export snapshot is indexed-validated and its normals recomputed. |
| Normal import/generation/orientation/remeshing constructors | Start uncertified unless exact complete inspections were supplied and no geometry-changing processing occurs. Conversion/export establishes indexed state, never infers CGAL from existence. |
| Full quality check or accepted unchanged repair output | Can establish strict CGAL state for the owned exact geometry. Escaped mutable objects require a newly owned/revalidated result for reusable certification. |
| Immutable lazy copy / const analysis materialization | Geometry unchanged; backing/cached normals/provenance retained. Dirty-source copies inherit invalidation rather than resurrecting stale inspection/property data. |

Even pure coordinate changes conservatively clear both levels: indexed version 1 includes finite/renderable coordinates, while the strict CGAL contract also depends on geometry, normals, self-intersection and volume. Setters invalidate before assignment, so a failed write cannot leave a false certificate. Filtering/import constructs a new owned object; no arbitrary inherited source certificate is attached.

No fingerprint was added. Provenance assumes the manifest and exact serialized geometry remain together under the existing export staging transaction. Same-shape in-bounds externally edited geometry/normals can evade these cheap checks; they are storage safety checks, not an authenticity or external-modification detector. External writers/edits must omit/invalidate certification to request full current validation. Normals are checked for shape and finiteness, not rederived to prove correspondence, which would defeat the optimization.

### Aggregate profiler changes

Existing detailed summaries, direct-child/self timing, top-five lists and metadata-only lazy-state counts remain. Added counts distinguish descriptors restored/still unopened, physical feature opening during loading/on demand, legacy eager opening, on-demand value read calls/bytes, cheap mesh checks, executed/skipped indexed validation, restored Unknown/IndexedValidated/CgalValidated provenance, restored/recomputed normals and vertex/face normal bytes read. Existing physical array/value counters still describe actual synchronous backend operations.

`FeatureActivity` is a small shared atomic diagnostic record inside `PocaZarrLoadReport`; callbacks retain only this record, never the report/output/scopes. `count()` can inspect retained demand activity while a report still exists, and `featureActivity()` permits an explicit retained snapshot after report destruction. No automatic per-feature/on-demand printing is added. The initial success summary is a snapshot, normally showing zero post-load demand; it does not update retroactively. First demand reads include opening time, so demand-open and demand-read inclusive totals overlap. Demand timers are flat lifetime aggregates outside the reconstruction scope tree; their self field equals the flat total. Successful opens/reads contribute totals; failures propagate without fabricated success counters.

Added timings cover mesh provenance, linear storage checks, normal-array reads/adoption, and on-demand feature opening/value reads. `Mesh topology validation` is zero-duration work when skipped, while the separate skip counter explains why. Existing normal-generation/adoption timing remains `Mesh display normals`, interpreted alongside restored/recomputed counts. Required normal reads count in generic Zarr traffic; bytes remain logical payload bytes, not decoded chunk/disk traffic. No measurement has been performed.

### Source fixture coverage (34 requested cases, not run)

| Cases | Existing source fixture coverage |
| --- | --- |
| 1-3 | `PocaZarrFeatureTests`: complete descriptor restoration, named MyData/count/mean/bins/bounds before opening, zero physical opens/reads; point/mesh fixtures check visible feature names and metadata-only reconstruction. |
| 4-6 | Feature first region read, reuse and exact signed-zero/scientific values; adapter mock checks factory/open counts, regions, descriptor getters and retained handle ownership. |
| 7-8 | Descriptor-less complete histogram opens without value reads; legacy display-only fixture retains sample path; new mesh/points have zero feature opens/reads during reconstruction. |
| 9-12 | Mesh vertex/face normal round-trip; exact restored/recomputed counters, legacy missing normals and unchanged equations. |
| 13-14 | Wrong vertex/face normal row count rejects; nonfinite supplied normal payload rejects. |
| 15-16 | Existing flat/smooth mode selection plus nonplanar distinction; deliberately supplied finite custom buffers survive untouched save/load/save, proving no regeneration. |
| 17-20 | Unknown/imported state, indexed-only promotion, complete strict tetrahedron CGAL promotion; open triangle fails strict contract and retains indexed level. |
| 21-25 | Real numeric stores persist/restore indexed/CGAL contracts; execute versus skip counters; missing certification, future version and unknown contract execute indexed validation. |
| 26-27 | Certified loads still report safety checks; existing certified corruption fixtures reject nonmonotonic offsets, cross-object and out-of-range face indices. |
| 28-30 | Face/vertex/offset setters invalidate; mutable CGAL access/subdivision invalidate; revalidation regains state; complete clean/repaired MeshRepair inspection survives unchanged output adoption; dirty copies retain invalidation. |
| 31-33 | Feature descriptor/open/read/demand counters and demand bytes; executed/skipped validation and restored/recomputed normals plus byte counts. |
| 34 | Existing throwing-reader lazy-state counting and real mesh/point completion snapshots stay nonmaterializing. |

Additional fixtures check invalid lightweight metadata, physical rank/count/type/domain/original-name mismatch deferred until first read, missing array first-read error/retry, reporter destruction, CGAL stronger contract failure, nonfinite normals and mutable-source normal-property invalidation. Existing manual flags/source registrations stay unchanged and OFF by default; the independent mock source project is neither configured nor built. OrganoGraph's existing persistence fixture uses the new invalidating geometry setters.

### Static audit and limits

Source review covers declarations/definitions/default-call compatibility, every indexed geometry use site, both repositories' mutable mesh access paths, command dispatch/ownership, existing registrations, complete CGAL acceptance, preserved indexed/normal equations and no backend ABI/scientific feature/rendering equation/child loading changes. No new files, manager classes, libraries or GUI actions were added. Static verification found 21 modified tracked files (no new files), all strict UTF-8/consistent CRLF with HEAD BOM status retained; 19 balanced C++ delimiter streams and 50 resolved quoted includes. Existing source/test registrations and default-call compatibility were inspected. git diff --check and added-line quote/include-artifact checks were clean. Normal-generation/materialization algorithms match HEAD after private-member renaming. These are source-only checks.

Normal buffers are copied temporarily for export, increasing peak memory by the cached float normal payload; mesh geometry/display remains resident. Missing normal metadata still uses the original CPU algorithm. Physical feature failures occur at demand time. Certification has no external tamper detector. Escaped mutable references conservatively prevent reusing certification and require fresh export normals. Existing low-level geometry mutation/render rebuilding conventions remain; no new renderer lifecycle was introduced. Validation-state updates/shared mesh inspection and other core/plugin state are not qualified for concurrent reconstruction.

The deferred handle initialization and diagnostic counters are individually synchronized, and required data/descriptor ownership stays local/shared immutable, which helps a later worker phase. Whole-child MyObject loading is still not worker-safe: the preceding NbObjects, Engine/plugin command installation, Qt/OpenGL, image diagnostic and report-map audit remains applicable. No MyMultipleObject parallelism, thread pool, general cache, compression/chunk redesign or GPU normal calculation was implemented.

Implementation files: existing ObjectListMesh header/main/indexed/persistence files; ObjectListBasicCommands; ZarrArrayAccess header/implementation; PocaZarrSchema header/implementation; PocaZarrFeatures; PocaZarrMeshExport/Load; PocaZarrLoadReport header/implementation. Fixture files: PocaZarrFeatureTests, PocaZarrMeshTests, PocaZarrPointsTests, backend_tests/ZarrArrayAccessMockTests and OrganoGraphPersistenceTests. Documentation: this file and CONTINUITY.md.

Suggested commit: perf(zarr): defer feature opening and reuse persisted mesh normals and validation.
