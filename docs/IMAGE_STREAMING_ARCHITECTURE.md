# Phase 5: regional image streaming

Source implementation dated 2026-10-02. Compilation, shader validation and runtime performance are **UNCONFIRMED**. No CMake configure/generate, compilation, linking, installation, application executable, Python script, benchmark, or test was run.

## Why whole-level rendering was insufficient

Choosing a stored pyramid level still allocates its entire texture. A 64 x 64 x 1937 XY-only coarse level retains pathological Z depth; hundreds of individually reasonable textures also exceed aggregate memory. Deep zoom previously could materialize a complete fine level when only a small region was visible. Phase 5 separates the scientific image from the resident display representation.

Scientific dimensions, pixels(), data(), getImage(), level-zero reload, existing analyses, segmentation, meshes, calibration and exports retain their meaning. Display state belongs to ImageDisplayCommand and its camera's LodUpdateManager. No TensorStore, OME/NGFF metadata, backend chunks or storage paths enter the renderer. The backend ABI, import selection, installed binaries and C:/tsb / C:/tsbd are untouched.

## Request and ownership contracts

ImageLodRequest now carries source level and dimensions, Region3D, target texture dimensions, physical resident bounds, scientific image bounds, frame, reduction mode, preview/detail intent, priority, visibility, resident-source identity, version, checked texture bytes, reader scratch and peak preparation estimate. Target dimensions describe display samples; sourceRegion describes storage voxels. Those values are independent.

Workers capture the scientific ImageInterface pointer and an immutable request. Cancellation is checked between fixed storage tiles. The request version and full target identity prevent old ROI/frame/mode results from becoming current. Command destruction cancels and waits for in-flight readers; ordinary view changes cancel without waiting. Image<T> clears commands before its callback/mutex members and scientific histogram are destroyed. Native storage callbacks execute outside the image metadata mutex, followed by a revision check. Resident region copying remains protected against pixel release.

ImageLodReady owns both its prepared payload and a shared RAII memory reservation. Draining the queue transfers that ownership; it does not release accounting. Consumers must retain the ready object/reservation while retaining its payload. The reservation releases when the last ready owner dies. A shared camera manager lifetime avoids dereferencing a destroyed camera.

Camera frame completion performs upload draining after all displayed families have registered visibility. Parent render contexts mark streaming as handled to avoid duplicate child planning. Standalone image commands use the same queue. All GL calls, bindless residency and eviction remain on the render thread.

## Conservative view and guard calculation

ImageViewRegion intersects six homogeneous frustum planes, six local image planes and optional six crop planes. Crop planes transform through the child's local-to-parent matrix. Triple-plane intersections recover the clipped convex polytope, including perspective, orthographic, rotated child, near-plane and camera-inside cases. A positive-vertex plane test rejects definitely offscreen boxes. Numerically inconclusive intersections conservatively enlarge to the image bbox.

The projected clipped footprint drives quality and priority. Bounds map into the chosen native level using actual dimensions and optional stored origin/spacing relative to scientific calibration and the image bbox. Floor/ceil and subtractive bounds checks produce a valid Region3D.

A detail region expands each axis by 25%, aligns to 16-voxel boundaries and clamps to source edges. Its reusable safe interior excludes 10% margins except at image boundaries. Active texture pitch tolerates a 1.25 coarsening / 0.5 refinement interval. Pending targets reuse the guard region, preventing small-pan version churn. A view outside the safe interior produces a new request.

## Regional preparation and progressive display

The planner prefers the coarsest native level retaining approximately 1.5 source samples per visible pixel. Deep zoom can select level zero. Resident pixels take precedence over stored/native data through a non-materializing readResidentRegion seam; existing full-resolution API callback precedence remains unchanged. Settled native selection honors pyramidalRenderingEnabled while the initial coarse preview remains bounded. Native-capable sources use readNativePyramidRegion; scientific level-zero sources use readFullResolutionRegion. Display tiles are at most 64 x 64 x 16 voxels. No regional native request calls getOrCreatePyramidLevel or materializes a whole native level.

The initial tier is a whole-image preview bounded to 64 per axis. Detail uses one guarded regional texture, bounded to 512 per axis at rest or 128 while interacting. Both are further constrained by GL_MAX_3D_TEXTURE_SIZE, per-image share and byte budgets. Small resident images of at most 1 MiB use a single direct full-resolution preview/copy, with no redundant detail tier.

Target pitch considers stored physical sampling where present. Independent axis caps and CPU/GPU byte admission reduce pathological Z depth even if every native pyramid level preserves all 1937 slices. The source remains unchanged. Two-dimensional images keep depth one. Frame requests read one scientific plane; positive native levels are used only when their Z count, origin and spacing preserve that plane.

RAW defaults to MIP: every contributing source voxel enters its display bin, so Z reduction does not skip sparse bright voxels. Average and Nearest remain available. Integral RAW Majority counts one bounded bin at a time, retaining first-source order for ties; oversized vote buffers fail explicitly. Floating Majority follows the existing nearest-style policy. LABEL display always uses nearest and never averages IDs, including when a request says Average. Scientific LABEL pixels/native levels/export policies are unaffected.

INT32 RAW display converts explicitly into float32 before R32F upload, retaining signed intensity rather than normalized integer pixel transfer. Positive INT32 label IDs use integer textures; nonpositive values remain display background. Scientific int32 data is unchanged. Float32 display/palette arithmetic cannot represent every large int32/uint32 value exactly; exact IDs remain in integer source/label textures.

A new GL texture and resident handle are created before old handles/textures are released. Admission includes simultaneous old/new bytes. Failed reads, canceled requests, refused admission and GL upload/residency errors retain the previous valid representation. A changed preview replaces the previous preview/detail only after success.

## Texture coordinate and ray mapping

Scientific normalized ray coordinates map to local physical position, then into detail bounds when inside, otherwise into preview bounds. No regional texture is stretched across the full image bbox. Single-array, single-label and multi-volume shaders share image_stream_sampling.glsl. ImageDescriptorGPU adds resident/preview bounds, handles and flags with checked 304-byte std430 stride and a 208-byte regional-field offset.

Bindless texture parameters stay fixed at GL_NEAREST. RAW linear interpolation is implemented in the shared sampler using eight clamped texel fetches; nearest settings and LABEL sampling remain nearest. Frame plane position uses scientific depth and calibrated bbox, independently of resident texture depth. Multi-frame descriptors carry each image's frame.

Ray sample counts derive from resident resolution and physical extent rather than full scientific dimensions; interactive counts cap at 192 and remain constrained by existing quality settings. This is approximate volume rendering, requiring visual validation for sparse signals, ROI seams and orientation changes.

Deploy the **complete shaders directory**, including image_stream_sampling.glsl, using the existing shader deployment workflow. ShaderSource.hpp expands relative fragment includes before the existing shader compiler. No deployment/build/install command was executed here.

## Budgets and residency

Defaults are centralized in ImageStreamPolicy.hpp, with no new preferences UI:

| Policy | Default |
| --- | --- |
| Volume textures per camera | 512 MiB |
| CPU preparing + ready, shared by managers | 256 MiB |
| One texture | 64 MiB |
| Reader/reducer tile scratch allowance | 1 MiB, plus declared adapter scratch |
| Upload drain | 4 results / nominal 16 MiB per frame |
| Preview / resting detail / interactive detail edge | 64 / 512 / 128 |
| Offscreen grace | 12 rendered frames |
| Guard / safe margin | 25% / 10% |

On NVX-capable drivers the camera lowers its GPU budget to at most one eighth of reported dedicated VRAM. Otherwise the portable source-configured budget applies. Each image's planned tier share is conservatively divided among the current/previous visible image count. Admission leaves replacement headroom of min(64 MiB, budget/4); transient and steady totals are both checked.

Visible/pinned images are retained while old/offscreen alternatives exist. Budget pressure evicts oldest invisible entries immediately; ordinary invisibility retains textures for 12 frames, then deletes preview/detail handles/textures while retaining command, LUT and feature state. Follow-up frames advance grace without queuing offscreen reads. Returning images obtain a preview before refinement.

Workers reserve the estimated peak before allocation or storage reads. Estimates include two typed output copies, Average accumulators, fixed tile scratch, declared reader scratch and bounded legacy whole-level copies. Ready transition releases temporary scratch and retains final bytes. Overflow checks protect dimensions, voxel counts, byte sums and region bounds. Unknown or oversized readers/refinements are refused. Preparation verifies the declared estimate before allocating its buffers; a released resident input cancels and requires replanning with storage scratch.

One result exceeding the nominal 16 MiB upload throttle is allowed alone to guarantee progress, but still respects the 64 MiB texture and aggregate residency budgets. GPU accounting covers volume textures, not LUT/feature textures, picking FBOs, path-tracing buffers, driver staging or other application geometry. CPU accounting covers display buffers/declared reader scratch, not preexisting scientific arrays or backend-internal codec/chunk caches.

## MyMultipleObject, picking and statistics

RAW and LABEL visibility is checked before createDisplay, including child model and crop transforms. Initialization/request counts are throttled; uninitialized or evicted offscreen children do not receive costly texture work. Hundreds of selected children can keep commands alive without retaining all volume textures.

Picking compares cached child/image identity, ordered hierarchy, selection, 16 transform values and six bbox coordinates. Only a changed snapshot rebuilds picking geometry; FBO resize handling remains. Snapshot scanning is still linear in image count.

Visible overlap grouping remains the existing pairwise connected-component grouping, potentially quadratic when many images overlap. It was left intact for later profiling, keeping this phase focused on regional reads and memory. The ordinary ImagesList renderer retains its 16-image shader array limit with an explicit error instead of array overflow. Existing mixed frame/volume batching and common-bbox assumptions still need manual checks.

Opening Zarr statistics now reads up to 4 x 4 x 4 distributed regions, each at most 8 cubed: at most 32,768 voxels / 128 KiB across five scalar types. It does not retain a huge complete coarsest level. Existing histogram provenance distinguishes measured level-zero values, bounded storage sample and metadata range. Valid OMERO display limits continue to win; single-level RAW/label opening policies from Phases 2/4 remain unchanged.

TIFF continues through the generic reader contract. Its current region adapter can decode complete planes repeatedly; it declares conservative plane scratch and unsafe refinements are refused. Very large TIFF/nonregional readers may be unable to provide even the initial preview within the budget. No TIFF strip/tile I/O rewrite was introduced. A native regional Zarr reader exercises the complete bounded path.

## Source tests and remaining manual coverage

ImageStreamingTests registers CPU checks in the existing Images Tests menu. Sources were added, reviewed and **not executed**.

| Requested cases | Source coverage |
| --- | --- |
| 1-12: offscreen/full/deep zoom, projections, rotation, camera-inside, crop, clamp and guard pans | ImageStreamingGeometryTests.cpp calls the shared geometry helpers |
| 13-15, 19-20, 32, 34, 36: regional/native reads, no materialization, five types, anisotropy, bounds, 2D/scientific invariants | ImageStreamingPreparationTests.cpp calls the real planner/preparer with lazy mock readers |
| 16-18: resident fallback, oversized rejection, logical GL dimension limit | Preparation fixtures; actual GL capability/upload is manual |
| 21-22: MIP bright voxel, RAW Majority and no invented LABEL IDs | Preparation reducer fixtures |
| 23: no offscreen initialization | Geometry source checks plus reviewed visibility-before-initialization wiring; actual GL allocation trace is manual |
| 24-26: grace, eviction order, visible/pinned protection | ImageStreamingBudgetTests.cpp calls real residency policy |
| 27-30: success/failure/cancel leases and newest-version publication | Reservation fixtures, callback destruction-order probe and actual LodUpdateManager worker supersession/failure source |
| 31: old texture survives failed replacement | Admission/failed-payload fixtures; actual GL failure injection is manual |
| 33: regional multi descriptor | Shared layout assertions and mapping fixture; GPU interpretation is manual |
| 35: bounded distributed histogram | Real sampling helper with recorded regions |
| 37: picking dirty/rebuild | Shared snapshot comparison helper; actual buffers/FBO resize are manual |

After the user's own compilation/deployment, validate:

1. **Hundreds of datasets:** overview, rotate and pan; only intersecting children initialize/refine; volume residency settles within the camera budget; picking geometry rebuilds only after selection/layout/bounds changes.
2. **2048 x 2048 x 1937 XY-only native pyramid:** opening stays unloaded, preview/detail Z are bounded; MIP preserves bright sparse slices and rotation remains responsive.
3. **Deep zoom to about 1%:** traces select fine/native level zero with a small padded ROI; no complete native/fine materialization; detail and preview align physically across translated/calibrated/rotated children.
4. **Pan during a blocked/slow read:** the render thread remains responsive, old data stays available, small pans reuse guard bands, newer ROI versions alone publish after completion.
5. **Leave and return to FOV:** rendering stops immediately; budget/grace releases invisible volume textures; returning images show preview then detail.
6. **All scalar types, labels and depth-one/frame views:** unchanged palette/scaleLUT/threshold/gamma/current-frame/selection; nearest labels, linear RAW toggles, signed intensity transfer, label borders and calibrated frames.
7. **Failures and budgets:** injected reader/GL errors retain valid textures, obsolete jobs release CPU reservations, dimensions respect queried GL limits, handles become nonresident before deletion.
8. **Compatibility:** Phase 1-4 native/generated APIs, scientific analyses and RAW/associated-label export still operate on authoritative data; TIFF reads remain within their declared scratch constraints.
9. **Shader variants:** MIP, direct, alpha, isosurface, frame, label and path tracing with preview-only/detail, ROI seams and crop; SSBO descriptor offsets/stride and picking FBO resize.
10. **Profile:** existing PerformanceProfiler categories cover visibility/planning, regional reads, resampling, upload and eviction. Existing lodDebug/debugPyramidalRendering gates plan/upload/budget/eviction diagnostics. Measure CPU ready/preparing peaks, actual VRAM (including other resources), read amplification and frame latency.

## File inventory

New files:

- poca_core/General/ImageStorageSampling.hpp.
- poca_opengl/OpenGL/ImageStreamPolicy.hpp, ImageStreamMemory.hpp, ImageVolumeResidency.hpp, ShaderSource.hpp.
- shaders/image_stream_sampling.glsl.
- poca_imageplugin/ImageViewRegion.hpp/.cpp, ImageRegionalLodPlanner.hpp/.cpp, ImageRegionalLodPreparation.hpp, ImageRegionalMajority.hpp, ImageDisplayStreaming.cpp, ImageRenderMetadata.hpp.
- poca_imageplugin/ImageStreamingTests.hpp/.cpp, ImageStreamingGeometryTests.cpp, ImageStreamingPreparationTests.cpp, ImageStreamingBudgetTests.cpp.
- poca/docs/IMAGE_STREAMING_ARCHITECTURE.md.

Modified files:

- poca_core/CMakeLists.txt, General/Image.hpp, Interfaces/ImageInterface.hpp.
- poca_opengl/CMakeLists.txt, OpenGL/Camera.hpp/.cpp, LodUpdateManager.hpp/.cpp, RenderCommandContext.hpp, Shader.hpp.
- shaders: alpha_blending_all/multi, direct_rendering_all/multi, frame_rendering_all/multi, isosurface_all/multi, maximum_intensity_projection_all/multi, path_tracing_all, label_rendering and frame_label_rendering (.frag).
- poca_imageplugin/CMakeLists.txt, ImageDisplayCommand.hpp/.cpp, ImagePyramidTests.cpp, ImageVolumeLodPreparation.hpp/.cpp, ImagesListCommands.hpp/.cpp, ImagesListMultiObjectDisplayCommand.hpp/.cpp.
- poca_loaderTiffFile/LoaderTiffFile.cpp (scratch declaration only).
- poca_loaderZarrFile/ZarrImageFactory.hpp (bounded opening sample).
- poca/docs/CHUNKED_ARRAY_ARCHITECTURE.md and CONTINUITY.md.

Prefixes: poca_core, poca_opengl and shaders are under poca/src; image/TIFF/Zarr plugins are under poca_extra/src.

Future evolution may replace the one-detail tier with a bounded brick cache or sparse texture implementation behind these request/residency contracts. This phase adds no sparse GPU textures, atlas/page table, remote storage, additional persistence, schema/backend work or dependency.
