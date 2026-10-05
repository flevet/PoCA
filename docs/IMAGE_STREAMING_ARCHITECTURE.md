# Phase 5 / 5.1 / 5.2: regional image streaming

Source implementation dated 2026-10-02. Compilation, shader validation and runtime performance are **UNCONFIRMED**. No CMake configure/generate, compilation, linking, installation, application executable, Python script, benchmark, or test was run.

## Phase 5.2: interaction-stable detail (2026-10-05)

This section supersedes Phase 5.1 pitch-based sampling invalidity, camera-based cancellation/publication, and interactive desired-detail replanning. Runtime visual behavior, responsiveness, Qt timing and GL/shader execution remain **UNCONFIRMED** and require manual validation.

- **Renderable versus optimal:** ImageStreamDetailState separates guarded spatial/source compatibility from pitch quality. A covered texture with matching source image, frame, scalar type, RAW/LABEL role, reduction, resident-source identity and scientific bounds remains renderable even when too coarse or too fine. Explicit scientific invalidation still clears its region. The existing 0.88/1.25 pitch thresholds now request replacement quality only.
- **Interaction reuse:** zoom-in keeps covered detail while it becomes softer; modest dezoom keeps over-resolved detail; rotation/pan recompute the conservative image-space requirement without invalidating merely because MVP changed. The existing 25% ROI guard, 10% safe inset and source-edge exemption are retained. A fixed one-plane frame has no Z guard to inset; matching-frame coverage uses its full plane extent, while wrong-frame identity still invalidates it. Leaving the safe interior selects coherent whole preview, including preview-based ray steps. No additional navigation tier or increased preview size was introduced.
- **Observation versus storage identity:** full MVP/viewport/crop/transform equality caches camera observation and increments a diagnostic camera generation. It does not identify storage data. Manager target equality uses source level, ROI/dimensions/bounds, frame, reduction, resident-source state, preview class and display dimensions/factors; viewGeneration is excluded. ImageStreamTarget performs source compatibility and coverage checks using the pending target's own native-level geometry.
- **Useful work survives:** covered queued/preparing/ready detail survives camera and quality changes, including a newly preferred source level. It finishes before further refinement. New input submits no detail reads; the unchanged 150 ms timer dispatches requestLodUpdate after settle. A materially uncovered ROI, incompatible source/frame/mode, invisibility, explicit scientific invalidation or teardown still retires storage versions. One in-flight reader plus at most one latest queued successor and reservation release remain intact.
- **Publication:** a completed result is rechecked against the latest source/requirement and immutable request geometry before admission or GL allocation. An older camera generation alone cannot reject it. During interaction a replacement cannot coarsen any axis of a covered resident; an already quality-optimal resident also takes precedence. Settled quality may replace an over-resolved detail. Failed/refused upload retains the old texture; successful upload/residency precedes handle replacement.
- **Shared shader contract:** existing hasDetail and streamFlags.x carry resident AND renderable, in single, array and MyMultipleObject paths. No new uniform, std430 field, shader quality policy or descriptor-layout change is needed. All 13 helper consumers were source-reviewed; LABEL sampling remains nearest and RAW reducers are untouched.
- **Diagnostics:** optional lodDebug/debugPyramidalRendering reuse messages report cameraGeneration, storageTargetInvalidated, renderability, optimality, refinement, quality and fallback/keep-detail reason on state transitions. Normal operation does not emit these logs.
- **Limits:** a conservative rotated requirement can legitimately leave a partial safe ROI; preview fallback is then intentional. Useful under-resolved work may publish and require another settled refinement. Budget refusal/eviction and unsupported source limits still apply. No visual/performance, real Qt, GPU publication or shader validation claim follows from source fixtures.

### Phase 5.2 source fixture coverage (not run)

| Requested cases | Source coverage and limits |
| --- | --- |
| 1–3: changed MVP, full-volume rotation, rotation inside guard | ImageStreamingInteractionTests uses real frustum intersection, camera observation, detail/target helpers and descriptor flags |
| 4, 8, 13: rotation/dezoom outside guard, obsolete ROI | Real coverage helpers, deferred submission, result rejection and manager version invalidation; actual timer/GL fallback remains manual |
| 5–7, 9: zoom pitch mismatch, refinement and covered dezoom | Production renderability/quality helper with under/over-resolved and unavailable-quality-plan cases |
| 10–11: wrong frame/source/reduction/type | Semantic stamp variants and prepared-result compatibility checks |
| 12, 14–15: old camera generation, useful in-flight work, bounded queue | ImageStreamingRequestReuseTests holds a real manager worker across 100 camera changes, preserves version, completes useful older-generation data, rejects uncovered results and balances leases |
| 16: atomic replacement | Existing scheduling fixture now preserves valid old detail after a refused upload; replacement policy checks prevent interactive downgrades; real GL commit is source-audited/manual |
| 17, 20: preview precedence and visible MyMultipleObject reuse | Sample-choice and shared descriptor flag fixtures; all three CPU uniform/descriptor writers use renderability; actual multiple-object rendering remains manual |
| 18–19, 24: nearest LABEL, RAW reductions, scientific independence | Existing typed preparation/reduction/resident-edit/frame fixtures retained; new reuse helpers do not mutate image data |
| 21–23: offscreen refinement, GPU/CPU budgets | Existing planner/culling/residency/reservation/version-supersession fixtures retained; new useful/rejected result fixture balances preparing/ready ownership |

### Manual Phase 5.2 acceptance (pending)

A. **Continuous rotation:** with a full/covering detailed ROI, rotate repeatedly and verify no coarse flash. With partial coverage, keep detail until the conservative requirement leaves its safe guard, then verify coherent fallback.
B. **Continuous zoom-in:** covered detail stretches/softens progressively without jumping to the 64-edge preview; settled finer detail replaces it.
C. **Zoom-out:** retain covered over-resolved detail; fallback only at the safe-coverage boundary; after settle, verify the larger appropriate detail.
D. **Rotate and zoom together:** verify no cancellation cascade, no interactive quality downgrade, bounded queued work and latest useful/optimal settled detail.
E. **Hundreds of images / MyMultipleObject:** verify visible detail reuse together with offscreen culling, budgets, eviction/grace, tiny-object overview, labels and stable picking.

Also validate wrong-frame/reduction transitions, all helper shader variants, read/upload rejection, unchanged X/Y/Z scaling and scientific analyses. Source fixtures have not been executed.

### Phase 5.2 changed-file inventory

- CONTINUITY.md
- poca/docs/IMAGE_STREAMING_ARCHITECTURE.md
- poca/src/poca_opengl/OpenGL/LodUpdateManager.cpp
- poca/src/shaders/image_stream_sampling.glsl
- poca_extra/src/poca_imageplugin/CMakeLists.txt
- poca_extra/src/poca_imageplugin/ImageStreamView.hpp
- poca_extra/src/poca_imageplugin/ImageStreamTarget.hpp (new)
- poca_extra/src/poca_imageplugin/ImageDisplayCommand.hpp
- poca_extra/src/poca_imageplugin/ImageDisplayCommand.cpp
- poca_extra/src/poca_imageplugin/ImageDisplayStreamView.cpp
- poca_extra/src/poca_imageplugin/ImageDisplayStreaming.cpp
- poca_extra/src/poca_imageplugin/ImagesListCommands.cpp
- poca_extra/src/poca_imageplugin/ImagesListMultiObjectDisplayCommand.cpp
- poca_extra/src/poca_imageplugin/ImageStreamingTests.hpp
- poca_extra/src/poca_imageplugin/ImageStreamingTests.cpp
- poca_extra/src/poca_imageplugin/ImageStreamingResponsivenessTests.cpp
- poca_extra/src/poca_imageplugin/ImageStreamingSchedulingTests.cpp
- poca_extra/src/poca_imageplugin/ImageStreamingInteractionTests.cpp (new)
- poca_extra/src/poca_imageplugin/ImageStreamingRequestReuseTests.cpp (new)

## Phase 5.1: responsiveness hardening (2026-10-05)

Historical Phase 5.1 description: Phase 5.2 above supersedes its reuse and generation/version policies. Its source I/O, startup, budgets and debounce remain applicable. Runtime behavior, Qt debounce timing, shader execution, GPU publication and performance still require manual validation.

- **Asymmetric pitch hysteresis:** an active pitch up to 1.25 times the desired pitch tolerates zoom-in delay; an active pitch below 0.88 times desired is promptly invalid on dezoom. Both values are centralized in ImageStreamPolicy.
- **Current-view validity:** CPU safe-volume coverage and pitch checks set detailUsableForCurrentView. Residency and validity are independent. The existing shared GLSL hasDetail uniform / streamFlags.x receives resident AND usable, consistently across all 13 single/array/multi fragment variants, including MIP, alpha/direct, frame and LABEL paths. Dezoom disables the central detail patch on the next drawn frame; the complete preview supplies coherent pixels while the old texture remains resident. Ray-step planning follows the sampled preview dimensions/bounds during fallback. Replacement publishes only after successful GL upload, matching request version and view generation.
- **Effective target first:** projected pixels times 1.5 oversampling are capped by 128 during interaction or 512 at rest, preview 64, and GL axis limits before native selection. The planner chooses the coarsest native source sufficient for that regional target. A 900-pixel overview therefore selects a roughly 128 source interactively; deep zoom still selects a regional level zero. Physical calibration and independent Z caps remain intact.
- **Debounce/coalescing:** one restartable 150 ms camera timer covers wheel, drag/pan/rotation/crop/transform gestures and release. Active input updates visibility, the desired target and validity, but submits no new detail preparation. First-preview jobs may start during interaction and survive camera movement. Settle wakes rendering through the existing requestLodUpdate command. Planning is cached by observed view, interaction state, publication and budget share.
- **Generation/version boundary:** each command observes MVP, viewport, crop/transform, scientific bounds, frame, mode and resident-source identity. A changed view invalidates old detail versions before upload, even if the batch submission budget is exhausted. One in-flight reader plus at most one latest queued target is allowed per image; busy images do not block other images' admitted work. Successful but obsolete results release payload/scratch leases before the reader is declared finished. The existing manager version remains the cancellation/publication authority; view generation supplements the render-thread guard.
- **Adaptive source slabs:** the centralized scratch byte target is 4 MiB, reduced if the admitted lease has less available scratch. Slabs prefer complete X rows, complete XY planes, then consecutive Z. A 700 x 700 x 41 uint16 source uses at most 11 useful full-XY slab reads rather than thousands of fixed tiles. TIFF keeps one session across those reads, decoding 41 planes once. Native readers use ordinary ImageInterface regions; no chunk, Zarr or TensorStore knowledge enters rendering. MIP/Average include all contributing voxels; nearest LABEL never averages or invents IDs.
- **TIFF opening/regions:** outOfCore=true creates an unloaded Image with reload, plane, region and request-local session callbacks. One metadata directory traversal validates homogeneous scalar planes without decoding pixels. A CPU-admitted sample reads up to three distributed planes in one session, retaining at most 24,576 sampled values for approximate histogram statistics. Scientific full access remains explicit. Region operations use one handle, seek once, decode requested planes sequentially and copy XY directly to the destination. Adjacent XY reads at the same Z reuse the decoded plane. outOfCore=false retains explicit eager scientific loading.

### TIFF first-display source trace

| Stage | Thread / potential cost | Phase 5.1 behavior |
| --- | --- | --- |
| Loader dispatch / TIFF metadata | Opening caller; file open and O(depth) directory traversal | One validation pass; no full-volume pixels |
| Opening statistics | Opening caller; seek/decode up to three complete planes, bounded sample copies and sample histogram | Shared CPU lease; at most 8192 samples per distinct plane; no full-resolution histogram or generated pyramid |
| Image / ImagesList / image-command/widget setup | GUI; metadata, parameters, histogram bins and ranges | RAW startup constructors/range access do not request scientific pixels |
| First createDisplay | Render thread; capability query, LUT and small feature textures, cube/VAO setup | No synchronous volume preparation; visibility checked before initialization |
| Visible preview planning/admission | Render thread then workers; metadata/frustum planning, queue and shared budget | Camera-independent bounded whole preview gets priority before detail |
| TIFF preview read/resampling | Worker; one open, forward directory walk, one decode per required Z, slab copies and reduction | No reopen per plane/tile; bounded live source/output buffers; Y/Z index arithmetic hoisted; cancellation before/after reads and during reduction |
| Prepared payload copy | Worker; bounded output copy and optional signed RAW float transfer | No scientific-volume memcpy, cache generation or immediate full-pixel release |
| Preview texture upload/first coarse draw | Render thread; bounded GL allocation/upload/bindless residency | Atomic successful publication; no GL on workers; frame completion schedules the draw |
| Regional refinement | Worker after last event settles | One latest guarded ROI; level zero remains regional |

There is no claim that all synchronous work or full-source I/O vanished. TinyTIFF offers the existing whole-plane decode and forward-directory APIs, not arbitrary bounded strip/tile decoding. Three sampled planes and metadata traversal can still delay the opening caller. MIP/Average without a native pyramid need one full source traversal before an exact coarse preview; it now runs asynchronously, bounded in memory, without eagerly loading and discarding scientific pixels. Time to first pixels still depends on the source and decoder. Sampled display bounds/statistics are approximate until explicitly recomputed from scientific data; a uniform sample cannot safely justify rejecting the entire image as constant.

The adapter declares conservative scratch of eight decoded-plane byte sizes. Oversized single planes are explicitly refused at bounded opening/display admission. Backend/decoder internal allocations remain conservatively estimated, not measured. Exact integral Majority still uses bounded per-bin votes and may rewind/redecode TIFF planes for nonmonotonic bin requests; oversized bins fail explicitly. This mode is not the one-pass MIP/Average slab path. Arbitrary scientific region calls each own their session. No native TIFF pyramid, random-access API, new codec/dependency or backend ABI has been invented.

### Phase 5.1 source fixtures (not run)

The existing Images Tests registry now also calls ImageStreamingResponsivenessTests, ImageStreamingSchedulingTests and ImageStreamingTiffTests. TIFF fixtures inject a fake adapter into the real session/factory; they verify control flow and counters, not real TinyTIFF codecs. Scheduling fixtures exercise real manager workers with fake CPU/upload callbacks, not GL. Existing geometry/preparation/budget fixtures remain registered.

| Requested Phase 5.1 cases | Source coverage / runtime limit |
| --- | --- |
| 1-4, 17-19: asymmetric reuse, invalid fine island, residency retained, safe pan / outside pan | Shared production pitch, coverage and sample-choice helpers; GPU visibility is manual |
| 5: preview descriptor selection when validity=false | Shared sample-choice helper and unchanged std430 descriptor flag; GLSL execution is manual |
| 6: successful upload activation only | Real manager failed/successful callback protocol plus source audit of GL commit point; real GL failure injection is manual |
| 7-11, 20: capped native target, 900/128 example, settled refinement, deep zoom/regional level zero, anisotropy | Real planner with unloaded mock native images; original preparation fixtures retain regional read checks |
| 12-14: repeated events/latest target, no interactive detail submission, settle submission | View-generation/submission helper, one-queued-target fixture; actual Qt timer timing is manual |
| 15-16: stale successful result / CPU lease release | Real worker version supersession, no cancellation check in the stale successful mock, forgotten-reader ownership |
| 21-22: nearest LABEL / 2D | Existing typed preparation/reducer/frame tests plus five-type TIFF depth-one fixtures |
| 23-24: TIFF open once / planes not reopened separately | Real region session with fake adapter counters; 700-stack preview uses one session and 41 decodes |
| 25-26: adaptive scratch bounds / useful source reads | Real slab policy and reduced-lease fixture; 700-stack sequential preparation |
| 27-28: scientific pixels and independent full access | Real lazy TIFF factory/five-type region layout, explicit scientific reload, copied callback factory, resident-edit fixtures |
| 29-30: MyMultipleObject offscreen / GPU residency | Existing geometry/culling and real residency policy fixtures; actual multiple-object GL allocation remains manual |

### Manual scenarios A-F (pending)

A. Open a normal 700 x 700 x 41 TIFF: check UI responsiveness, sampled histogram provenance, first coarse pixels, one preview session / 41 decoded planes and no eager full-volume load followed by discard.
B. Zoom continuously: preview or valid resident detail stays responsive; no new detail I/O per wheel tick; one latest settled target refines; genuine deep zoom reaches only a padded level-zero ROI.
C. Strongly dezoom from a small fine ROI: the next frame is coherent whole preview, with no central fine island; larger/coarser replacement becomes active after upload.
D. Rapidly alternate zoom/dezoom: at most one latest queued target per image, obsolete successful reads never publish, preparing/ready reservations return to baseline.
E. Use the 2048 x 2048 x 1937 to 64 x 64 x 1937 native pyramid: interactive source selection is coarse, settled source can refine, Z textures stay bounded, no unnecessary level-zero reads.
F. MyMultipleObject with many images: offscreen images stay unrefined, tiny visible objects stay coarse, busy readers do not prevent other previews, fine jobs wait for settle, residency/picking remain stable.

Also validate small guarded pans, larger pans, rotation inside a resident source volume, frame/crop/transform changes, all scalar types, nearest labels, reader/upload errors and every shared-helper shader variant.

## Why whole-level rendering was insufficient

Choosing a stored pyramid level still allocates its entire texture. A 64 x 64 x 1937 XY-only coarse level retains pathological Z depth; hundreds of individually reasonable textures also exceed aggregate memory. Deep zoom previously could materialize a complete fine level when only a small region was visible. Phase 5 separates the scientific image from the resident display representation.

Scientific dimensions, pixels(), data(), getImage(), level-zero reload, existing analyses, segmentation, meshes, calibration and exports retain their meaning. Display state belongs to ImageDisplayCommand and its camera's LodUpdateManager. No TensorStore, OME/NGFF metadata, backend chunks or storage paths enter the renderer. The backend ABI, import selection, installed binaries and C:/tsb / C:/tsbd are untouched.

## Request and ownership contracts

ImageLodRequest now carries source level and dimensions, Region3D, target texture dimensions, physical resident bounds, scientific image bounds, frame, reduction mode, preview/detail intent, priority, visibility, resident-source identity, version, checked texture bytes, reader scratch and peak preparation estimate. Target dimensions describe display samples; sourceRegion describes storage voxels. Those values are independent.

Workers capture the scientific ImageInterface pointer and an immutable request. Cancellation is checked before/after adaptive source slabs and during reduction. The request version and full target identity prevent old ROI/frame/mode results from becoming current. Command destruction cancels and waits for in-flight readers; incompatible source state or uncovered pending ROI cancels without waiting; harmless camera changes retain useful work. Image<T> clears commands before its callback/mutex members and scientific histogram are destroyed. Native storage callbacks execute outside the image metadata mutex, followed by a revision check. Resident region copying remains protected against pixel release.

ImageLodReady owns both its prepared payload and a shared RAII memory reservation. Draining the queue transfers that ownership; it does not release accounting. Consumers must retain the ready object/reservation while retaining its payload. The reservation releases when the last ready owner dies. A shared camera manager lifetime avoids dereferencing a destroyed camera.

Camera frame completion performs upload draining after all displayed families have registered visibility. Parent render contexts mark streaming as handled to avoid duplicate child planning. Standalone image commands use the same queue. All GL calls, bindless residency and eviction remain on the render thread.

## Conservative view and guard calculation

ImageViewRegion intersects six homogeneous frustum planes, six local image planes and optional six crop planes. Crop planes transform through the child's local-to-parent matrix. Triple-plane intersections recover the clipped convex polytope, including perspective, orthographic, rotated child, near-plane and camera-inside cases. A positive-vertex plane test rejects definitely offscreen boxes. Numerically inconclusive intersections conservatively enlarge to the image bbox.

The projected clipped footprint drives quality and priority. Bounds map into the chosen native level using actual dimensions and optional stored origin/spacing relative to scientific calibration and the image bbox. Floor/ceil and subtractive bounds checks produce a valid Region3D.

A detail region expands each axis by 25%, aligns to 16-voxel boundaries and clamps to source edges. Its reusable safe interior excludes 10% margins except at image boundaries. Active texture pitch tolerates at most 1.25 times desired on zoom-in and at least 0.88 times desired on dezoom. Current-view source compatibility and coverage decide sampling independently of quality/residency; pitch only decides refinement. Covered rotations, zooms and small pans reuse resident safe bounds. A view outside the safe interior produces a new request.

## Regional preparation and progressive display

The planner caps projected-pixel oversampling by the effective display edge before selecting the coarsest sufficient native level. Deep zoom can select level zero. Resident pixels take precedence over stored/native data through a non-materializing readResidentRegion seam; existing full-resolution API callback precedence remains unchanged. Settled native selection honors pyramidalRenderingEnabled while the initial coarse preview remains bounded. Native-capable sources use readNativePyramidRegion; scientific level-zero sources use readFullResolutionRegion. Adaptive slabs fit a centralized 4 MiB scratch target, reduced to available reserved bytes. No regional native request calls getOrCreatePyramidLevel or materializes a whole native level.

The initial tier is a whole-image preview bounded to 64 per axis. Detail uses one guarded regional texture, bounded to 512 per axis for settled refinement. The planner retains its 128 interactive cap, but active input submits no new detail work and retains the latest settled-quality target. Both are further constrained by GL_MAX_3D_TEXTURE_SIZE, per-image share and byte budgets. Small resident images of at most 1 MiB use a single direct full-resolution preview/copy, with no redundant detail tier.

Target pitch considers stored physical sampling where present. Independent axis caps and CPU/GPU byte admission reduce pathological Z depth even if every native pyramid level preserves all 1937 slices. The source remains unchanged. Two-dimensional images keep depth one. Frame requests read one scientific plane; positive native levels are used only when their Z count, origin and spacing preserve that plane.

RAW defaults to MIP: every contributing source voxel enters its display bin, so Z reduction does not skip sparse bright voxels. Average and Nearest remain available. Integral RAW Majority counts one bounded bin at a time, retaining first-source order for ties; oversized vote buffers fail explicitly. Floating Majority follows the existing nearest-style policy. LABEL display always uses nearest and never averages IDs, including when a request says Average. Scientific LABEL pixels/native levels/export policies are unaffected.

INT32 RAW display converts explicitly into float32 before R32F upload, retaining signed intensity rather than normalized integer pixel transfer. Positive INT32 label IDs use integer textures; nonpositive values remain display background. Scientific int32 data is unchanged. Float32 display/palette arithmetic cannot represent every large int32/uint32 value exactly; exact IDs remain in integer source/label textures.

A new GL texture and resident handle are created before old handles/textures are released. Admission includes simultaneous old/new bytes. Failed reads, canceled requests, refused admission and GL upload/residency errors retain the previous valid representation. A changed preview replaces the previous preview/detail only after success.

## Texture coordinate and ray mapping

Scientific normalized ray coordinates map to local physical position, then into detail bounds only when CPU source/coverage renderability is true and the sample is inside; otherwise they use preview bounds. No regional texture is stretched across the full image bbox. Single-array, single-label and multi-volume shaders share image_stream_sampling.glsl. ImageDescriptorGPU adds resident/preview bounds, handles and flags with checked 304-byte std430 stride and a 208-byte regional-field offset.

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
| Reader/reducer source scratch target | 4 MiB, plus declared adapter scratch |
| Upload drain | 4 results / nominal 16 MiB per frame |
| Preview / resting detail / interactive detail edge | 64 / 512 / 128 |
| Offscreen grace | 12 rendered frames |
| Guard / safe margin | 25% / 10% |

On NVX-capable drivers the camera lowers its GPU budget to at most one eighth of reported dedicated VRAM. Otherwise the portable source-configured budget applies. Each image's planned tier share is conservatively divided among the current/previous visible image count. Admission leaves replacement headroom of min(64 MiB, budget/4); transient and steady totals are both checked.

Visible/pinned images are retained while old/offscreen alternatives exist. Budget pressure evicts oldest invisible entries immediately; ordinary invisibility retains textures for 12 frames, then deletes preview/detail handles/textures while retaining command, LUT and feature state. Follow-up frames advance grace without queuing offscreen reads. Returning images obtain a preview before refinement.

Workers reserve the estimated peak before allocation or storage reads. Estimates include two typed output copies, Average accumulators, byte-bounded source scratch, declared reader scratch and bounded legacy whole-level copies. Ready transition releases temporary scratch and retains final bytes. Overflow checks protect dimensions, voxel counts, byte sums and region bounds. Unknown or oversized readers/refinements are refused. Preparation verifies the declared estimate before allocating its buffers; a released resident input cancels and requires replanning with storage scratch.

One result exceeding the nominal 16 MiB upload throttle is allowed alone to guarantee progress, but still respects the 64 MiB texture and aggregate residency budgets. GPU accounting covers volume textures, not LUT/feature textures, picking FBOs, path-tracing buffers, driver staging or other application geometry. CPU accounting covers display buffers/declared reader scratch, not preexisting scientific arrays or backend-internal codec/chunk caches.

## MyMultipleObject, picking and statistics

RAW and LABEL visibility is checked before createDisplay, including child model and crop transforms. Initialization/request counts are throttled; uninitialized or evicted offscreen children do not receive costly texture work. Hundreds of selected children can keep commands alive without retaining all volume textures.

Picking compares cached child/image identity, ordered hierarchy, selection, 16 transform values and six bbox coordinates. Only a changed snapshot rebuilds picking geometry; FBO resize handling remains. Snapshot scanning is still linear in image count.

Visible overlap grouping remains the existing pairwise connected-component grouping, potentially quadratic when many images overlap. It was left intact for later profiling, keeping this phase focused on regional reads and memory. The ordinary ImagesList renderer retains its 16-image shader array limit with an explicit error instead of array overflow. Existing mixed frame/volume batching and common-bbox assumptions still need manual checks.

Opening Zarr statistics now reads up to 4 x 4 x 4 distributed regions, each at most 8 cubed: at most 32,768 voxels / 128 KiB across five scalar types. It does not retain a huge complete coarsest level. Existing histogram provenance distinguishes measured level-zero values, bounded storage sample and metadata range. Valid OMERO display limits continue to win; single-level RAW/label opening policies from Phases 2/4 remain unchanged.

TIFF continues through the generic reader/session contract, with sequential plane decoding and conservative declared scratch; unsafe refinements are refused. Very large TIFF/nonregional readers may be unable to provide even the initial preview within the budget. No TIFF strip/tile I/O rewrite was introduced. A native regional Zarr reader exercises the complete bounded path.

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

## Phase 5.1 changed-file inventory

- CONTINUITY.md
- poca_extra/src/poca_imageplugin/CMakeLists.txt
- poca_extra/src/poca_imageplugin/ImageDisplayCommand.cpp
- poca_extra/src/poca_imageplugin/ImageDisplayCommand.hpp
- poca_extra/src/poca_imageplugin/ImageDisplayStreaming.cpp
- poca_extra/src/poca_imageplugin/ImageDisplayStreamView.cpp
- poca_extra/src/poca_imageplugin/ImageRegionalLodPlanner.cpp
- poca_extra/src/poca_imageplugin/ImageRegionalLodPreparation.hpp
- poca_extra/src/poca_imageplugin/ImageRegionalMajority.hpp
- poca_extra/src/poca_imageplugin/ImageRegionalReduction.hpp
- poca_extra/src/poca_imageplugin/ImagesListCommands.cpp
- poca_extra/src/poca_imageplugin/ImagesListCommands.hpp
- poca_extra/src/poca_imageplugin/ImagesListMultiObjectDisplayCommand.cpp
- poca_extra/src/poca_imageplugin/ImagesListMultiObjectDisplayCommand.hpp
- poca_extra/src/poca_imageplugin/ImageSourceSlab.hpp
- poca_extra/src/poca_imageplugin/ImageStreamingBudgetTests.cpp
- poca_extra/src/poca_imageplugin/ImageStreamingPreparationTests.cpp
- poca_extra/src/poca_imageplugin/ImageStreamingResponsivenessTests.cpp
- poca_extra/src/poca_imageplugin/ImageStreamingSchedulingTests.cpp
- poca_extra/src/poca_imageplugin/ImageStreamingTests.cpp
- poca_extra/src/poca_imageplugin/ImageStreamingTests.hpp
- poca_extra/src/poca_imageplugin/ImageStreamingTiffTests.cpp
- poca_extra/src/poca_imageplugin/ImageStreamView.hpp
- poca_extra/src/poca_imageplugin/ImageVolumeLodPreparation.cpp
- poca_extra/src/poca_loaderTiffFile/CMakeLists.txt
- poca_extra/src/poca_loaderTiffFile/LoaderTiffFile.cpp
- poca_extra/src/poca_loaderTiffFile/LoaderTiffFile.hpp
- poca_extra/src/poca_loaderTiffFile/TiffStorageImage.hpp
- poca/docs/IMAGE_STREAMING_ARCHITECTURE.md
- poca/src/poca_core/CMakeLists.txt
- poca/src/poca_core/General/Image.hpp
- poca/src/poca_core/General/TiffImageIO.hpp
- poca/src/poca_core/General/TiffRegionReader.hpp
- poca/src/poca_core/Interfaces/ImageInterface.hpp
- poca/src/poca_opengl/OpenGL/Camera.cpp
- poca/src/poca_opengl/OpenGL/Camera.hpp
- poca/src/poca_opengl/OpenGL/ImageStreamPolicy.hpp
- poca/src/poca_opengl/OpenGL/LodUpdateManager.cpp
- poca/src/poca_opengl/OpenGL/LodUpdateManager.hpp
- poca/src/shaders/image_stream_sampling.glsl
