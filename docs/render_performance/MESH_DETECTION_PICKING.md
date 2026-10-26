# Mesh/detection picking — Stage 1 source checkpoint

2026-10-26. Starting revision: `1fdbe6f`. Specification: `Codex_PoCA_Demand_Driven_Mesh_Detection_Picking.md` supplied by the user. This change is instrumentation only. Existing profiler improvements and the legacy mesh/detection rendering, allocations, readback and result handling remain in place.

## Stage 1 — source findings and implementation

Actual call graph traced in the current sources:

| Flow | Current functions and observable behavior |
| --- | --- |
| Ordinary onscreen paint | `Camera::paintGL` -> `Camera::drawElements` -> global `display` dispatch -> `MyObject::executeCommand` sets camera model -> target `display` -> visible `drawElements`, then `drawPicking`. ObjectList can create its pick target in either drawing routine; DetectionSet draws only when its target already exists. |
| Hover | `Camera::mouseMoveEvent` makes the context current -> `dispatchProfiledPick` wraps the existing global `pick` -> target derived `execute` first calls `BasicDisplayCommand::execute` -> `pick` reads the previously drawn FBO -> derived result construction -> camera tooltip/picked-point consumption. |
| Click | `Camera::mouseReleaseEvent` makes the context current -> same global path, separately classified left/right -> existing selection/zoom result consumption. Right click subsequently issues `doubleClickCamera`; the distinct ObjectList `m_pickOneObject` preview is unchanged. |
| Resize | `Camera::resizeEvent` -> existing `updatePickingBuffer` broadcast -> `BasicDisplayCommand::updatePickingFBO` creates/replaces per-command targets, including commands which subsequently draw nothing. Resize now has a `pick_resize` diagnostic row. |
| Multi-object | Existing parent family commands can handle global bounding-box picks and suppress children. `MyMultipleObject` retains its family-handled/selected-child policy. Instrumentation counts only actual child visits/draws; it does not force child dispatch. |

Updated source references (one-based lines at this checkpoint):

| Owner | Exact source/function references |
| --- | --- |
| Profiler | `poca/src/poca_core/General/PerformanceProfiler.cpp:218` `RenderFrame`; `:307` `pickingDraw`; `:315` `pickFboChange`; `:334` `PickScope`; `:368` `renderCsvHeader`. |
| Base command | `poca/src/poca_opengl/OpenGL/BasicDisplayCommand.cpp:186` `execute`; `:234` `pick`; `:367` `releasePickingGpuProfile`; `:382` `updatePickingFBO`. |
| Camera sampler | `poca/src/poca_opengl/OpenGL/Camera.cpp:624` `beginPickingGpuTimer`; `:672` `pollPickingGpuTimers`; `:702` `releasePickingGpuTimers`; `:727` `dispatchProfiledPick`. |
| Camera input/paint | `poca/src/poca_opengl/OpenGL/Camera.cpp:952` `paintGL`; `:2325` `mouseMoveEvent`; `:2550` `mouseReleaseEvent`; `:2907` `resizeEvent`. |
| Mesh/object | `poca/src/poca_objectlistplugin/ObjectListDisplayCommand.cpp:184` typed-context `execute`; `:368` `display`; `:377` `drawElements`; `:625` `drawPicking`; `:666` unchanged `displayZoomToBBox`. |
| Detections | `poca/src/poca_detectionsetplugin/DetectionSetDisplayCommand.cpp:94` typed-context `execute`; `:202` `display`; `:214` `drawElements`; `:251` `drawPicking`. |
| Registered fixture | `poca_extra/src/poca_imageplugin/ImageStreamingTests.cpp:78` `runRenderProfilerTests`; `:241` existing `registerImageStreamingTests`. |

Visible drawing and picking share the camera matrices, buffers/features/clipping and surrounding GL state. Picking still follows normal component drawing and still restores the legacy default FBO/depth/polygon/etc. behavior; this checkpoint does not repair or redesign that state handling. `BasicDisplayCommand::pick` binds the stored target, reads `(x, m_hImage - y - 1)`, optionally saves the full image, releases, and performs the original `int(pixel - 1)` conversion. No fresh picking render occurs on this legacy path.

Transform warning for the later stage: `MyObject::executeCommand("display")` multiplies the camera model by the object model. `MyMultipleObject` child dispatch saves/restores the camera model specifically around `display`. The equivalent model push is not established by `pick`. Existing FBO contents therefore embody the previous display state, not necessarily the current pick-event matrices. Stage 2 must establish a correct scoped per-child transform before drawing on demand. Neither conditionally allowed object-dispatch file was edited.

Only three GUI global `pick` construction/dispatch sites were found in the inspected `poca/src` and `poca_extra/src` C++ trees: hover, left release, right release in Camera. Existing no-camera/programmatic command overloads remain possible. The two derived commands provide a separate component-level `pick_event` scope for those callers (camera ID/viewport 0 when unavailable). Such a scope is not a whole-scene programmatic-dispatch measurement, and multiple programmatic component records must not be represented as one global event. Legacy context assumptions/results are retained at this stage.

### Changes by owner/method

- `PerformanceProfiler.hpp/.cpp`: append 56 metric columns (26 per target family plus four input types), retain all 69 previous metric names/positions, existing shared instance/identities/paint timers/history/CSV queue/stdout summary. Schema 2 appends Unix-microsecond timestamp, GPU source-record ID and GPU picking status after metrics. `RenderFrame` accepts explicit non-paint kinds without joining them into an enclosing paint or incrementing draw/paint counters. `PickScope` attributes actual existing audited GL draw sites to the active target family without adding them a second time to `picking_gl_draws`. Fixed scalar process-wide FBO census remains valid across enable/disable; no new mutexes or logger.
- `BasicDisplayCommand.hpp/.cpp`: identify only the two target command names; copies retain type but own no census/query-camera resources. `updatePickingFBO` observes valid real targets at the existing allocation sites, success/reallocation counts, observed depth bits and requested R32F payload; restore the additional diagnostic draw-FBO binding. `pick` measures the existing pixel/full-image readbacks and successful/failed binds, records decoded hit/miss without changing IDs. Destruction/freeGPU retire census and release target-family camera query resources through guarded `QPointer`s. Other plugins keep their base command behavior and are excluded from new family totals.
- `Camera.hpp/.cpp`: retain the existing global pick dispatch/input policy in `dispatchProfiledPick`; identify hover/left/right separately. Two feature-bearing picking shader templates classify their original GL draw calls; all shaders/draw arguments/transforms are unchanged. Camera owns eight two-query timestamp slots; `PickingGpuScope` encloses only target clear + shader draw. Runtime capability/context/thread guards, asynchronous availability checks, source-ID result rows, enable-session separation, exact-context cleanup and resize scopes are added.
- `ObjectListDisplayCommand.cpp` / `DetectionSetDisplayCommand.cpp`: before the original base execution, add a diagnostic-only component pick-event scope if there is no GUI event scope. At `drawPicking` entry, start the family CPU/draw scope; observe the original bind/clear, and bracket clear/draw with the camera GPU scope. Keep every original visible/picking call, ID/result path and FBO ownership/order.
- `poca_extra/src/poca_imageplugin/ImageStreamingTests.cpp`: extend the existing `runRenderProfilerTests` fixture with picking-only family/event separation, no global double-counting, component visit/render distinctions, delayed GPU source identity, unavailable status, FBO census/unknown bytes/release and disabled silence. Adapt its schema-field assertion to appended picking metadata. Existing tests/registration remain intact; no GL samples are fabricated by these synthetic checks.

### Metric interpretation

For each `objectlist_pick_*` / `detectionset_pick_*` prefix:

- `draw_invocations` counts entries including histogram/no-resource early exits; `gl_draws` counts the actual audited GL calls, including each buffer chunk, not attempted high-level draws. `components_rendered` counts entered routines which actually submitted at least one GL draw. `components_visited` counts base `execute("pick")` visits before picking enable/coordinate checks.
- `cpu_us` encloses the entire picking routine including bind/clear/shader/release and any allocation inside that routine. `fbo_outside_draw_cpu_us` separately encloses FBO construction/replacement outside that routine (e.g. ObjectList `drawElements`, resize). Add these two disjoint metrics to estimate targeted paint picking/allocation CPU share; do not add other inclusive frame timers. Query profiling/diagnostic overhead is included and requires the on/off comparison below.
- `fbo_allocations` includes successful valid initial creations and replacements; `fbo_reallocations` is the subset replacing a previously valid counted target. Eligible components, invalid Qt targets and the distinct 400x400 preview are excluded. No new FBO is created for instrumentation.
- `fbo_live`, `fbo_peak_live`, `fbo_requested_bytes`, `fbo_unknown_payloads` are process-wide family resource gauges copied into each completed record, not per-camera deltas. Peak live is process-lifetime, including disabled periods. Do not sum gauges across rows; use snapshots/maxima. Stdout reports their interval maxima.
- Requested bytes are `width * height * (4 + ceil(observed_depth_bits / 8))`, reflecting R32F color plus the actual depth precision queried at an enabled allocation. They exclude driver tiling/alignment/bookkeeping and are NOT physical VRAM. Valid targets created while disabled have unknown payload rather than fabricated bytes; their live count is still retained. Enable before loading/resizing, require `fbo_unknown_payloads=0` for complete byte reporting. Individual allocation events outside a known camera record use `kind=pick_resource`, camera ID 0; resize/paint allocations inherit the enclosing camera record.
- `fbo_bind_attempts`, `fbo_binds` (successful), `fbo_bind_failures`, `fbo_clears` distinguish attempted legacy render/read binds from success and actual issued clears. These counters exclude Qt-internal FBO construction binds, diagnostic depth-inspection binds and release/default-FBO rebinding; they are not an all-GL-bind trace. Legacy code still clears/draws after a failed bind; the counters expose that existing behavior. A capture with failures cannot establish correct picking results.
- `readpixels_calls` / `readpixels_us` time the existing synchronous read calls only. Full-frame `saveImage` adds its existing second read; image encoding/logging are outside that metric. `read_hits` / `read_misses` classify the original decoded ID after an actual read, not semantic correctness. No-read/disabled/missing-resource visits are not invented misses; compare visits to reads and validate IDs manually.

### GPU sampling and limitations

Capability requires the exact camera context current on its GUI thread, desktop OpenGL >=3.3 or `GL_ARB_timer_query`, required entry points and nonzero timestamp counter bits. Otherwise `*_gpu_unavailable` and `GPU_PICK_TIMING_UNAVAILABLE` explicitly describe missing measurement. Capability has NOT been observed on this machine.

Pool: eight pairs = at most 16 live query objects/camera; at most one sampled interval per family on every thirtieth render/event record; at most 128 submitted intervals (256 timestamp submissions) per camera lifetime, including after freeGPU/re-enable. Both families get an opportunity even in a 351-mesh/117-detection paint. Samples intentionally favor the first eligible component of each type, so they are not a random or representative estimate of an entire heterogeneous family. Restart/recreate the camera for a fresh budget/capture.

The start timestamp is immediately before the legacy clear; the end is after its shader draw and before FBO release. No `glFinish`, additional pixel readback or blocking query-result request is introduced. On later targeted invocations, poll at most once per record, at most two availability checks per pending pair; read numeric timestamps only after BOTH are available. Keep subsequent paints running to allow pending results to resolve. No polling timer or artificial scene redraw was added.

`*_gpu_supported` counts capable intervals; `*_gpu_query_count` counts actual timestamp submissions (normally two per sampled interval) in its SOURCE record; `*_gpu_skipped` counts unsampled capable intervals and separately discarded pending intervals in `gpu_discarded` rows. `kind=gpu_sample` contains one completed interval, type-specific `*_gpu_sample_count=1`, elapsed nanoseconds and `gpu_source_record_id` referencing its original record. Join by source ID and camera ID to classify paint vs explicit event, viewport and scenario. GPU result rows must never be treated as paints or attributed to the polling frame. Session changes discard pending old-session results rather than measuring them in a new capture. Discarded/pending samples are not zero-time samples; report submitted/completed/discarded counts separately.

Samples do NOT cover all 468 operations and do not represent full-frame GPU picking totals. Report sample counts, latency distribution and coverage as sampled intervals / capable intervals, separated by family and source kind. The pre-existing `gpu_timing_available=0` / stdout `gpu_timing_unavailable=1` remain the original full-frame-GPU limitation; the appended `gpu_pick_timing_status` and per-family fields describe the new targeted samples. Existing `compare_csv.py` deliberately still reports no whole-frame GPU execution measurement and filters `kind=paint`; it is unchanged and does not summarize picking GPU/event rows.

Queries are released on camera context destruction, camera destruction, or target command freeGPU/destruction for all observed still-live cameras. Cleanup uses only the owning exact current context, temporarily makes that camera current if necessary, and restores the prior context/surface. If the owning context cannot be made current, deletion is skipped; the slot remains until a later valid cleanup or context destruction. If Qt destroys the context, its query objects are reclaimed by GL and stale handles are reset before using a new context. Query cleanup/result pending behavior and multi-camera lifecycle still require runtime validation. No queries/GL capability checks/CSV rows/logging/clock samples/new heap allocations occur in disabled draw instrumentation; disabled overhead has NOT been benchmarked. Resource lifecycle census performs fixed scalar bookkeeping while disabled.

## Stage 1 measured baseline — NOT MEASURED

| Required result | Actual result at this checkpoint |
| --- | --- |
| Dataset/scene/viewport/configuration capture | NOT CAPTURED |
| Paint CPU median/p95 and sample count | NOT MEASURED |
| Type-separated picking CPU_us and frame CPU share | NOT MEASURED |
| Actual GL draw/clear/bind/readback counts per paint/event | NOT MEASURED |
| FBO allocations/reallocations/live/peak/requested attachment bytes | NOT MEASURED |
| Hover/left/right dispatch CPU latency and ID correctness | NOT MEASURED |
| GPU capability / per-family sample distributions | UNCONFIRMED / NOT MEASURED; zero real captured samples |
| Profiler on/off overhead | NOT MEASURED |
| Before/after CSV comparison and measured speedup | BLOCKED: no baseline, Stage 2 not entered |

Blocker: repository/user instructions explicitly make implementation source-only and prohibit build/configure, PoCA, tests/benchmarks/helper/Python execution unless separately requested. The request acknowledges unavailable runtime and requires the hard gate. No executable was built or launched and no data was loaded. Static counts or the earlier streaming 468-draw observation are not timings or this baseline. This source checkpoint remains in the working tree; no commit was created. Preserve/commit it and its captures before applying Stage 2.

### Precise user-run capture recipe to unlock Stage 2

1. Build **existing configured PoCA Debug and Release trees**, rebuilding all affected DLL consumers of core/OpenGL headers, including both target plugins and image plugin test consumers. Use the existing toolchain/configuration; do not use stale plugins. Suggested command template only: `cmake --build '<existing-PoCA-build>' --config Release` (and Debug). No command in this recipe was executed by Codex.
2. Run the existing **Preferences > Tests > images > Render profiler, overlap and batch decisions** fixture (`image_render_performance`). Registration is `registerImageStreamingTests` in `ImageStreamingTests.cpp`, source listed in `poca_extra/src/poca_imageplugin/CMakeLists.txt` and registered through `ImagePyramidTests.cpp` -> `ImagePlugin::registerTests` in `ImagePlugin.cpp`. Its added values are synthetic invariants, never profiler results.
3. Before application launch, create an output directory and set a fresh absolute CSV filename, e.g. PowerShell `$env:POCA_RENDER_PROFILE_CSV = 'D:\captures\2026-10-26-legacy-single-settled-release.csv'`. Launch your actual configured PoCA executable, optionally redirect stdout/stderr to a matching `.log`. Enable existing **Preferences > Verbose > Verbose**, then **Add verbose type... > renderPerformance**. It remains the ONLY enable switch. Enable before loading scenes; if any family `fbo_unknown_payloads` is nonzero, resize to a different viewport and back while enabled, then collect the final matched viewport. Use fresh process/camera per capture to reset GPU sampling budget and process-wide peaks.
4. Capture `MyObject` with one mesh plus one DetectionSet, then `MyMultipleObject` with 2, 10 and 117 datasets where available. For the reference 117 case, document the actual component inventory (expected scene has 351 meshes and 117 DetectionSets), not assumed allocations. Record unavailable scenes explicitly rather than substituting another scene.
5. For EACH scene capture grid/bbox OFF and customary settings, with identical physical/logical viewport dimensions, device pixel ratio, camera/model/projection, active/selected children, feature thresholds, point/shape modes, clipping, shader settings and stream state. Save timestamped settings notes and a screenshot/selection reference table. Keep existing image/LOD settings unchanged.
6. Separate fresh traces for: (a) settled/no mouse events, discard 30 warmup paints and retain at least 120 paints; (b) resize; (c) rotation/zoom/pan; (d) hover; (e) left/right selection and double-click zoom; (f) object/feature changes. Collect settled and active OME-Zarr streaming separately. This code adds no repaint timer; stationary windows may paint only on demand. Use an existing explicit redraw/display pathway or record an existing repeatable redraw trigger, keep mouse stationary, report that trigger and collect 120 ACTUAL `kind=paint` records. Standalone `kind=draw` records or repeated screenshots are not stationary onscreen paints. If that count cannot be achieved, report the limitation and actual count.
7. After GPU sampling paints/events, permit later ordinary paints so completed queries can be polled, then close normally to drain the CSV queue. Reject capture files with CSV/schema errors, `csv_dropped`, failed FBO binds, missing source IDs or unknown payloads for a claimed complete attachment-byte total. Different schema versions require separate fresh files.
8. Analyze paint rows ONLY for paint CPU median and nearest-rank p95. Sum per-family `cpu_us + fbo_outside_draw_cpu_us`, divide by `sum(cpu_ms) * 1000` for the measured targeted frame CPU fraction. Report each family draw/clear/bind/allocation/read count, sum counters and divide by actual paint count for per-paint rates; report gauge maxima/snapshots separately. Include resize/resource rows when reporting lifecycle allocations. For `pick_event` rows, separate hover/left/right/other counters; report event cpu_ms median/p95, components visited/rendered, draws, reads and read CPU_us. Do not average event rows into paints.
9. Join `gpu_sample` to its original source row, separate family/source kind, convert nanoseconds to milliseconds, report mean/median/p95/min/max, actual completed sample count, timestamp-submission count, skipped/discarded/pending counts and coverage. Never multiply the sampled mean by 468 and call it measured GPU frame cost. Report unavailable capability explicitly.
10. Repeat the same workload with renderPerformance disabled and an existing external CPU/GPU tracing tool or other independent frame measurement, if feasible, to quantify instrumentation overhead. Disabled profiling intentionally creates no CSV. Report missing overhead comparison honestly.
11. For each tested center/edge/background/overlap hit, record selected object/point index or no-hit, info text, picked centroid/point coordinates and selection box using the existing UI/command results; save same-camera screenshots. Include the correctness cases listed below. Return the CSVs, logs, settings/viewport/scene notes, selection reference and build/test outcomes. Those actual instrumented LEGACY captures are the input needed before Stage 2.

## Stage 2 — NOT STARTED (hard gate)

No demand-driven rendering change, shared scratch FBO, ID/order/state restoration change or transform propagation has been implemented. Target commands still own their legacy full-viewport targets and render picking on ordinary onscreen paints; hover/click remains synchronous and reads the legacy buffer. Scratch-FBO owner/lifetime, fresh draw-before-read, exact per-child transform, no-hit validation and GL restoration are pending design/implementation after the measured gate, not claims of this change. Neither `MyObject.cpp` nor `MyMultipleObject.cpp` was edited.

No before/after CSV comparison exists. No CPU/GPU improvement or selection/pixel equivalence is claimed.

## Regression status

| Verification | Result |
| --- | --- |
| Strict production allowlist and complete changed-file inventory | PASS, static diff inspection |
| Legacy target visible display/drawElements/freeGPUMemory bodies | PASS, unchanged relative to HEAD after newline normalization |
| Existing result/ID conversion/saveImage calls and command ordering | PASS, static inspection; runtime equivalence UNVERIFIED |
| Metric enum/label count (125), two 26-metric family blocks, schema (143 fields) | PASS, static consistency checks |
| C++ delimiter balance, UTF-8/CRLF and original BOM status | PASS, static checks |
| `git diff --check` and quoting/merge artifact searches | PASS |
| Debug / Release build and affected DLL load | NOT RUN |
| Registered profiler/picking accounting fixture | ADDED, NOT RUN |
| Profiler-disabled silence/no GL/query/FBO overhead | Source guards inspected; runtime/overhead NOT RUN |
| Legacy picking CPU/GPU performance/capability capture | NOT RUN / NOT MEASURED |
| Stage 2: stationary paints zero targeted pick draws/binds/clears/allocations | NOT RUN; Stage 2 gated |
| ID/text/point/bbox/screenshot equivalence | NOT RUN |
| Point/shape, 2D/3D, overlap/blank/edge, togglePicking, histogram/feature/clipping | NOT RUN |
| MyObject/multi 1/2/10/117, transformed/grid children, active-child/reorder/remove | NOT RUN |
| Resize then pick before paint, non-unit DPI, saveImage, hover/left/right/double-click | NOT RUN |
| freeGPU, close/reopen/context loss, two cameras, GL state/leaks | NOT RUN |
| Offscreen screenshot/movie and unchanged image/LOD/uploads/depth-clear counts | NOT RUN; unrelated source untouched |

No CMake/configure/compiler/build/test/PoCA/Python/helper/benchmark/dependency execution occurred. Static verification is not runtime validation.

## Complete changed-file inventory

Production (all explicitly allowlisted):

1. `poca/src/poca_core/General/PerformanceProfiler.hpp`
2. `poca/src/poca_core/General/PerformanceProfiler.cpp`
3. `poca/src/poca_opengl/OpenGL/BasicDisplayCommand.hpp`
4. `poca/src/poca_opengl/OpenGL/BasicDisplayCommand.cpp`
5. `poca/src/poca_opengl/OpenGL/Camera.hpp`
6. `poca/src/poca_opengl/OpenGL/Camera.cpp`
7. `poca/src/poca_objectlistplugin/ObjectListDisplayCommand.cpp`
8. `poca/src/poca_detectionsetplugin/DetectionSetDisplayCommand.cpp`

Registered relevant test: `poca_extra/src/poca_imageplugin/ImageStreamingTests.cpp`.

Documentation: `poca/docs/render_performance/README.md` (appended only), `poca/docs/render_performance/MESH_DETECTION_PICKING.md` (this report).

Mandatory workspace briefing: `CONTINUITY.md`.

Conditional dispatch-file edits: NONE. No shader/scientific/image/LOD/compositor/batching/CMake/test-registration/dependency/generated-tree changes. `compare_csv.py` is unchanged; its existing paint filter excludes event/GPU rows.

## Residual costs and review boundaries

Per-hover component visits/readbacks can still be high; actual counts are NOT MEASURED. `glReadPixels` stays synchronous. Legacy per-paint picking cost and per-command FBO memory remain until Stage 2. Unrelated plugin picking (e.g. Delaunay/Voronoi display routines found by source search) is out of scope, not newly timed as either target family; it may still use per-paint picking. Existing image-family profiler fields retain their prior scope and are not relabelled as global FBO totals. No context safety/selection bug discovered during inspection was opportunistically fixed.

## Diff receipt

Tracked-file output of `git diff --stat` at the source checkpoint (the new report is untracked and is listed separately in the complete inventory):

```text
 CONTINUITY.md                                      |  67 +++++++---
 poca/docs/render_performance/README.md             |   8 ++
 poca/src/poca_core/General/PerformanceProfiler.cpp | 144 +++++++++++++++++++--
 poca/src/poca_core/General/PerformanceProfiler.hpp |  56 +++++++-
 .../DetectionSetDisplayCommand.cpp                 |  15 +++
 .../ObjectListDisplayCommand.cpp                   |  15 +++
 .../src/poca_opengl/OpenGL/BasicDisplayCommand.cpp |  68 +++++++++-
 .../src/poca_opengl/OpenGL/BasicDisplayCommand.hpp |  11 ++
 poca/src/poca_opengl/OpenGL/Camera.cpp             | 129 +++++++++++++++++-
 poca/src/poca_opengl/OpenGL/Camera.hpp             |  37 +++++-
 .../src/poca_imageplugin/ImageStreamingTests.cpp   |  64 ++++++++-
 11 files changed, 572 insertions(+), 42 deletions(-)
```

`git diff --check`: PASS. New/untracked report plus the tracked paths above make 12 changed paths, including eight allowlisted production files. Source-only checks and the complete runtime limitations are stated above; no Stage 2 diff exists.

## Diagnostic follow-up — 2026-10-26

User-supplied Visual Studio diagnostics after checkpoint `2fd59e2` establish a failed user build (superseding unknown build status for that attempt). This follow-up is source-only; no rebuild, link, test or runtime capture has been performed by Codex, and Stage 2 remains gated.

Source corrections, confined to the existing production allowlist:

- `BasicDisplayCommand.cpp:33`: include Windows/GLEW before Qt, matching neighboring OpenGL implementation files. The new framebuffer-extension calls previously had no GLEW declarations in this translation unit; Windows OpenGL 1.1 declarations alone are insufficient. No GL call or framebuffer operation was changed.
- `Camera.cpp:730` and `:2935`: use `dynamic_cast<::MyMultipleObject*>`; the class is declared in the global namespace in `poca_core/Objects/MyMultipleObject.hpp`, not `poca::core`. Picking dispatch and resize instrumentation remain identical.
- `PerformanceProfiler.hpp:120` and `PerformanceProfiler.cpp:218`: restore the original explicit eight-argument `RenderFrame` entrypoint, delegating to the existing extended constructor with `kind=nullptr, active=true`. The extended constructor now requires `kind`, preventing overload ambiguity; nine-argument event calls still default `active=true`, and ten-argument conditional scopes remain supported. This restores the exact legacy constructor symbol reported missing without changing frame identity, timers, object layout or metric schema. All affected consumers still need a consistent rebuild; preserving a symbol does not validate stale binaries against changed record layouts.

The diagnostic list also contains IntelliSense-only missing new profiler members/enum values and excessive initializer errors. Before editing, the reported `C:/DevC++/PoCA/src` profiler header/source, Camera and BasicDisplayCommand source hashes matched this checkout. The on-disk header contains every named missing member and all 125 enum entries; a header search under that reported tree found one PerformanceProfiler.hpp. Those IntelliSense diagnostics are inconsistent with current on-disk declarations. Stale IntelliSense state or a different selected include/build artifact remains UNCONFIRMED; no duplicate declarations, build-system changes or cache deletions were added to conceal it.

`Camera::PickingGpuScope` constructor/destructor/stop and `BasicDisplayCommand::profilePickingCamera` definitions already exist in the registered `poca_opengl` sources. Their unresolved symbols are consistent with an old archive surviving the reported source compilation failures; the exact artifact-selection cause and final link success are UNCONFIRMED. Rebuild core/OpenGL and affected DLL consumers from the updated tree using the existing configured toolchain, then refresh/rescan IntelliSense if its active-error list remains inconsistent. No generated directories were modified.

Static verification: four production files changed (+12/-3), source constructor declarations/definitions and 8/9/10-argument call resolution inspected; existing eight-argument nested-frame and explicit-event fixtures remain unchanged and NOT RUN. All 125 profiler metrics/schema, rendering bodies, ID/result handling, query pool and stage gate are retained. CRLF/UTF-8/BOM checks, lexical delimiter checks, exact method-body comparison and `git diff --check` are clean. This supersedes any interpretation of the earlier static PASS rows as proof of compilation. Runtime selection/performance remains NOT VERIFIED / NOT MEASURED. Earlier line-reference/diff tables describe the original checkpoint; the new overload shifts subsequent profiler source lines by five and the include addition shifts base-command lines by three.

Follow-up changed files: PerformanceProfiler.hpp/.cpp, BasicDisplayCommand.cpp, Camera.cpp, this report, and mandatory CONTINUITY.md. No CMake, target registration, generated artifacts or unrelated files changed.
