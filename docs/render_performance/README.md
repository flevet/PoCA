# PoCA 3D rendering profiler and targeted fixes

2026-10-26. **SOURCE-ONLY: builds, source fixtures, GL regression, screenshots and before/after timings NOT RUN / NOT MEASURED.** Initial root/poca/poca_extra status was clean; both source trees are in one Git checkout, main at bf59ca99c9b814e4c0fef602a589c73001281df1. Existing user changes were preserved (none were present initially).

## Stage outcomes

The shared profiler was implemented and statically checked first, before any rendering behavior changes. Its initial checkpoint had 67 metric fields; the final schema has 69. No instrumented baseline was built or measured. The pre-change HEAD does not contain this new profiler, so an uninstrumented run must not be presented as a matching profiler baseline.

| Item | Status | Implementation / deliberate limits |
| --- | --- | --- |
| A | completed/not runtime tested | Existing process-wide PerformanceProfiler extended; per-frame CPU/counters/queues, bounded history, asynchronous CSV, source fixture. Compilation NOT RUN. Optional GPU queries deferred: gpu_timing_unavailable=1. |
| B1 | completed/not runtime tested | Both-off geometry loop skipped; all-child flag inspection preserves differing child options. Conservative 8-corner XY rejection with line expansion margin. Owned bounded exact geometry cache, context cleanup and freeGPU invalidation. Original text/facing generation and draw order retained. |
| B2 | completed/not runtime tested | NDEBUG GL_CHECK_ERRORS is a true no-op unless POCA_ENABLE_RELEASE_GL_CHECKS is defined. Debug/glAssert and explicit recovery checks retained. Debug/Release compilation NOT RUN. |
| B3 | completed/not runtime tested | Maintained visible/forced counts; original previous-frame stabilization and forced predicate retained. Slow-reference/randomized 320-entry fixture and zero-share-scan profiler assertion added, NOT RUN. bytes/admit/eviction policy unchanged. |
| B4 | completed/not runtime tested | ImageDisplayCommand/Atrous empty picking FBO allocation/clear/readback replaced by deterministic no-hit on pick. Resize notifications allocate nothing. Existing downstream depth/fill/default-FBO state retained. Functional multi-object picking untouched. |
| B5 | completed/not runtime tested | Exact ordered topology key and original DFS order; descriptors/handles rebuilt every frame. Independent graph-reference cache fixtures for 1/10/100/300 layouts, NOT RUN. |
| C1 | completed/not runtime tested | Before shared draws, compare selected raw lists' actual shader/gamma/steps/frame/index. Explicit supported modes; whole-family child fallback for incompatibility. Unknown mode cannot silently map to MIP. |
| C2 | completed/not runtime tested | Whole-family fallback whenever existing filtered/segmentation enable parameters are true. Existing child pipeline owns exact filtered-then-segmentation layering/model/crop; no duplicate shared draws. |
| C3 | completed/not runtime tested | Conservative family preflight prevents >1024 candidates reaching MIP/direct shared shaders, plus actual-batch defensive invariant. Existing child renderer is the fallback; no truncation/chunking. |
| C4 | completed/not runtime tested | Shader math, adaptive/fixed stepping, cube geometry, ordering, science, LOD and scene depth clears preserved. These costs are measured, not rewritten. |

Stage validations were static: inspect declarations/call sites/ownership and run git diff --check/artifact searches after each edit stage. Final checks: 23 allowlisted files (20 C++ streams), balanced delimiters, strict UTF-8/CRLF, all 21 tracked BOM states equal raw HEAD blobs, 69 matching metric labels/references. Debug diagnostics and shared unit-cube initialization compare exactly with HEAD. Audited Camera/ImagesList scene clear masks/order compare exactly with HEAD. These checks do not establish runtime correctness.

## Enable, capture, disable

1. Use existing **Preferences > Verbose > Verbose** to enable output.
2. Use **Preferences > Verbose > Add verbose type...**, entering exactly **renderPerformance**. Empty/general verbose-type selection does not enable it. imagePerformance/lodDebug/debugPyramidalRendering and the Performance widget remain independent; leave detailed diagnostics off during timing comparisons.
3. Optionally set the CSV path before launch (absolute path, existing parent directory, a fresh filename per process/capture):

    $env:POCA_RENDER_PROFILE_CSV = 'D:\captures\poca-after.csv'
    & 'C:\path\to\poca.exe' *> 'D:\captures\poca-after.log'

   Replace the executable path with your configured PoCA executable. Search stdout capture for **[PoCA][RenderPerformance]**; this uses the same std::cout sink as existing PoCA image diagnostics, not a new log destination. CSV export does not require console visibility. The environment variable is read on enable; change it by disabling/re-enabling profiling. The equivalent persisted keys are Preferences.verbose=true and Preferences.verboseTypes including "renderPerformance"; existing Engine API is setVerbose(true), addVerboseType("renderPerformance").
4. Disable via **Preferences > Verbose > Verbose**, or clear/remove the type. The next paint/draw observes the change and drains/stops CSV. Normal application quit also drains it before Qt/plugin teardown. Force-killing a process can lose the pending tail. An invalid/unwritable path or mismatched existing header diagnoses export failure without changing rendering.

Only enabled profiling collects render records. Counter loops have no locks or allocations; each disabled event/timer returns at its opt-in guard. Enabled frames use the same process-wide profiler object stored on QCoreApplication, including across static-library/plugin DLLs. The optional CSV thread is isolated diagnostic I/O; streaming worker architecture is unchanged. Sixteen simultaneous nested frames use fixed storage; deeper reentrant frames use owned temporary records and report render_frame_stack_spills, without losing/cross-attributing frames.

## Records, timers, counters and CSV

CPU intervals use steady_clock. cpu_ms covers the whole paintGL, or standalone drawElements. Time around OpenGL calls is CPU wall time, **not GPU execution time**. GPU queries were not added; gpu_timing_available=0 / gpu_timing_unavailable=1 is explicit.

Each paint or standalone draw has a monotonic process-lifetime frame ID, non-reused process-lifetime camera ID, object kind, attached child count, framebuffer dimensions and interaction state. Same-camera onscreen drawElements inside paint contributes to that paint's record rather than creating a second paint. Standalone/offscreen draws and nested other-camera paints have distinct records; nested completion/CSV order can differ from entry-ID order. Queue -1 means not sampled; otherwise queue_camera (log) / camera_id (CSV) identifies the manager whose actual queued/ready/preparing lengths were sampled under its existing mutex.

CPU scopes cover drawElements, scene preparation, grid geometry generation, overlay drawing, multi-child dispatch, visibility, LOD planning, list/multi display, overlap grouping, regional upload/publication and functional multi picking FBO work. These scopes are inclusive across different metrics and must not be summed as disjoint costs. Nested scopes of the same metric count only the enclosing interval. Independent imagePerformance samples are not added into render counters.

These are **audited renderer draw calls**, not a complete all-plugins GL count: live glDraw sites in Camera.cpp/.hpp helpers, ImagesListCommands, ImagesListMultiObjectDisplayCommand, ImageDisplayCommand and RealTimeImageFilteringCommand. Raw single/multi, nonraw, picking and other audited helper draws are distinct. Frame geometry uploads refer to bbox/grid face/grid line buffers, not debug points, text glyphs or every unrelated plugin buffer. Descriptor calls/bytes are actual SSBO glBufferData/glBufferSubData sites. Each actual remaining picking FBO allocation/clear is counted; the two empty owners perform none.

visibility_calls/hits/misses and visible_images/culled_images count calls/results, **not unique images**. Repeated visibility/LOD state updates are retained. lod_request_attempts counts request() entries; lod_requests_enqueued counts new immutable targets, deduplicated/priority requeues/continuation enqueues are separate. lod_requests_deferred counts the GUI interaction-policy submission rejection. worker_admission_deferred/rejected count worker admission attempts, not unique images. worker_preparations/preparation_us are asynchronous preparation callbacks (including incremental bricks), not physical I/O or GUI draws. manager_redraw_requests is a process-wide manager event count that can originate from publication or workers; stream_redraw_requests/scheduled are GUI camera events. ready_uploads_processed counts actual current visible upload callback invocations.

texture3d_allocation_calls counts actual audited glTexImage3D calls; texture3d_upload_chunks counts actual regional glTexSubImage3D calls. texture3d_submitted_bytes counts payload bytes submitted at those calls (including filtered full-texture transfers); texture3d_uploaded_bytes counts successfully checked regional transfers. Successful filtered transfer bytes are **not_instrumented**, since their existing producer does not check success and no GL queries were added. Physical disk bytes, other-plugin GL calls, glyph-internal draw counts and GPU execution are **not_instrumented**. residency_map_entries_examined counts actual traversal visits (frame, bytes, admission/eviction/grace walks); imageShare itself adds zero visits after B3. Other remaining traversals can still dominate that total.

One aggregate line is emitted per >=1-second completed-frame activity interval; no idle timer or per-image logs. Counts/times are totals across all completed records since the prior line; batch_max_images and frame_cache_entries/bytes are maxima. CPU mean/median/p95/max uses only paints among the last min(512, records completed in that interval). Median averages the middle two; p95 is nearest rank ceil(.95*N). paint_samples exposes the sample count. Standalone draws are excluded from paint time statistics. Divide pure-paint totals by paintgl_calls for per-paint averages; mixed standalone records must first be separated using CSV. between_* counts and CPU timings (including on-demand picking) happened outside GUI frames and are not assigned to a camera; worker_* and manager_redraw_requests are process-wide interval deltas, not per-frame metrics.

History holds at most 512 completed records. The background CSV queue holds at most 256 rows, no per-call flush/formatting on the GL thread. Slow disk causes explicit csv_dropped rather than blocking GL; reject captures with drops for per-frame comparisons. CSV rows contain GUI frame counters only, with one row per completed paint/standalone draw when the exporter keeps up. Prefix plus 69 numeric metric fields (schema version 1):

    schema,frame_id,camera_id,kind,offscreen,object_kind,children,width,height,interactive,cpu_ms,queued,ready,preparing,gpu_timing_available,paintgl_calls,draw_elements_onscreen,draw_elements_offscreen,recompute_frame_calls,child_frame_recomputes,grid_face_draws,grid_line_draws,bbox_draws,raw_volume_gl_draws_single,raw_volume_gl_draws_multi,nonraw_volume_gl_draws,other_audited_gl_draws,picking_gl_draws,shared_descriptor_glBufferData_calls,shared_descriptor_glBufferData_bytes,shared_descriptor_glBufferSubData_calls,shared_descriptor_glBufferSubData_bytes,frame_geometry_glBufferData_calls,frame_geometry_glBufferData_bytes,overlap_pair_tests,overlap_cache_hits,overlap_cache_misses,batch_count,batch_max_images,visible_images,culled_images,child_display_dispatches,picking_fbo_creates,picking_fbo_clears,visibility_calls,visibility_cache_hits,visibility_cache_misses,image_share_calls,residency_map_entries_examined,lod_request_attempts,lod_requests_enqueued,lod_requests_deduplicated,lod_priority_requeues,lod_continuations_enqueued,ready_uploads_processed,texture3d_allocation_calls,texture3d_upload_chunks,texture3d_uploaded_bytes,stream_redraw_requests,stream_redraw_scheduled,depth_clears,frame_cache_hits,frame_cache_misses,frame_cache_entries,frame_cache_bytes,multi_batch_fallback_incompatible_settings,multi_batch_fallback_overlays,multi_batch_fallback_limit,draw_elements_us,grid_preparation_us,grid_draw_us,child_dispatch_us,visibility_us,lod_planning_us,images_list_us,multi_display_us,overlap_us,texture_upload_us,picking_us,scene_preparation_us,object_display_us,lod_requests_deferred,texture3d_submitted_bytes,render_frame_stack_spills

## Source fixtures and user-run build steps (all NOT RUN)

No CMake/configure/build, PoCA/test/benchmark/Python/helper execution, dependency installation, backend rebuild or generated-tree writes occurred. No Debug/Release pass/fail or pixel correctness claim is made. Existing C:/tsb and C:/tsbd were inspected read-only and identify backend builds, not PoCA builds; they were not used.

After reviewing the source, use your **existing configured PoCA** build trees/toolchain. Suggested templates only (not commands executed here):

    cmake --build '<existing-PoCA-Release-build>' --config Release
    cmake --build '<existing-PoCA-Debug-build>' --config Debug

Rebuild all consumers of PerformanceProfiler/Camera headers, including executable, image and Atrous plugins; do not mix stale plugin binaries with the changed static-core class layouts. Use existing optional CUDA/Qt/dependency configuration; no new options/dependencies were introduced. To opt into Release routine GL checks, define POCA_ENABLE_RELEASE_GL_CHECKS for the desired targets through your existing local build settings; this task adds no project-wide option.

In **Preferences > Tests > images**, run **Render profiler, overlap and batch decisions** (ID image_render_performance). It covers independent graph-reference groups/order, cache hits/invalidation and zero hit pair checks, supported/unsupported settings/1024 decisions, nested frame/camera attribution, disabled silence/no CSV, reset/maxima, zero residency share scans, bounded 512-record history and actual background CSV header.

Run **Regional streaming and budgets** (ID image_regional_streaming): existing fixtures remain registered and now include the independent original slow-share formula, fixed-seed 320+ entry sequences across eight frames, repeated true/false touches, force toggles, insert/remove, budget changes, empty frames and eviction grace. Assert share equality for both high-resolution input values/all IDs after each operation. Run **Preferences > Tests > camera > Camera zoom and persistence** for existing checks plus pure overlay-culling/exact-generated-cache-key fixtures. These CPU fixtures are registered sources; none was executed. Existing GPU/picking fixtures remain manual.

## Manual GL regression and capture protocol (NOT RUN)

Use the same datasets, exact camera/view/viewport, crop/clip, grid flags, shader/gamma/steps and force-L0 settings in Debug and Release. Capture before-fix screenshots from an instrumented baseline if available; none was captured in this task. This final checkout includes the fixes, so an after-only capture is not a baseline or a speedup measurement.

| Scene | Actions and expected comparison |
| --- | --- |
| MyObject | Open ordinary RAW 3D; RAW+label; 2D/frame; filtered image + segmentation. Compare intensities/LUT/gamma/threshold, crop/clip, frame text, overlays and click selection. |
| MyMultipleObject separated | Use 2, 10, 50, 100, ~300 volumes; arrange a grid, then show only a small subset. Compare complete output, selection, child transforms and counters. Disable enabled filtering/label overlay flags on all images to exercise shared RAW batching when Atrous is installed. |
| Overlap | Two overlapping volumes, a connected chain, fully overlapping volumes; compare against uncached grouping/reference output without splitting the connected group. Include reorder/remove, moving child and changed detail ROI/resident bounds. |
| Grid/bbox | Both OFF, each ON separately, both ON; selected-child changes, heterogeneous flags, fonts on/off, rotations, zoom, crop, layout changes and resize. Check text/legend, facing, thickness, colors and boundary-touching lines. Close/reopen camera and issue freeGPU; check resource cleanup. |
| Streaming | Native pyramid and no pyramid; force-L0 on/off; navigation then >=150 ms settled. Verify preview retention, final detail/L0 and unchanged upload/request counts in settled frames. |
| Fallbacks | Different child gamma/steps/frame indices; path tracing/custom mode; overlays on/off; Atrous present/absent. Verify entire ImagesList family uses child rendering, labels/overlays appear once, order/model/crop/picking remain correct. |
| Capacity | MIP/direct 1024 candidates accepted; >1024 candidates use child fallback before any shared draw. The existing per-child RAW renderer still explicitly rejects >16 simultaneous renderable RAW images in one list; this task does not remove that separate limit or silently truncate it. |
| Picking/export | Click RAW/label/filtered MyObject and multi objects; single image commands retain no-hit, multi picker retains actual object selection. Resize before first pick. Export screenshot/movie/offscreen; inspect GL diagnostics and repeat camera close/reopen for leaks. |

Discard 30 warmup paint frames, record >=120 settled paints, then collect a separate matched navigation/resize trace. Reject CSV captures with csv_dropped > 0 or export errors. Use a fresh file per process to avoid reset frame-ID collisions. Optional standard-library-only summarizer (created, NOT RUN):

    python poca/docs/render_performance/compare_csv.py D:/captures/poca-before.csv D:/captures/poca-after.csv --warmup 30 --samples 120
    python poca/docs/render_performance/compare_csv.py D:/captures/nav-before.csv D:/captures/nav-after.csv --trace --warmup 30 --samples 0

For multiple cameras add --camera when the equivalent ID is the same, or use --before-camera / --after-camera to select different equivalent IDs; alternatively capture one camera at a time. Summarizer reports paint median/p95 and actual per-paint counters; GPU timing remains NOT MEASURED. A true before/after needs an instrumented pre-fix build/capture; no matching baseline artifact was produced here under source-only policy.

## Before/after: NOT MEASURED

No wall-clock/FPS/GPU/upload measurements or screenshots exist for this task. The following are **source-derived expected counts**, conditional on the scene reaching the indicated path; they are not profiler output:

| Work | Before source | After source expectation |
| --- | --- | --- |
| All 300 children grid+bbox OFF | Per-child recompute still traversed | 0 child_frame_recomputes / grid/bbox draws; separate initial frame-text recompute remains |
| Nonempty unchanged child overlays ON | Up to 3N child glBufferData calls per overlay pass | 0 per-child uploads after cache warmup; initial/current-child uncached updates can still contribute up to 6 total calls, independent of N |
| imageShare, N entries | 2N entry visits per call | 0 per-share visits; one frame aggregate traversal and original admission/grace/bytes walks remain |
| 300 ready visible disjoint RAWs, compatible shared path | 44,850 pair tests per topology rebuild | 0 pair tests on exact cache hit; miss retains original tests/groups/order |
| Ordinary single-image/filtered empty picking | Empty picking FBO allocation/clear work | 0 such FBO creates/clears; functional multi picking remains on demand |

## File-by-file inventory

All production edits are within the task's explicit allowlist. ImageDisplayCommand.hpp is necessary to expose a non-reused render identity plus resource-change version for cache invalidation; streaming lodRequestId is unchanged and no handles are cached. Existing ImageCameraStreamingTests.cpp is a permitted relevant image fixture. compare_csv.py is the permitted optional stdlib summarizer. The sole non-allowlist workspace file is CONTINUITY.md, required by the user's workspace ledger rules.

| Tree | File | Why |
| --- | --- | --- |
| poca | src/poca_core/General/PerformanceProfiler.hpp | Frame/metric/scoped timing/CSV interfaces and bounded shared state. |
| poca | src/poca_core/General/PerformanceProfiler.cpp | Opt-in collection, IDs, intervals, CSV I/O and shutdown. |
| poca | src/poca_core/Objects/MyMultipleObject.cpp | Measurement of existing child dispatch only. |
| poca | src/poca_opengl/OpenGL/Camera.hpp | Owned frame cache, context cleanup API and audited helper counters. |
| poca | src/poca_opengl/OpenGL/Camera.cpp | Frame boundaries/scopes, exact overlay cache, conservative culling and cleanup. |
| poca | src/poca_opengl/OpenGL/GLBuffer.hpp | Release diagnostic macro only. |
| poca | src/poca_opengl/OpenGL/ImageVolumeResidency.hpp | O(1) shares and truthful traversal counts; admission formulas unchanged. |
| poca | src/poca_opengl/OpenGL/LodUpdateManager.cpp | Actual request/worker/upload/queue/redraw event measurement only. |
| poca | docs/render_performance/README.md | Scope, configuration, schema, verification and regression report. |
| poca | docs/render_performance/compare_csv.py | Optional offline CSV comparison; not executed. |
| poca_extra | src/poca_imageplugin/ImageDisplayCommand.cpp | Actual draws and empty-pick no-hit/state preservation. |
| poca_extra | src/poca_imageplugin/ImageDisplayCommand.hpp | Non-reused render identity and separate monotonic resource-version accessor/state. |
| poca_extra | src/poca_imageplugin/ImageDisplayStreaming.cpp | Visibility/planning measurement and resource invalidation version. |
| poca_extra | src/poca_imageplugin/ImageDisplayStreamView.cpp | LOD planning CPU measurement only. |
| poca_extra | src/poca_imageplugin/ImageDisplayStreamUpload.cpp | Actual allocation/chunk/bytes/timing and final resource-version publication. |
| poca_extra | src/poca_imageplugin/ImagesListCommands.cpp | Audited single RAW draws, list CPU time and existing depth clear measurement. |
| poca_extra | src/poca_imageplugin/ImagesListMultiObjectDisplayCommand.hpp | CPU topology/settings responsibilities owned by existing command. |
| poca_extra | src/poca_imageplugin/ImagesListMultiObjectDisplayCommand.cpp | Topology cache, preflight/fallback/limit, actual draw/descriptor/picking counters and GPU-free cache cleanup. |
| poca_extra | src/poca_atrousplugin/RealTimeImageFilteringCommand.cpp | Audited draws/transfers and deterministic empty-pick handling. |
| poca_extra | src/poca_imageplugin/ImageStreamingBudgetTests.cpp | Independent original share formula and repeatable event equivalence fixture. |
| poca_extra | src/poca_imageplugin/ImageStreamingTests.cpp | Existing registry profiler/CSV/topology/settings fixtures. |
| poca_extra | src/poca_imageplugin/ImageCameraStreamingTests.cpp | Conservative overlay culling and exact owned-geometry key fixtures. |
| workspace | CONTINUITY.md | Mandatory dated/provenance continuity ledger. |

## Remaining risks and explicit diff review

Disjoint RAW groups still issue one volume draw per group/image. Connected overlap groups still perform per-fragment image loops; no measured GPU speedup is claimed for 300 overlapping images. CPU frame/text/legend generation remains per visible child despite GPU buffer reuse. Exact view-state changes conservatively repopulate the overlay cache. Depth clears, share stabilization and residency grace redraws remain; settling can require extra frames. Filtering flags cause conservative whole-family fallback even before filter textures exist, trading batching for the established overlay semantics. >1024 family preflight may decline disjoint candidates that would fit individually; no output is silently truncated. The separate per-child >16 RAW limit remains an explicit existing error.

The cache is capped at 512 child entries and 64 MiB of tracked entry/owned CPU capacity/requested GL payload, plus bounded map/driver bookkeeping; oversized geometry uses original uncached buffers. Cache bytes are not physical driver VRAM telemetry. Entries are purged on child removal, all overlays disabled, context teardown/camera destruction and freeGPU; GL names are uniquely owned and released on the GUI thread with the camera context current/restored. Descriptors, textures and scientific data are not retained in the topology cache.

Source diff review: no scientific coordinate/voxel/data-model changes, no changed shader source/math, sampling/gamma/compositing/threshold/clip/depth policy, image quality target or LOD algorithm changes; no skipped visibility touches; no new dependencies; no unrelated GUI/CMake/toolchain/generated-tree changes. No additional image texture or picking FBO allocations were introduced. The **authorized B1 owned overlay cache necessarily adds bounded GL buffers/CPU keys**; profiler adds bounded diagnostic memory and optional CSV I/O. Thus a blanket claim of zero new resource allocations would be false. No per-image/unconditional diagnostic logging or GL synchronization/readback/query was added; CSV failures are diagnosed only after export opt-in. Actual pixel equivalence, resource-pressure behavior, plugin ABI rebuilding, GL errors and runtime performance remain unverified until the user-run matrix passes.

## Mesh/detection picking — separate Stage 1 checkpoint (2026-10-26)

[Stage 1 findings, complete diff inventory, capture recipe and regression matrix](MESH_DETECTION_PICKING.md).

This subsequent change extends the existing profiler with type-separated mesh/detection picking counters, explicit pick-event/resource rows and bounded asynchronous GPU timestamp samples. CSV schema 2 appends fields while retaining all previous columns. Earlier sections above describe the preceding profiler/optimization checkpoint; the linked report documents the new targeted GPU measurements and their limitations, including the unchanged whole-frame `gpu_timing_available=0` field. `compare_csv.py` remains paint-only and unchanged.

**Baseline CPU/GPU cost: NOT MEASURED. Stage 2: NOT STARTED.** Source-only workspace policy prevents compilation/runtime capture here. Legacy per-paint picking/FBO allocations and synchronous hover/click behavior remain; a real instrumented legacy capture and selection reference are required before demand-driven behavior changes. No measured speedup is claimed.

## Mesh/detection picking — Stage 2 stop receipt (2026-10-26)

The supplied IND83 baseline authorizes Stage 2 investigation: 332 redundant picking draws and ~18 ms CPU work per onscreen pass, with 332 persistent targets. The 249-vs-83 mesh resize discrepancy is explained by ObjectListsCommands displaying all entries but forwarding resize/pick only to the current list.

Stage 2 was stopped under the user's equivalence rule: SSAO's component-specific display bypasses the normal object-model push, and DetectionSet's legacy picking pass changes the framebuffer seen by subsequent visible child draws. An unconditional on-demand replacement cannot claim unchanged selection/visible rendering across those paths. The draft production/test changes were restored; Stage 1 behavior/profiler remain intact. See [MESH_DETECTION_PICKING_STAGE2.md](MESH_DETECTION_PICKING_STAGE2.md) for exact call sites, required decision and NOT RUN matrix. No compilation, updated runtime validation, comparable Stage 2 capture or measured improvement occurred.

## GPU/component picking disabled — diagnostic policy (2026-10-27)

The replacement request supersedes demand-driven Stage 2. Production source now retains the full legacy implementation behind one central default-false constant, poca::core::POCA_ENABLE_PICKING in General/PickingPolicy.hpp. Component/camera picking, ID framebuffers, readbacks and picking resize dispatch are guarded; visible rendering and non-picking SSAO/filter/export/explicit preview targets remain active. SSAO explicitly binds its geometry target when picking is disabled.

[Picking shutdown implementation, complete source inventory and NOT RUN matrix](PICKING_DISABLED.md). The 305-dataset runtime/build remains untested under the never-compile workspace rule; source changes are retained. Set the constant true and consistently rebuild consumers to restore legacy picking. No demand-driven or shared-target mechanism was introduced, and no measured improvement is claimed.

## Generic collection planning and shared image update (2026-11-02)

This section supersedes the earlier whole-family compatibility and 1024-candidate descriptions for collection **display**. Starting source: `270317432f0f793c6d35cd9f3dd683b8bc6b2a8e`. Implementation is source-only: no configure, compilation, tests, PoCA, helper scripts, GL captures or benchmark runs were performed. Build success, pixel equivalence, context recreation and performance acceptance remain UNCONFIRMED. No FPS or speedup is claimed.

### Baseline audit and implementation

The baseline shared image provider rejected the complete ImagesList family for missing providers, incompatible settings, live overlays or more than 1024 selected RAW candidates, before visible spatial partitions. RAW preparation traversed the collection again. Cached overlap topology and shared cube geometry already existed, but each batch uploaded its complete descriptor array every frame. The MIP shader kept 1024 maxima per fragment and sampled every image at every union-ray step. Legacy child display composes scene * parent * child; shared providers previously cancelled the parent with inverse(parent) * child.

`CollectionRenderPlan` now inventories arbitrary child components and one existing BasicComponentList level without image/plugin dependencies. Each runtime item carries child/component/sub-element identity, current selection, local/parent/world and projected bounds, effective matrices, disposition and provider-populated resource/LOD metadata. Bounds use all eight corners. References are borrowed for one synchronous frame and never serialized. Unknown components remain normal plugin dispatch. Disabled batching retains lightweight legacy child ordering without enumerating sub-elements. Empty lists are safe.

`MyMultipleObject::executeGlobalCommand` creates this plan before parent/shared display commands, installs it in CommandExecutionContext, then dispatches remaining children in the established stable far-to-near order. `MyObject` skips only fully handled components, while ImagesListCommands skips individually handled images in RAW, ordinary and overlay passes. Family markers remain available for existing complete-family providers, are checked for the requested family in component-specific dispatch, and do not suppress non-render commands. No Qt operation or new command/macro/serialization API was introduced.

The existing image command owns `DisplayPlan`. It observes each image through existing visibility/crop/residency functions before initialization, LOD submission, descriptor gathering and overlap work. Each image keeps its own selection, histogram, thresholds, LUT, interpolation and texture handles. Screen-connected regions with conflicting settings or unsupported providers/overlays remain entirely child-rendered; unrelated compatible regions can still batch. Conservative projected barriers prevent newly shared regions from crossing overlapping legacy geometry/plugin draws. Individual labels keep their existing renderer. 2D/frame, path-traced/custom and unsupported overlay regions retain legacy dispatch. Disabled/culled/deferred items are distinguished from actual draws; no whole-family claim is made after partial work.

Shared RAW gathering consumes item indices instead of rescanning child lists. Existing spatial overlap topology, adaptive ray-step calculation, shared unit cube, blend/depth policy, root crop/clip uniforms and end depth clear remain. Shared matrices now compose parent * child consistently with legacy display; geometry providers receive only the minimal matching child-local matrix correction and live batching-option gate. Their GPU algorithms and finer display toggles are unchanged. Picking remains centrally disabled.

MIP now keeps one scalar maximum per image and skips sample loops for conservatively nonintersecting image-local rays. The exact repeated-add sample grid, per-channel thresholds/LUT calculations, missing-channel sentinel treatment and final channel addition order are retained. Other volume shaders are unchanged. There is no MIP local-array channel limit. Direct/label shader limits and the device SSBO block capacity are checked on visible spatial groups; safely disjoint groups can split, inseparable over-limit groups delegate without truncation.

SSBO storage is keyed by non-reused image identity, reallocated only for capacity growth and updated only over the changed descriptor-row span. Static payloads have no SSBO upload. Transform changes update matrices/bounds without requesting voxel reload solely for movement; existing projected-LOD decisions may legitimately request a different level/ROI. Other children retain textures. Removed descriptor batches are pruned. The shared command owns its buffers and context cleanup connection; explicit freeGPU/context teardown also releases live image/list GPU state through existing commands, with owning-context switching/restoration. Frame plans release borrowed provider references even on exceptional exit. No second streaming/resource manager was added.

Label streaming now reads the existing configured `labelMode` instead of unconditionally substituting Nearest. Existing reducers are reused, Average is rejected for categorical labels, and no voxel values, label IDs, pyramid generation/storage or reduction algorithms changed. Existing initialization (4) and LOD submission (4 interactive / 16 settled) budgets, pending-target deduplication, progressive refinement and force L0 remain.

### Source inventory

| Files | Responsibility |
| --- | --- |
| poca_core/General/CollectionRenderPlan.hpp, .cpp (new); core CMakeLists.txt | Plugin-neutral frame inventory, bounds, child order and element ownership; source registration. |
| poca_core/Objects/MyMultipleObject.cpp, MyObject.cpp; poca_opengl/OpenGL/RenderCommandContext.hpp | Frame context, remaining-child/component dispatch and render-only filtering. |
| poca_core/General/PerformanceProfiler.hpp, .cpp | Eight appended scalar counters; telemetry CSV schema 6. |
| ObjectListMultiObjectDisplayCommand.cpp; DetectionSetMultiObjectDisplayCommand.cpp; VoronoiMultiObjectDisplayCommand.cpp; DelaunayTriangulationMultiObjectDisplayCommand.cpp | Child-local transform composition and existing batching-option checks only. |
| poca_imageplugin/ImagesListCollectionPlan.cpp (new); ImagesListMultiObjectDisplayCommand.hpp, .cpp | Image eligibility/projected regions, shared RAW gather, descriptor spans and owned context cleanup. |
| poca_imageplugin/ImagesListCommands.hpp, .cpp | Skip only frame-owned individual images; ordinary single-object behavior retained. |
| poca_imageplugin/ImageDisplayCommand.hpp; ImageDisplayStreaming.cpp | Already-observed view request entrypoint and configured categorical reduction. |
| poca/src/shaders/maximum_intensity_projection_multi.frag | Scalar per-channel maxima and conservative image-local ray rejection. |
| poca_imageplugin/CollectionRenderingTests.cpp (new); ImageStreamingTests.hpp, .cpp; image CMakeLists.txt | Existing fixture registration, CPU ownership/transform/partition/reference cases. |
| poca/docs/render_performance/README.md; workspace CONTINUITY.md | Report, unrun acceptance/benchmark procedure and required continuity record. |

### Verification and deliberate limitations

The existing **images > Render profiler, overlap and batch decisions** fixture now includes heterogeneous children with two RAW plus one label image, a flat image, mesh, DetectionSet, future-plugin component and no-image child; independent A/B/C selection and arbitrary parent/child rotations; eight-corner bounds; partial versus complete ownership; non-render forwarding; mixed gamma/unsupported projected regions; 1100 MIP descriptors; direct/device-capacity spatial partitions and inseparable fallback; unchanged/transform/histogram descriptor spans; exact CPU MIP loop-schedule reference. These are authored CPU fixtures, **NOT RUN**, and do not establish GL draw or streaming equivalence. Existing regional streaming/navigation/camera fixtures remain available, also NOT RUN.

The legacy per-child ImagesList RAW renderer still explicitly rejects more than 16 simultaneously renderable RAW images in one list. Supported shared MIP is no longer constrained to that number, but an unsupported/inseparable region delegated to such an oversized legacy list can still raise that pre-existing error. This is an explicit remaining capability limitation, not silent truncation. No universal arbitrary-mode/count or completed runtime definition-of-done claim is made.

Planning is a lightweight per-frame inventory; transformed metadata is recalculated on camera/transform changes. Topology can still require quadratic pair checks on misses, and CPU descriptor payloads are still gathered before exact span comparison. DetectionSet/Voronoi/Delaunay retain their existing geometry-buffer update behavior. Shared display requires the current owning camera context; simultaneous unrelated contexts and full camera/shader recreation require manual validation. The existing SSAO-specific component dispatch convention is not refactored. No new nonblocking whole-frame GPU timer was added: gpu_timing_available remains 0.

New scalar counter meanings: collection_candidate_items / collection_selected_items cover generic items; image_plan_visible covers provider visibility observations; image_plan_batched counts RAW elements actually issued in shared draws; image_plan_fallback counts selected items delegated to child regions plus individual visible label dispatches (not measured GL draw count); image_plan_groups counts accepted regions including culled/deferred-only regions; shared_descriptor_cache_hits counts identical batch payloads; shared_geometry_allocations counts two initial cube buffers. Existing draw, descriptor allocation/upload bytes, overlap, visibility, LOD, texture/upload, child dispatch and CPU interval counters remain the actual work evidence. Compare common columns across CSV versions, not numeric column positions.

### Reproducible before/after procedure (NOT RUN)

1. Build/deploy the baseline commit and this checkout separately with the same existing Release toolchain and dependencies. Include the modified shader and rebuild all consumers of changed core/plugin headers. Use the same driver, RTX 3090 24 GB, dataset, viewport, camera state, crop/clip, interpolation, LUTs, modes, ray steps, force L0 and streaming budgets. Keep batchComponentRendering enabled for the shared-path comparison; repeat with it disabled and with a single MyObject. Existing OME-Zarr auto-batching defaults/loaders were not changed.
2. Enable renderPerformance and use a fresh POCA_RENDER_PROFILE_CSV path per process/capture. Record cold loading/preview readiness separately from warmed rendering. Compare the motivating 305-child / approximately 1220-image scene with meshes OFF and ON. Also use heterogeneous/no-image children, different channel counts, mixed 2D/3D, independent selections and 1100 selected RAW images. No implementation constants depend on these scene counts.
3. After preview/detail settles, discard 30 warmup paints and collect at least 120 stationary paints. Capture separate matched 30-second traces of continuous rotation, translation, overview-to-close zoom, repeated rotation/translation of one child, and image/mesh toggling during navigation. Toggle only B/RAW2 in the three-child table and verify A/C textures retain their handles and have no unrelated allocation/upload work. Track both cold and warm GPU memory with the same external GPU tool; cache byte counters are not physical VRAM.
4. Use the existing compare_csv.py on before/after captures (see command templates above), separately for stationary and trace data. Report CPU paint p50/p95, actual RAW/nonraw/geometry draw counts, batch sizes, descriptor allocation/upload calls and bytes, cache hits, pair tests, texture allocations/upload bytes, LOD enqueue/dedup/deferred counters and frame cadence/FPS. Reject dropped/export-error CSV captures. Export profiler histories for new columns if the older comparison utility does not summarize them. Measure GPU rendering with an external nonblocking Nsight/driver capture; do not add per-frame glFinish. No whole-frame GPU timing is provided by this change.
5. Compare stationary fully refined screenshots against legacy batching-OFF output and baseline batching-ON output separately. Cover RAW MIP/direct/alpha/isurface, frame/2D/path tracing, overlapping labels/RAWs, mixed gamma/steps, live overlays, translucent/wireframe meshes, root plus child rotation, crop/clip, perspective/orthographic, SSAO, offscreen export and resize. Assert original categorical IDs and configured Nearest/MIP/Majority reduction results independently of screenshot similarity. Inspect depth/blending/overlay order and check every selected element exactly once.
6. Run the existing image profiler, regional streaming/budget, navigation lifecycle/rendering and camera fixtures plus the new registered CPU cases. Exercise rapid zoom/force L0/refinement, removal/addition, freeGPU, close/reopen and context recreation. Picking must remain disabled under the current policy. Record build/test results and GL diagnostics explicitly before accepting runtime equivalence.

Before/after CPU, FPS, GPU, VRAM, preview readiness and screenshot results: **NOT MEASURED**. Remaining source cost candidates are connected-group ray sampling, miss-time overlap analysis, CPU descriptor gathering and unchanged non-image buffer updates; none is a measured bottleneck. The smallest justified next step is the matched profiler/GPU capture above, then optimize the dominant measured cost while retaining this equivalence matrix.
