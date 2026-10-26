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
