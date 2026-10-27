# Compile-time GPU/component picking shutdown

2026-10-27. User request: c4db6295-7177-466b-908b-a8f27af8299f/Pasted text.txt. This supersedes the abandoned demand-driven Stage 2 task and its selection/SSAO equivalence gate. **Production C++ changes are implemented and retained.** No demand-driven renderer, shared picking target, replacement selection algorithm, CMake option or GUI setting was added.

## Policy and behavior

The one central constant is `poca::core::POCA_ENABLE_PICKING = false`, in `poca/src/poca_core/General/PickingPolicy.hpp`. It is included through BasicDisplayCommand and Camera, and is directly includable from core and plugins through the existing core include root. Set it to true and consistently rebuild all affected OpenGL/plugin/header consumers to restore the retained legacy path. No original method, class or public signature was removed or changed; two protected base-command guard overloads were added. No CMake change is required to include the header.

When false:

- BasicDisplayCommand `pick()` immediately sets ID -1 and returns before framebuffer binding, both `glReadPixels` calls, or saveImage work. `updatePickingFBO()` immediately returns without creating/replacing a target. Its original bodies remain beneath the guards.
- `skipDisabledPickingCommand()` clears the stored ID for disabled command execution. Pick, picking-buffer updates, current-picked-ID/info queries and set-picked-ID commands return immediately. Typed requests replace reused result objects with the existing no-hit contract: an empty CommandExecutionResult, without selected ID/info/point/bbox/preview payloads.
- Mouse-derived `doubleClickCamera` is disabled. An explicit request carrying `objectID` can still use the original object preview/zoom path and its separate 400x400 framebuffer. It does not require or allocate an ID-picking framebuffer.
- All 20 discovered component `drawPicking()` implementations return before their original code. All 21 live calls are flag-gated, so ordinary display bypasses the methods entirely. Eager `drawElements` target-allocation calls are also flag-gated; the base allocation guard independently protects every caller.
- The three camera picking-shader templates return before shader lookup, GL state changes, buffer access or draws. Picking GPU query submission/poll methods are guarded; query cleanup remains active. The profiler implementation, schema and existing counters are retained unchanged.
- `Camera::dispatchProfiledPick()` preserves input-event profiling, clears camera picked-info/points and the returned result, then returns without component dispatch. Hover-linked Voronoi selection is gated. Right-click picked-object preview dispatch and world-position mouse double-click selection are gated; navigation, ROI handling, wheel zoom, transformations and ordinary redraw remain active.
- The entire picking resize scope/broadcast is skipped. Camera offscreen framebuffer resize, SSAO framebuffer resize, matrix recalculation, maximizing/resizing and update scheduling are retained.
- Multi-object ObjectList/DetectionSet/Delaunay/Voronoi bounding-box selection and merged ImagesList GPU selection are blocked at their command entry, before candidate traversal, selection mutation, context acquisition, private VAO creation or FBO use. The base bounding-box pick helper also independently returns no-hit.

No component is hidden or removed. Normal display and visible draw functions, buffers/features, image/mesh/detection shaders and data computations are retained. Selection overlays driven by a picked ID intentionally have no picked ID while disabled. Table/data selection and normal commands are not replaced with another mouse-picking mechanism.

## Guarded component owners

Core plugins: ObjectList, ObjectListMultiObject, DetectionSet, DetectionSetMultiObject, DelaunayTriangulation, DelaunayTriangulationMultiObject, VoronoiDiagram, VoronoiMultiObject, ColocTesseler and ObjectColocalization.

Extra plugins: ImageDisplay, ImagesListMultiObject, ATrousDisplay, RealTimeImageFiltering, SkeletonDisplay, TrackSet, TrackSetPalmTracer, NeurosphereDisplay, NanoSynAtlasDisplay, NanoSynAtlas3D, SpineObjectCommands, GraphAnalysis, Organoid and Ticker.

ATrous/Organoid's currently commented or empty picking bodies remain intact, with their live display calls bypassed. The FitEllipsoid files contain commented picking calls only and were left unchanged. VMAS/NeuroExpansion have no active GPU picking implementation or own pick-result branch; any inherited base allocation/read requests are covered centrally. UI consumers of get-picked-ID already check result availability/validity and receive no-hit; they were inspected without GUI edits.

The whole-tree readback audit found exactly two `glReadPixels` calls, both inside guarded BasicDisplayCommand::pick. The only persistent ID-picking framebuffer construction is in guarded BasicDisplayCommand::updatePickingFBO. The only pick-shader lookups are in the three guarded Camera templates. No other picking renderer/allocation/readback implementation was found in either source tree.

## SSAO and normal framebuffer preservation

With the policy false, Camera::drawSSAO explicitly binds `m_fboGeometry` before the object display dispatch and again before DetectionSet display dispatch. This makes the required target explicit after removing the picking pass's framebuffer side effects. SSAO shaders, passes, mathematical operations, model transforms, scene-depth clears and child dispatch are unchanged. The two binds are conditional on disabled picking; true retains the legacy target behavior.

Normal image path-tracing/filtered-image targets, heatmap target, camera offscreen/export target, SSAO geometry/noise/blur/halo/silhouette/lighting targets, plot framebuffer and explicit ObjectList preview/zoom framebuffer remain active. Their allocation, rendering, export and cleanup implementations were not disabled. A framebuffer named m_pickOneObject is the explicit object preview, not a persistent ID target; it is intentionally preserved.

Existing picking FBO destruction/freeGPU, resource census retirement, GPU timer cleanup and context ownership rules remain unchanged. Cleanup is not short-circuited by the policy. A consistent false build starts with null per-component pick targets and cannot allocate their attachments through the guarded code paths.

## Verification

Source-only checks actually performed:

- 24 complete component source files compare to HEAD after removing only the new guard lines/conditions; final newlines/trailing whitespace are normalized for that comparison. All original implementation code remains available when true.
- 20 picking methods, 21 call sites and three camera shader overloads have guards; base FBO constructor/readback bodies are retained exactly beneath their entry guards.
- Base destruction/census/query cleanup, Camera paintGL/drawElements/wheel/query cleanup bodies match HEAD. SSAO matches HEAD after excluding only the two explicit target binds and their comment.
- One default-false constant, one guarded picking FBO constructor, two guarded pixel readbacks and fully guarded picking resize scope were verified by whole-tree source searches.
- All 30 changed/new C++ streams passed lexical delimiter, strict UTF-8, CRLF and original tracked BOM checks. `git diff --check` and targeted quoting/merge artifact searches passed.
- No loader, 305-dataset layout, scientific data/coordinates, geometry, shader, pyramid, LOD, residency, general dispatcher, dependency or generated build-tree file was edited.

The existing registered `image_render_performance` fixture in ImageStreamingTests.cpp was extended with disabled-policy checks: direct pick/saveImage and resize without a component/camera/context; clearing a deliberately stale stored ID and reused result payloads; safe missing parameters; retained normal display and explicit object-ID preview command eligibility; active freeGPU cleanup. It skips these diagnostic-only assertions when the flag is true, so it does not execute a legacy GL path without a context. Its registration and CMake membership are unchanged.

| Build/runtime acceptance | Status |
| --- | --- |
| Debug / Release build and consistent updated DLL load | NOT RUN — explicit never-compile workspace instruction |
| Existing fixture plus new disabled-picking assertions | ADDED, NOT RUN |
| Open merged OME-Zarr with 305 datasets / 915 meshes / 305 DetectionSets | NOT RUN |
| Measured zero mesh/detection/all-component picking draws | NOT RUN; guarded out by source |
| Measured zero picking allocations/reallocations/live attachments/readbacks | NOT RUN; guarded out by source |
| Fullscreen resize/maximize and zero picking work | NOT RUN; picking scope/broadcast guarded, normal resize retained |
| Hover/click/direct pick safety and no-hit results | NOT RUN; source guards and fixture added |
| Normal 2D/3D images/meshes/detections/streaming/navigation | NOT RUN; original rendering logic retained |
| SSAO, normal filtering/export and explicit preview/zoom | NOT RUN; non-picking buffers retained, minimal SSAO bind correction |
| True-policy legacy runtime behavior | NOT RUN; legacy bodies retained and compared statically |

No configure/build/compiler/linker, PoCA, test, benchmark, Python/helper executable, dependency installation or backend rebuild was run. PowerShell and Git were used only for source editing and static verification. No measured timing, VRAM reduction, GPU speedup or visual-equivalence claim is made. Runtime acceptance requires a user-built consistent diagnostic binary; production changes are not reverted because that runtime is unavailable.

## Exact retained C++ file inventory

- `poca_extra/src/poca_atrousplugin/ATrousDisplayCommand.cpp`
- `poca_extra/src/poca_atrousplugin/RealTimeImageFilteringCommand.cpp`
- `poca_extra/src/poca_imageplugin/ImageDisplayCommand.cpp`
- `poca_extra/src/poca_imageplugin/ImagesListMultiObjectDisplayCommand.cpp`
- `poca_extra/src/poca_imageplugin/ImageStreamingTests.cpp`
- `poca_extra/src/poca_nanosynatlasplugin/NanoSynAtlas3DCommand.cpp`
- `poca_extra/src/poca_nanosynatlasplugin/NanoSynAtlasDisplayCommand.cpp`
- `poca_extra/src/poca_nanosynatlasplugin/SpineObjectCommands.cpp`
- `poca_extra/src/poca_neurosphereplugin/NeurosphereDisplayCommand.cpp`
- `poca_extra/src/poca_organoidsimulationplugin/GraphAnalysisCommand.cpp`
- `poca_extra/src/poca_organoidsplugin/OrganoidCommand.cpp`
- `poca_extra/src/poca_skeletonplugin/SkeletonDisplayCommand.cpp`
- `poca_extra/src/poca_tickersplugin/TickerCommand.cpp`
- `poca_extra/src/poca_tracksetpalmtracerplugin/TrackSetPalmTracerCommand.cpp`
- `poca_extra/src/poca_tracksetplugin/TrackSetCommand.cpp`
- `poca/src/poca_coloctesselerplugin/ColocTesselerDisplayCommand.cpp`
- `poca/src/poca_core/General/PickingPolicy.hpp`
- `poca/src/poca_delaunaytriangulationplugin/DelaunayTriangulationDisplayCommand.cpp`
- `poca/src/poca_delaunaytriangulationplugin/DelaunayTriangulationMultiObjectDisplayCommand.cpp`
- `poca/src/poca_detectionsetplugin/DetectionSetDisplayCommand.cpp`
- `poca/src/poca_detectionsetplugin/DetectionSetMultiObjectDisplayCommand.cpp`
- `poca/src/poca_objectcolocalizationplugin/ObjectColocalizationCommands.cpp`
- `poca/src/poca_objectlistplugin/ObjectListDisplayCommand.cpp`
- `poca/src/poca_objectlistplugin/ObjectListMultiObjectDisplayCommand.cpp`
- `poca/src/poca_opengl/OpenGL/BasicDisplayCommand.cpp`
- `poca/src/poca_opengl/OpenGL/BasicDisplayCommand.hpp`
- `poca/src/poca_opengl/OpenGL/Camera.cpp`
- `poca/src/poca_opengl/OpenGL/Camera.hpp`
- `poca/src/poca_voronoidiagramplugin/VoronoiDiagramDisplayCommand.cpp`
- `poca/src/poca_voronoidiagramplugin/VoronoiMultiObjectDisplayCommand.cpp`

This list contains 29 production files (including the new central policy header) and one existing registered fixture. Documentation updates are CONTINUITY.md, an append to poca/docs/render_performance/README.md and this new report. The prior untracked MESH_DETECTION_PICKING_STAGE2.md audit and earlier README/ledger work are preserved as historical material; their abandoned scope is superseded by this request. Two unchanged header status/cache markers from the previous turn are not production content changes in this task.

No commit was created. Potential commit: `Disable GPU picking behind a central compile-time diagnostic flag`.
