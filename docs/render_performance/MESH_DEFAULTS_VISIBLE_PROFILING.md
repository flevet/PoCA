# Multi-object mesh defaults and visible rendering profiling

2026-10-27. Implemented C++ source; compilation, fixtures, PoCA execution, visual regression and performance captures **NOT RUN / NOT MEASURED**. Workspace source-only rules apply. No generated/build/dependency/backend state was changed.

## Part A: precedence and scope

`MyMultipleObject` construction visits every child component and recursively visits `BasicComponentList` entries. The stable core `CommandableObject::name()` identifies `ObjectListMesh`, independent of editable component/list names. Only `pointRendering`, `outlinePointRendering` and `ellipsoidRendering` are loaded as false through the existing command parameter interface, without recording or GPU work. Missing children/components/lists are safe. There is no core-to-geometry/plugin dependency.

The hook runs once, only for more than one child, independently of batching. Both `Engine::generateMultipleObject` (Data Assembler) and `Engine::assembleMultipleObject` reach it after child commands exist. OME-Zarr `loadObject` restores child characteristics before constructing the collection. Later parent manifest restoration changes the parent transform/layout/hierarchy/current child; parent command restoration enumerates parent owners, not child owners. Thus saved child ON values yield to initial OFF in a newly opened collection. Standalone and one-child collections retain their previous command settings. No loader or save-format changes were necessary.

`ObjectListsWidget::update` and `ObjectListWidget::update` already block signals and read each current component's real command parameters into the three buttons. Existing GUI/recorded commands remain usable after initialization; switching lists/children and rendering never reapplies defaults. Shape/fill/alpha/selection/visibility/skeleton/link/DetectionSet/image settings and global preferences are untouched by production code. Children retain existing parent ownership; no new resource ownership was introduced.

## Part B: measurement definitions

Schema **3** contains 468 metric columns plus the existing 18 identity/status columns = **486 fields**. Existing field identities and counters remain intact. Four fixed blocks use these prefixes:

- `objectlist_individual_`: standalone or per-child renderer; use `object_kind` to distinguish `MyObject` from multi-object fallback/disabled batching.
- `objectlist_shared_`: shared family attempt/handling and its actual submissions.
- `detectionset_individual_` and `detectionset_shared_`: equivalent DetectionSet paths; no DetectionSet defaults changed.

Each block has display CPU microseconds, framebuffer-offscreen calls, visited/selected/submitted components, unselected/no-domain observations, build and transform costs, decision calls/reasons, and per-pass CPU/draw/submitted/OFF/empty counters. Passes are `points`, `outline`, `opaque`, `translucent`, `ellipsoids`, `skeleton`, `links`, `bbox`; irrelevant combinations remain zero. Shared `ranges` counts range invocations (including synthetic transparency ranges), while shared DetectionSet ranges describe contributing source ranges. OFF/empty values describe pass opportunities at the instrumented boundary, not unique object totals.

Timings are steady-clock **inclusive CPU wall time**, including driver submission cost, not GPU execution. `display_us` contains build/decision/transform/pass times. Shared translucent time includes the unchanged transparent traversal/sort plus its draw passes; same-metric nested timers suppress duplication. Individual 2D helper/shape timings similarly suppress same-metric nesting. Do not add these inclusive values, or add child dispatch to component display, as disjoint frame costs. Independent pass timings do not include all state setup/teardown; inspect the enclosing display time too. Existing general Performance-widget scoped measurements remain intact.

Actual GL accounting:

- Direct `glDrawArrays`, `glDrawElements`, `glDrawElementsInstanced` and `glMultiDrawArrays` sites in the allowed ObjectList source are counted immediately beside submission. One multi-draw API invocation is one `gl_draws`; `draw_ranges` also counts its constituent ranges.
- Camera's already-audited sphere/simple/uniform helpers increment `other_audited_gl_draws` at actual internal GL call sites. `VisibleHelperScope` measures their scoped deltas, including empty-helper early returns and DetectionSet's existing two draws for `RandomOneColor`. Primitive metadata comes from the same buffers/branch. No Camera/header/wrapper modification was needed. The helper subset overlaps the existing global counter: **do not add helper draws to `other_audited_gl_draws` again**.
- `indexed_triangle_indices` and `array_triangle_vertices` count submitted GL_TRIANGLES elements separately. `submitted_triangles` divides those counts by three; `wire_triangle_elements` identifies polygon-line submissions. Multi-draw triangle ranges retain their existing triangle-aligned counts.
- Points count GL_POINTS inputs, not expanded sphere triangles. Lines count GL_LINES vertices. Ellipsoids retain GL_QUADS: `submitted_quad_indices`, `submitted_quads` and `ellipsoid_instances` include instances and repeated submissions; no invented triangle equivalent is reported. Per-pass `submitted` counts input elements multiplied by instances/calls, not visible pixels, surviving vertices or fragments.
- `components_drawn` means a component contributed to an actual submission, including sentinel/filtered data later discarded by shaders. Shared ObjectList source membership is inspected only with profiling enabled, including mesh skeleton/link metadata; individual counts use actual GL deltas. Shared source census occurs on a successfully prepared family; rejected attempts are represented by fallback counters and the subsequent individual renderer.

There is **no component frustum test** in these audited visible paths. `no_frustum_test` explicitly counts those candidates. Shared triangle spans test selection; `triangle_visibility_checks`/`triangle_unselected_spans` describe that predicate, not camera culling. Point/outline/ellipsoid/skeleton/link submissions can still contain off-camera or shader-discarded inputs. Primitive counts measure all that submitted work, including outside-view geometry; no new culling or speculative screen-visibility classification was added. `offscreen` record identity and `offscreen_calls` instead identify offscreen framebuffer rendering.

Scene overhead retains existing child-dispatch, grid-draw, bbox-GL and camera counters. New local/collection bounds and collection grid/preparation counters time the existing traversal, transform/depth preparation and grid work. `bbox_preparation_*` measures selection-box preparation separately from actual bbox helper draws. Between-frame work retains existing `between_*` reporting rather than being attributed to a fictitious frame.

Shared `builds` and `build_us` identify first build/rebuild attempts; geometry cardinalities describe completed buffer preparation. `objectlist_shared_build_admitted_estimated_peak_bytes` is the **existing six-copy admission estimate**, not measured allocation/residency; memory-rejected builds do not publish an admitted size. DetectionSet `build_bytes` is the combined submitted vector payload, excluding allocator overhead. Individual ObjectList `build_ellipsoid_instances` exposes retained ellipsoid allocation even with its draw switch OFF. Shared points/outlines still allocate while OFF; shared ellipsoids already consult their individual flags. Allocation policies were not changed; pursue an allocation optimization only after measuring build costs/cardinalities.

## Actual fallback provenance

`multi_batch_fallback_limit` currently has **two** increment sites: `ImagesListMultiObjectDisplayCommand::compatibleForDisplay`'s `!common.fitsShader(selectedRaw)` capacity branch, and the pre-existing ObjectList rebuild `!withinBudget` branch. It is not an exclusive mesh-memory diagnosis. Both existing increments remain; `objectlist_shared_fallback_memory` now identifies the latter exactly.

ObjectList `can_batch_*` observes its existing child-presence/dimension checks. DetectionSet observes child presence and matching normal/color availability. `fallback_settings` identifies those existing compatibility failures, or DetectionSet's reference `pointRendering=false` return; it does not invent stricter compatibility checks. `missing_command` records existing missing-reference observations. `fallback_build` identifies an existing failed rebuild/empty-range return, with the memory reason additionally distinguished. `fallback` counts actual unsuccessful display attempts; decision calls/reasons can nest during a rebuild and must not be summed as unique failed frames. `handled` records publication of the existing family-handled result. `objectlist_batching_disabled`/`detectionset_batching_disabled` identify the root dispatch policy when shared commands are not installed.

## GPU status and exact capture method

No new normal-rendering GPU queries were introduced. `*_gpu_deferred` marks submitting display invocations with unavailable GPU timing; it is not milliseconds or zero GPU cost. Existing picking metrics/queries remain untouched and picking remains `POCA_ENABLE_PICKING=false`.

The established query pool and teardown belong to Camera and are explicitly picking-family scoped. Reusing it would mislabel normal work as picking. These component classes have no allowed header changes to store context-owned pending queries; adding a second query/context lifecycle to core would introduce new GL ownership/dependencies. Accurate shared family totals also span interleaved mesh/decorations and child/image work. This task therefore does not add unsafe/nested queries, change ordering, or extrapolate representative samples.

For GPU comparison, capture a **single matched settled frame** in a context-aware OpenGL GPU debugger, then separately capture a matched navigation/upload frame. Request per-event GPU durations and inspect the actual GL draw events using `objectRendering*`, `multiObjectObjectRendering*`, `sphereRendering*`, `multiObjectSphereRendering*` and `*3DInstanceRendering*` shaders; include SSAO geometry variants when used. Record event IDs, draw mode/count/indexing/instances, renderer, framebuffer and duration. Sum only disjoint draw-event intervals for a family/pass; do not sum nested ranges or extrapolate one mesh to a whole scene. Save the capture plus GPU/driver/build/shader mode/camera/settings, and mark unsupported/unavailable timings explicitly. Keep picking off and do not insert `glFinish`, `glGetError`, blocking query reads or new debug groups into the rendering code.

## Capture and acceptance checklist (runtime checks unrun)

Enable existing Preferences verbose type **renderPerformance** and optionally set `POCA_RENDER_PROFILE_CSV` to a fresh absolute filename before launching the user's rebuilt application. Disable/normal quit drains the existing writer. Fixed storage remains 512 history records, 256 CSV queue rows and 16 inline nested frames with the existing spill path. New scopes allocate no per-component diagnostic storage or strings. Labels are generated once on report/CSV access. Console summaries retain the existing >=1-second interval and suppress zero-valued new family fields; CSV retains all zero columns. Logging loss/errors stay explicit; reject comparisons with CSV drops.

1. Use the same instrumented executable for decorations manually ON versus initial OFF. Hold dataset/camera/viewport/background/clipping/mesh/image/shader settings constant. This compares rendering settings, not an algorithmic optimization.
2. Inspect actual three boolean parameters on all child meshes; separately verify shape/visibility/DetectionSet states. Run Data Assembler batch ON/OFF, standalone, one-child, the real 83-dataset/249-mesh store, and 305 datasets if resources permit. Switch lists/children, re-enable each setting and confirm it stays enabled after GUI refresh and subsequent frames.
3. Verify volumes with enabled meshes on white/black backgrounds in MIP/direct modes; verify screenshot/offscreen/SSAO paths. Picking draw/FBO/read counters must remain zero. No visual validation has been performed here.
4. Discard at least 30 warmup paints and collect at least 120 matched paints. Separate `kind=paint`, camera/object identity and `offscreen`; identify no-upload frames using zero `ready_uploads_processed`, `texture3d_upload_chunks` and `texture3d_submitted_bytes`. Report navigation/streaming (interaction, queues, requests/uploads) separately. A no-upload frame can still have pending work; retain the queue state.
5. Compare frame CPU median/p95 and actual rendered FPS separately, per-family display/build/transform/pass CPU, GL calls/ranges, points/outline/triangle/quads/instances, OFF/empty/selection counts, family handling/fallback reasons, scene/grid/bounds/bbox costs, and external GPU event durations where available. No CPU/primitive estimate is a measured GPU/FPS result.
6. Existing registered fixtures were expanded, **not executed**: `image.pocaZarrDatasetContainer` covers seeded globals, standalone/one-child/two-child batch ON/OFF, editable settings, saved-ON reopening, unchanged globals/shape/DetectionSet and synthetic 83 x 3 parameter inspection; `image_render_performance` covers disabled silence, exact synthetic family/helper/instanced/indexed/wire/multi-range arithmetic, CSV row/header matching and existing history/logging bounds. Fixtures do not replace real GUI/GL/store checks.
7. Debug/Release builds, fixture execution, live checkbox synchronization, actual 83/305 workloads and external GPU/FPS comparison remain **unrun**. All binaries/plugins consuming the changed shared profiler layout must be rebuilt together by the user; no build-system/dependency change is required.

## Exact files changed by this task

| File | Part / responsibility |
| --- | --- |
| `poca/src/poca_core/Objects/MyMultipleObject.cpp` | A: sole initialization hook; B: existing scene/bounds/grid/disabled-batch diagnostics |
| `poca/src/poca_core/General/PerformanceProfiler.hpp` | B: fixed family/pass metrics, safe metadata view, scoped CPU/helper accounting |
| `poca/src/poca_core/General/PerformanceProfiler.cpp` | B: accounting, schema 3 names/rows and sparse interval summaries |
| `poca/src/poca_objectlistplugin/ObjectListDisplayCommand.cpp` | B: individual normal display/build/pass/direct/helper/bbox instrumentation |
| `poca/src/poca_objectlistplugin/ObjectListMultiObjectDisplayCommand.cpp` | B: shared decisions/build/transform/source census and all normal range submissions |
| `poca/src/poca_detectionsetplugin/DetectionSetDisplayCommand.cpp` | B: individual normal display/build/helper instrumentation |
| `poca/src/poca_detectionsetplugin/DetectionSetMultiObjectDisplayCommand.cpp` | B: shared decisions/build/transform/source/helper instrumentation |
| `poca_extra/src/poca_imageplugin/ImageStreamingTests.cpp` | Existing deterministic profiler fixture extension, unrun |
| `poca_extra/src/poca_loaderZarrFile/PocaZarrDatasetContainerTests.cpp` | Existing defaults/persistence fixture extension, unrun |
| `poca/docs/render_performance/MESH_DEFAULTS_VISIBLE_PROFILING.md` | This report |
| `CONTINUITY.md` | Required continuity ledger |

Earlier working changes to shared ObjectList, Atrous, shared ImagesList, the OME loader and container fixture were retained. No new production files/headers/dependencies, shader/geometry/scientific coordinate changes, picking/SSAO changes, image/LOD/culling/depth-clear changes or build-system edits were introduced.

Static validation: `git diff --check`, metric/label/CSV-loop alignment, safe empty-buffer metadata reads, preserved GL/Camera draw/clear/FBO/polygon call sequence (including the prior shared GL_FILL addition), source API/ownership/dispatch inspection, and strict UTF-8/non-BOM/CRLF checks. These checks do not establish compilation or runtime correctness.
