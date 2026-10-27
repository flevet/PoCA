# ObjectListMesh visibility and lazy decorations

2026-10-27. Source changes only. Builds, tests, application/helper execution, visual comparisons and GL captures: NOT RUN.

## Exact modified-file inventory

- poca/src/poca_objectlistplugin/ObjectListDisplayCommand.cpp
- poca/src/poca_objectlistplugin/ObjectListDisplayCommand.hpp
- poca/src/poca_core/General/PerformanceProfiler.cpp
- poca/src/poca_core/General/PerformanceProfiler.hpp
- poca/docs/render_performance/OBJECTLIST_MESH_VISIBILITY_LAZY_BUFFERS.md
- CONTINUITY.md

## A: conservative surface rejection

The existing display command caches the union of getBBoxMeshes(), rather than using the localization/component box, which may be smaller than a reconstructed surface. Object/bound/offset counts, finite extents, ordering and triangle alignment are checked. Empty ranges are skipped; invalid metadata draws conservatively.

The test precedes createDisplay(). Projection matches objectRendering.vs: model position first, then projection times view. The active camera model includes child/parent placement; no image scale is added. All eight corners must lie beyond the same one of six OpenGL homogeneous clip planes. A 1e-4 margin scales with absolute transform contributions, including cancellation from large translations. Non-affine/singular/non-finite matrices, invalid viewports, eye-plane ambiguity and near-plane straddling bypass rejection.

Only surfaces are suppressed. Independently enabled points, outlines, ellipsoids, skeletons/links and selection boxes retain their draw path. Without those passes, a rejected component returns before GPU construction. Visibility is recomputed every display; only geometry bounds are cached. freeGPU and surface reconstruction invalidate the cache. explode supplies its displayed triangle bounds and invalidates per-object range correspondence.

Offscreen/SSAO, 2D and diagnostic picking=true keep the original path; zoom preview retains its separate entrypoint. Rotation, translation, affine nonuniform/negative scale and both projections follow the same mathematical test. Runtime equivalence is NOT VERIFIED. No selection, scientific geometry or feature values are changed.

## B: full detail retained; bounded diagnostic histogram

Subpixel omission is DEFERRED at the visual/performance gate. No measurement establishes material triangle reduction or that planning/coalescing beats one draw on these workloads. No omission, hysteresis state, proxy, decimation, multidraw planner or replacement geometry is introduced.

Range-order evidence inspected in the current source:

- ObjectListMesh.cpp::processSurfaceMesh appends each mesh's face vertices and terminal triangle offset, then stores its vertex bounds. The triangle-array constructor also records bounds and offsets in input mesh order.
- generateTriangles copies m_triangles.getData() unchanged; createDisplay uploads that sequence.
- ObjectListMeshPersistence.cpp expands indexed faces within each object, appends firstTriangles and stores that object's indexed-vertex bounds. IndexedMeshGeometry::validateStorage confines face indices to that object's vertex interval and limits total faces to uint32_t/3.
- Cached metadata checks terminal size equality, aligned monotonic offsets and signed GLsizei capacity; empty ranges are skipped.

These contracts do not justify unconditional renderer-wide omission: explode updates displayed vertices without updating scientific per-object boxes, sorted draws can use an element buffer, and special passes/feature domains require exclusions. A later implementation must validate uploaded per-object containment, exclude modified/indexed draw layouts, measure range planning/coalescing costs, and pass the 1px/2px visual and zoom-restoration gates.

With renderPerformance enabled only, normal multi-child displays sample at most 128 evenly spaced nonempty meshes/list. Counters distinguish <1px, 1-2px, >=2px and unknown footprint, plus triangles in the sampled <1px ranges. These are SAMPLE budgets, not omitted/submitted/saved geometry or extrapolated totals. No per-triangle scan, CGAL materialization, per-object logs or camera-visibility cache is added. Omission and retained-multidraw-range counters remain zero.

## C: lazy decoration buffers

Separate ready flags own points, outline points and ellipsoids. First materialization occurs in the display callback on the current GL context using current features/selection. OFF/ON reuses valid buffers. Enabling decorations does not regenerate the triangle VBO. Histogram/LUT/selection commands refresh resident feature streams; freeGPU/surface reconstruction resets readiness. Translucent sorting skips uninitialized surfaces and never creates a point index buffer while points are disabled or absent.

Point position+feature buffers are created only when needed. Unused point normals are no longer generated; point picking IDs remain conditional on the existing switch. Surface normals/features/picking IDs remain. Ellipsoids reuse original PCA/centroid mathematics and wait until enabled and major-axis data exists.

Outline construction was previously commented out. Enabling it now uses the existing matching generateOutlineLocs/getOutlineLocsFeatureInSelection APIs. No previous outline allocation savings are claimed.

Triangle-level histograms have no defined per-object decoration feature mapping: surfaces remain current, but decoration builds/draws wait for an object histogram instead of reading uninitialized/stale feature VBOs. No scientific feature values change.

GL handles remain owned by ObjectListDisplayCommand. Cleanup includes normal/decorative feature buffers required for invalidation; command copies start with independent GPU state. Shared small-collection batching is unchanged and can still eagerly build points/outlines. Its memory guard and handled/fallback decisions remain unchanged.

## Measurements

| Quantity | Attachment baseline: 83 children | After changes |
|---|---:|---|
| steady paintGL CPU | approximately 32-35 ms/frame | NOT MEASURED |
| first mesh-list build CPU | 26,404,505 us | NOT MEASURED |
| surface GL draws/frame | 249 | NOT MEASURED |
| submitted surface triangles/frame | 33,418,730 | NOT MEASURED |
| initial point vertices built | 16,814,479 | NOT MEASURED |
| actual GPU allocations/upload bytes | not supplied | NOT MEASURED |
| GPU elapsed time / validated FPS | unavailable | NOT MEASURED |

Source behavior: eligible rejected surfaces submit no surface triangles. Fully rejected undecorated lists construct no display buffers. Disabled individual point/outline/ellipsoid families allocate zero corresponding buffers until enabled. All-visible overview surfaces remain full detail. These are code-path properties, not workload measurements or speedup claims.

## Profiler semantics

Existing opt-in profiler extends to CSV schema 4: 600 metrics, 618 columns. Use a fresh CSV path; existing schema-mismatch checks reject old headers. GPU time remains unavailable; no queries/waits/readbacks were added. Existing frame/RAW/DetectionSet/streaming/picking counters remain.

frustum_tests/culled/bypassed/deferred_builds count selected-list tests, surface rejections, conservative/unsupported visits and rejected visits lacking a first build. Deferred-build counts are visits, not unique lists. Existing components_drawn, GL and submitted-triangle counters still describe actual submissions.

BuildUs aggregates surface/decoration builds and existing skeleton builds. SurfaceBuildUs, PointBuildUs, OutlineBuildUs and EllipsoidBuildUs separate family CPU costs. Display/pass scopes overlap these timers: do not add inclusive scopes. Family allocation counters describe initial glBufferData storage submissions; upload bytes describe new-family payload, not driver residency or GPU elapsed time. Selection-box storage and later feature-only uploads are outside those family counters.

Deferred point bytes estimate eventual position+feature payload, excluding eliminated normals and disabled picking IDs. Deferred ellipsoid bytes estimate transform+feature payload. Recorded only at an actual surface build, they describe deferred requests, not resident-memory savings. OutlineDormant counts OFF initial builds and claims no avoided legacy outline bytes.

## Runtime acceptance still required: all NOT RUN

Use matching rebuilt binaries and identical viewport/settings for the 83-child overview, few-visible pan, deep zoom, continuous rotation/zoom and busy/empty streaming queues. Compare CPU median/p95, actual GL/triangle counters, family build CPU/payload, GL errors and external GPU timings.

Repeat with two-child Data Assembler (batch ON/OFF) and standalone objects; use 305 children only with adequate memory. Cover both projections, child transforms, negative/near-plane geometry, resize/maximize, clipping, transparency, wireframe, backgrounds, feature/selection/LUT changes, all three decoration toggles, freeGPU/rebuild and explode followed by navigation.

Compare offscreen export/zoom preview/thumbnail/SSAO output, zero picking metrics, RAW visibility/polygon state, CSV integrity and stable memory. Shared paths retain their existing behavior. There is no omission switch yet because B has no omission path; use the sample histogram to qualify a later experiment. No regression-free runtime result, CPU/GPU speedup or 60 FPS result is claimed.


Static verification completed: git diff --check; strict UTF-8/non-BOM/CRLF; lexical delimiter counts; helper declarations/definitions; 116 enum/label entries per visible family; unchanged pass-block layout; 600 metrics/618 columns; source comparison preserving complete picking/zoom-preview bodies and original ellipsoid mathematics. Existing profiler fixture derives CSV column counts dynamically. These static checks do not establish compilation or runtime correctness.
