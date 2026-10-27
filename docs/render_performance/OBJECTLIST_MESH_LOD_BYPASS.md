# ObjectListMesh: bypass unused per-object LOD planning

2026-10-27 [USER/CODE] Source correction to the attached
Codex_PoCA_Stop_Wasted_Subpixel_LOD_Planning_20261027.md request.
This supersedes the always-enabled histogram sampling described in
OBJECTLIST_MESH_VISIBILITY_LAZY_BUFFERS.md; its frustum and lazy-buffer work remains.

## Production changes

- poca/src/poca_objectlistplugin/ObjectListDisplayCommand.cpp
- poca/src/poca_core/General/PerformanceProfiler.hpp
- poca/src/poca_core/General/PerformanceProfiler.cpp

The local kEnableObjectListPerObjectSubpixelLod constant is false.
ObjectListDisplayCommand::display uses if constexpr around the sole
profileMeshSizes call. The default path performs only an O(1) candidate check
and aggregate counter increment while profiling. It creates no LOD timer,
projected-size samples, range lists, per-object masks or per-frame vectors.
No camera-dependent visibility cache or new eligibility heuristic is added.

The bypass is after the existing component frustum decision and before
drawElements, createDisplay and surface GL submission. Offscreen components
still return before initialization when no independent decoration pass is
needed. The existing one-time cached union-bounds preparation remains: it
inspects mesh metadata once per visibility invalidation, not on every frame.
Warm frustum tests still project just eight component-box corners.

## Why it always reached full drawing

The current profileMeshSizes implementation is diagnostic only. Its entry
requires active profiling, a multi-child camera object, 3D and shapeRendering.
It increments LodFullDrawFallbacks unconditionally, starts LodPlanningUs,
then checks mesh/range validity and viewport size. Eligible lists sample at
most 128 nonempty object ranges, projecting their eight box corners and
recording a size histogram. Invalid/near-plane cases record unknown samples.
Neither successful samples nor subpixel samples select or omit geometry.
The routine returns void and never changes draw state. drawElements always
uses the original full buffer via glDrawElements or glDrawArrays.

Thus the observed full draws are deliberate, not evidence of a failed
omission threshold, range fragmentation or missing validated offsets.
The histogram remains unchanged behind the local flag for controlled future
experiments; setting the flag true does not enable a new omission algorithm.
No cheap proven benefit estimate exists in this code, so the experiment stays OFF.

## Profiler semantics and comparison

One VisibleMetric is appended: LodFastBypasses / lod_fast_bypasses.
The existing family-block schema emits objectlist_individual_lod_fast_bypasses;
other family counters remain zero unless explicitly recorded. CSV rows are
version 5 because the family blocks gained a column. Prior names and pass-block
ordering are retained; column offsets after a family block change.
Use a fresh CSV or the existing header mismatch rejection will disable export.

A fast bypass counts the same candidates as the old diagnostic entry:
onscreen, non-SSAO, surfaceVisible, active visible profiling, multi-child camera,
3D and shapeRendering. It counts skipped planning attempts, not confirmed draws;
missing histogram display bounds can still prevent a subsequent draw.
Actual GL calls and submitted geometry continue to come from visibleDraw.
Frustum-culled surfaces never count a planning bypass.
No default fast bypass increments LodFullDrawFallbacks or omitted-triangle counters.

All baseline values below are supplied evidence, not measurements performed here.
Expectations require the matching 83-dataset view/settings and rebuilt binaries.

| Metric | Supplied baseline | Default-OFF source expectation / new measurement |
| --- | --- | --- |
| lod_tests | 121,877 / 7 frames (~17,411/frame) | 0 from this scanner; NOT MEASURED |
| lod_planning_us | 96,155 / 7 frames (~13.74 ms/frame) | 0 from this scanner; NOT MEASURED |
| lod_full_draw_fallbacks | 1,743 / 7 frames (249/frame) | 0 from this scanner; NOT MEASURED |
| lod_fast_bypasses | Not previously available | ~249/frame in matching overview; NOT MEASURED |
| individual_gl_draws | 1,743 / 7 frames (249/frame) | Same surface draw branches; NOT MEASURED |
| individual_submitted_triangles | ~33,418,730/frame | Same submitted geometry; NOT MEASURED |
| first ObjectListMesh buffer build | ~26.4 s before lazy buffers, ~14.1 s afterward | Lazy code unchanged; NOT MEASURED |
| overview visible mesh lists | 249 | Same visibility decision; NOT MEASURED |
| zoomed culled / drawn lists | 221 culled / 28 drawn, ~3.74M triangles | Same component culling; NOT MEASURED |
| paintGL median / p95 | Not supplied; observed frame CPU ~39.9 -> ~58.8 ms was not a controlled comparison | NOT MEASURED |

CPU timer counters are CPU elapsed microseconds, not GPU execution time.
No FPS or measured CPU speedup is claimed.

## Preservation and regression acceptance

Static diff inspection: drawElements, surfaceFrustumStatus, prepareMeshVisibility,
profileMeshSizes, all lazy decoration builders and ownership are unchanged.
The ObjectList header, shared renderer/admission guard, DetectionSet, RAW images,
picking constant/guards, polygon correction, shaders, geometry, transforms,
features, saved data, texture uploads, streaming and GPU residency are untouched.
No new resources, Qt actions, commands, dependencies or CMake options.

Completed static checks: ten relevant method bodies compared with HEAD;
117 matching profiler enum/name fields with prior enum ordering preserved;
strict UTF-8/non-BOM/CRLF for all three production streams; git diff --check
and targeted quoting/merge-artifact searches clean. C++17 is already required.

Compilation, automated tests, application execution and visual validation:
NOT RUN under the workspace source-only policy. No ObjectList-specific test
target exists in the current plugin CMake source registration; no synthetic
test mirroring the flag was added. Static checks do not verify runtime behavior.

Unrun acceptance matrix from the request:

1. Same 83-dataset overview/camera/settings: compare representative no-upload
   paint rows and aggregate intervals, lod counters, 249 draws and ~33.4M triangles.
   Compute paintGL median/p95 from matching paint rows; compare surfaces visually.
2. Zoom/pan: verify meaningful frustum_tests/frustum_culled (historically 221/249),
   reduced draws and deferred builds for unseen components.
3. Deep zoom and return: nuclei/Voronoi geometry and feature colors stay intact.
4. Point, outline-point and ellipsoid toggles: initial OFF remains lazy;
   each ON materializes correctly and OFF hides it; standalone defaults stay independent.
5. Orthographic/perspective, transformed/rotated/scaled children, wireframe,
   translucency, black/white backgrounds and viewport maximization:
   unchanged surfaces, polygon fill boundaries and RAW visibility.
6. Two-child Data Assembler and only if memory safely permits, the 305-dataset
   merged scene: retain bounded shared-mesh rejection and individual drawing.
7. Confirm POCA_ENABLE_PICKING=false counters remain zero; compare shared RAW,
   DetectionSet, depth clears, first-build/lazy allocation counters and streaming
   counters with equivalent settings.
