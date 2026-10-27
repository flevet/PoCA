# Mesh/detection picking — Stage 2 stopped at the equivalence gate

2026-10-26. Starting checkpoint: `6373dc2`, clean working tree. The user authorized Stage 2 using the supplied IND83 baseline and explicitly required stopping an affected change if exact selection or GL ownership could not be guaranteed. **Stage 2 behavior was not retained.** An unconditional implementation was drafted and audited, then its ten production-file changes and registered-test additions were restored to the starting source after the SSAO conflict below was found. The existing Stage 1 profiler, rendering, FBO ownership, picking and tests remain intact.

Only this report, an appended README section and the mandatory continuity ledger remain changed. No compilation, test/runtime/helper execution, generated/dependency edits, external-source copies or commit occurred. There is no comparable Stage 2 capture and no measured improvement claim.

## Stage 1 evidence

Evidence: the user-supplied `C:/Users/flevet100p/Downloads/PoCA_Stage1_Picking_Baseline_IND83_2026-10-26.md`, derived there from `Texte collé(20261026-163244).txt`. The underlying stdout was not independently available in this turn. The original specification was read from `C:/Users/flevet100p/Downloads/Codex_PoCA_Demand_Driven_Mesh_Detection_Picking.md`. Document instructions were distinguished from the current request; embedded build/capture recipes were not treated as permission to compile.

| Supplied baseline, 83 children | ObjectList/Mesh | DetectionSet | Combined |
| --- | ---: | ---: | ---: |
| Component count | 249 | 83 | 332 |
| GL picking draws / onscreen pass | 249 | 83 | 332 |
| Picking draws over 407 onscreen passes | 101,343 | 33,781 | 135,124 |
| CPU picking work / onscreen pass | 12.682 ms | 5.327 ms | 18.009 ms |
| Initial/live persistent picking FBOs | 249 | 83 | 332 |
| Final requested attachment bytes | 848,239,084 | 516,297,516 | 1,364,536,600 |
| Readbacks / summed readback CPU | 2,656 / 489.630 ms | 2,656 / 489.756 ms | 5,312 / 979.386 ms |
| GPU intervals actually sampled | 13 | 13 | 26 |

Final requested payload is 1,301.32 MiB, not measured physical driver VRAM. The supplied sparse GPU averages (29.22 and 6.77 microseconds) must not be summed or extrapolated to a whole-frame GPU cost. The global picking counter also includes 32 unrelated draws. Separate bind/clear totals are not supplied in the analysis and are not invented.

The capture includes 409 paints, 407 onscreen drawing passes, 56 hover dispatches and no clicks. Streaming was unfinished. Pooled per-frame CPU median/p95, exact viewport/settings, profiler overhead, independently checked IDs and screenshot/selection equivalence are unavailable. The supplied ~101.99 ms post-initialization interval-weighted paint mean is not a pooled median/p95; no aligned CPU-picking percentage is asserted.

## Current call graph

| Flow | Verified current source |
| --- | --- |
| Ordinary paint | `Camera::paintGL` -> `Camera::drawElements` -> normal global display -> `MyObject::executeCommand("display")` multiplies the object model -> component visible drawing followed by picking drawing/FBO work. `ObjectListDisplayCommand::display` is at line 368 and its eager allocation at 382; DetectionSet display/picking is at 202/251. |
| Hover/click | Camera mouse handlers make the context current -> `dispatchProfiledPick` (`Camera.cpp:727`) -> unchanged global pick/current-child/family dispatch -> derived `execute` calls the base -> `BasicDisplayCommand::pick` (`:237`) reads the existing target -> original ID/info/point/bbox result consumption. |
| Resize | `Camera::resizeEvent` broadcasts `updatePickingBuffer` -> list/current-child routing -> `BasicDisplayCommand::updatePickingFBO` (`:385`) allocates/replaces targets. |
| SSAO paint | `Camera::drawSSAO` (`Camera.cpp:1323`) binds its geometry FBO, then calls **component-specific** display at `:1382-1386`; this does not use normal global display's object-model push. DetectionSet is dispatched without `offscreen=true`, so its ordinary picking pass still executes and ends by binding the camera default FBO (`DetectionSetDisplayCommand.cpp:285`). |

Visible drawing and picking currently share the GL depth/cull/blend/polygon/point state and the framebuffer transition left by the picking pass. Therefore deleting only the picking draw is not sufficient to establish unchanged visible rendering.

## Resize discrepancy resolved by source routing

`poca/src/poca_objectlistplugin/ObjectListsCommands.cpp:79` handles typed `display` by iterating **all** contained ObjectLists. Other commands, including `updatePickingBuffer` and `pick`, forward only to `currentObjectList()`. `BasicComponentList` also exposes current-entry forwarding overloads.

Consequently all three mesh lists per child acquire targets during display, while resize reaches one current mesh list per child: **249 live mesh targets, 83 replacements per resize**. DetectionSet is a single component per child, and all 83 receive resize. This explains the observed accounting; it is not independent proof of a selection defect. No dispatch or resize behavior was changed.

## Concrete blocker: two incompatible display-state contracts

### 1. Normal display and SSAO do not propagate the same model

- `poca/src/poca_core/Objects/MyObject.cpp:355` pushes `cameraModel * objectModel` for `executeCommand("display")`.
- The normal global display path executes that push before component rendering, and `MyMultipleObject` saves/restores the camera model around child display.
- `MyObject::executeCommandOnSpecificComponent`, at `MyObject.cpp:394`, instead obtains the component and calls its command directly. It does not execute the object's display model push.
- `MyMultipleObject.cpp:95`/`:402` forwards component-specific display to children with camera-model save/restore, but the child's specific-component method still does not perform the object-model multiplication.
- SSAO uses this specific-component route (`Camera.cpp:1382-1386`). Normal global rendering uses the other route.

A fresh on-demand renderer that unconditionally composes the normal root/child model can therefore select a different screen location from the SSAO display/picking path. Capturing a matrix from the previous paint would violate the requested live-state contract and would still fail before a first post-resize paint. No supplied transformed-child/SSAO selection reference resolves this discrepancy. Applying the normal model push to SSAO display would alter visible rendering/child display semantics, which this task forbids.

### 2. Legacy DetectionSet picking changes the next visible framebuffer

Inside `Camera::drawSSAO`, the geometry FBO is bound before the DetectionSet display dispatch. The command omits `offscreen=true`. Thus each visible DetectionSet executes its picking pass, which ends with:

    glBindFramebuffer(GL_FRAMEBUFFER, _cam->defaultFramebufferObject());

On the multi-child path, subsequent visible component draws can consequently see the default FBO rather than the geometry FBO. Removing the picking routine leaves a different draw target for those draws. The actual pixel impact was not measured here, but the GL-state/control-flow difference is present in source.

Silently repairing SSAO target restoration or normalizing its child-display transforms would change an out-of-scope rendering contract. Keeping an ordinary picking pass for SSAO would violate the user's mandatory zero target-picking draws/binds/clears requirement. Reintroducing legacy per-component picking allocations as an automatic exception would also fail the requested resource behavior. No such exception or broader refactor was installed.

The user's explicit stop rule therefore applies. The shared on-demand implementation affected both display contracts and was withdrawn as one coherent change. This avoids leaving an implementation whose equivalence is known to be unproven for a required existing path.

## Draft design audited, then withdrawn

The draft used one camera/exact-context-owned lazy R32F+depth scratch FBO and private VAO, synchronous clear/draw/read before the unchanged result builders, live component membership/model resolution, scoped GL state restoration, deterministic invalid/no-hit handling, target-only lazy resize and shared resource counters in the existing profiler. CPU fixtures covered transforms, removal/reorder, ID decoding, contextless requests and shared-resource accounting.

Static checks on that draft confirmed the normal display paths contained no target picking draws/binds/clears/allocations, fourteen protected original method/result bodies remained unchanged apart from removing ObjectList's eager allocation, and all 125 existing metric labels remained in order. **Those checks did not establish SSAO equivalence.** The final audit found the conflicts above. The draft APIs, shared-resource schema/metrics, fixtures and behavior changes have all been removed from the production/test sources. The retained profiler is still the original Stage 1 schema and sampling implementation.

## Verification status

| Check | Final result |
| --- | --- |
| Existing Stage 1 production/test source retained | PASS, restored and compared against starting HEAD |
| Resize-dispatch explanation | PASS, source tracing; no routing change |
| Original allowlist and rendering/science/image/LOD/shader/dispatch boundary | PASS, no retained production changes |
| `git diff --check`, UTF-8/CRLF/BOM and targeted draft/artifact checks | PASS, static |
| Stage 2 demand-driven behavior | NOT IMPLEMENTED; stopped at equivalence gate |
| Debug / Release builds and updated DLL load | NOT RUN; explicit never-compile workspace instruction |
| Stage 2 registered fixtures | NOT RUN; withdrawn with their dependent draft APIs |
| Zero-pick stationary paint, scratch allocations/resize/live memory | NOT RUN |
| Expected ID/info/point/bbox equality and visible screenshots | NOT RUN |
| Hover/left/right/double-click/preview/saveImage | NOT RUN |
| Shape/point, 2D/3D, feature thresholds, clipping, hide/disable | NOT RUN |
| Single/multi 1/2/10/83/117, grid/transformed/active/reordered/removed children | NOT RUN |
| Non-unit DPI, freeGPU, context loss/reopen, two cameras, GL errors/leaks | NOT RUN |
| Offscreen/movie/SSAO/image/LOD/scene-depth visual equivalence | NOT RUN |
| Comparable Stage 2 CPU/GPU/overhead capture and before/after CSV | NOT RUN / NOT MEASURED |

Runtime discovery found an existing running `C:/DevC++/PoCA/bin/bin/Release/poca.exe`; its Release executable directory contains only `poca.exe`, and no standalone relevant test executable was found in the checkout/reported source tree. That binary predates this turn's source draft. The relevant existing fixture is registered as `image_render_performance` through `ImageStreamingTests.cpp` -> `ImagePyramidTests.cpp:140` -> `ImagePlugin.cpp:176`. The withdrawn fixture extensions could not execute without rebuilding source/DLLs, and compilation is prohibited. Running the old GUI would not validate Stage 2. No executable was launched or driven.

## Required decision before resuming

Resolve the SSAO contract explicitly: either authorize a separately reviewed correction of its model/framebuffer behavior with matching visual/selection references, or explicitly narrow the required Stage 2 coverage. The current all-path zero-picking and unchanged-rendering requirements were not silently weakened. Even after that decision, exact runtime selection and GL lifecycle acceptance require a consistent user-built binary and the original test matrix.

A comparable Stage 2 capture must use the same IND83 viewport/camera/current lists/thresholds/clipping/grid/bbox/image settings and separate settled/streaming scenarios. The sparse baseline GPU samples remain unsuitable for a whole-frame GPU estimate. Actual hover work/readbacks and other plugins' ordinary picking remain residual costs; no measured reduction or speedup is claimed.

## Final changed-file inventory

1. `CONTINUITY.md` — canonical findings, stop decision and validation state.
2. `poca/docs/render_performance/README.md` — appended Stage 2 stop receipt.
3. `poca/docs/render_performance/MESH_DETECTION_PICKING_STAGE2.md` — this report.

No production or test source change is retained. No shader, image, LOD, scientific-data, CMake, generated, dependency, unrelated plugin, compare script or object-dispatch file was changed. No commit was created. Potential documentation commit: `Document the SSAO equivalence blocker for demand-driven picking`.

### Git receipt

The final tracked-file diff stat (the new report remains untracked and is listed separately above):

```text
 CONTINUITY.md                          | 54 +++++++++++++++++++---------------
 poca/docs/render_performance/README.md |  6 ++++
 2 files changed, 36 insertions(+), 24 deletions(-)
```

`git diff --check`: PASS. The new report also has strict UTF-8/CRLF and no trailing whitespace. No production/test content diff remains. Git status additionally reports `poca/src/poca_detectionsetplugin/DetectionSetDisplayCommand.hpp` and `poca/src/poca_objectlistplugin/ObjectListDisplayCommand.hpp` as modified despite empty diffs and filtered hashes identical to HEAD (`7a51931ab76e4925908e7c554be6d857b2f67ec9` and `bfd1dc4b899cc5f639744c086f141b15a48736e7`). This stat/cache discrepancy was not addressed by modifying the Git index or staging anything.
