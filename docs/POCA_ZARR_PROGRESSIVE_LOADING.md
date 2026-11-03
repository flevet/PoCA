# OME-Zarr positioning preflight and progressive collection loading

2026-11-03 — D278 supersedes the D277 scheduling/display lifecycle described here. Source changes and fixtures are authored; compilation, fixture execution, GUI behavior and corrected timings are UNCONFIRMED.

## Phase A files

- poca/src/poca_core/General/DatasetPlacement.hpp/.cpp: extracted DataAssembler grid, existing stb rectangle packing and grouped circle calculations. Integer packing rejects overflow/exhaustion. Circle rectangles are centered on their circles; group envelopes enclose their corners.
- poca/src/poca_core/General/PlacementPreviewWidget.hpp/.cpp: the existing painted preview, now shared by DataAssembler and Zarr; a small Qt adapter translates circle labels/descriptors.
- poca/src/poca/Widgets/DatasetAssemblerWidget.cpp: calls those shared implementations.
- poca_extra/src/poca_loaderZarrFile/PocaZarrPositioning.hpp/.cpp: metadata inventory, diagnostics and one selected set of target positions.
- PocaZarrPositioningDialog.hpp/.cpp in the same plugin: Keep, grid, packing, hierarchy groups, preview and placement parameters, Apply versus Cancel.
- poca/src/poca_core/Objects/MyObject.hpp/.cpp; PocaZarrDatasetManifest.cpp and its declaration in PocaZarrDataset.hpp: authoritative matrix restoration and translation-only placement.
- PocaZarrPositioningTests.cpp; existing PocaZarrTestSupport.hpp; test declarations/registry and source lists.

## Metadata and coordinate semantics

Preflight reads root/child manifests, ordered identities, transforms, hierarchy and saved grid boxes. Images contribute their level-zero array **shape JSON** and NGFF metadata. No backend numeric array is opened and no image chunk is read for preflight. Quantitative groups contribute bounding_box when available; feature arrays, vertices, faces, offsets and meshes are never read for placement detection.

Each MyObject contributes one footprint: the union of its available component bounds. RAW and LABEL components are not counted as different datasets. All eight corners are transformed using the saved child model. Layout/overlap use collection-local XY, consistent with MyMultipleObject's existing grid/bounds convention. Parent × child world bounds are also recorded. Current image geometry is voxel XYZ; NGFF spacing/origin remain calibration and are not multiplied into geometry a second time.

An X sweep stops at disjoint intervals. Significant overlap is intersection area / min(child XY areas) >= 0.25, with 1e-6 ratio tolerance. Statistics include known footprints, pairs, involved children, XY-only/3D pairs, nearly identical boxes and inconclusive children. Grid cells are checked against transformed bounds and never supply missing geometry bounds.

**Legacy limitation:** existing mesh exports lack a metadata bounding box. An image+mesh child has a known image footprint and incomplete total extents. The dialog explains that placement cannot guarantee separation outside known footprints. A child with no usable XY footprint disables repositioning choices. Keep and Cancel remain available. Incomplete metadata prompts for an explicit decision even when no overlap was established; it is never called overlap-free. No new manifest field/version or guessed geometry bounds were introduced.

Targets are calculated once and validated before publication. Only XY translation changes; Z registration is retained. Grid, packing and circle calculations share one implementation with DataAssembler.

## Transforms and persistence

MyObject::restoreModelTransform retains the exact model, saved rotation and inverted-translation editing state. A residual affine matrix reconciles editing state with the authoritative model, including scale/shear. Subsequent manual translation/rotation retains that residual. Singular editing rotations fail explicitly.

offsetModelTranslation pre-multiplies the model by the displacement and subtracts it from the editing translation. Rotation and linear columns stay unchanged. Placement follows original transform/characteristic restoration. There is no recomputeGrid/resetModelMatrices call.

Opening writes nothing to the source store. Explicit export uses existing model/rotation/translation and grid_boxes persistence; the manifest remains **0.2**. The approved target boxes enter the recorded open CommandInfo, so macro replay uses the same positioning owner without prompting. A ready-child selection made during loading stays current at completion.

## Phase B files

- Both poca/include/LoaderInterface.hpp and poca_extra/include/LoaderInterface.hpp: optional progressive extension; original LoaderInterface vtable retained.
- poca/src/poca_core/General/ProgressiveObjectLoad.hpp: application-thread Qt handoff/progress/completion contract, registered with existing AUTOMOC.
- Engine.hpp/.cpp: optional GUI capability dispatch; synchronous loading APIs remain unchanged.
- MyMultipleObject.hpp/.cpp and MyMultipleObjectLoading.cpp: ordered ownership transfer, expected identities, metadata envelope, hierarchy indices, incomplete state/command guards and weak cancellation on parent destruction.
- PocaZarrProgressiveLoad.hpp/.cpp: one persistent preparation thread, a bounded FIFO and ordered GUI assembly.
- PocaZarrDatasetLoader.cpp and PocaZarrDataset.hpp: synchronous and progressive paths share child finalization. reconstruct() still returns a complete unregistered dataset.
- LoaderZarr.hpp, LoaderZarrObject.cpp, LoaderZarrGui.cpp, PocaZarrDatasetExporter.cpp: routing, recorded-placement replay, action state and incomplete export rejection.
- MainWindow.hpp/.cpp and MainWindowLoading.cpp: one viewer/registration, progress/cancellation, incoming-child display, persistent incomplete diagnostics and action guards.
- PocaZarrProgressiveTests.cpp; existing test support/declarations/registry; core/app/plugin CMake source lists.

## Threads, ownership and rendering integration

One persistent std::thread performs preflight, waits for the unchanged positioning decision, then prepares all children and the parent payload in manifest order. A mutex-protected FIFO holds at most two prepared payloads. A condition variable supplies backpressure. The GUI pops one payload and wakes the worker BEFORE assembly, attachment or Qt signals. The producer can prepare N+1 during GUI commit/display work for N; it does not depend on GUI consumption notifications to start another preparation. After preparing the final payload the producer waits for the GUI completion/cancellation acknowledgement, so queued CPU cleanup stays on the worker even if final assembly fails. Only producer exit is joined. Queued notifications are coalesced, and consumption returns to the event loop after each child.

Worker preparation reads JSON and independent read-only storage handles, constructs storage-backed images/points/features and indexed CPU meshes. The inspected constructors own their palettes/histograms/geometry; they install no plugin commands or GL resources. Buffers and thread-local reports are independent. Legacy coarse-statistics reads may consult existing atomic Engine logging flags; no Engine/plugin registration runs there. A diagnostic rejects GUI-thread preparation or an active GL context.

The GUI uses the same child finalizer as synchronous loading: Engine assembly, plugin commands and characteristic restoration. It then applies the precomputed translation. The first ready child creates the parent with complete expected identities and hierarchy indices. Only an actual ready prefix is stored; there are no null/dummy children. The parent is registered once and receives one MDI camera.

Appending preserves old child pointers, transforms, components and GPU resources, and emits no immediate redraw. Progressive MainWindow callbacks schedule one single-shot refresh per 100 ms burst, refresh the Objects/Properties panels once per burst, and share the existing camera/image-stream redraw gate. Progress counters still advance on every commit; user-driven updates are unchanged. No progressive GUI callback issues createDisplay. Existing selected/visible lazy render commands and image streaming prepare resources with the current camera context. Parent aggregate components and characteristics wait until all children exist. Ready-child selections/visibility are retained.

The existing image provider operates on the ready prefix: its frame-local descriptors, stable image identities, resource versions and existing initialization/LOD budgets do not require aggregate scientific metadata. Projected regions crossing child-rendered geometry use the existing ordering barriers and legacy fallback. Whole-buffer geometry batchers remain completion-gated: selected geometry uses cached individual lazy resources during loading, followed by the normal shared transition at completion. Appending never frees/rebuilds an existing geometry buffer. No new batches/resource manager, shader, image representation, streaming budget, Force L0 policy, visibility default or scientific algorithm was introduced. RAW/LABEL associations and mesh-decoration policy follow normal finalization.

The complete available metadata envelope seeds initial framing once. Attachment never recreates/recenters the camera. Incomplete extents explicitly limit preview/framing to known metadata and loaded bounds; inspect final framing manually.

## Cancellation, errors and incomplete data

Cancellation retains a viewable **incomplete prefix**, sets the cancellation predicate under the queue mutex and wakes both positioning and queue-slot waits. A backend read/child CPU construction remains non-interruptible; normal cancellation emits finished only after the producer exits, while the GUI continues processing events. Queued CPU payloads are discarded outside the queue mutex. Closing/deleting the parent revokes borrowed pointers through the existing weak callback. Forced destruction of the job clears its receiver under the same mutex and detaches the thread: the thread owns only shared CPU pipeline state and never captures the job, parent or widget. QObject removes queued calls for the destroyed receiver. Normal completion/cancellation uses the exit-only join; it never joins a live preparation between datasets.

A child failure stops loading and reports manifest index, name and cause. Later indices are never shifted/skipped. Parent-finalization failure is distinct from a failed child. Properties retains ready/pending/failed counts and the cause after the transient status control disappears.

Loading/Cancelled/Failed roots reject aggregate analysis, layout changes and whole-dataset export. Commands use PoCA error reporting plus LoadingCommandRejection; direct assembly/export APIs fail explicitly. There is no resume/retry protocol: address the cause, close and reopen.

## Authored fixtures

The existing opt-in POCA_ZARR_EXPORT_SOURCE_TESTS menu gains:

1. **PoCA-Zarr positioning preflight**: identical positions, three merged grids, separated/XY-only children, RAW/LABEL grouping, Keep versus Cancel, rotations/nonunit scale, unequal dimensions, uncertain bounds, legacy empty grid metadata, no numeric preflight reads, preview targets, later edits and explicit save/reopen. Byte snapshots enforce untouched input stores.
2. **PoCA-Zarr progressive reconstruction**: 20 children with images/labels, a mesh, DetectionSet and parent features; early handoff, event-loop heartbeat, ready-prefix display dispatch exactly once, one registration, stable addresses/order, hierarchy/current state, lazy arrays, wrong-thread attachment rejection, cancellation between commits/in flight, in-flight parent destruction, named failure at child 7, preserved user selection/visibility, synchronous equivalence, preflight cancellation and single-object routing.
3. **PoCA-Zarr progressive viewer (20 datasets)**: returns the first prefix to the existing Tests-menu MDI viewer, with 250 ms preparation delays. The 5×4 grid must populate while the camera is navigable. Fixture backing stores outlive the viewer.

**No fixtures or existing suites were compiled or executed.** No configuration/generation, build, app, test/helper executable or backend rebuild ran. Static inspection cannot establish GPU equivalence, responsiveness, visible-child timing, deployment compatibility or race freedom.

## Reproducible 305-object acceptance / benchmark

Use an available runtime containing these sources and a copy of Merged.ome.zarr (expected 305 identities: 83+117+105). Record revision, hardware, runtime configuration, store location, cache conditions, RAW/LABEL modes, camera, LOD policy and memory budget.

1. Preserve a source-store hash inventory. Time interactive preflight. Confirm metadata-only reads, overlap statistics and the choice before reconstruction. Cancel once: no viewer/root or source changes should remain.
2. Test Keep against synchronous transforms/order/hierarchy. Reopen and choose grid or packing. Inspect rotation/scale, unequal dimensions and incomplete-bound diagnostics.
3. Capture progressive logs: metadata preflight ms, first viewer ready ms, first submitted frame ms, per-child preparation/GUI commit ms, complete ms. **Handoff/submitted-frame timing does not prove visible content.** Separately record the first visibly rendered child and successful camera interaction, both before full reconstruction.
4. Navigate continuously while the grid fills. Confirm one viewer, no automatic recentering/duplicates, correct raw/label and mesh settings, then 305/305. Check Force L0/pyramids separately from reconstruction progress.
5. Cancel during slow preparation, verify a viewable incomplete prefix and export rejection. Close during a read and check worker/callback/GL cleanup. Corrupt a copied child's payload: confirm exact index/name/cause and no later commits.
6. Explicitly export the completed repositioned collection to a different store. Reopen synchronously and progressively; compare models, hierarchy, features and rendering. A legacy mesh-bound warning can recur because the format was not extended.
7. Compare that same completed, repositioned store via synchronous non-recordable macro open and interactive progressive open. Alternate three runs per path with stated cold/warm-cache conditions. Record medians for total/first-visible/first-interactive times, maximum GUI stall/commit, peak process/GPU memory, and final count using existing profiler/OS tools.
8. Recheck the original hash inventory. Only the explicit export destination may change.

| Metric | Before | After | Acceptance |
|---|---|---|---|
| Preflight / numeric reads | UNMEASURED | UNMEASURED | No full image/mesh/feature materialization |
| First visible child / camera interaction | UNMEASURED | UNMEASURED | Before full reconstruction |
| Total reconstruction | UNMEASURED | UNMEASURED | Record regressions/improvements |
| GUI stall / commit | UNMEASURED | UNMEASURED | No long continuous blocking |
| Peak RAM / GPU memory | UNMEASURED | UNMEASURED | No accidental L0 materialization |
| Final children / equivalence | UNVERIFIED | UNVERIFIED | 305, equivalent science/display state |

No actual merged-store path or runnable acceptance receipt was supplied. No performance numbers or passing-test claim is made.

## D278 performance correction and validation status (2026-11-03)

The smallest geometry choice is the temporary cached individual path, then the normal shared path once the collection is complete. The current shared mesh/detection commands build whole buffers and validate the current inventory; merely removing isLoadComplete() would require repeated reconstruction or risk omitting newly appended children. Their completion gate remains intact. Reusing individual resources avoids a new batching subsystem, preserves ordering/settings/fallback, and avoids repeatedly initializing the existing prefix. This choice still needs measured acceptance: a single visible mesh build can stall the GUI, and the completed shared geometry needs one initial build. These remaining costs are not claimed to be acceptable without runtime measurements.

Additional changed responsibilities:

- poca/src/poca_core/General/ProgressiveLoadProfile.hpp/.cpp: per-opening thread-safe trace/counters and scoped CPU timings around existing GPU setup; registered in the core source lists.
- poca/src/poca_core/Objects/MyMultipleObject.hpp/.cpp and MyMultipleObjectLoading.cpp: safe image-only ready-prefix policy, coalesced append behavior, shared profile ownership; geometry/scientific completion guards retained.
- poca/src/poca/Widgets/MainWindowLoading.cpp: lazy display, single-shot panel bursts and reuse of camera streaming redraw gating, final panel refresh.
- poca/src/poca_opengl/OpenGL/Camera.cpp: count actual gated redraw schedules only for a progressive profile.
- poca/src/poca_objectlistplugin and poca/src/poca_detectionsetplugin, individual/shared display commands: timing scopes around existing lazy initialization or completed shared rebuild; geometry algorithms unchanged.
- poca_extra/src/poca_imageplugin/ImageDisplayCommand.cpp and ImageDisplayStreamUpload.cpp: initialization/stream-transfer CPU timing; ImagesList collection plan/provider: ready-prefix eligibility and existing ordering fallback.
- poca_extra/src/poca_loaderZarrFile/PocaZarrProgressiveLoad.hpp/.cpp and PocaZarrProgressiveTests.cpp: persistent producer/queue/lifetime and regression fixtures.
- poca_extra/src/poca_imageplugin/CollectionRenderingTests.cpp: partial-prefix render inventory, stable identities/selections/transforms, geometry barriers and unchanged descriptor transition checks.

### Instrumentation semantics

Pipeline records have a job identity, steady-clock t_ms relative to creation, thread ID and event/index. Events cover preflight, positioning decision, pipeline start, each preparation, enqueue/dequeue, GUI assembly, attachment, GPU setup, first redraw schedule and full reconstruction. All gated redraw schedules and submitted frames are counted; only the first is logged individually. Texture-transfer scopes accumulate CPU time without per-brick logs. Detailed per-child reconstruction logs remain buffered in their bounded payload rather than replaying large reports to the GUI console.

The collection_complete summary reports full reconstruction wall_ms and pipeline_ms (the latter starts AFTER preflight and positioning), worker_busy_ms, queue_slot_wait_ms, gui_commit_ms, gpu_cpu_ms, peak_queue_depth, peak_accounted_queue_bytes, redraw_schedules, scene_redraws and submitted_frames. A second first_completed_collection_frame summary waits until Camera::paintGL has actually drawn a completed scene (past the MDI startup gate), includes the initial final-geometry transition and stops progressive tracing. A hidden viewer may not submit that final frame; reconstruction completion remains separately logged. Streaming can continue after either event.

GPU timings measure CPU work issuing setup/upload commands, not GPU hardware execution time. Queue bytes sum existing non-materializing component memorySize() values recursively plus buffered report bytes. They are accounting estimates: object/allocator/backend overhead is excluded, and individual component counters can saturate at UINT_MAX. Queue cardinality is the hard bound (two waiting payloads, one serial producer, one GUI consumer); no fixed process-RAM ceiling is claimed. Use process/GPU measurements for actual peak RAM/VRAM.

first_submitted_frame_not_visibility can be an empty frame before textures or the MDI startup gate are ready. It MUST NOT be reported as first visible content. Record first visibly rendered dataset and first successful interaction separately from a GUI capture; no shader or picking-readback instrumentation was added to infer visibility.

### Regression fixtures authored, not executed

The existing manual progressive fixture now injects 15 ms preparation and 20 ms GUI assembly delays. During the first GUI attachment/display callback it holds the GUI stack open, without processEvents(), while the worker prepares additional children and fills both queue slots. It checks preparation[1] precedes the end of GUI commit[0], the worker progresses during simulated display initialization, one stable worker ID, queue peak exactly two, positive backpressure time, manifest order/exactly-once attachment and an event-loop heartbeat between commits.

Dedicated cases cancel with zero, one and two queued children, destroy the Engine parent while the queue is full, and forcibly destroy the loading job while preparation is gated. They check prompt return, cancellation state/prefix, queue bounds, producer-state release and no late attachment. Existing synchronous equivalence, RAW/LABEL/storage semantics, visibility/current-child/transform preservation, failure identity, incomplete analysis/export rejection, wrong-thread attachment guard and viewer fixtures are retained. The worker guard uses a separate fixture root whose lifetime spans the entire job, rather than borrowing a viewer concurrently with destruction.

CPU render fixtures inspect append-only ready inventories, unchanged existing elements/models/selections and descriptor spans, completion gates, and the ordering-barrier partition. Actual GL resource reuse, visual equivalence, camera responsiveness and full wall-time performance require the manual viewer/real-store checks; CPU fixtures do not establish them.

### Timings and remaining acceptance

The following baseline numbers are supplied in the task, for only the first 33 children of an 83-child store. The underlying execution log and a compatible executable/store path were not supplied in this chat; the baseline is not independently reproduced here.

| Metric | Supplied baseline | Corrected run |
|---|---:|---:|
| Preflight, all 83 children (separate phase) | about 17.1 s | UNMEASURED; implementation unchanged |
| Median preparation, first 33 | about 5.59 s / child | UNMEASURED |
| Preparation accumulated, first 33 | about 186.9 s | UNMEASURED |
| Median GUI commit, first 33 | about 484 ms / child | UNMEASURED |
| GUI commit accumulated, first 33 | about 17.3 s | UNMEASURED |
| Individual mesh initialization/render spikes | about 200–350 ms | UNMEASURED; lazy selected mesh work remains |
| Full 83-child synchronous wall time | NOT PROVIDED | UNMEASURED |
| Full 83-child progressive wall time | NOT PROVIDED | UNMEASURED |
| First visible / first interactive dataset | NOT PROVIDED | UNMEASURED |

Run the existing opt-in Tests-menu fixtures using a compatible user-built runtime, then alternate at least three synchronous and corrected-progressive opens of the SAME 83-child store with the same cache conditions, camera, placement choice, component visibility and LOD budgets. Exclude dialog dwell from pipeline throughput; retain complete-open timing as a separate measure. Record time to first visible content and full reconstruction/final submitted scene separately. Compare scientific reconstruction and final rendering; navigate, change selection, cancel and close during loading. Capture queue depth/memory estimates together with real process RAM/VRAM, GUI stalls and existing render counters (individual builds/shared rebuilds, descriptor uploads, RAW multi draws). Existing data must not be re-exported or changed for the comparison.

The supplied preparation times suggest preparation remains the dominant measured stage, but corrected throughput, backend/render contention, total speedup/regression and the acceptability of remaining individual mesh stalls are UNCONFIRMED. No compilation, CMake generation, tests, PoCA execution, backend execution or benchmark occurred here. The task's measured-performance definition of done remains pending; source inspection alone cannot satisfy it.
