# Progressive OME-Zarr reset and loading follow-up (D279)

Date: 2026-11-03. Source delivery; runtime acceptance PENDING.

## Visualization findings and corrections

Three concrete source defects affect Reset Viewer. Their presence is established by source inspection; identifying which dominated the supplied real-store disappearance still requires the requested reproduction.

1. MainWindow::resetViewer calls Camera::resetProjection and then MdiChild::resetViewer. The latter changed a QSlider range/value with its valueChanged connection active. Its slot unconditionally called applyFramePlane, including when the slider was hidden in 3D mode. It also compared a relative slider index to absolute collection Z limits. With positive Z origins (or a shrinking range), Reset could immediately install a thin crop after fitting the whole collection. During partial loading, setCurrentFrame was rejected but zoomToBoundingBox still applied the thin crop; this could leave camera cropping and image frame state inconsistent.
2. Camera::zoomToBoundingBox synchronously called repaint while resetProjection had not yet restored its final crop flag/planes. A frame could observe an intermediate crop, touch residency and cancel view requests.
3. The fit used unrotated XY extents although Reset retains the camera rotation. Long/deep boxes viewed obliquely, and portrait viewports, could remain partially outside the image frustum.

MdiChild now blocks slider signals during range refresh, clamps relative indices, and explicitly reapplies a slice only in slice mode. The slider slot also checks slice mode. Camera zoom schedules update after state changes, and reset clears interaction/settle state and requests LOD reevaluation. Existing eye orientation is retained. A small camera geometry calculation fits all eight oriented box corners to the active viewport, with a safe physical eye distance for orthographic and perspective projection.

MyMultipleObject continues to union the immutable metadata pending envelope with eight-corner transformed ready-child bounds. Completion clears the pending envelope and uses actual child bounds. Attachment does not recenter the camera. A read-only pendingLoadBounds accessor supports diagnostics; the bounds contract and child transformations are unchanged.

The incomplete command allowlist now admits updatePickedObject, getInfoObjectCurrentlyPicked, setCurrentFrame and togglePicking. Existing analysis, layout and whole-dataset export rejection remains. No image pixels, LUTs, interpolation, persisted scientific features, image selections or child transforms are modified by the reset corrections.

## Visibility diagnosis

Enable global verbose output and use the existing Add verbose type... action to add zarrLoadVisibility, then move the camera and use Reset Viewer at 1, 10 and 40 ready children and after completion.

For the next eight rendered frames, the camera logs collection bounds, pending envelope, ready/expected counts, projection/view/parent matrices, camera translation, crop state and clip planes. ImageDisplayCommand logs image identity/name, selected state, effective transform, local-to-crop transform, source bounds/frame, uncropped frustum visibility, cropped visibility, initialization, desired/pending/displayed pyramid levels, displayed dimensions, preview/detail texture readiness, Force L0 and request failure state.

Opt-in readbacks around RAW list draws, shared RAW batches and individual LABEL draws/composites report changed RGB pixels. They preserve read framebuffer and pixel-pack state, exclude label bounding-box overlays and fail explicitly on GL readback errors. Multisampled/empty targets report that contribution readback is unavailable. A changed image-pass framebuffer establishes contribution at that pass, not survival after later geometry/overlays. Inspect final framebuffer images too. A draw counter or frameSwapped is never reported as visible content.

Diagnostic readback is synchronous and can be expensive; do not enable zarrLoadVisibility during timing comparisons. Reset can be repeated to start another capture window if texture recovery occurs later.

## Partial collection rendering

The existing shared image provider already accepts the ready prefix. Its descriptors/identities and LOD planner do not need pending children or aggregate scientific features. However, parent-level image draws cannot cross projected footprints of child-rendered transparent geometry or incompatible/plugin draws without changing compositing order. Existing ordering barriers and the individual fallback remain intact. Image-only compatible projected regions can still batch during partial loading. This follow-up does not remove barriers, reorder transparent operations or introduce a renderer.

Cold individual ObjectList initialization is now admitted once per progressive scene frame, after the first frame's opportunity to submit image previews. Components with no display bounds or no requested visible pass do not consume the allowance. Existing warm resources keep drawing in their established order. Cold initialization pauses during camera interaction and resumes with the existing settle update. Deferred cold components request the existing coalesced streaming redraw. Selected geometry refines throughout loading, rather than waiting for completion; no visibility flags are toggled.

The temporary behavior is explicit: a newly selected/attached cold geometry component can be absent for several frames while it waits its turn. A single admitted component's existing initialization remains synchronous and can still stall. The allowance is a count bound, not a hard millisecond guarantee, and does not chunk a large mesh upload. Offscreen and SSAO passes retain their existing behavior. Completion removes this progressive-only admission limit. Existing image initialization/request/upload/memory budgets are unchanged.

ObjectList no longer generates or uploads per-triangle picking IDs while the existing global POCA_ENABLE_PICKING switch is false. Visible geometry, normals and features are unchanged. Surface allocation/upload counters reflect the removed ID buffer.

The existing 100 ms single-shot loader panel refresh and common camera/image redraw gate are retained. No attachment-triggered immediate paint or eager createDisplay was introduced. Camera/user input updates remain immediate.

## Background reconstruction

No preparation thread, queue size, scheduling or ownership protocol was changed. The single producer, preflight approval, manifest order, two-slot FIFO, release-before-GUI-commit and terminal acknowledgement remain the D278 implementation.

loadComponents already measures image initialization, quantitative component loading, mesh construction/validation, persisted normals and feature/histogram restoration. Progressive loading previously kept those reports in each Prepared log and discarded them. Use Add verbose type... to enable the explicit zarrLoadPerformance category and print each child report after GUI finalization. Reports identify the child, component path/kind and each stage's inclusive and self seconds; nested inclusive timings must not be summed. The outer Dataset reconstruction interval spans queue dwell and GUI finalization too; use worker component/stage scopes and worker_busy_ms for preparation costs.

Image and ImagesList stages now carry identities too. Mesh geometry values are split into vertex allocation, face allocation, vertex reads and face reads. Existing stages distinguish metadata/opening, offset validation, cheap storage checks, topology validation/provenance, persisted normal reads versus generation, triangle/auxiliary construction, triangle-z, feature adoption and Histogram/MyData work.

The source audit found two avoidable copies: offset vectors were copied into their geometry owner, and every indexed vertex/face row was read into a scratch buffer then copied to its final vector. Offsets now move into the owner; unchanged chunk reads target the final contiguous std::array vectors directly, with compile-time packing checks. This removes 24*(vertices+faces) bytes of post-read element copying and two 1.5 MiB scratch vectors, plus the offset copies. Allocation/read counts and data types/order remain unchanged. Runtime savings are UNMEASURED.

Expensive topology validation, persisted-normal checks, display-triangle expansion and feature/histogram restoration have not been weakened or replaced without stage measurements. Unknown provenance still validates; certified geometry still receives storage safety checks. No backend/format changes and no additional loading threads.

Per-component/stage timings for the real 83-child store are NOT PROVIDED and UNMEASURED here. Worker busy time alone does not identify whether reads/decompression, validation, triangle expansion, normals or features dominate. Use the exposed reports before selecting a larger algorithmic optimization.

## Measurements

These are user-supplied baseline values, not reproduced in this session.

| Metric | Supplied previous progressive run | After this change |
|---|---:|---:|
| Full wall time | 560.3 s | UNMEASURED |
| Pipeline time | 489.2 s | UNMEASURED |
| Worker busy | 476.4 s | UNMEASURED |
| Accumulated GUI commit | 26.2 s | UNMEASURED |
| Reported GPU initialization | 26.2 s | UNMEASURED |
| Scene redraws | 786 | UNMEASURED |
| Submitted frames | 1,110 | UNMEASURED |
| Maximum recorded frame | approximately 936 ms | UNMEASURED |
| Queue-slot wait | 0 ms | UNMEASURED |
| Previous full synchronous load | NOT PROVIDED | UNMEASURED |
| Per-component reconstruction | NOT PROVIDED | UNMEASURED |

ProgressiveLoadProfile additionally reports accumulated scene_cpu_ms and max_scene_cpu_ms across actual onscreen scene preparation/draw submission/overlays. These are CPU wall durations, not GPU hardware timers. gpu_cpu_ms likewise measures CPU resource setup/transfer submission and can include driver waits. Worker, GUI and renderer times overlap and must not be added to estimate wall time.

For a rendering-suppressed diagnostic launch, set POCA_ZARR_SUPPRESS_RENDERING=1 in that process environment. While the collection state is Loading, onscreen scene work is suppressed; the existing producer, GUI attachment, positioning and progress remain active. Normal rendering resumes after completion/cancellation/failure. This switch is read on paint, creates no persistent setting, and must be unset for normal visualization/tests. It is deliberately diagnostic; its first completed scene can require resources not initialized during loading.

Compare normal and suppressed pipeline/worker/commit timings to separate reconstruction from renderer contention. Run the same store and selected components/placement/camera/cache conditions, alternate runs, retain logs and source hashes, and record dialog dwell separately. Also compare a complete synchronous open through the existing synchronous Engine/loader API or macro path; the interactive multi-child file open remains progressive. Do not re-export the store or change LOD budgets for the comparison. Report full reconstruction and first actual completed image frame separately.

## Regression coverage and results

Authored, registered, NOT EXECUTED:

- Tests > camera > Progressive Reset Viewer (actual image pixels): application-level Camera plus MdiChild reset sequence at 1/10/40/83 children, RAW/LABEL and shared/individual paths, camera rotation/translation first, positive-Z hidden-slider regression, metadata-envelope lifetime and transform/frame/selection preservation. Only one known bright image is selected, preventing other children or their refinement from masquerading as image recovery. Final framebuffer RGB differences must show that image.
- Camera zoom/persistence fixture: all eight deep-box corners in landscape/portrait orthographic/perspective frusta after rotated fitting.
- Progressive loader fixture: navigation commands are not rejected while analysis/export remain rejected; existing stable producer and preparation/GUI/display overlap, cancellation and ownership tests remain.
- Mesh fixture: exact double-coordinate, face and offset-vector equality after direct reads; component stage report presence. Existing persisted normals, feature values and malformed-geometry checks remain.

Real-store Reset Viewer at the four checkpoints, pyramidal recovery/Force L0, selected mixed meshes, final compositing, working-producer overlap, normal/suppressed timing and synchronous comparison are PENDING. No compatible executable/store path was supplied after the request for those paths. No compile, CMake generation, tests, app, helper, benchmark or backend execution occurred. Static verification cannot establish runtime visibility, GPU equivalence or a loading speedup.

## Shared mesh limitation

The conservative 256 MiB estimate and fallback remain. Independently bounded groups require a larger ownership/range/admission change than this fix; see [bounded mesh grouping investigation](POCA_ZARR_BOUNDED_MESH_GROUPING.md). No claim is made that shared meshes now fit this store.