# Camera projection scale and physical distance

2026-10-05 — source-only hardening. Compilation, Qt/OpenGL behavior, movie
timing and visual acceptance remain unverified.

## State and operations

`m_distanceOrtho` is projection half-extent on the smaller viewport dimension.
`m_cameraDistance` is signed physical eye-to-center distance for both projections.
`StateCamera::m_eye` is an orientation anchor, not the translated physical eye.
`getCameraDistance()` returns stored distance; `getCameraPosition()` and
`updateCamera()` use the shared physical-position calculation.

- Orthographic `zoomBy` adds the existing delta and clamps extent to 0.0001.
  Orthographic `setDistanceOrtho` clamps/sets extent. Both recalculate projection
  without changing physical distance, eye anchor, orientation or view matrix.
- Perspective `zoomBy` retains signed distance addition, eye/view rebuilding and
  equivalent extent tracking. Perspective `setDistanceOrtho` maps extent to
  signed distance for existing movie/path compatibility.
- Projection switching converts once at the scene center plane. To perspective:
  distance = signed(extent * factorH / tan(halfFov)); extent then tracks
  abs(distance) * tan(halfFov). To orthographic: extent =
  abs(distance) * tan(halfFov) / factorH; physical distance stays fixed.
  factorH is 1 for landscape/square and height/width for portrait viewports.
  Center/orientation stay fixed; only entering perspective rebuilds the view.
  Setting the currently active projection remains a no-op.
- Constructor and `zoomToBoundingBox(..., true)` fit using
  max(perspective-equivalent distance, half-diagonal * 1.01 + 0.001).
  Existing constructor extent and bounding-box XY fit calculations are retained.
  Fit/reset initializes both once; later ortho zoom remains independent.
  `resetProjection` still delegates to bounding-box fit.
- Near = max(0.001, abs(distance)/1000); far = abs(distance) + 4*sceneRadius.
  The existing scene-radius calculation is retained. Ortho extent is excluded.
  At a centered bounding-box fit, the eye lies outside the bounding sphere
  with room for the positive near plane through rotation.
- Trackball, pivot, model translation and panning are unchanged.
  Screen-pixel/gizmo scaling still uses ortho extent. Ctrl+Z captures and
  restores both independent distances exactly without projection conversion.

`CameraGeometry.hpp` holds pure calculations used by production and fixtures.
`CameraZoomPersistence.hpp` holds JSON policy; `CameraPersistence.cpp` bridges
that policy to Camera. No UI slots or dependencies were added.

## JSON and partial restoration

MainWindow, MainFilterWidget and CreateMovie screenshot capture now call
`saveZoomState`, writing `distanceOrtho` and the new optional `cameraDistance`.
Other camera fields and existing command dispatch are retained.

| Active projection / selection | Extent | Physical distance |
| --- | --- | --- |
| Ortho, zoom only | Restore extent | Keep |
| Ortho, view only | Keep | Restore distance |
| Ortho, view + zoom | Restore extent | Restore independently |
| Perspective, zoom selected | Restore extent if present | Restore distance |
| Perspective, view without zoom | Keep | Keep |
| Neither view nor zoom | Keep | Keep |

Legacy JSON lacking cameraDistance derives distanceOrtho/tan(halfFov) once
when physical-distance restoration is selected. Legacy ortho uses the old
positive radius; legacy perspective preserves current sign. Ortho zoom-only
legacy loads keep current physical distance. An explicit new distance wins over
perspective-equivalent setter conversion. Wrong JSON types remain errors.

MainWindow restores after the selected view/rotation/translation fields and
before crop/fit. `zoomToBoundingBox(..., false)` cannot overwrite either value.
Explicit `_reset=true` still supersedes saved zoom with a fresh fit.
Crop/rotation/translation flags and distanceOrthoOriginal handling are retained.
Projection type remains unsaved: select the intended mode before loading.

## Movie and legacy paths

CreateMovie restores the first saved position's zoom metadata once. Later
orthographic interpolation changes extent while keeping one physical radius.
Perspective interpolation still maps extent to signed distance.
CreateMovieWidget position apply restores new/legacy metadata through the
shared full-restoration policy. MainWindow's two legacy JSON path entrypoints
also restore the first position once for traveling paths.

Camera's path initialization/frame writes use `applyDistanceOrtho`, sharing
the public setter's calculation without an extra matrix/projection rebuild
before the existing batched orientation update. Scale-only tuple paths keep
the current ortho radius. A pre-existing limitation remains: the two-state path
overload initializes m_statesPath but not the tuple samples consumed by its
timer callback. This task does not repair that older path mechanism.

`computeCameraProxyPoint` keeps orientation * (2 * distanceOrtho). This is the
trajectory/arc-length scale proxy, not the physical eye. Interpolation, timing,
proxy shape and capture are retained. Later ortho keyframe physical distances
are not interpolated; explicit position apply can restore another radius.
Center/orientation travel may still move the eye with a fixed radius.

## Streaming and limits

With fixed bounds/orientation/distance, ortho zoom changes XY projection only.
A fully depth-visible aligned volume has a nested XY source footprint with
unchanged source Z interval. A rotated volume can still change local Z footprint
because shrinking XY planes intersect it differently; this is valid geometry.

No production streaming, readers, residency, scientific level-zero, image
bounding-box/scaling/calibration, spatial metadata, TensorStore/backend, Zarr
or TIFF implementation changed. No GL work moved to workers. ScatterplotGL
and ObjectListDisplayCommand local projections are untouched.

Projection continuity is approximate away from the scene center plane.
Switching into perspective after extreme zoom can put the eye inside the
scene; existing perspective zoom/clipping policy remains. Explicit file/legacy
distances restore faithfully instead of silently applying safe-fit bounds.
Arbitrary transforms/pivots may move bounds away from the centered fit.
Zero-size viewport handling and live Qt/widget undo require runtime checks.

## Source fixtures (written and registered, never executed)

Existing registry entry: camera / Camera zoom and persistence;
ID `camera_zoom_geometry`. Fixtures construct no live QOpenGLWidget.

| Requested regression | Fixture |
| --- | --- |
| 1–4: ortho zoom/setter/eye/near/far | ImageCameraZoomTests: checkOrthographicZoom |
| 5–7: perspective motion, transitions, both round trips, portrait/landscape | ImageCameraZoomTests: checkPerspectiveAndTransitions |
| 8–9: safe fit/deep box/zoom after fit | ImageCameraZoomTests: checkFitAndUndo |
| 10: exact independent undo values | ImageCameraZoomTests: checkFitAndUndo; actual capture/restore audited |
| 11–12: new/legacy JSON, partial flags, sign, malformed types | ImageCameraPersistenceTests |
| 13–14: ortho/perspective movie interpolation | ImageCameraZoomTests: checkMovieScales |
| 15: nested XY, stable depth/Z, zoom-out | ImageCameraStreamingTests |
| 16: unchanged image dimensions/calibration/pixels | ImageCameraStreamingTests; production diff scope audit |

Helpers are exercised in source fixtures; event delivery, widget matrices,
GPU rendering, movie frames and performance remain unverified.

## Coupling audit and changed files

Original coupling: constructor; getter; zoomBy; setDistanceOrtho; bounding-box
fit; projection switch; legacy animateCameraPath initialization/frame writes.
Dependent users audited: clipping; updateCamera/updateCameraEyeUp; physical
position; resetProjection; rotation/panning; screen-pixel/gizmo scaling; undo;
all JSON capture/load paths; MainWindow path readers; CreateMovie apply,
playback and proxy. Remaining conversions occur only in fit, perspective
operations, projection transitions and explicit legacy restoration.

Changed files, relative to repository root:

- poca/src/poca_opengl/OpenGL/Camera.hpp, Camera.cpp, CameraGeometry.hpp (new),
  CameraZoomPersistence.hpp (new), CameraPersistence.cpp (new).
- poca/src/poca_opengl/CMakeLists.txt (source lists only).
- poca/src/poca/Widgets/MainWindow.cpp, MainFilterWidget.cpp.
- poca_extra/src/poca_createmovieplugin/CreateMovieCommand.cpp, CreateMovieWidget.cpp.
- poca_extra/src/poca_imageplugin/ImageCameraZoomTests.cpp (new),
  ImageCameraPersistenceTests.cpp (new), ImageCameraStreamingTests.cpp (new),
  ImageStreamingTests.hpp, ImageStreamingTests.cpp, CMakeLists.txt (source lists).
- poca/docs/CAMERA_ARCHITECTURE.md, CONTINUITY.md.

## Manual validation after your build

A. Ortho: zoom very far in/out; depth stays visible and physical position and
near/far remain fixed.

B. Rotate strongly after zoom; radius remains fixed without unexpected clipping.
Check normal screen-plane panning.

C. Phase-5 streaming: aligned depth-visible volume, zoom in/out; XY ROI changes
without Z jumps caused by moving near/far.

D. Ortho -> Perspective -> Ortho and reverse; check center-plane apparent scale
in landscape and portrait windows.

E. Save/load new positions from both UIs and old JSON. Check zoom-only, view-only,
full restore, crop/rotation/translation flags, explicit reset and Ctrl+Z.

F. CreateMovie: capture/apply, orthographic zoom/travel, legacy paths, stable
physical radius, correct scale interpolation, perspective movies and proxy shape.

G. Perspective zoom still physically moves the eye. Check switching after deep
zoom and unchanged image calibration/scaling.

No CMake configure/generate, compilation, linking, installation,
application executable, Python script, benchmark, or test was run.

