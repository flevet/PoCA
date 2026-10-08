# Inspector sizing in the current PoCA GUI

2026-10-08. Source-only horizontal milestone and vertical follow-up.
Runtime behavior was not verified.

## Current architecture and horizontal milestone

The Objects dock retains its 300 logical-pixel minimum, ObjectsDockTabs,
vertical Objects/inspector splitter, Properties/Controls/Camera/ROI tabs and
resizable Controls/Camera scroll areas. Core plugin insertion still creates
nested tabs; MainWindow applies InspectorSizing after Engine::addGUI.
No docking framework, polling, ownership or command dispatch changes.

Horizontal milestone aff1894 made labels/combos/plots/containers shrinkable.
QCustomPlot's preferred and minimum hints both derive from its internal layout;
QCPHistogram additionally advertises a 400-pixel preferred width. Long combo
names, labels and inactive tab/stack pages previously propagated large widths.
OrganoGraph's component form wraps long rows; X/Y/Z selectors have separate
rows; category/plot-mode/distribution/transfer/action controls use compact rows.
Full names, model data, tooltips, zoom and selection behavior are preserved.

InspectorSizing.hpp/.cpp remains the only sizing utility in poca_qt:

- makeHorizontallyShrinkable sets horizontal Ignored and minimumWidth(0),
  preserving vertical policy/stretch and height-for-width flags.
- configureInspectorCombo uses AdjustToMinimumContentsLengthWithIcon with
  a shared 12-character hint; full names and item data are unchanged.
- configureInspectorLabel shrinks identifiers and optionally wraps explanations.
- configureInspectorTabs applies horizontal shrinking to tab/stack containers,
  enables tab scroll buttons and elides captions after plugin insertion.
- configureInspectorPlot, added in this follow-up, accepts QWidget only and
  reuses horizontal shrinking. It sets vertical Preferred, zero vertical policy
  stretch and one shared 200-pixel minimum. Preferred retains GrowFlag/ShrinkFlag;
  natural sizeHint remains in effect, with no fixed or maximum height added.
  Parent layouts assign spare height to a spacer below the controls.

poca_qt has no QCustomPlot, QtDataVisualization or OrganoGraph dependency.
QCPHistogram owns its local Ignored/Preferred policy because poca_qt already
links poca_plot; a reverse dependency would create a cycle. FilterHistogramWidget
retains its QHBoxLayout plot stretch: that stretch is horizontal. Its existing
caller-supplied maximum for compact feature histogram rows is unchanged.

## Confirmed vertical causes and fix

OrganoGraph's QCustomPlot had a 350-pixel minimum and vertical Expanding;
the stack also used vertical Expanding. The optional 3D page had a 350-pixel
minimum; its page/native container used vertical Expanding. In addition,
addWidget(m_plotStack, 1) made the plot stack the sole positive-stretch child,
taking spare height ahead of the trailing empty filler widget.

The 2D plot, stack, optional EmbeddingScatter3DWidget and native window container
now use configureInspectorPlot. Both pages and the stack have compatible
Ignored/Preferred policies and 200-pixel minima. QCustomPlot's ordinary natural
height hint is below this floor; the effective preferred height is 200 pixels.
The stack still aggregates its pages' hints, but no inactive 350-pixel page
remains. No global stack subclass or preferred-height cap was introduced.

The stack is inserted with zero layout stretch. The empty filler is replaced
by addStretch(1) after identification, transfer and plot-action controls. Thus
normal spare panel height goes below the content. A parent can still explicitly
allocate more height to the plot: Preferred permits growth without a maximum.
The native window container keeps its ownership and zero-margin filling layout.
No axes/ranges/data/LUT/scatter/embedding/selection/command algorithms changed.

## Other exact sites audited

The shared 200-pixel minimum follows existing K-Ripley, TrackSet and NanoSynAtlas
one-region plot sizing; it provides a usable inspector plot without a 350+ floor.

| Plot site | Vertical follow-up |
| --- | --- |
| Voronoi characteristics | Expanding and minimum 400 become shared Preferred/200; spare grid row stretch 1 and existing outer filler stretch 1. |
| ClusterVisu Monte Carlo | Expanding becomes shared Preferred/200; local empty filler becomes trailing stretch 1. |
| LatentSpace plot | Expanding and minimum 420 become shared Preferred/200; plot stretch 1 removed, trailing stretch 1 after status. |
| K-Ripley | Existing Preferred/200 now uses helper; expanding spare-space filler retained. |
| TrackSet MSD | Existing Preferred/200 now uses helper; expanding spare-space filler retained. |
| NanoSynAtlas one-region | Existing Preferred/200 now uses helper; expanding spare-space filler retained. |
| DetectionSet cleaner plots | Already Ignored/Preferred with 150-pixel minima; expanding statistics widget/filler; source audited, unchanged. |
| QCPHistogram / feature histogram rows | Already Preferred vertically; 30-pixel preferred hint and horizontal stretch; source audited, unchanged. |

Horizontal combo/label/name/feature-selector policies are unchanged. TransferFeature
selectors, wrapped K-Ripley results and LatentSpace status remain as in aff1894.

## Manual source fixtures (updated, not run)

OrganoGraphSizingTests.cpp remains registered through the existing TestRegistry
and ORGANO_GUI_SOURCE_TESTS option, default OFF. No framework/files were added.

The previous fixture incorrectly required OrganoGraph vertical Expanding. It now
requires Preferred for plot, stack, optional 3D page and native container; zoom
interaction is asserted separately. Its artificial 1400-pixel width hint retains
horizontal coverage while its vertical hint is reduced from 350 to 200.

Coverage:

1. OrganoGraph QCustomPlot and stack remain horizontally shrinkable; long names,
   combo model values/tooltips, dynamic labels and hidden wide tabs retain coverage.
2. Plot/stack vertical Preferred, zero policy stretch and compact 200-pixel minima.
3. Usable nonzero minimum/effective preferred heights, with no maximum height.
4. Optional 3D page/native container have matching compact policies/minima.
5. OrganoGraph plot layout stretch zero; only the final spacer gets stretch 1.
6. Taller dock keeps the plot compact instead of assigning all spare height.
7. Fixture-only parent stretch makes 2D/3D pages grow and return to compact height;
   GrowFlag/ShrinkFlag and native container geometry remain checked.
8. Shared QCustomPlot fixture resets a legacy 420-pixel minimum, stays compact
   beside a spacer and permits explicit growth/shrinking; histogram stays Preferred.
9. Static source audit verifies the six other plugin sites use the same helper;
   DetectionSet and histogram source already use compatible vertical policies.

The fixture shows a temporary dock and delivers resize/layout events only when
explicitly run by the user. It was not configured, compiled or executed here.

## Vertical follow-up working set

No new classes/files/dependencies or CMake registration changes.

- poca/src/poca_qt/Widgets/InspectorSizing.hpp and InspectorSizing.cpp
- poca_extra/src/poca_organographplugin/OrganoGraphWidget.cpp,
  EmbeddingScatter3DWidget.cpp and OrganoGraphSizingTests.cpp
- poca/src/poca_voronoidiagramplugin/VoronoiDiagramWidget.cpp
- poca/src/poca_kripleyplugin/KRipleyWidget.cpp
- poca_extra/src/poca_clustervisuplugin/ClusterVisuWidget.cpp
- poca_extra/src/poca_tracksetplugin/TrackSetWidget.cpp
- poca_extra/src/poca_nanosynatlasplugin/NanoSynAtlasWidget.cpp
- poca_extra/src/poca_latentspaceexplorationplugin/LatentSpaceExplorationWidget.cpp
- poca/docs/GUI_INSPECTOR_SIZING.md and CONTINUITY.md

The full horizontal milestone working set is available in commit aff1894.

## Limitations and source verification

Shrinking stops at the explicit minimum; the existing Controls scroll area
serves smaller available heights and tall control lists. Plot growth requires
the containing layout to deliberately allocate height; making the normal panel
taller primarily grows the bottom spacer. Natural hints can exceed 200 for
unusual plot content/fonts/DPI; there is no cap or active-page stack subclass.
QCustomPlot axes/legends can still clip at narrow sizes; no plot layout algorithm
was changed. Native 3D platform/context behavior and font/style effects need
later runtime verification. Some legacy plugin/Camera/ROI rows and captions
remain wide, using existing AsNeeded horizontal scrolling where required.
Later dynamically inserted tabs need the same existing horizontal policy call.

Static checks cover helper declarations/call sites, retained PoCA::Qt links,
zero plot stretch/spare-space allocation, fixture assertions, unchanged scientific
and storage/loading paths, diff/quoting artifacts, UTF-8/BOM preservation and CRLF.
The vertical follow-up verified 13 changed files: strict UTF-8, consistent CRLF,
unchanged HEAD BOM status and clean diff/artifact checks; all seven plugin Qt
consumer links and existing helper/manual-fixture registrations remain valid.
The prior horizontal audit covered 29 files, 53 quoted includes and 10 Qt links.
No ADS, external dependency, dock/tab architecture or command/macro changes.
No CMake, build, application, tests, Python or helper executable was run.
Runtime behavior was not verified.
