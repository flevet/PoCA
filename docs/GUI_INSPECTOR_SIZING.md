# Inspector sizing in the current PoCA GUI

2026-10-08. Source-only change; runtime behavior was not verified.

## Static audit and causes

The Objects dock keeps its existing 300 logical-pixel minimum. It contains
ObjectsDockTabs, a vertical Objects/inspector splitter, Properties/Controls/
Camera/ROI Manager tabs, and the existing Controls and Camera scroll areas.
Both scroll areas already use setWidgetResizable(true).

Plugin insertion in poca::core::utils::addWidget/addSingleTabWidget creates
nested QTabWidgets; addWidget may put a vertical layout directly on a tab
container. OrganoGraph also creates its own nested tabs. MainWindow applies
the sizing policy after Engine::addGUI, covering those existing containers
without changing core insertion APIs or introducing a Core -> Qt dependency.

The main sources of excessive horizontal requirements were:

- QCustomPlot::minimumSizeHint and sizeHint both return its plot layout's
  minimum outer size. Expanding horizontal policies retain that minimum.
- PoCA's QCPHistogram additionally advertises a 400-pixel preferred width;
  FilterHistogramWidget previously reapplied an Expanding policy.
- OrganoFeatureSelector inherited QComboBox's content sizing. X/Y/Z selectors
  sat in one horizontal row and could each advertise a long feature name.
- Dataset path/name labels used Preferred sizing without clipping or wrapping.
- Tabs and stacked pages aggregate content hints, including inactive pages.
  Inspector tab bars disabled scroll buttons.
- OrganoGraph's centroid/Voronoi options, category choices, LUT/plot-mode
  controls, distribution controls and transfer targets were crowded into
  long rows. Plot actions also summed several button minima.

The audit inspected size policies, explicit sizes, hints, layouts, scroll
areas and plugin selectors/plots in both repositories. Fixed icon sizes and
vertical plot minimum heights remain useful constraints.

## Minimal reusable policy

poca_qt/Widgets/InspectorSizing.hpp/.cpp is the only new production source
pair. The existing ButtonLayer, CustomColorDialog and PerformanceWidget own
other responsibilities; none was an appropriate owner for inspector sizing.

The poca::qt functions configure existing Qt objects:

- makeHorizontallyShrinkable: horizontal Ignored policy and minimumWidth(0),
  preserving the complete vertical policy, stretch and height-for-width flags.
- configureInspectorCombo: AdjustToMinimumContentsLengthWithIcon with a
  shared 12-character hint, followed by horizontal shrinking. This offers
  useful text beside an inspector row label without scanning the longest
  feature name. The hint is not a fixed pixel minimum.
- configureInspectorLabel: horizontal shrinking with optional word wrapping.
  Identifier labels clip and callers maintain dynamic tooltips.
- configureInspectorTabs: shrink existing tab/tab-bar/stack containers,
  enable tab scroll buttons and elide tab captions. The traversal is limited
  to these container types; it does not change every descendant's policy.

Callers opt in explicitly. No event filter, polling, subclass framework,
fixed sidebar maximum, ownership change or QCustomPlot dependency was added
to this policy. Apply configureInspectorTabs after inserting plugin widgets.

## OrganoGraph and plot changes

X/Y/Z now have one selector per row. The component form uses WrapLongRows;
centroid and Voronoi controls stack. Category and plot-mode controls use
small grids, distributions and transfer choices stack, and plot actions
use two rows. The symbol selector has explicit stretch so a spacer cannot
consume the space freed by ignoring its size hint.

All OrganoGraph combo boxes use the policy, with feature selectors configured
by their existing subclass. Full item labels/data/category menus are preserved.
The selected feature tooltip now includes the full name and description.
Coverage/symbol labels wrap; identity/color labels shrink. Existing plot
identity tooltips remain managed by the identification code.

The QCustomPlot, plot stack, optional EmbeddingScatter3DWidget and its native
window container shrink horizontally. The main plot/stack retain vertical
expansion and plot-height constraints. The plot gets explicit layout stretch.
Axes, interactions, selection, commands and scientific/rendering algorithms
are unchanged.

Other exact plot-policy cases were adjusted in DetectionSet (three cleaner
plots), Voronoi, K-Ripley, ClusterVisu, TrackSet, NanoSynAtlas and LatentSpace.
K-Ripley result and LatentSpace status labels wrap. TransferFeature component
selectors use compact sizing. Number-only selectors were left alone.

QCPHistogram owns its plot-specific policy so all its PoCA instances benefit.
FilterHistogramWidget stops overwriting that policy and gives the plot stretch.
This remains in poca_plot because poca_qt already links poca_plot; linking back
would create a cycle. Third-party qcustomplot.cpp/.h are unchanged.

## Manual source fixtures

OrganoGraphSizingTests.cpp uses the existing plugin TestRegistry with the new
ORGANO_GUI_SOURCE_TESTS option, default OFF. Its manual GUI action covers:

1. Compact combo minimum hint after inserting a 512-character item; resetting
   a previous explicit 1200-pixel minimum.
2. Exact feature/item labels, current data and full feature tooltip.
3. Long label shrinking in a real parent layout, with unchanged text.
4. A 1400-pixel plot layout hint accepted by a narrow plot/stack.
5. Combo/label and plot/container growth when parent width increases.
6. OrganoGraph's layout minimum and hidden 1600-pixel tab-page isolation from
   the dock, with tab navigation available and the dock minimum still 300.
7. Vertical policy preservation, existing plot zoom interaction, and the
   QCPHistogram 400-pixel hint remaining preferred rather than required.

These fixtures are source only. They were not configured, compiled or run.
The manual hierarchy fixture shows a temporary dock and delivers layout events
only when a user explicitly runs that registered test.

## Files

Added:
- poca/src/poca_qt/Widgets/InspectorSizing.hpp
- poca/src/poca_qt/Widgets/InspectorSizing.cpp
- poca_extra/src/poca_organographplugin/OrganoGraphSizingTests.cpp
- poca/docs/GUI_INSPECTOR_SIZING.md

Modified:
- poca/src/poca/Widgets/MainWindow.cpp
- poca/src/poca/Widgets/MainFilterWidget.cpp
- poca/src/poca_qt/CMakeLists.txt
- poca/src/poca_plot/Plot/QCPHistogram.cpp
- poca/src/poca_plot/Plot/FilterHistogramWidget.cpp
- poca/src/poca_detectionsetplugin/DetectionSetWidget.cpp
- poca/src/poca_voronoidiagramplugin/VoronoiDiagramWidget.cpp
- poca/src/poca_kripleyplugin/KRipleyWidget.cpp and CMakeLists.txt
- poca_extra/src/poca_clustervisuplugin/ClusterVisuWidget.cpp and CMakeLists.txt
- poca_extra/src/poca_tracksetplugin/TrackSetWidget.cpp and CMakeLists.txt
- poca_extra/src/poca_nanosynatlasplugin/NanoSynAtlasWidget.cpp and CMakeLists.txt
- poca_extra/src/poca_latentspaceexplorationplugin/LatentSpaceExplorationWidget.cpp and CMakeLists.txt
- poca_extra/src/poca_transferfeatureplugin/TransferFeatureWidget.cpp and CMakeLists.txt
- poca_extra/src/poca_organographplugin/OrganoFeatureSelector.cpp
- poca_extra/src/poca_organographplugin/OrganoGraphWidget.cpp
- poca_extra/src/poca_organographplugin/EmbeddingScatter3DWidget.cpp
- poca_extra/src/poca_organographplugin/OrganoGraphPlugin.cpp
- poca_extra/src/poca_organographplugin/CMakeLists.txt
- CONTINUITY.md

## Deliberate limitations and verification

This is not a complete GUI rewrite. Some legacy plugin/Camera/ROI controls
still have wide multi-column rows or long button/check-box captions. Existing
scroll areas retain AsNeeded horizontal scrolling for those genuine child
constraints. Long OrganoGraph controls use more vertical space, served by
the existing Controls scroll area. Plot heights are unchanged.

QCustomPlot's internal layout still calculates its natural minimum, so axes,
legends or long in-plot text may clip at very narrow sizes. No axis/legend
algorithm was changed. Native 3D context/platform behavior and font/DPI/style
effects require later runtime inspection. Tab/stack traversal runs at current
GUI assembly points; later dynamically inserted tabs need the same policy.

Static checks cover source registration, PoCA::Qt include/link availability,
unchanged command/scientific/storage paths, quoting artifacts, diff whitespace,
UTF-8, BOM preservation and CRLF. The completed audit verified 29 files, 25 retained HEAD BOM statuses, 53 quoted includes and 10 Qt consumer links; diff and artifact checks are clean. No ADS or new external dependency was introduced.
No CMake configuration, build, application, test, Python or helper executable
was run. Runtime behavior was not verified.