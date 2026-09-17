set(targetname poca_voronoidiagramplugin)

set(VORONOI_DIAGRAM_PLUGIN_HEADER_FILES
  VoronoiDiagramPlugin.hpp
  VoronoiDiagramDisplayCommand.hpp
  VoronoiMultiObjectDisplayCommand.hpp
  VoronoiDiagramBasicCommands.hpp
  VoronoiDiagramParamDialog.hpp
  VoronoiDiagramWidget.hpp
  VoronoiDiagramCharacteristicsCommands.hpp
  VoronoiCommandContext.hpp
  )

set(VORONOI_DIAGRAM_SOURCE_HEADER_FILES
  VoronoiDiagramPlugin.cpp
  VoronoiDiagramDisplayCommand.cpp
  VoronoiMultiObjectDisplayCommand.cpp
  VoronoiDiagramBasicCommands.cpp
  VoronoiDiagramParamDialog.cpp
  VoronoiDiagramWidget.cpp
  VoronoiDiagramCharacteristicsCommands.cpp
)

set(header_list
  ${VORONOI_DIAGRAM_PLUGIN_HEADER_FILES}
  )

set(source_list
  ${VORONOI_DIAGRAM_SOURCE_HEADER_FILES}
  )

set(moc_header_list
  VoronoiDiagramParamDialog.hpp
  VoronoiDiagramPlugin.hpp
  VoronoiDiagramWidget.hpp
)

poca_add_plugin(${targetname}
    SOURCES ${header_list} ${source_list} ${moc_header_list}
    USE_CUDA
    LINK_LIBRARIES
        PoCA::Core
        PoCA::Geometry
        PoCA::OpenGL
        PoCA::Plot
        PoCA::Qt
        Qt5::Core
        Qt5::Gui
        Qt5::Widgets
        Qt5::OpenGL
        Qt5::PrintSupport
        PoCA::CGAL
)

install(TARGETS ${targetname} LIBRARY DESTINATION bin/plugins RUNTIME DESTINATION bin/plugins)
