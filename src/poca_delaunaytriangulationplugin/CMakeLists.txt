set(targetname poca_delaunaytriangulationplugin)

set(DELAUNAY_TRIANGULATION_PLUGIN_HEADER_FILES
  DelaunayTriangulationPlugin.hpp
  DelaunayTriangulationDisplayCommand.hpp
  DelaunayTriangulationMultiObjectDisplayCommand.hpp
  DelaunayTriangulationBasicCommands.hpp
  DelaunayTriangulationWidget.hpp
  DelaunayTriangulationParamDialog.hpp
  )

set(DELAUNAY_TRIANGULATION_SOURCE_HEADER_FILES
  DelaunayTriangulationPlugin.cpp
  DelaunayTriangulationDisplayCommand.cpp
  DelaunayTriangulationMultiObjectDisplayCommand.cpp
  DelaunayTriangulationBasicCommands.cpp
  DelaunayTriangulationWidget.cpp
  DelaunayTriangulationParamDialog.cpp
)

set(header_list
  ${DELAUNAY_TRIANGULATION_PLUGIN_HEADER_FILES}
  )

set(source_list
  ${DELAUNAY_TRIANGULATION_SOURCE_HEADER_FILES}
  )

set(moc_header_list
  DelaunayTriangulationParamDialog.hpp
  DelaunayTriangulationPlugin.hpp
  DelaunayTriangulationWidget.hpp
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
)

install(TARGETS ${targetname} LIBRARY DESTINATION bin/plugins RUNTIME DESTINATION bin/plugins)
