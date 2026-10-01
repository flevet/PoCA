set(targetname poca_dbscanplugin)

set(DBSCAN_PLUGIN_HEADER_FILES
  DBSCANPlugin.hpp
  DBSCANWidget.hpp
  DBSCANCommand.hpp
  DBSCANCommandContext.hpp
  )

set(DBSCAN_SOURCE_HEADER_FILES
  DBSCANPlugin.cpp
  DBSCANWidget.cpp
  DBSCANCommand.cpp
)

set(header_list
  ${DBSCAN_PLUGIN_HEADER_FILES}
  )

set(source_list
  ${DBSCAN_SOURCE_HEADER_FILES}
  )

set(moc_header_list
  DBSCANPlugin.hpp
  DBSCANWidget.hpp
)

poca_add_plugin(${targetname}
    SOURCES ${header_list} ${source_list} ${moc_header_list}
    USE_CUDA
    LINK_LIBRARIES
        PoCA::Core
        PoCA::Geometry
        PoCA::OpenGL
        PoCA::Plot
        Qt5::Core
        Qt5::Gui
        Qt5::Widgets
        Qt5::OpenGL
        Qt5::PrintSupport
)

install(TARGETS ${targetname} LIBRARY DESTINATION bin/plugins RUNTIME DESTINATION bin/plugins)
