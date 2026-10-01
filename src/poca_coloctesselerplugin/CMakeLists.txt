set(targetname poca_coloctesselerplugin)

set(COLOC_TESSELER_PLUGIN_HEADER_FILES
  ColocTesselerPlugin.hpp
  ColocTesselerDisplayCommand.hpp
  ColocTesselerBasicCommands.hpp
  ColocTesselerWidget.hpp
  ColocTesseler.hpp
  MyTableWidget.hpp
  )

set(COLOC_TESSELER_SOURCE_HEADER_FILES
  ColocTesselerPlugin.cpp
  ColocTesselerDisplayCommand.cpp
  ColocTesselerBasicCommands.cpp
  ColocTesselerWidget.cpp
  ColocTesseler.cpp
  MyTableWidget.cpp
)

set(header_list
  ${COLOC_TESSELER_PLUGIN_HEADER_FILES}
  )

set(source_list
  ${COLOC_TESSELER_SOURCE_HEADER_FILES}
  )

set(moc_header_list
  ColocTesselerPlugin.hpp
  ColocTesselerWidget.hpp
)

poca_add_plugin(${targetname}
    SOURCES ${header_list} ${source_list} ${moc_header_list}
    USE_CUDA
    CUDA_RESOLVE_DEVICE_SYMBOLS
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
