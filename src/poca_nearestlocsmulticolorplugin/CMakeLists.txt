set(targetname poca_nearestlocsmulticolorplugin)

set(NEAREST_LOCS_MULTICOLOR_PLUGIN_HEADER_FILES
  NearestLocsMultiColorPlugin.hpp
  NearestLocsMultiColorCommands.hpp
  NearestLocsMultiColorWidget.hpp
  )

set(NEAREST_LOCS_MULTICOLOR_SOURCE_HEADER_FILES
  NearestLocsMultiColorPlugin.cpp
  NearestLocsMultiColorCommands.cpp
  NearestLocsMultiColorWidget.cpp
)

set(header_list
  ${NEAREST_LOCS_MULTICOLOR_PLUGIN_HEADER_FILES}
  )

set(source_list
  ${NEAREST_LOCS_MULTICOLOR_SOURCE_HEADER_FILES}
  )

set(moc_header_list
  NearestLocsMultiColorPlugin.hpp
  NearestLocsMultiColorWidget.hpp
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
