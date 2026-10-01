set(targetname poca_kripleyplugin)

set(KRIPKEY_PLUGIN_HEADER_FILES
  KRipleyPlugin.hpp
  KRipleyWidget.hpp
  KRipley.hpp
  )

set(KRIPKEY_SOURCE_HEADER_FILES
  KRipleyPlugin.cpp
  KRipleyWidget.cpp
  KRipley.cpp
)

set(header_list
  ${KRIPKEY_PLUGIN_HEADER_FILES}
  )

set(source_list
  ${KRIPKEY_SOURCE_HEADER_FILES}
  )

set(moc_header_list
  KRipleyPlugin.hpp
  KRipleyWidget.hpp
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
