set(targetname poca_objectcolocalizationplugin)

set(OBJECT_COLOCALIZATION_PLUGIN_HEADER_FILES
  ObjectColocalizationPlugin.hpp
  ObjectColocalization.hpp
  ObjectColocalizationCommands.hpp
  ObjectColocalizationWidget.hpp
  ObjectColocalizationInfoDialog.hpp
  )

set(OBJECT_COLOCALIZATION_SOURCE_HEADER_FILES
  ObjectColocalizationPlugin.cpp
  ObjectColocalization.cpp
  ObjectColocalizationCommands.cpp
  ObjectColocalizationWidget.cpp
  ObjectColocalizationInfoDialog.cpp
)

set(header_list
  ${OBJECT_COLOCALIZATION_PLUGIN_HEADER_FILES}
  )

set(source_list
  ${OBJECT_COLOCALIZATION_SOURCE_HEADER_FILES}
  )

set(moc_header_list
  ObjectColocalizationInfoDialog.hpp
  ObjectColocalizationPlugin.hpp
  ObjectColocalizationWidget.hpp
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
        PoCA::CGAL
)

install(TARGETS ${targetname} LIBRARY DESTINATION bin/plugins RUNTIME DESTINATION bin/plugins)
