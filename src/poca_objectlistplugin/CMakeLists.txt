set(targetname poca_objectlistplugin)

set(OBJECT_LIST_PLUGIN_HEADER_FILES
  SortableFloatItem.hpp
  ObjectListBasicCommands.hpp
  ObjectListPlugin.hpp
  ObjectListDisplayCommand.hpp
  ObjectListMultiObjectDisplayCommand.hpp
  ObjectListWidget.hpp
  ObjectListsWidget.hpp
  ObjectListParamDialog.hpp
  ObjectListsCommands.hpp
  )

set(OBJECT_LIST_PLUGIN_SOURCE_HEADER_FILES
  ObjectListBasicCommands.cpp
  ObjectListPlugin.cpp
  ObjectListDisplayCommand.cpp
  ObjectListMultiObjectDisplayCommand.cpp
  ObjectListMultiObjectTransparency.cpp
  ObjectListWidget.cpp
  ObjectListsWidget.cpp
  ObjectListParamDialog.cpp
  ObjectListsCommands.cpp
)

set(header_list
  ${OBJECT_LIST_PLUGIN_HEADER_FILES}
  )

set(source_list
  ${OBJECT_LIST_PLUGIN_SOURCE_HEADER_FILES}
  )

set(moc_header_list
  ObjectListParamDialog.hpp
  ObjectListPlugin.hpp
  ObjectListWidget.hpp
  ObjectListsWidget.hpp
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
        PoCA::OpenMP
        PoCA::CGAL
        PoCA::OpenGLDeps
)

install(TARGETS ${targetname} LIBRARY DESTINATION bin/plugins RUNTIME DESTINATION bin/plugins)
