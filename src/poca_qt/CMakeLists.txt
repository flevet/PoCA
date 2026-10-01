set(targetname poca_qt)

set(QT_WIDGET_HEADER_FILES
  Widgets/ButtonLayer.hpp
  Widgets/CustomColorDialog.hpp
  Widgets/PerformanceWidget.hpp
  )

set(QT_WIDGET_SOURCE_FILES
  Widgets/ButtonLayer.cpp
  Widgets/CustomColorDialog.cpp
  Widgets/PerformanceWidget.cpp
  )

set(header_list
  ${QT_WIDGET_HEADER_FILES}
  )

set(source_list
  ${QT_WIDGET_SOURCE_FILES}
  )

set(moc_header_list
  Widgets/ButtonLayer.hpp
  Widgets/CustomColorDialog.hpp
)

poca_add_static_library(${targetname}
    ALIAS PoCA::Qt
    SOURCES ${header_list} ${source_list} ${moc_header_list}
    PUBLIC_INCLUDE_DIRECTORIES
        ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC_LINK_LIBRARIES
        PoCA::Core
        PoCA::Plot
        PoCA::PluginAPI
        Qt5::Core
        Qt5::Gui
        Qt5::Widgets
        Qt5::OpenGL
        Qt5::PrintSupport
)
