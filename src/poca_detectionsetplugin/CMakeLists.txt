set(targetname poca_detectionsetplugin)

set(DETECTION_SET_PLUGIN_HEADER_FILES
  DetectionSetPlugin.hpp
  DetectionSetBasicCommands.hpp
  DetectionSetDisplayCommand.hpp
  DetectionSetMultiObjectDisplayCommand.hpp
  GaussianDisplayCommand.hpp
  HeatMapDisplayCommand.hpp
  DetectionSetWidget.hpp
  CleanerCommand.hpp
  DetectionSetCommandContext.hpp
  )

set(DETECTION_SET_PLUGIN_SOURCE_FILES
  DetectionSetPlugin.cpp
  DetectionSetBasicCommands.cpp
  DetectionSetDisplayCommand.cpp
  DetectionSetMultiObjectDisplayCommand.cpp
  GaussianDisplayCommand.cpp
  HeatMapDisplayCommand.cpp
  DetectionSetWidget.cpp
  CleanerCommand.cpp
)

set(header_list
  ${DETECTION_SET_PLUGIN_HEADER_FILES}
  )

set(source_list
  ${DETECTION_SET_PLUGIN_SOURCE_FILES}
  )

set(moc_header_list
  DetectionSetPlugin.hpp
  DetectionSetWidget.hpp
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
        PoCA::OpenMP
)

install(TARGETS ${targetname} LIBRARY DESTINATION bin/plugins RUNTIME DESTINATION bin/plugins)
