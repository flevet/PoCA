set(targetname poca)

set(OTHER_HEADER_FILES
)

set(SMLM_OBJECT_HEADER_FILES
  Objects/SMLM_Object/SMLMObject.hpp
  )

set(SMLM_OBJECT_SOURCE_FILES
  Objects/SMLM_Object/SMLMObject.cpp
  )

set(COLOC_OBJECT_HEADER_FILES
  Objects/Coloc_Object/ColocObject.hpp
  )

set(COLOC_OBJECT_SOURCE_FILES
  Objects/Coloc_Object/ColocObject.cpp
  )

set(WIDGET_HEADER_FILES
  Widgets/MainFilterWidget.hpp
  Widgets/MdiChild.hpp
  Widgets/ColocalizationChoiceDialog.hpp
  Widgets/MacroWidget.hpp
  Widgets/MainWindow.hpp
  Widgets/MergeDatasetsChoiceDialog.hpp
  Widgets/PythonWidget.hpp
  Widgets/ROIGeneralWidget.hpp
  Widgets/ReorganizeRenderingWidget.hpp
  Widgets/ColorButtonGridWidget.hpp
  Widgets/DatasetAssemblerWidget.hpp
  )

set(WIDGET_SOURCE_FILES
  Widgets/MainFilterWidget.cpp
  Widgets/MdiChild.cpp
  Widgets/ColocalizationChoiceDialog.cpp
  Widgets/MacroWidget.cpp
  Widgets/MainWindow.cpp
  Widgets/MergeDatasetsChoiceDialog.cpp
  Widgets/PythonWidget.cpp
  Widgets/ROIGeneralWidget.cpp
  Widgets/ReorganizeRenderingWidget.cpp
  Widgets/ColorButtonGridWidget.cpp
  Widgets/DatasetAssemblerWidget.cpp
 )

set(header_list
  ${OTHER_HEADER_FILES}
  ${SMLM_OBJECT_HEADER_FILES}
  ${COLOC_OBJECT_HEADER_FILES}
  ${WIDGET_HEADER_FILES}
  )

set(source_list
  ${SMLM_OBJECT_SOURCE_FILES}
  ${COLOC_OBJECT_SOURCE_FILES}
  ${WIDGET_SOURCE_FILES}
  )

set(main_CPP
  main.cpp
  )

set(moc_header_list
  Widgets/ColocalizationChoiceDialog.hpp
  Widgets/ColorButtonGridWidget.hpp
  Widgets/DatasetAssemblerWidget.hpp
  Widgets/MacroWidget.hpp
  Widgets/MainFilterWidget.hpp
  Widgets/MainWindow.hpp
  Widgets/MdiChild.hpp
  Widgets/MergeDatasetsChoiceDialog.hpp
  Widgets/PythonWidget.hpp
  Widgets/ROIGeneralWidget.hpp
  Widgets/ReorganizeRenderingWidget.hpp
)

poca_add_executable(${targetname}
    SOURCES ${header_list} ${source_list} ${moc_header_list} ${main_CPP}
    USE_CUDA
    LINK_LIBRARIES
        PoCA::Qt
        PoCA::OpenGL
        PoCA::Geometry
        PoCA::Plot
        PoCA::Core
        Qt5::Core
        Qt5::Gui
        Qt5::Widgets
        Qt5::OpenGL
        Qt5::PrintSupport
        PoCA::CGAL
)

install(TARGETS ${targetname} RUNTIME DESTINATION bin)
