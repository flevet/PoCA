set(targetname poca_plot)

set(PLOT_HEADER_FILES
  Plot/CustomizedSlider.hpp
  Plot/FilterHistogramWidget.hpp
  Plot/Icons.hpp
  Plot/QCPGraphWithColor.hpp
  Plot/QCPHistogram.hpp
  Plot/qcustomplot.h
  Plot/Misc.h
  )

set(PLOT_SOURCE_FILES
  Plot/CustomizedSlider.cpp
  Plot/FilterHistogramWidget.cpp
  Plot/QCPGraphWithColor.cpp
  Plot/QCPHistogram.cpp
  Plot/qcustomplot.cpp
  Plot/Misc.cpp
)

set(header_list
  ${PLOT_HEADER_FILES}
  )

set(source_list
  ${PLOT_SOURCE_FILES}
  )

set(moc_header_list
  Plot/CustomizedSlider.hpp
  Plot/FilterHistogramWidget.hpp
  Plot/QCPHistogram.hpp
  Plot/qcustomplot.h
)

poca_add_static_library(${targetname}
    ALIAS PoCA::Plot
    SOURCES ${header_list} ${source_list} ${moc_header_list}
    PUBLIC_INCLUDE_DIRECTORIES
        ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC_LINK_LIBRARIES
        PoCA::Core
        PoCA::PluginAPI
        Qt5::Core
        Qt5::Gui
        Qt5::Widgets
        Qt5::OpenGL
        Qt5::PrintSupport
)
