set(targetname poca_loadercsv)

set(LOADER_HEADER_FILES
  LoaderCSVFile.hpp
  OpenFileDialog.hpp
  )

set(LOADER_SOURCE_FILES
  LoaderCSVFile.cpp
  OpenFileDialog.cpp
)

set(header_list
  ${LOADER_HEADER_FILES}
  )

set(source_list
  ${LOADER_SOURCE_FILES}
  )

set(moc_header_list
  LoaderCSVFile.hpp
  OpenFileDialog.hpp
)

poca_add_plugin(${targetname}
    SOURCES ${header_list} ${source_list} ${moc_header_list}
    USE_CUDA
    LINK_LIBRARIES
        PoCA::Core
        PoCA::Geometry
        PoCA::OpenGL
        Qt5::Core
        Qt5::Widgets
)

install(TARGETS ${targetname} LIBRARY DESTINATION bin/plugins RUNTIME DESTINATION bin/plugins)
