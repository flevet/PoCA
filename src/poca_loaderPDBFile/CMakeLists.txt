set(targetname poca_loaderpdb)

set(LOADER_HEADER_FILES
  LoaderPDBFile.hpp
  )

set(LOADER_SOURCE_FILES
  LoaderPDBFile.cpp
)

set(header_list
  ${LOADER_HEADER_FILES}
  )

set(source_list
  ${LOADER_SOURCE_FILES}
  )

set(moc_header_list
  LoaderPDBFile.hpp
)

poca_add_plugin(${targetname}
    SOURCES ${header_list} ${source_list} ${moc_header_list}
    USE_CUDA
    LINK_LIBRARIES
        PoCA::Core
        PoCA::Geometry
        PoCA::OpenGL
        PoCA::Gemmi
        Qt5::Core
        Qt5::Widgets
)

install(TARGETS ${targetname} LIBRARY DESTINATION bin/plugins RUNTIME DESTINATION bin/plugins)
