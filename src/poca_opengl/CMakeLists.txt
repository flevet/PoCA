set(targetname poca_opengl)

set(MISC_HEADER_FILES
  Cuda/Misc.h
  )

set(MISC_SOURCE_FILES
  Cuda/Misc.cpp
  )

set(OPENGL_HEADER_FILES
  OpenGL/BasicDisplayCommand.hpp
  OpenGL/Camera.hpp
  OpenGL/GLBuffer.hpp
  OpenGL/Helper.h
  OpenGL/LodUpdateManager.hpp
  OpenGL/MultiPrimitiveData.hpp
  OpenGL/SsaoShader.hpp
  OpenGL/Shader.hpp
  OpenGL/TextDisplayer.hpp
  OpenGL/RenderCommandContext.hpp
  )

set(OPENGL_SOURCE_FILES
  OpenGL/BasicDisplayCommand.cpp
  OpenGL/Camera.cpp
  OpenGL/Helper.cpp
  OpenGL/LodUpdateManager.cpp
  OpenGL/SsaoShader.cpp
  OpenGL/TextDisplayer.cpp
)

set(GENERAL_HEADER_FILES
  General/Roi.hpp
  )

set(GENERAL_SOURCE_FILES
  General/Roi.cpp
  )

set(PLOT_HEADER_FILES
  Plot/ScatterplotGL.hpp
   )

set(PLOT_SOURCE_FILES
  Plot/ScatterplotGL.cpp
)

set(INTERFACES_HEADER_FILES
  Interfaces/CameraInterface.hpp
  )

set(FONTSTASH_HEADER_FILES
  fontstash/fontstash.h
  fontstash/gl3corefontstash.h
  fontstash/stb_truetype.h
  )

set(header_list
  ${MISC_HEADER_FILES}
  ${OPENGL_HEADER_FILES}
  ${GENERAL_HEADER_FILES}
  ${PLOT_HEADER_FILES}
  ${INTERFACES_HEADER_FILES}
  ${FONTSTASH_HEADER_FILES}
  )

set(source_list
  ${MISC_SOURCE_FILES}
  ${OPENGL_SOURCE_FILES}
  ${GENERAL_SOURCE_FILES}
  ${PLOT_SOURCE_FILES}
  )

set(cuda_list
  Cuda/Misc.cu
  )

set(moc_header_list
  OpenGL/Camera.hpp
  Plot/ScatterplotGL.hpp
)

poca_add_static_library(${targetname}
    ALIAS PoCA::OpenGL
    SOURCES ${header_list} ${source_list} ${moc_header_list}
    USE_CUDA
    CUDA_RESOLVE_DEVICE_SYMBOLS
    CUDA_SOURCES ${cuda_list}
    PUBLIC_INCLUDE_DIRECTORIES
        ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC_LINK_LIBRARIES
        PoCA::Core
        PoCA::Geometry
        Qt5::Core
        Qt5::OpenGL
        PoCA::OpenGLDeps
    PRIVATE_LINK_LIBRARIES
        PoCA::OpenMP
)
