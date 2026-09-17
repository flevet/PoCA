set(targetname poca_core)

set(MISC_HEADER_FILES
  Cuda/CoreMisc.h
  Cuda/BasicOperationsImage.h
  Cuda/ConnectedComponents.h
  )

set(DESIGN_PATTERNS_HEADER_FILES
  DesignPatterns/ChangeManagerSingleton.hpp
  DesignPatterns/MacroRecorderSingleton.hpp
  DesignPatterns/MediatorWObjectFWidget.hpp
  DesignPatterns/Observer.hpp
  DesignPatterns/Subject.hpp
  )

set(DESIGN_PATTERNS_SOURCE_FILES
  DesignPatterns/ChangeManagerSingleton.cpp
  DesignPatterns/MacroRecorderSingleton.cpp
  DesignPatterns/MediatorWObjectFWidget.cpp
  DesignPatterns/Subject.cpp
)

set(FIT_HEADER_FILES
  Fit/lmcurve.h
  Fit/lmmin.h
  Fit/lmstruct.h
  )

set(FIT_SOURCE_FILES
  Fit/lmcurve.c
  Fit/lmmin.c
)

set(GENERAL_HEADER_FILES
  General/ArrayStatistics.hpp
  General/BasicComponent.hpp
  General/BasicComponentList.hpp
  General/Command.hpp
  General/CommandableObject.hpp
  General/EquationFit.hpp
  General/Engine.hpp
  General/Histogram.hpp
  General/Image.hpp
  General/ImagesList.hpp
  General/json.hpp
  General/JsonCommandContext.hpp
  General/Misc.h
  General/MyArray.hpp
  General/MyData.hpp
  General/Palette.hpp
  General/PerformanceProfiler.hpp
  General/PluginList.hpp
  General/PythonInterpreter.hpp
  General/Region3D.hpp
  General/ranker.h
  General/Scatterplot.hpp
  General/TestRegistry.hpp
  General/TiffImageIO.hpp
  General/Vec2.hpp
  General/Vec3.hpp
  General/Vec4.hpp
  General/Vec6.hpp
  General/stb_rect_pack.h
  )

set(GENERAL_SOURCE_FILES
  General/ArrayStatistics.cpp
  General/BasicComponent.cpp
  General/CommandableObject.cpp
  General/EquationFit.cpp
  General/Engine.cpp
  General/ImagesList.cpp
  General/Misc.cpp
  General/MyData.cpp
  General/Palette.cpp
  General/PerformanceProfiler.cpp
  General/PluginList.cpp
  General/PythonInterpreter.cpp
  General/Scatterplot.cpp
  General/TestRegistry.cpp
)

set(INTERFACES_HEADER_FILES
  Interfaces/BasicComponentInterface.hpp
  Interfaces/CommandableObjectInterface.hpp
  Interfaces/HistogramInterface.hpp
  Interfaces/ImageInterface.hpp
  Interfaces/MyObjectInterface.hpp
  Interfaces/PaletteInterface.hpp
  Interfaces/ROIInterface.hpp
  Interfaces/SubjectInterface.hpp
  Interfaces/ScatterplotInterface.hpp
  )

set(MY_OBJECT_HEADER_FILES
  Objects/MyObject.hpp
  Objects/MyMultipleObject.hpp
  Objects/MyObjectDisplayCommand.hpp
  Objects/ObjectCommandContext.hpp
  )

set(MY_OBJECT_SOURCE_FILES
  Objects/MyObject.cpp
  Objects/MyMultipleObject.cpp
  Objects/MyObjectDisplayCommand.cpp
  )

set(header_list
  ${MISC_HEADER_FILES}
  ${DESIGN_PATTERNS_HEADER_FILES}
  ${FIT_HEADER_FILES}
  ${GENERAL_HEADER_FILES}
  ${INTERFACES_HEADER_FILES}
  ${MY_OBJECT_HEADER_FILES}
  )

set(source_list
  ${DESIGN_PATTERNS_SOURCE_FILES}
  ${FIT_SOURCE_FILES}
  ${GENERAL_SOURCE_FILES}
  ${MY_OBJECT_SOURCE_FILES}
  )

if(POCA_HAS_CUDA)
  # Host-side CUDA regression tests: use the normal C++ compiler.
  list(APPEND source_list Cuda/FaceConnectedComponentTests.cpp)
endif()

set(cuda_list
  Cuda/CoreMisc.cu
  Cuda/BasicOperationsImage.cu
  Cuda/ConnectedComponents.cu
  Cuda/FaceConnectedComponents.cu
  Cuda/FillHoles.cu
  )

poca_add_static_library(${targetname}
    ALIAS PoCA::Core
    SOURCES ${header_list} ${source_list}
    USE_CUDA
    CUDA_SOURCES ${cuda_list}
    PUBLIC_INCLUDE_DIRECTORIES
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/poca_opengl
    PUBLIC_LINK_LIBRARIES
        Qt5::Core
        Qt5::Widgets
        PoCA::GLM
        TinyTIFF::TinyTIFF
        PoCA::OpenGLDeps
    PRIVATE_LINK_LIBRARIES
        PoCA::OpenMP
)
