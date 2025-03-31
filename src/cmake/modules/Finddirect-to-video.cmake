# - Try to find the direct-to-video library
# Once done this will define
#
#  direct-to-video_FOUND - system has direct-to-video
#  direct-to-video_INCLUDE_DIR - direct-to-video include directory
#  direct-to-video_LIB - direct-to-video library directory
#  direct-to-video_LIBRARIES - direct-to-video libraries to link

if(direct-to-video_FOUND)
    return()
endif()

# We prioritize libraries installed in /usr/local with the prefix .../direct-to-video-*, 
# so we make a list of them here
file(GLOB lib_glob "/usr/local/lib/direct-to-video-*")
file(GLOB inc_glob "/usr/local/include/direct-to-video-*")

# Find the library with the name "direct-to-video" on the system. Store the final path
# in the variable direct-to-video_LIB
find_library(direct-to-video_LIB_DEBUG 
    # The library is named "direct-to-video", but can have various library forms, like
    # libdirect-to-video.a, libdirect-to-video.so, libdirect-to-video.so.1.x, etc. This should
    # search for any of these.
    NAMES direct-to-video
    # Provide a list of places to look based on prior knowledge about the system.
    # We want the user to override /usr/local with environment variables, so
    # this is included here.
    HINTS
        ${direct-to-video_DIR}
        ${direct-to-video_DIR}
        $ENV{direct-to-video_DIR}
        $ENV{direct-to-video_DIR}
        ENV direct-to-video_DIR
    # Provide a list of places to look as defaults. /usr/local shows up because
    # that's the default install location for most libs. The globbed paths also
    # are placed here as well.
    PATHS
        /usr
        /usr/local
        /usr/local/lib
        ${lib_glob}
    # Constrain the end of the full path to the detected library, not including
    # the name of library itself.
    PATH_SUFFIXES 
        lib
)

# Find the library with the name "direct-to-video" on the system. Store the final path
# in the variable direct-to-video_LIB
find_library(direct-to-video_LIB_RELEASE 
    # The library is named "direct-to-video", but can have various library forms, like
    # libdirect-to-video.a, libdirect-to-video.so, libdirect-to-video.so.1.x, etc. This should
    # search for any of these.
    NAMES direct-to-video
    # Provide a list of places to look based on prior knowledge about the system.
    # We want the user to override /usr/local with environment variables, so
    # this is included here.
    HINTS
        ${direct-to-video_DIR}
        ${direct-to-video_DIR}
        $ENV{direct-to-video_DIR}
        $ENV{direct-to-video_DIR}
        ENV direct-to-video_DIR
    # Provide a list of places to look as defaults. /usr/local shows up because
    # that's the default install location for most libs. The globbed paths also
    # are placed here as well.
    PATHS
        /usr
        /usr/local
        /usr/local/lib
        ${lib_glob}
    # Constrain the end of the full path to the detected library, not including
    # the name of library itself.
    PATH_SUFFIXES 
        lib
)

# Find the path to the file "source_file.hpp" on the system. Store the final
# path in the variables direct-to-video_INCLUDE_DIR. The HINTS, PATHS, and
# PATH_SUFFIXES, arguments have the same meaning as in find_library().
find_path(direct-to-video_INCLUDE_DIR source_file.hpp
    HINTS
        ${direct-to-video_DIR}
        ${direct-to-video_DIR}
        $ENV{direct-to-video_DIR}
        $ENV{direct-to-video_DIR}
        ENV direct-to-video_DIR
    PATHS
        /usr
        /usr/local
        /usr/local/include
        ${inc_glob}
    PATH_SUFFIXES 
        include
)


# Check that both the paths to the include and library directory were found.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(direct-to-video
    "\ndirect-to-video not found --- You can download it using:\n\tgit clone 
    https://github.com/mmorse1217/cmake-project-template\n and setting the 
    direct-to-video_DIR environment variable accordingly"
    direct-to-video_LIB_DEBUG direct-to-video_LIB_RELEASE direct-to-video_INCLUDE_DIR)

# These variables don't show up in the GUI version of CMake. Not required but
# people seem to do this...
mark_as_advanced(direct-to-video_INCLUDE_DIR direct-to-video_LIB_DEBUG direct-to-video_LIB_RELEASE)

# Finish defining the variables specified above. Variables names here follow
# CMake convention.
set(direct-to-video_INCLUDE_DIRS ${direct-to-video_INCLUDE_DIR})
set(direct-to-video_LIBRARY_DEBUG ${direct-to-video_LIB_DEBUG})
set(direct-to-video_LIBRARY_RELEASE ${direct-to-video_LIB_RELEASE})

# If the above CMake code was successful and we found the library, and there is
# no target defined, lets make one.
if(direct-to-video_FOUND AND NOT TARGET direct-to-video::direct-to-video)
    add_library(direct-to-video::direct-to-video UNKNOWN IMPORTED)
    # Set location of interface include directory, i.e., the directory
    # containing the header files for the installed library
    set_target_properties(direct-to-video::direct-to-video PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${direct-to-video_INCLUDE_DIRS}"
        )

    # Set location of the installed library
    set_target_properties(direct-to-video::direct-to-video PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
        IMPORTED_LOCATION "${direct-to-video_LIBRARY_DEBUG}"
        )
		
	# Set location of the installed library
    set_target_properties(direct-to-video::direct-to-video PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
        IMPORTED_LOCATION "${direct-to-video_LIBRARY_RELEASE}"
        )
endif()