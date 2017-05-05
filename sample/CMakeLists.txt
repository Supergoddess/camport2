cmake_minimum_required(VERSION 2.8)

option(ENABLE_PCL "Enable PCL" OFF)

set(ALL_SAMPLES
    SimpleView_Callback
    SimpleView_FetchFrame
    SimpleView_Registration
    SimpleView_TriggerMode
    SimpleView_MultiDevice
    DumpAllFeatures
    )

set(SAMPLES_USED_PCL
    DepthToWorld
    SimpleView_Point3D
    )

set(TARGET_LIB tycam)
set(INCLUDE_PATH ../include)

# ========================================
# === common, pcl is too slow, so build a
# === lib to speed up
# ========================================
set(COMMON_SOURCES
    common/PointCloudViewer.cpp
    )

add_library(sample_common STATIC ${COMMON_SOURCES})
include_directories(${INCLUDE_PATH})

# ========================================
# === OpenCV
# ========================================
find_package(OpenCV REQUIRED)
if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV library not found")
else()
    include_directories(${OpenCV_INCLUDE_DIRS})
    include_directories(${OpenCV2_INCLUDE_DIRS})
    # if(CMAKE_BUILD_TYPE STREQUAL Debug)
    # link_directories(${OpenCV_LIB_DIR_DBG})
    # else()
    # link_directories(${OpenCV_LIB_DIR_OPT})
    # endif()
    link_directories(${OpenCV_LIB_DIR})
endif()

# ========================================
# === PCL
# ========================================
# find_package(PCL 1.3 REQUIRED COMPONENTS common io)
if (ENABLE_PCL)
find_package(PCL)
if (NOT PCL_FOUND)
    message(WARNING "PCL library not found, some samples will not be compiled")
else()
    add_definitions(-DHAVE_PCL)
    include_directories(${PCL_INCLUDE_DIRS})
    link_directories(${PCL_LIBRARY_DIRS})
    # target_link_libraries(MY_TARGET_NAME ${PCL_LIBRARIES})
    set(ALL_SAMPLES ${ALL_SAMPLES} ${SAMPLES_USED_PCL})
endif()
endif()

foreach(sample ${ALL_SAMPLES})
    file(GLOB sources ${sample}/*.cpp)
    add_executable(${sample} ${sources})
    add_dependencies(${sample} sample_common ${TARGET_LIB})
    target_link_libraries(${sample} sample_common ${TARGET_LIB} ${OpenCV_LIBS} ${PCL_LIBRARIES})
    set_target_properties(${sample} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/)
    # install(TARGETS ${sample} RUNTIME DESTINATION samples/)
endforeach()

