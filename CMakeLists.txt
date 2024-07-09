cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

project(Kans)
set(CMAKE_CXX_STANDARD 14)
set(OpenCV_DIR /Users/quannguyennam/Documents/Projects/KANS/opencv/build)


find_package(Torch REQUIRED)
find_package(OpenCV 4 REQUIRED )


# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Add matplotlibcpp include directory
include_directories(${CMAKE_SOURCE_DIR}/matplotlib-cpp)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${TORCH_LIBRARIES})

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET ${PROJECT_NAME}
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:${PROJECT_NAME}>)
endif (MSVC)
