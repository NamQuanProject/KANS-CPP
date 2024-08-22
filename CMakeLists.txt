cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

project(Kans)

# Set paths for OpenCV, SFML, and matplotlib-cpp

set(CMAKE_CXX_STANDARD 14)
set(OpenCV_DIR /Users/quannguyennam/Documents/Projects/KANS/opencv/build)
set(SFML_ROOT "/Users/quannguyennam/Documents/Projects/KANS/SFML-2.6.1/build")
set(matplotlib_DIR /Users/quannguyennam/Documents/Projects/KANS/matplotlib-cpp)

# Find required packages
find_package(Torch REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(SFML 2.6 COMPONENTS system window graphics REQUIRED)

# Python dependencies
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS} ${SFML_INCLUDE_DIR})
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${Python3_NumPy_INCLUDE_DIRS})
include_directories(${matplotlib_DIR})

# Add the main executable
add_executable(${PROJECT_NAME} main.cpp)

# Link libraries
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${TORCH_LIBRARIES} sfml-system sfml-window sfml-graphics sfml-network sfml-audio ${Python3_LIBRARIES})

# Set C++ standard to C++17
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

# Handle MSVC specific settings
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET ${PROJECT_NAME}
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:${PROJECT_NAME}>)
endif (MSVC)
