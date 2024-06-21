cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

project(Kans)
set(CMAKE_CXX_STANDARD 14)


find_package(Torch REQUIRED)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Set linker flags to link libraries unconditionally
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(Kans
    "${TORCH_LIBRARIES}"
    
)

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)




if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET main
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:main>)
endif (MSVC)