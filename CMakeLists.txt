cmake_minimum_required(VERSION 3.15)
project(GPUDestruction VERSION 1.0)

# C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# GLAD
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/vendor/glad/include)
# GLFW
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/vendor/glfw/include)
# GLM
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/vendor/glm)

# where .lib
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/vendor/glfw/lib-vc2022) 


# create executable
add_executable(GPUDestruction
    main.cpp
    vendor/glad/src/glad.c
)

# link library
target_link_libraries(GPUDestruction
    glfw3.lib       # GLFW
    opengl32.lib    # Windows default OpenGL
)

#  copy glfw3.dll to exe
add_custom_command(TARGET GPUDestruction POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/vendor/glfw/lib-vc2022/glfw3.dll"
    $<TARGET_FILE_DIR:GPUDestruction>
    COMMENT "Copying glfw3.dll to output directory"
)