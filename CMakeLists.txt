# CMake 最低版本要求
cmake_minimum_required(VERSION 3.15)

# 專案名稱
project(GPUDestruction VERSION 1.0)

# 設定 C++ 標準為 C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ----------------------------------------------------
# 找到我們的函式庫
# ----------------------------------------------------

# GLAD
# 將 glad 的 include 目錄加入
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/vendor/glad/include)

# GLFW
# 將 glfw 的 include 目錄加入
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/vendor/glfw/include)
# 告訴 CMake 去哪裡找 .lib 檔案
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/vendor/glfw/lib-vc2022) 
# (如果你的 VS 是 2019，請改成 lib-vc2019)

# ----------------------------------------------------
# 建立可執行檔
# ----------------------------------------------------

# 我們的可執行檔叫做 "GPUDestruction"
# 它是由 main.cpp 和 glad.c 編譯而來
add_executable(GPUDestruction
    main.cpp
    vendor/glad/src/glad.c  # <-- 必須把 glad.c 也一起編譯
)

# ----------------------------------------------------
# 連結函式庫
# ----------------------------------------------------

# 告訴連結器 (Linker) 我們需要哪些函式庫
target_link_libraries(GPUDestruction
    glfw3.lib       # GLFW
    opengl32.lib    # Windows 預設的 OpenGL 函式庫
)

# ----------------------------------------------------
# (Pro-Tip) 自動複製 DLL
# ----------------------------------------------------
# 這會在每次建置成功後，自動把 glfw3.dll 複製到你的執行檔旁邊
add_custom_command(TARGET GPUDestruction POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/vendor/glfw/lib-vc2022/glfw3.dll"
    $<TARGET_FILE_DIR:GPUDestruction>
    COMMENT "Copying glfw3.dll to output directory"
)