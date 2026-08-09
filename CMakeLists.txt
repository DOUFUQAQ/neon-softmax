cmake_minimum_required(VERSION 3.16)
project(kunpeng_softmax C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# 打开告警：kernel 代码最怕"悄悄的类型提升"
if(MSVC)
    add_compile_options(/W3)
else()
    add_compile_options(-Wall -Wextra)
endif()

# 公共头文件目录
include_directories(${CMAKE_SOURCE_DIR}/include)

# 源文件
set(SOFTMAX_SOURCES
    src/softmax_scalar_f32.c
    src/softmax_neon_f32.c
    src/softmax_neon_f16.c
    src/exp_poly.c
)

# 静态库
add_library(softmax STATIC ${SOFTMAX_SOURCES})

# 检测是否为 AArch64 平台
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    message(STATUS "AArch64 detected -> NEON path enabled (USE_AARCH64_NEON)")
    target_compile_definitions(softmax PRIVATE USE_AARCH64_NEON)
    target_link_libraries(softmax PUBLIC m)
else()
    message(STATUS "Non-AArch64 platform (${CMAKE_SYSTEM_PROCESSOR}) -> NEON falls back to scalar")
    target_link_libraries(softmax PUBLIC m)
endif()

# 测试程序
add_executable(test_softmax tests/test_softmax.c)
target_link_libraries(test_softmax softmax m)

# 基准测试程序
add_executable(bench_softmax bench/bench_softmax.c)
target_link_libraries(bench_softmax softmax m)

# ctest 集成：断言失败时 CI 会真的红掉
enable_testing()
add_test(NAME softmax_unit COMMAND test_softmax)
