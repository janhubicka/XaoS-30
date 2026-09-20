cmake_minimum_required(VERSION 3.21)
project(XaoSModern VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# A plain single-config build is an application build, so make it fast by default.
# Debug/sanitizer builds remain explicit opt-ins.
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(XAOS_BUILD_GUI "Build the Qt 6 desktop/mobile application" ON)
option(XAOS_BUILD_TOOLS "Build benchmark/headless helper executables" OFF)
option(XAOS_NATIVE "Optimize for this computer (binaries are not portable)" OFF)
option(XAOS_FAST_MATH "Enable aggressive floating-point optimizations in release builds" ON)
option(XAOS_LTO "Enable link-time optimization in release builds" ON)
option(XAOS_SANITIZE "Enable address/undefined sanitizers" OFF)
option(XAOS_TSAN "Enable thread sanitizer instead" OFF)
option(BUILD_TESTING "Build the regression-test executable" OFF)
if(XAOS_SANITIZE AND XAOS_TSAN)
  message(FATAL_ERROR "Use separate builds for ASan/UBSan and TSan")
endif()

set(XAOS_IPO_SUPPORTED OFF)
if(XAOS_LTO AND NOT XAOS_SANITIZE AND NOT XAOS_TSAN)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT XAOS_IPO_SUPPORTED OUTPUT XAOS_IPO_ERROR LANGUAGES CXX)
  if(NOT XAOS_IPO_SUPPORTED)
    message(WARNING "LTO requested but unavailable: ${XAOS_IPO_ERROR}")
  endif()
endif()

function(xaos_optimize_release_target target)
  if(XAOS_SANITIZE OR XAOS_TSAN)
    return()
  endif()

  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # CMake's Release defaults already contain -O3 on the common toolchains, but
    # spell it out so custom toolchain files do not silently reduce optimization.
    target_compile_options(${target} PRIVATE
      "$<$<CONFIG:Release>:-O3>"
      "$<$<CONFIG:RelWithDebInfo>:-O3>")
    if(XAOS_FAST_MATH)
      target_compile_options(${target} PRIVATE
        "$<$<CONFIG:Release>:-ffast-math>"
        "$<$<CONFIG:Release>:-fno-finite-math-only>"
        "$<$<CONFIG:Release>:-ffp-contract=fast>"
        "$<$<CONFIG:RelWithDebInfo>:-ffast-math>"
        "$<$<CONFIG:RelWithDebInfo>:-fno-finite-math-only>"
        "$<$<CONFIG:RelWithDebInfo>:-ffp-contract=fast>")
    endif()
  elseif(MSVC)
    target_compile_options(${target} PRIVATE
      "$<$<CONFIG:Release>:/O2>"
      "$<$<CONFIG:RelWithDebInfo>:/O2>")
    if(XAOS_FAST_MATH)
      target_compile_options(${target} PRIVATE
        "$<$<CONFIG:Release>:/fp:fast>"
        "$<$<CONFIG:RelWithDebInfo>:/fp:fast>")
    endif()
  endif()

  if(XAOS_IPO_SUPPORTED)
    # CMake emits the compiler/linker-specific LTO flags (-flto for GCC/Clang,
    # /GL+/LTCG for MSVC) only for optimized application configurations.
    set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE)
  endif()
endfunction()

find_path(GMP_INCLUDE_DIR gmp.h REQUIRED)
find_library(GMP_LIBRARY NAMES gmp REQUIRED)
find_package(Threads REQUIRED)

add_library(xaos_core
  src/autopilot.cpp src/axis.cpp src/executor.cpp src/fast_mandel.cpp src/formulae.cpp
  src/palette.cpp src/renderer.cpp src/simd.cpp)
target_include_directories(xaos_core PUBLIC include ${GMP_INCLUDE_DIR})
target_link_libraries(xaos_core PUBLIC ${GMP_LIBRARY} Threads::Threads)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # Strict builds keep scalar/SIMD contraction identical. Fast release builds
  # deliberately allow contraction/reassociation for maximum throughput.
  if(NOT XAOS_FAST_MATH OR XAOS_SANITIZE OR XAOS_TSAN)
    target_compile_options(xaos_core PUBLIC -ffp-contract=off)
  endif()
  target_compile_options(xaos_core PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
  if(XAOS_NATIVE)
    target_compile_options(xaos_core PRIVATE -march=native)
  endif()
  if(XAOS_SANITIZE)
    target_compile_options(xaos_core PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(xaos_core PUBLIC -fsanitize=address,undefined)
  elseif(XAOS_TSAN)
    target_compile_options(xaos_core PUBLIC -fsanitize=thread -fno-omit-frame-pointer)
    target_link_options(xaos_core PUBLIC -fsanitize=thread)
  endif()
elseif(MSVC)
  target_compile_options(xaos_core PUBLIC /fp:strict)
endif()
xaos_optimize_release_target(xaos_core)

# GCC's slim LTO objects force every consumer of xaos_core through the LTO
# plugin. Keep a native copy beside the LTO IR so the deliberately non-LTO
# regression executable can link quickly and, on MinGW, without duplicate
# COMDAT/template definitions. The shipped app still links the same objects
# through LTO.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND XAOS_IPO_SUPPORTED)
  target_compile_options(xaos_core PRIVATE
    "$<$<CONFIG:Release>:-ffat-lto-objects>"
    "$<$<CONFIG:RelWithDebInfo>:-ffat-lto-objects>")
endif()

# The default palette is a compatibility artifact: its 65,534 entries must be
# bit-identical to the original XaoS floating-point mksmooth() loop. Fast-math
# reassociation changes a few entries, so keep this tiny presentation-only TU
# strict while the expensive fractal kernels remain aggressively optimized.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set_source_files_properties(src/palette.cpp PROPERTIES
    COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")
  # Scalar and explicit AVX2/NEON implementations intentionally use identical
  # operation order so resume/reuse is backend-independent. Compiler-level
  # reassociation breaks that contract on ARM64; the intrinsics themselves
  # already express the optimized vector operations explicitly.
  set_source_files_properties(src/simd.cpp PROPERTIES
    COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")
elseif(MSVC)
  set_source_files_properties(src/palette.cpp PROPERTIES
    COMPILE_OPTIONS "/fp:precise")
  set_source_files_properties(src/simd.cpp PROPERTIES
    COMPILE_OPTIONS "/fp:precise")
endif()

# Benchmark/test executables are host tools. Android only builds the Qt app.
if(NOT ANDROID)
  if(XAOS_BUILD_TOOLS)
    add_executable(xaos-bench app/bench.cpp)
    target_link_libraries(xaos-bench PRIVATE xaos_core)
    xaos_optimize_release_target(xaos-bench)
    add_executable(xaos-kernel-bench app/kernel_bench.cpp)
    target_include_directories(xaos-kernel-bench PRIVATE ${GMP_INCLUDE_DIR})
    target_link_libraries(xaos-kernel-bench PRIVATE ${GMP_LIBRARY})
    xaos_optimize_release_target(xaos-kernel-bench)
    add_executable(xaos-formula-kernel-bench app/formula_kernel_bench.cpp)
    target_link_libraries(xaos-formula-kernel-bench PRIVATE xaos_core)
    xaos_optimize_release_target(xaos-formula-kernel-bench)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND
       (NOT XAOS_FAST_MATH OR XAOS_SANITIZE OR XAOS_TSAN))
      target_compile_options(xaos-kernel-bench PRIVATE -ffp-contract=off)
      target_compile_options(xaos-formula-kernel-bench PRIVATE -ffp-contract=off)
    endif()
  endif()
  include(CTest)
  if(BUILD_TESTING)
    add_executable(xaos-tests tests/core_tests.cpp)
    target_link_libraries(xaos-tests PRIVATE xaos_core)
    # The test driver is a large template-heavy translation unit. Optimizing the
    # driver itself buys no application performance and makes edit/build cycles
    # much slower; it still exercises the separately optimized xaos_core library.
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
      target_compile_options(xaos-tests PRIVATE -O0)
      if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND XAOS_IPO_SUPPORTED)
        target_link_options(xaos-tests PRIVATE -fno-lto)
      endif()
    elseif(MSVC)
      target_compile_options(xaos-tests PRIVATE /Od)
    endif()
    set_property(TARGET xaos-tests PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
    add_test(NAME core COMMAND xaos-tests)
    set_tests_properties(core PROPERTIES TIMEOUT 120)
  endif()
endif()

if(XAOS_BUILD_GUI)
  find_package(Qt6 6.2 REQUIRED COMPONENTS Widgets)
  if(ANDROID)
    find_package(Qt6 6.2 REQUIRED COMPONENTS Sensors)
  endif()
  # qt_add_executable is required for Qt's Android deployment metadata/APK target,
  # and is equally valid for the existing desktop platforms.
  qt_add_executable(xaos-modern MANUAL_FINALIZATION app/qt_main.cpp app/qt_executor.hpp)
  target_link_libraries(xaos-modern PRIVATE xaos_core Qt6::Widgets)
  xaos_optimize_release_target(xaos-modern)
  if(ANDROID)
    target_link_libraries(xaos-modern PRIVATE Qt6::Sensors)
  endif()
  set_target_properties(xaos-modern PROPERTIES
    WIN32_EXECUTABLE TRUE
    MACOSX_BUNDLE TRUE)
  if(ANDROID)
    set_target_properties(xaos-modern PROPERTIES
      QT_ANDROID_PACKAGE_NAME "org.xaos.xaos30"
      QT_ANDROID_MIN_SDK_VERSION 28
      QT_ANDROID_TARGET_SDK_VERSION 36
      QT_ANDROID_VERSION_NAME "${PROJECT_VERSION}"
      QT_ANDROID_VERSION_CODE 1)
  endif()
  qt_finalize_target(xaos-modern)
  if(NOT ANDROID)
    install(TARGETS xaos-modern RUNTIME DESTINATION bin BUNDLE DESTINATION .)
  endif()
endif()

if(NOT ANDROID AND XAOS_BUILD_TOOLS)
  install(TARGETS xaos-bench RUNTIME DESTINATION bin)
endif()
