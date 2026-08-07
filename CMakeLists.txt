cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(Resonantia VERSION 0.1.0 LANGUAGES CXX CUDA)

# ============================================================================
# 编译标准: C++20 + CUDA20
# ============================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================================
# CUDA 架构: SM87 (Jetson AGX Orin)
# ============================================================================
set(VOXMUTATIO_CUDA_ARCHITECTURES "87" CACHE STRING
    "CUDA architectures for Resonantia (87 is Jetson AGX Orin)")
set(CMAKE_CUDA_ARCHITECTURES "${VOXMUTATIO_CUDA_ARCHITECTURES}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_87,code=sm_87 --expt-relaxed-constexpr -O3 --use_fast_math -Xptxas -O3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DDEBUG")

# ============================================================================
# 依赖
# ============================================================================
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# Find cuFFT and cuSOLVER
find_library(CUFFT_LIBRARY cufft PATHS ${CUDAToolkit_LIBRARY_DIR})
find_library(CUSOLVER_LIBRARY cusolver PATHS ${CUDAToolkit_LIBRARY_DIR})

# ============================================================================
# 包含目录
# ============================================================================
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

# ============================================================================
# 核心静态库 (voxmutatio — voice conversion engine)
# ============================================================================
file(GLOB_RECURSE VOXMUTATIO_SOURCES
    "src/core/*.cpp"
    "src/core/*.cu"
    "src/autograd/*.cpp"
    "src/autograd/*.cu"
    "src/separation/*.cpp"
    "src/separation/*.cu"
    "src/content/*.cpp"
    "src/content/*.cu"
    "src/f0/*.cpp"
    "src/f0/*.cu"
    "src/index/*.cpp"
    "src/index/*.cu"
    "src/synthesizer/*.cpp"
    "src/synthesizer/*.cu"
    "src/training/*.cpp"
    "src/training/*.cu"
    "src/pipeline/*.cpp"
    "src/io/*.cpp"
    "src/webui/*.cpp"
)

add_library(voxmutatio STATIC ${VOXMUTATIO_SOURCES})
target_link_libraries(voxmutatio PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::cufft
    Threads::Threads
)
target_include_directories(voxmutatio PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ============================================================================
# 工具
# ============================================================================
add_executable(vc_convert tools/vc_convert.cpp)
target_include_directories(vc_convert PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(vc_convert PRIVATE voxmutatio CUDA::cudart CUDA::cublas)
set_target_properties(vc_convert PROPERTIES
    BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}"
    INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}"
    LINK_FLAGS "-Wl,--disable-new-dtags"
)

add_executable(vc_train tools/vc_train.cpp)
target_include_directories(vc_train PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(vc_train PRIVATE voxmutatio CUDA::cudart CUDA::cublas)
set_target_properties(vc_train PROPERTIES
    BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}"
    INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}"
    LINK_FLAGS "-Wl,--disable-new-dtags"
)

add_executable(vc_preprocess tools/vc_preprocess.cpp)
target_include_directories(vc_preprocess PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(vc_preprocess PRIVATE voxmutatio CUDA::cudart CUDA::cublas)
set_target_properties(vc_preprocess PROPERTIES
    BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}"
    INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}"
    LINK_FLAGS "-Wl,--disable-new-dtags"
)

add_executable(vc_serve tools/vc_serve.cpp)
target_include_directories(vc_serve PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(vc_serve PRIVATE voxmutatio CUDA::cudart CUDA::cublas)
set_target_properties(vc_serve PROPERTIES
    BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}"
    INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}"
    LINK_FLAGS "-Wl,--disable-new-dtags"
)

add_executable(vc_batch tools/vc_batch.cpp)
target_include_directories(vc_batch PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(vc_batch PRIVATE voxmutatio CUDA::cudart CUDA::cublas)
set_target_properties(vc_batch PROPERTIES
    BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}"
    INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}"
    LINK_FLAGS "-Wl,--disable-new-dtags"
)

add_executable(vc_probe tools/vc_probe.cpp)
target_include_directories(vc_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(vc_probe PRIVATE voxmutatio CUDA::cudart CUDA::cublas)
set_target_properties(vc_probe PROPERTIES
    BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}"
    INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}"
    LINK_FLAGS "-Wl,--disable-new-dtags"
)

    add_executable(test_model_load tools/test_model_load.cpp)
    target_include_directories(test_model_load PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(test_model_load PRIVATE voxmutatio CUDA::cudart)
    set_target_properties(test_model_load PROPERTIES
        BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}"
        INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}"
        LINK_FLAGS "-Wl,--disable-new-dtags"
    )

    add_executable(build_index tools/build_index.cpp)
    target_include_directories(build_index PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(build_index PRIVATE voxmutatio CUDA::cudart CUDA::cublas)
    set_target_properties(build_index PROPERTIES
        BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}"
        INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}"
        LINK_FLAGS "-Wl,--disable-new-dtags"
    )

# ============================================================================
# 测试
# ============================================================================
option(VOXMUTATIO_BUILD_TESTS "Build test executables" ON)

if(VOXMUTATIO_BUILD_TESTS)
    enable_testing()

    # Fetch Catch2
    include(FetchContent)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.4.0
    )
    FetchContent_MakeAvailable(Catch2)

    # Test: Core
    add_executable(test_core tests/test_core.cpp)
    target_link_libraries(test_core PRIVATE voxmutatio CUDA::cudart CUDA::cublas Catch2::Catch2WithMain)
    target_include_directories(test_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME test_core COMMAND test_core)

    # Test: Audio I/O
    add_executable(test_audio_io tests/test_audio_io.cpp)
    target_link_libraries(test_audio_io PRIVATE voxmutatio CUDA::cudart CUDA::cublas Catch2::Catch2WithMain)
    target_include_directories(test_audio_io PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME test_audio_io COMMAND test_audio_io)

    # Test: F0
    add_executable(test_f0 tests/test_f0.cpp)
    target_link_libraries(test_f0 PRIVATE voxmutatio CUDA::cudart CUDA::cublas Catch2::Catch2WithMain)
    target_include_directories(test_f0 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME test_f0 COMMAND test_f0)

    # Test: Index
    add_executable(test_index tests/test_index.cpp)
    target_link_libraries(test_index PRIVATE voxmutatio CUDA::cudart CUDA::cublas Catch2::Catch2WithMain)
    target_include_directories(test_index PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME test_index COMMAND test_index)

    # Test: Pipeline (integration)
    add_executable(test_pipeline tests/test_pipeline.cpp)
    target_link_libraries(test_pipeline PRIVATE voxmutatio CUDA::cudart CUDA::cublas Catch2::Catch2WithMain)
    target_include_directories(test_pipeline PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME test_pipeline COMMAND test_pipeline)

    # Test: HuBERT Inference (end-to-end with real model)
    add_executable(test_hubert_inference tests/test_hubert_inference.cpp)
    target_link_libraries(test_hubert_inference PRIVATE voxmutatio CUDA::cudart CUDA::cublas Catch2::Catch2WithMain)
    target_include_directories(test_hubert_inference PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME test_hubert_inference COMMAND test_hubert_inference WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

    # Test: RMVPE F0 alignment (end-to-end with real model)
    add_executable(test_rmvpe_alignment tests/test_rmvpe_alignment.cpp)
    target_link_libraries(test_rmvpe_alignment PRIVATE voxmutatio CUDA::cudart CUDA::cublas Catch2::Catch2WithMain)
    target_include_directories(test_rmvpe_alignment PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME test_rmvpe_alignment COMMAND test_rmvpe_alignment WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

    # Test: VITS synthesizer alignment (end-to-end with real model)
    add_executable(test_vits_alignment tests/test_vits_alignment.cpp)
    target_link_libraries(test_vits_alignment PRIVATE voxmutatio CUDA::cudart CUDA::cublas Catch2::Catch2WithMain)
    target_include_directories(test_vits_alignment PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME test_vits_alignment COMMAND test_vits_alignment WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

    add_executable(test_vits_gen_debug tests/test_vits_gen_debug.cpp)
    target_link_libraries(test_vits_gen_debug PRIVATE voxmutatio CUDA::cudart CUDA::cublas Catch2::Catch2WithMain)
    target_include_directories(test_vits_gen_debug PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    # Test: End-to-end voice conversion (highest priority integration test)
    add_executable(test_e2e_conversion tests/test_e2e_conversion.cpp)
    target_link_libraries(test_e2e_conversion PRIVATE voxmutatio CUDA::cudart CUDA::cublas Catch2::Catch2WithMain)
    target_include_directories(test_e2e_conversion PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME test_e2e_conversion COMMAND test_e2e_conversion WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

    # Test: Feature index retrieval (RVC weighted k-NN)
    add_executable(test_index_retrieval tests/test_index_retrieval.cpp)
    target_link_libraries(test_index_retrieval PRIVATE voxmutatio CUDA::cudart CUDA::cublas Catch2::Catch2WithMain)
    target_include_directories(test_index_retrieval PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME test_index_retrieval COMMAND test_index_retrieval WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

    # Test: Autograd gradient check (training foundation, spec 002)
    add_executable(test_autograd tests/test_autograd.cpp)
    target_link_libraries(test_autograd PRIVATE voxmutatio CUDA::cudart CUDA::cublas Catch2::Catch2WithMain)
    target_include_directories(test_autograd PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME test_autograd COMMAND test_autograd WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

    # Test: Differentiable mel loss (training signal, spec 002)
    add_executable(test_mel_loss tests/test_mel_loss.cpp)
    target_link_libraries(test_mel_loss PRIVATE voxmutatio CUDA::cudart CUDA::cublas Catch2::Catch2WithMain)
    target_include_directories(test_mel_loss PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME test_mel_loss COMMAND test_mel_loss WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

    # Test: Trainable autograd decoder vs inference decoder (spec 002)
    add_executable(test_generator_trainer tests/test_generator_trainer.cpp)
    target_link_libraries(test_generator_trainer PRIVATE voxmutatio CUDA::cudart CUDA::cublas Catch2::Catch2WithMain)
    target_include_directories(test_generator_trainer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME test_generator_trainer COMMAND test_generator_trainer WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endif()
