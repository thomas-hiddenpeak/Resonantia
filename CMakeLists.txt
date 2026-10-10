cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

# CUDA architecture selection must be available before project() enables the
# CUDA language; otherwise a fresh configure asks nvcc to guess an architecture
# before the Jetson SM87 default has been applied.
set(VOXMUTATIO_CUDA_ARCHITECTURES "87" CACHE STRING
    "CUDA architectures for Resonantia (87 is the Jetson Orin baseline)")
set(CMAKE_CUDA_ARCHITECTURES "${VOXMUTATIO_CUDA_ARCHITECTURES}" CACHE STRING
    "CUDA architectures selected by VOXMUTATIO_CUDA_ARCHITECTURES" FORCE)
if(NOT DEFINED CMAKE_CUDA_COMPILER AND "$ENV{CUDACXX}" STREQUAL "")
    find_program(_VOXMUTATIO_NVCC NAMES nvcc HINTS /usr/local/cuda/bin)
    if(_VOXMUTATIO_NVCC)
        set(CMAKE_CUDA_COMPILER "${_VOXMUTATIO_NVCC}" CACHE FILEPATH
            "CUDA compiler selected for Resonantia")
    endif()
endif()

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
# CUDA 架构: SM87 (Jetson Orin baseline)
# ============================================================================
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
# 内部组件与兼容静态库
# ============================================================================
# Keep every runtime translation unit explicit.  The OBJECT targets make the
# dependency graph reviewable while voxmutatio remains the single public static
# archive consumed by existing callers.
function(resonantia_add_component target)
    add_library(${target} OBJECT ${ARGN})
    target_include_directories(${target} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    )
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--use_fast_math;-Xptxas=-O3>
    )
endfunction()

resonantia_add_component(resonantia_core_obj
    src/core/device.cpp
    src/core/types.cpp
)
target_link_libraries(resonantia_core_obj PRIVATE CUDA::cudart)

resonantia_add_component(resonantia_io_obj
    src/io/audio_io.cpp
    src/io/bounded_audio_io.cpp
    src/io/safetensors.cpp
)
target_link_libraries(resonantia_io_obj PRIVATE resonantia_core_obj)

resonantia_add_component(resonantia_cuda_ops_obj
    src/content/cuda/kernels.cu
)
target_link_libraries(resonantia_cuda_ops_obj PRIVATE
    resonantia_core_obj
    CUDA::cudart
    CUDA::cublas
    CUDA::cufft
)

resonantia_add_component(resonantia_content_obj
    src/content/hubert_encoder.cpp
    src/content/wavlm_encoder.cpp
)
target_link_libraries(resonantia_content_obj PRIVATE
    resonantia_cuda_ops_obj
    resonantia_io_obj
)

resonantia_add_component(resonantia_f0_obj
    src/f0/fcpe.cpp
    src/f0/rmvpe.cpp
    src/f0/rmvpe_ops.cu
)
target_link_libraries(resonantia_f0_obj PRIVATE
    resonantia_cuda_ops_obj
    resonantia_io_obj
    CUDA::cudart
)

resonantia_add_component(resonantia_index_obj
    src/index/cuda_flat_index.cpp
)

resonantia_add_component(resonantia_synthesizer_obj
    src/synthesizer/synthesizer.cpp
    src/synthesizer/vits_ops.cu
)
target_link_libraries(resonantia_synthesizer_obj PRIVATE
    resonantia_core_obj
    resonantia_cuda_ops_obj
    resonantia_io_obj
    CUDA::cudart
)

resonantia_add_component(resonantia_separation_obj
    src/separation/roformer.cu
    src/separation/sep_registry.cpp
    src/separation/separator.cu
    src/separation/stft.cu
    src/separation/vad.cu
)
target_link_libraries(resonantia_separation_obj PRIVATE
    resonantia_core_obj
    resonantia_io_obj
    CUDA::cudart
    CUDA::cublas
    CUDA::cufft
)

resonantia_add_component(resonantia_autograd_obj
    src/autograd/autograd.cu
)
target_link_libraries(resonantia_autograd_obj PRIVATE
    resonantia_core_obj
    CUDA::cudart
    CUDA::cublas
)

resonantia_add_component(resonantia_training_obj
    src/training/discriminator.cpp
    src/training/fine_tuning.cpp
    src/training/gan_trainer.cpp
    src/training/generator_trainer.cpp
    src/training/mel_loss.cpp
    src/training/posterior_encoder.cpp
)
target_link_libraries(resonantia_training_obj PRIVATE
    resonantia_core_obj
    resonantia_io_obj
    resonantia_separation_obj
    resonantia_autograd_obj
    CUDA::cudart
)

resonantia_add_component(resonantia_pipeline_obj
    src/pipeline/bounded_pipeline.cpp
    src/pipeline/pipeline.cpp
)
target_include_directories(resonantia_pipeline_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(resonantia_pipeline_obj PRIVATE
    resonantia_core_obj
    resonantia_io_obj
    resonantia_content_obj
    resonantia_f0_obj
    resonantia_index_obj
    resonantia_synthesizer_obj
)

resonantia_add_component(resonantia_webui_obj
    src/webui/server.cpp
)
target_link_libraries(resonantia_webui_obj PRIVATE Threads::Threads)

set(_resonantia_runtime_components
    resonantia_core_obj
    resonantia_io_obj
    resonantia_cuda_ops_obj
    resonantia_content_obj
    resonantia_f0_obj
    resonantia_index_obj
    resonantia_synthesizer_obj
    resonantia_separation_obj
    resonantia_autograd_obj
    resonantia_training_obj
    resonantia_pipeline_obj
    resonantia_webui_obj
)

set(_resonantia_runtime_objects)
foreach(component IN LISTS _resonantia_runtime_components)
    list(APPEND _resonantia_runtime_objects $<TARGET_OBJECTS:${component}>)
endforeach()

add_library(voxmutatio STATIC ${_resonantia_runtime_objects})
set_target_properties(voxmutatio PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(voxmutatio
    PUBLIC
        # Public headers expose CudaBuffer and therefore require both the
        # CUDA runtime headers and cudart at downstream compile/link time.
        CUDA::cudart
    PRIVATE
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
function(resonantia_add_app_support target source)
    add_library(${target} STATIC ${source})
    target_include_directories(${target} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    )
    # The Options headers include public voxmutatio types, so consumers need
    # the complete facade usage requirements as part of this interface.
    target_link_libraries(${target}
        PUBLIC voxmutatio
        PRIVATE CUDA::cublas
    )
endfunction()

resonantia_add_app_support(resonantia_vc_batch_app_support
    tools/app_support/vc_batch_app.cpp)
resonantia_add_app_support(resonantia_vc_convert_app_support
    tools/app_support/vc_convert_app.cpp)
resonantia_add_app_support(resonantia_vc_preprocess_app_support
    tools/app_support/vc_preprocess_app.cpp)
add_library(resonantia_training_control STATIC tools/app_support/training_cancel.cpp)
target_include_directories(resonantia_training_control PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(resonantia_training_control PUBLIC Threads::Threads)

resonantia_add_app_support(resonantia_vc_train_app_support
    tools/app_support/vc_train_app.cpp)
target_link_libraries(resonantia_vc_train_app_support PRIVATE resonantia_training_control)

add_library(resonantia_web_support STATIC
    tools/web/http_transport.cpp
    tools/web/workflow_support.cpp
    tools/web/web_service.cpp
)
target_include_directories(resonantia_web_support PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
target_link_libraries(resonantia_web_support PRIVATE
    resonantia_training_control
    voxmutatio
    Threads::Threads
)

function(resonantia_add_tool target source)
    add_executable(${target} ${source})
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${target} PRIVATE
        voxmutatio
        CUDA::cudart
        ${ARGN}
    )
    set_target_properties(${target} PROPERTIES
        BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}"
        INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}"
        LINK_FLAGS "-Wl,--disable-new-dtags"
    )
endfunction()

resonantia_add_tool(vc_convert tools/vc_convert.cpp
    resonantia_vc_convert_app_support CUDA::cublas)
resonantia_add_tool(vc_train tools/vc_train.cpp
    resonantia_vc_train_app_support CUDA::cublas)
resonantia_add_tool(vc_preprocess tools/vc_preprocess.cpp
    resonantia_vc_preprocess_app_support CUDA::cublas)
resonantia_add_tool(vc_serve tools/vc_serve.cpp
    resonantia_web_support CUDA::cublas)
resonantia_add_tool(vc_batch tools/vc_batch.cpp
    resonantia_vc_batch_app_support CUDA::cublas)
resonantia_add_tool(vc_probe tools/vc_probe.cpp CUDA::cublas)
resonantia_add_tool(test_model_load tools/test_model_load.cpp)
resonantia_add_tool(build_index tools/build_index.cpp CUDA::cublas)

# ============================================================================
# 测试
# ============================================================================
option(VOXMUTATIO_BUILD_TESTS "Build reproducible contract test executables" ON)
option(VOXMUTATIO_ENABLE_MODEL_TESTS
       "Build/register tests that require ignored model weights and reference fixtures" OFF)

option(VOXMUTATIO_ENABLE_SLOW_WEB_TESTS
       "Register opt-in real 120-second Web body deadline acceptance test" OFF)

if(VOXMUTATIO_BUILD_TESTS)
    enable_testing()
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ResonantiaTests.cmake)
endif()
