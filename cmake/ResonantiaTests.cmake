include_guard(GLOBAL)

set(_resonantia_test_output_dir "${CMAKE_BINARY_DIR}/Testing/Artifacts")
file(MAKE_DIRECTORY "${_resonantia_test_output_dir}")

# Prefer a system package, but keep a reproducible fallback for developer and CI
# machines. The archive is content-addressed; no branch or mutable Git tag is run.
find_package(Catch2 3.4 CONFIG QUIET)
if(NOT Catch2_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        Catch2
        URL https://github.com/catchorg/Catch2/archive/refs/tags/v3.4.0.tar.gz
        URL_HASH SHA256=122928b814b75717316c71af69bd2b43387643ba076a6ec16e7882bfb2dfacbb
        DOWNLOAD_EXTRACT_TIMESTAMP FALSE
    )
    FetchContent_MakeAvailable(Catch2)
endif()

function(resonantia_add_test_executable target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE
        voxmutatio CUDA::cudart CUDA::cublas Catch2::Catch2WithMain)
    target_include_directories(${target} PRIVATE ${PROJECT_SOURCE_DIR}/include)
endfunction()

function(resonantia_add_process_test_executable target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE Threads::Threads)
endfunction()

function(resonantia_register_test name target)
    cmake_parse_arguments(ARG "NO_GPU_LOCK" "TEST_SPEC;TIMEOUT"
                          "LABELS;ASSETS;ARGS" ${ARGN})

    set(missing_assets)
    foreach(asset IN LISTS ARG_ASSETS)
        if(NOT EXISTS "${asset}")
            list(APPEND missing_assets "${asset}")
        endif()
    endforeach()
    if(missing_assets)
        list(JOIN missing_assets ", " missing_text)
        message(STATUS "CTest: not registering ${name}; missing local assets: ${missing_text}")
        return()
    endif()

    set(test_command "$<TARGET_FILE:${target}>")
    if(ARG_TEST_SPEC)
        list(APPEND test_command "${ARG_TEST_SPEC}")
    endif()
    list(APPEND test_command ${ARG_ARGS})
    add_test(NAME ${name} COMMAND ${test_command})
    set_tests_properties(${name} PROPERTIES
        # Test fixtures use paths such as ../models and ../tests/fixtures.
        # Anchor them to the source tests directory so an out-of-tree build
        # behaves identically whether its binary directory is inside the
        # checkout or in an unrelated temporary directory.
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/tests"
        ENVIRONMENT "RESONANTIA_TEST_OUTPUT_DIR=${_resonantia_test_output_dir}"
        LABELS "${ARG_LABELS}"
        TIMEOUT "${ARG_TIMEOUT}"
    )
    if(NOT ARG_NO_GPU_LOCK)
        set_tests_properties(${name} PROPERTIES RESOURCE_LOCK voxmutatio_gpu)
    endif()
endfunction()

set(_contract_targets
    test_core
    test_audio_io
    test_content_kernels
    test_f0
    test_index
    test_pipeline
    test_autograd
)
foreach(target IN LISTS _contract_targets)
    resonantia_add_test_executable(${target} "${PROJECT_SOURCE_DIR}/tests/${target}.cpp")
endforeach()

resonantia_register_test(test_core test_core LABELS "contract;gpu" TIMEOUT 60)
resonantia_register_test(test_audio_io test_audio_io LABELS "contract;gpu" TIMEOUT 60)
resonantia_register_test(test_content_kernels test_content_kernels
                         LABELS "contract;gpu" TIMEOUT 60)
resonantia_register_test(test_f0 test_f0 LABELS "contract;gpu" TIMEOUT 60)
resonantia_register_test(test_index test_index LABELS "contract;gpu" TIMEOUT 60)
resonantia_register_test(test_pipeline_smoke test_pipeline TEST_SPEC "~[contract]"
                         LABELS "contract;gpu" TIMEOUT 60)
resonantia_register_test(test_autograd test_autograd LABELS "contract;gpu" TIMEOUT 180)

resonantia_add_process_test_executable(
    test_cli_contract "${PROJECT_SOURCE_DIR}/tests/test_cli_contract.cpp")
add_dependencies(test_cli_contract vc_convert vc_batch vc_train vc_preprocess)
resonantia_register_test(test_cli_contract test_cli_contract
    ARGS "$<TARGET_FILE:vc_convert>" "$<TARGET_FILE:vc_batch>"
         "$<TARGET_FILE:vc_train>" "$<TARGET_FILE:vc_preprocess>"
    LABELS "contract;cli" TIMEOUT 60 NO_GPU_LOCK)

resonantia_add_process_test_executable(
    test_web_contract "${PROJECT_SOURCE_DIR}/tests/test_web_contract.cpp")
target_link_libraries(test_web_contract PRIVATE resonantia_web_support)
add_dependencies(test_web_contract vc_serve)
resonantia_register_test(test_web_contract test_web_contract
    ARGS "$<TARGET_FILE:vc_serve>"
    LABELS "contract;web" TIMEOUT 60 NO_GPU_LOCK)

resonantia_add_test_executable(test_legacy_webserver_contract
    "${PROJECT_SOURCE_DIR}/tests/test_legacy_webserver_contract.cpp")
resonantia_register_test(test_legacy_webserver_contract test_legacy_webserver_contract
    LABELS "contract;web;api" TIMEOUT 30 NO_GPU_LOCK)

add_executable(test_realtime_benchmark_report
    "${PROJECT_SOURCE_DIR}/tests/test_realtime_benchmark_report.cpp"
    "${PROJECT_SOURCE_DIR}/tools/realtime/benchmark_report.cpp"
)
target_include_directories(test_realtime_benchmark_report PRIVATE
    "${PROJECT_SOURCE_DIR}"
)
target_link_libraries(test_realtime_benchmark_report PRIVATE Catch2::Catch2WithMain)
resonantia_register_test(test_realtime_benchmark_report test_realtime_benchmark_report
    LABELS "contract;realtime" TIMEOUT 30 NO_GPU_LOCK)

if(NOT VOXMUTATIO_ENABLE_MODEL_TESTS)
    message(STATUS
        "Model/reference tests are disabled. Set VOXMUTATIO_ENABLE_MODEL_TESTS=ON "
        "to build them; only tests whose local assets exist will be registered.")
    return()
endif()

set(_model_targets
    test_hubert_inference
    test_rmvpe_alignment
    test_vits_alignment
    test_vits_gen_debug
    test_e2e_conversion
    test_long_audio_chunking
    test_fcpe_alignment
    test_wavlm_alignment
    test_index_retrieval
    test_mel_loss
    test_generator_trainer
    test_gan
    test_separation
    test_vits_stream_characterization
)
foreach(target IN LISTS _model_targets)
    resonantia_add_test_executable(${target} "${PROJECT_SOURCE_DIR}/tests/${target}.cpp")
endforeach()
target_link_libraries(test_separation PRIVATE CUDA::cufft)

set(MODELS "${PROJECT_SOURCE_DIR}/models")
set(FIXTURES "${PROJECT_SOURCE_DIR}/tests/fixtures")
set(HUBERT "${MODELS}/hubert_base/model.safetensors")
set(RMVPE "${MODELS}/rmvpe.safetensors")
set(G_V1 "${MODELS}/pretrained/pretrained/f0G40k.safetensors")
set(G_V2 "${MODELS}/pretrained_v2/pretrained_v2/f0G40k.safetensors")
set(D_V2 "${MODELS}/pretrained_v2/pretrained_v2/f0D40k.safetensors")
set(SPEECH "${FIXTURES}/speech_librispeech.wav")

resonantia_register_test(test_pipeline_contract test_pipeline TEST_SPEC "[contract]"
    LABELS "integration;model;gpu" TIMEOUT 300
    ASSETS "${HUBERT}" "${RMVPE}" "${G_V1}" "${G_V2}" "${SPEECH}")
resonantia_register_test(test_hubert_inference test_hubert_inference
    LABELS "integration;model;gpu" TIMEOUT 300
    ASSETS "${HUBERT}" "${SPEECH}" "${FIXTURES}/hubert_ref_layer12.bin"
           "${FIXTURES}/hubert_ref_v1_final256.bin")
resonantia_register_test(test_rmvpe_alignment test_rmvpe_alignment
    LABELS "integration;model;gpu" TIMEOUT 300
    ASSETS "${RMVPE}" "${SPEECH}" "${FIXTURES}/rmvpe_ref_salience.bin"
           "${FIXTURES}/rmvpe_ref_f0.bin")
resonantia_register_test(test_vits_alignment test_vits_alignment
    LABELS "integration;model;gpu" TIMEOUT 600
    ASSETS "${G_V2}" "${FIXTURES}/vits_ref_phone.bin" "${FIXTURES}/vits_ref_pitch.bin"
           "${FIXTURES}/vits_ref_m_p.bin" "${FIXTURES}/vits_ref_z.bin"
           "${FIXTURES}/vits_ref_nsff0.bin" "${FIXTURES}/vits_ref_audio.bin")
resonantia_register_test(test_vits_gen_debug test_vits_gen_debug
    LABELS "integration;model;gpu;debug" TIMEOUT 600
    ASSETS "${G_V2}" "${FIXTURES}/vits_ref_nsff0.bin" "${FIXTURES}/vits_ref_har.bin"
           "${FIXTURES}/vits_ref_z.bin" "${FIXTURES}/vits_ref_convpre.bin"
           "${FIXTURES}/vits_ref_ups0.bin" "${FIXTURES}/vits_ref_noiseconv0.bin"
           "${FIXTURES}/vits_ref_stage0.bin")
resonantia_register_test(test_e2e_conversion test_e2e_conversion
    LABELS "integration;model;gpu" TIMEOUT 900
    ASSETS "${HUBERT}" "${RMVPE}" "${G_V2}" "${SPEECH}"
           "${FIXTURES}/vits_ref_audio.bin")
resonantia_register_test(test_long_audio_chunking test_long_audio_chunking
    LABELS "integration;model;gpu;long" TIMEOUT 1200
    ASSETS "${HUBERT}" "${RMVPE}" "${G_V2}" "${SPEECH}")
resonantia_register_test(test_fcpe_alignment test_fcpe_alignment
    LABELS "integration;model;gpu" TIMEOUT 300
    ASSETS "${MODELS}/fcpe/model.safetensors" "${SPEECH}"
           "${FIXTURES}/fcpe_ref_mel.bin" "${FIXTURES}/fcpe_ref_f0.bin")
resonantia_register_test(test_wavlm_alignment test_wavlm_alignment
    LABELS "integration;model;gpu" TIMEOUT 300
    ASSETS "${MODELS}/wavlm/model.safetensors" "${SPEECH}"
           "${FIXTURES}/wavlm_ref_last.bin")
resonantia_register_test(test_index_retrieval test_index_retrieval
    LABELS "integration;model;gpu" TIMEOUT 300
    ASSETS "${FIXTURES}/hubert_ref_layer12.bin")
resonantia_register_test(test_vits_stream_characterization test_vits_stream_characterization
    LABELS "integration;model;gpu;realtime" TIMEOUT 300
    ASSETS "${G_V2}")
resonantia_register_test(test_mel_loss test_mel_loss
    LABELS "integration;training;gpu" TIMEOUT 300 ASSETS "${SPEECH}")
resonantia_register_test(test_generator_trainer test_generator_trainer
    LABELS "integration;training;model;gpu" TIMEOUT 1800
    ASSETS "${G_V2}" "${SPEECH}" "${FIXTURES}/vits_ref_z.bin"
           "${FIXTURES}/vits_ref_nsff0.bin" "${FIXTURES}/vits_ref_audio.bin")
resonantia_register_test(test_gan test_gan
    LABELS "integration;training;model;gpu" TIMEOUT 1800
    ASSETS "${G_V2}" "${D_V2}" "${SPEECH}" "${FIXTURES}/vits_ref_z.bin"
           "${FIXTURES}/vits_ref_nsff0.bin" "${FIXTURES}/vits_ref_m_p.bin"
           "${FIXTURES}/vits_ref_logs_p.bin")

resonantia_register_test(test_separation_stft test_separation TEST_SPEC "[stft]"
    LABELS "integration;fixture;gpu" TIMEOUT 300 ASSETS "${SPEECH}")
resonantia_register_test(test_separation_spec test_separation TEST_SPEC "[spec]"
    LABELS "integration;fixture;gpu" TIMEOUT 300 ASSETS "${SPEECH}")
resonantia_register_test(test_separation_umx test_separation TEST_SPEC "[umx]"
    LABELS "integration;model;gpu" TIMEOUT 900
    ASSETS "${MODELS}/separation/umxhq_vocals.safetensors"
           "${FIXTURES}/separation/umx_mix_mag.bin"
           "${FIXTURES}/separation/umx_model_out.bin"
           "${FIXTURES}/separation/umx_input_wave.bin"
           "${FIXTURES}/separation/umx_vocal_wave.bin")
resonantia_register_test(test_separation_roformer test_separation TEST_SPEC "[roformer]"
    LABELS "integration;model;gpu" TIMEOUT 1800
    ASSETS "${MODELS}/separation/dereverb_roformer.safetensors"
           "${MODELS}/separation/dereverb_roformer_config.i64"
           "${MODELS}/separation/dereverb_roformer_dim_inputs.i64"
           "${MODELS}/separation/dereverb_roformer_freq_indices.i64"
           "${MODELS}/separation/dereverb_roformer_num_bands_per_freq.i64"
           "${FIXTURES}/separation/rof_input_wave.bin"
           "${FIXTURES}/separation/rof_band_split_in.bin"
           "${FIXTURES}/separation/rof_band_split.bin"
           "${FIXTURES}/separation/rof_block_final.bin"
           "${FIXTURES}/separation/rof_mask.bin"
           "${FIXTURES}/separation/rof_output_wave.bin")
resonantia_register_test(test_separation_vad test_separation TEST_SPEC "[vad]"
    LABELS "integration;model;gpu" TIMEOUT 300
    ASSETS "${MODELS}/vad/silero_vad.safetensors"
           "${FIXTURES}/vad/vad_input.bin" "${FIXTURES}/vad/vad_probs.bin")
