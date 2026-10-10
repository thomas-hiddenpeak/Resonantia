include_guard(GLOBAL)

set(_resonantia_test_output_dir "${CMAKE_BINARY_DIR}/Testing/Artifacts")
file(MAKE_DIRECTORY "${_resonantia_test_output_dir}")

include("${PROJECT_SOURCE_DIR}/cmake/ResonantiaCatch2.cmake")

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
    test_bounded_pipeline_dsp
    test_autograd
)
foreach(target IN LISTS _contract_targets)
    resonantia_add_test_executable(${target} "${PROJECT_SOURCE_DIR}/tests/${target}.cpp")
endforeach()

resonantia_register_test(test_core test_core LABELS "contract;gpu" TIMEOUT 60)
resonantia_register_test(test_audio_io test_audio_io LABELS "contract;gpu" TIMEOUT 60)

# The private bounded file seam is deliberately host-only. Compile its source
# directly so this contract neither links the CUDA facade nor consumes the GPU
# resource lock.
add_executable(test_bounded_audio_io
    "${PROJECT_SOURCE_DIR}/tests/test_bounded_audio_io.cpp"
    "${PROJECT_SOURCE_DIR}/src/io/bounded_audio_io.cpp"
)
target_include_directories(test_bounded_audio_io PRIVATE
    "${PROJECT_SOURCE_DIR}/src"
)
target_link_libraries(test_bounded_audio_io PRIVATE Catch2::Catch2WithMain)
resonantia_register_test(test_bounded_audio_io test_bounded_audio_io
    LABELS "contract;io;host" TIMEOUT 60 NO_GPU_LOCK)

resonantia_register_test(test_content_kernels test_content_kernels
                         LABELS "contract;gpu" TIMEOUT 60)
resonantia_register_test(test_f0 test_f0 LABELS "contract;gpu" TIMEOUT 60)
resonantia_register_test(test_index test_index LABELS "contract;gpu" TIMEOUT 60)
resonantia_register_test(test_pipeline_smoke test_pipeline TEST_SPEC "~[contract]"
                         LABELS "contract;gpu" TIMEOUT 60)
target_include_directories(test_bounded_pipeline_dsp PRIVATE
    "${PROJECT_SOURCE_DIR}/src")
resonantia_register_test(test_bounded_pipeline_dsp test_bounded_pipeline_dsp
    LABELS "contract;pipeline;gpu" TIMEOUT 60)
resonantia_register_test(test_autograd test_autograd LABELS "contract;gpu" TIMEOUT 180)

# Recompile the two temporary-buffer CUDA operation files with a private test-only
# failure seam. The production component objects never receive this definition.
add_executable(test_cuda_failure
    "${PROJECT_SOURCE_DIR}/tests/test_cuda_failure.cpp"
    "${PROJECT_SOURCE_DIR}/tests/cuda_failure_injection.cpp"
    "${PROJECT_SOURCE_DIR}/src/f0/rmvpe_ops.cu"
    "${PROJECT_SOURCE_DIR}/src/synthesizer/vits_ops.cu"
)
target_compile_definitions(test_cuda_failure PRIVATE
    VOXMUTATIO_TEST_CUDA_FAILURE_INJECTION=1)
target_compile_options(test_cuda_failure PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--use_fast_math;-Xptxas=-O3>)
target_include_directories(test_cuda_failure PRIVATE
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_SOURCE_DIR}/tests"
)
target_link_libraries(test_cuda_failure PRIVATE
    CUDA::cudart Catch2::Catch2WithMain)
resonantia_register_test(test_cuda_failure test_cuda_failure
    LABELS "contract;gpu;cuda-failure" TIMEOUT 60)

# Exercise the public Synthesizer exception boundary with the same private
# CUDA failure seam.  The production facade is not linked here: these sources
# are recompiled only for this test target, so the injection macro cannot leak
# into normal executables.
add_executable(test_synth_cuda_failure
    "${PROJECT_SOURCE_DIR}/tests/test_synth_cuda_failure.cpp"
    "${PROJECT_SOURCE_DIR}/tests/cuda_failure_injection.cpp"
    "${PROJECT_SOURCE_DIR}/src/io/safetensors.cpp"
    "${PROJECT_SOURCE_DIR}/src/content/cuda/kernels.cu"
    "${PROJECT_SOURCE_DIR}/src/synthesizer/vits_ops.cu"
    "${PROJECT_SOURCE_DIR}/src/synthesizer/synthesizer.cpp"
)
target_compile_definitions(test_synth_cuda_failure PRIVATE
    VOXMUTATIO_TEST_CUDA_FAILURE_INJECTION=1)
target_compile_options(test_synth_cuda_failure PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--use_fast_math;-Xptxas=-O3>)
target_include_directories(test_synth_cuda_failure PRIVATE
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_SOURCE_DIR}/tests"
)
target_link_libraries(test_synth_cuda_failure PRIVATE
    CUDA::cudart CUDA::cublas CUDA::cufft Catch2::Catch2WithMain)
resonantia_register_test(test_synth_cuda_failure test_synth_cuda_failure
    LABELS "contract;gpu;cuda-failure" TIMEOUT 60)

resonantia_add_process_test_executable(
    test_cli_contract "${PROJECT_SOURCE_DIR}/tests/test_cli_contract.cpp")
target_include_directories(test_cli_contract PRIVATE
    "${PROJECT_SOURCE_DIR}"
    "${PROJECT_SOURCE_DIR}/include")
add_dependencies(test_cli_contract vc_convert vc_batch vc_train vc_preprocess
                 build_index)
resonantia_register_test(test_cli_contract test_cli_contract
    ARGS "$<TARGET_FILE:vc_convert>" "$<TARGET_FILE:vc_batch>"
         "$<TARGET_FILE:vc_train>" "$<TARGET_FILE:vc_preprocess>"
         "$<TARGET_FILE:build_index>"
    LABELS "contract;cli" TIMEOUT 60 NO_GPU_LOCK)

resonantia_add_process_test_executable(
    test_web_contract "${PROJECT_SOURCE_DIR}/tests/test_web_contract.cpp")
target_link_libraries(test_web_contract PRIVATE resonantia_web_support)
add_dependencies(test_web_contract vc_serve)
resonantia_register_test(test_web_contract test_web_contract
    ARGS "$<TARGET_FILE:vc_serve>"
    LABELS "contract;web" TIMEOUT 60 NO_GPU_LOCK)

# Keep real two-minute wall-clock validation out of the default fast suite.
if(VOXMUTATIO_ENABLE_SLOW_WEB_TESTS)
    resonantia_register_test(test_web_body_deadline test_web_contract
        ARGS "$<TARGET_FILE:vc_serve>" "--body-deadline-only"
        LABELS "contract;web;host;slow" TIMEOUT 130 NO_GPU_LOCK)
endif()

resonantia_add_process_test_executable(test_web_transaction_failure
    "${PROJECT_SOURCE_DIR}/tests/test_web_transaction_failure.cpp")
target_link_libraries(test_web_transaction_failure PRIVATE resonantia_web_support)
resonantia_register_test(test_web_transaction_failure test_web_transaction_failure
    LABELS "contract;web;host" TIMEOUT 30 NO_GPU_LOCK)

resonantia_add_process_test_executable(test_web_job_lifecycle
    "${PROJECT_SOURCE_DIR}/tests/test_web_job_lifecycle.cpp")
target_link_libraries(test_web_job_lifecycle PRIVATE resonantia_web_support)
resonantia_register_test(test_web_job_lifecycle test_web_job_lifecycle
    LABELS "contract;web;host" TIMEOUT 30 NO_GPU_LOCK)

# Response barriers exist only in this private recompilation. No system-call
# interposition, real CLI worker, or production runtime test callback is used.
resonantia_add_process_test_executable(test_web_job_ownership
    "${PROJECT_SOURCE_DIR}/tests/test_web_job_ownership.cpp")
target_sources(test_web_job_ownership PRIVATE
    "${PROJECT_SOURCE_DIR}/tools/web/http_transport.cpp"
    "${PROJECT_SOURCE_DIR}/tools/web/workflow_support.cpp"
    "${PROJECT_SOURCE_DIR}/tools/web/web_service.cpp")
target_include_directories(test_web_job_ownership PRIVATE "${PROJECT_SOURCE_DIR}")
target_compile_definitions(test_web_job_ownership PRIVATE
    RESONANTIA_TEST_JOB_OWNERSHIP=1)
target_link_libraries(test_web_job_ownership PRIVATE voxmutatio resonantia_training_control)
resonantia_register_test(test_web_job_ownership test_web_job_ownership
    LABELS "contract;web;host" TIMEOUT 15 NO_GPU_LOCK)

resonantia_add_process_test_executable(test_web_process_contract
    "${PROJECT_SOURCE_DIR}/tests/test_web_process_contract.cpp")
target_link_libraries(test_web_process_contract PRIVATE resonantia_web_support)
resonantia_register_test(test_web_process_contract test_web_process_contract
    LABELS "contract;web;host" TIMEOUT 10 NO_GPU_LOCK)

resonantia_add_process_test_executable(test_web_result_contract
    "${PROJECT_SOURCE_DIR}/tests/test_web_result_contract.cpp")
target_link_libraries(test_web_result_contract PRIVATE resonantia_web_support)
resonantia_register_test(test_web_result_contract test_web_result_contract
    LABELS "contract;web;host" TIMEOUT 5 NO_GPU_LOCK)

resonantia_add_process_test_executable(test_web_train_cancel
    "${PROJECT_SOURCE_DIR}/tests/test_web_train_cancel.cpp")
target_link_libraries(test_web_train_cancel PRIVATE resonantia_web_support)
resonantia_register_test(test_web_train_cancel test_web_train_cancel
    LABELS "contract;web;host" TIMEOUT 30 NO_GPU_LOCK)

# A private source recompilation supplies the deterministic material-commit
# pause. Production resonantia_web_support/vc_serve never receive this macro.
resonantia_add_process_test_executable(test_web_train_cancel_commit
    "${PROJECT_SOURCE_DIR}/tests/test_web_train_cancel.cpp")
target_sources(test_web_train_cancel_commit PRIVATE
    "${PROJECT_SOURCE_DIR}/tools/web/http_transport.cpp"
    "${PROJECT_SOURCE_DIR}/tools/web/workflow_support.cpp"
    "${PROJECT_SOURCE_DIR}/tools/web/web_service.cpp")
target_include_directories(test_web_train_cancel_commit PRIVATE "${PROJECT_SOURCE_DIR}")
target_compile_definitions(test_web_train_cancel_commit PRIVATE
    RESONANTIA_TEST_MATERIAL_COMMIT_GATE=1)
target_link_libraries(test_web_train_cancel_commit PRIVATE voxmutatio resonantia_training_control)
resonantia_register_test(test_web_train_cancel_commit test_web_train_cancel_commit
    LABELS "contract;web;host" TIMEOUT 30 NO_GPU_LOCK)

# Pause inside final three-artifact publication without syscall interposition.
resonantia_add_process_test_executable(test_web_train_cancel_publish
    "${PROJECT_SOURCE_DIR}/tests/test_web_train_cancel.cpp")
target_sources(test_web_train_cancel_publish PRIVATE
    "${PROJECT_SOURCE_DIR}/tools/web/http_transport.cpp"
    "${PROJECT_SOURCE_DIR}/tools/web/workflow_support.cpp"
    "${PROJECT_SOURCE_DIR}/tools/web/web_service.cpp")
target_include_directories(test_web_train_cancel_publish PRIVATE "${PROJECT_SOURCE_DIR}")
target_compile_definitions(test_web_train_cancel_publish PRIVATE
    RESONANTIA_TEST_TRAINING_PUBLISH_GATE=1)
target_link_libraries(test_web_train_cancel_publish PRIVATE voxmutatio resonantia_training_control)
resonantia_register_test(test_web_train_cancel_publish test_web_train_cancel_publish
    LABELS "contract;web;host" TIMEOUT 15 NO_GPU_LOCK)

resonantia_add_process_test_executable(test_training_cancel_protocol
    "${PROJECT_SOURCE_DIR}/tests/test_training_cancel_protocol.cpp")
target_link_libraries(test_training_cancel_protocol PRIVATE resonantia_web_support resonantia_training_control)
resonantia_register_test(test_training_cancel_protocol test_training_cancel_protocol
    LABELS "contract;web;host" TIMEOUT 30 NO_GPU_LOCK)

# The remaining test retains real option parsing and the actual CLI pre-cancel
# entry. It still requires the production CUDA build, not GPU test execution.
resonantia_add_process_test_executable(test_training_cancel
    "${PROJECT_SOURCE_DIR}/tests/test_training_cancel.cpp")
target_link_libraries(test_training_cancel PRIVATE resonantia_web_support resonantia_vc_train_app_support)
add_dependencies(test_training_cancel vc_train)
resonantia_register_test(test_training_cancel test_training_cancel
    ARGS "$<TARGET_FILE:vc_train>" LABELS "contract;web;host" TIMEOUT 30 NO_GPU_LOCK)

# The production main is linked unchanged to host throwing app stubs. The
# fixture does not link the CUDA facade or add hooks to the production CLI.
add_executable(vc_train_entry_exception_fixture
    "${PROJECT_SOURCE_DIR}/tools/vc_train.cpp"
    "${PROJECT_SOURCE_DIR}/tests/vc_train_entry_exception_fixture.cpp")
target_link_libraries(vc_train_entry_exception_fixture PRIVATE resonantia_training_control)
resonantia_add_process_test_executable(test_vc_train_entry
    "${PROJECT_SOURCE_DIR}/tests/test_vc_train_entry.cpp")
target_link_libraries(test_vc_train_entry PRIVATE resonantia_web_support)
add_dependencies(test_vc_train_entry vc_train_entry_exception_fixture)
resonantia_register_test(test_vc_train_entry test_vc_train_entry
    ARGS "$<TARGET_FILE:vc_train_entry_exception_fixture>"
    LABELS "contract;cli;host" TIMEOUT 10 NO_GPU_LOCK)

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
target_include_directories(test_rmvpe_alignment PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_include_directories(test_fcpe_alignment PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_include_directories(test_long_audio_chunking PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(test_separation PRIVATE CUDA::cufft)

set(MODELS "${PROJECT_SOURCE_DIR}/models")
set(FIXTURES "${PROJECT_SOURCE_DIR}/tests/fixtures")
set(HUBERT "${MODELS}/hubert_base/model.safetensors")
set(RMVPE "${MODELS}/rmvpe.safetensors")
set(FCPE "${MODELS}/fcpe/model.safetensors")
set(WAVLM "${MODELS}/wavlm/model.safetensors")
set(G_V1 "${MODELS}/pretrained/pretrained/f0G40k.safetensors")
set(G_V2 "${MODELS}/pretrained_v2/pretrained_v2/f0G40k.safetensors")
set(D_V2 "${MODELS}/pretrained_v2/pretrained_v2/f0D40k.safetensors")
set(SPEECH "${FIXTURES}/speech_librispeech.wav")

resonantia_add_process_test_executable(
    test_build_index_contract
    "${PROJECT_SOURCE_DIR}/tests/test_build_index_contract.cpp")
add_dependencies(test_build_index_contract build_index)
resonantia_register_test(test_build_index_contract test_build_index_contract
    ARGS "$<TARGET_FILE:build_index>" "${HUBERT}" "${SPEECH}"
         "${FIXTURES}/speech_01.wav"
    LABELS "integration;model;gpu;index" TIMEOUT 300
    ASSETS "${HUBERT}" "${SPEECH}" "${FIXTURES}/speech_01.wav")

resonantia_add_process_test_executable(
    test_vc_convert_bounded_contract
    "${PROJECT_SOURCE_DIR}/tests/test_vc_convert_bounded_contract.cpp")
add_dependencies(test_vc_convert_bounded_contract vc_convert vc_batch)
resonantia_register_test(
    test_cli_bounded_callers test_vc_convert_bounded_contract
    ARGS "$<TARGET_FILE:vc_convert>" "$<TARGET_FILE:vc_batch>" "${HUBERT}"
         "${RMVPE}" "${G_V2}" "${SPEECH}"
    LABELS "integration;model;gpu;cli;bounded-file" TIMEOUT 300
    ASSETS "${HUBERT}" "${RMVPE}" "${G_V2}" "${SPEECH}")

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
resonantia_register_test(test_bounded_file_resources test_long_audio_chunking
    TEST_SPEC "Bounded file working set is duration independent"
    LABELS "integration;model;gpu;long;resource" TIMEOUT 1200
    ASSETS "${HUBERT}" "${RMVPE}" "${G_V2}" "${SPEECH}")
resonantia_register_test(test_bounded_file_fcpe_v1 test_long_audio_chunking
    TEST_SPEC "FCPE v1 bounded file output matches the legacy buffer reference"
    LABELS "integration;model;gpu;long;bounded-file;fcpe;v1" TIMEOUT 1200
    ASSETS "${HUBERT}" "${FCPE}" "${G_V1}" "${SPEECH}")
resonantia_register_test(test_bounded_file_wavlm_preflight test_long_audio_chunking
    TEST_SPEC "WavLM bounded profile fails before payload or staging I/O"
    LABELS "integration;model;gpu;bounded-file;wavlm;contract" TIMEOUT 300
    ASSETS "${WAVLM}" "${RMVPE}" "${G_V2}")
resonantia_register_test(test_fcpe_alignment test_fcpe_alignment
    LABELS "integration;model;gpu" TIMEOUT 300
    ASSETS "${FCPE}" "${SPEECH}"
           "${FIXTURES}/fcpe_ref_mel.bin" "${FIXTURES}/fcpe_ref_f0.bin")
resonantia_register_test(test_wavlm_alignment test_wavlm_alignment
    LABELS "integration;model;gpu" TIMEOUT 300
    ASSETS "${WAVLM}" "${SPEECH}"
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
