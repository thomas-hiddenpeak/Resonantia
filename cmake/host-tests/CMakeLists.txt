cmake_minimum_required(VERSION 3.24)
project(ResonantiaHostContracts LANGUAGES CXX)

# Reuse existing Linux host test sources without enabling CUDA or changing the
# production build. This is a bounded CI subset, not the full device suite.
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "The host contract subset requires Linux")
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
find_package(Threads REQUIRED)
enable_testing()
get_filename_component(RESONANTIA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
include("${RESONANTIA_ROOT}/cmake/ResonantiaCatch2.cmake")

add_library(host_transport STATIC
    "${RESONANTIA_ROOT}/src/io/audio_io.cpp"
    "${RESONANTIA_ROOT}/tools/web/http_transport.cpp"
    "${RESONANTIA_ROOT}/src/separation/sep_registry.cpp"
    "${RESONANTIA_ROOT}/tools/app_support/training_cancel.cpp")
target_include_directories(host_transport PUBLIC
    "${RESONANTIA_ROOT}" "${RESONANTIA_ROOT}/include")
target_link_libraries(host_transport PUBLIC Threads::Threads)
add_library(host_workflow STATIC "${RESONANTIA_ROOT}/tools/web/workflow_support.cpp")
target_link_libraries(host_workflow PUBLIC host_transport)
add_library(host_web STATIC "${RESONANTIA_ROOT}/tools/web/web_service.cpp")
target_link_libraries(host_web PUBLIC host_workflow)

function(register_host_contract name timeout)
    add_test(NAME ${name} COMMAND ${name} ${ARGN})
    set_tests_properties(${name} PROPERTIES
        WORKING_DIRECTORY "${RESONANTIA_ROOT}/tests"
        LABELS "contract;host" TIMEOUT "${timeout}"
        ENVIRONMENT "CUDA_VISIBLE_DEVICES=;OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
endfunction()

foreach(name IN ITEMS test_web_job_lifecycle test_web_result_contract test_web_train_cancel test_web_audio_upload)
    add_executable(${name} "${RESONANTIA_ROOT}/tests/${name}.cpp")
    target_link_libraries(${name} PRIVATE host_web)
    register_host_contract(${name} 30)
endforeach()

add_executable(test_web_job_ownership
    "${RESONANTIA_ROOT}/tests/test_web_job_ownership.cpp"
    "${RESONANTIA_ROOT}/tools/web/web_service.cpp")
target_compile_definitions(test_web_job_ownership PRIVATE RESONANTIA_TEST_JOB_OWNERSHIP=1)
target_link_libraries(test_web_job_ownership PRIVATE host_workflow)
register_host_contract(test_web_job_ownership 15)

add_executable(test_web_train_cancel_commit
    "${RESONANTIA_ROOT}/tests/test_web_train_cancel.cpp"
    "${RESONANTIA_ROOT}/tools/web/web_service.cpp"
    "${RESONANTIA_ROOT}/tools/web/workflow_support.cpp")
target_compile_definitions(test_web_train_cancel_commit PRIVATE RESONANTIA_TEST_MATERIAL_COMMIT_GATE=1)
target_link_libraries(test_web_train_cancel_commit PRIVATE host_transport)
register_host_contract(test_web_train_cancel_commit 30)

add_executable(test_web_train_cancel_publish
    "${RESONANTIA_ROOT}/tests/test_web_train_cancel.cpp"
    "${RESONANTIA_ROOT}/tools/web/web_service.cpp"
    "${RESONANTIA_ROOT}/tools/web/workflow_support.cpp")
target_compile_definitions(test_web_train_cancel_publish PRIVATE RESONANTIA_TEST_TRAINING_PUBLISH_GATE=1)
target_link_libraries(test_web_train_cancel_publish PRIVATE host_transport)
register_host_contract(test_web_train_cancel_publish 15)

add_executable(test_training_cancel_protocol
    "${RESONANTIA_ROOT}/tests/test_training_cancel_protocol.cpp")
target_link_libraries(test_training_cancel_protocol PRIVATE host_workflow)
register_host_contract(test_training_cancel_protocol 30)

add_executable(test_bounded_audio_io
    "${RESONANTIA_ROOT}/tests/test_bounded_audio_io.cpp"
    "${RESONANTIA_ROOT}/src/io/bounded_audio_io.cpp")
target_include_directories(test_bounded_audio_io PRIVATE "${RESONANTIA_ROOT}/src")
target_link_libraries(test_bounded_audio_io PRIVATE Catch2::Catch2WithMain)
register_host_contract(test_bounded_audio_io 60)

add_executable(test_legacy_webserver_contract
    "${RESONANTIA_ROOT}/tests/test_legacy_webserver_contract.cpp"
    "${RESONANTIA_ROOT}/src/webui/server.cpp")
target_include_directories(test_legacy_webserver_contract PRIVATE "${RESONANTIA_ROOT}/include")
target_link_libraries(test_legacy_webserver_contract PRIVATE Catch2::Catch2WithMain)
register_host_contract(test_legacy_webserver_contract 30)

add_executable(vc_train_entry_exception_fixture
    "${RESONANTIA_ROOT}/tools/vc_train.cpp"
    "${RESONANTIA_ROOT}/tests/vc_train_entry_exception_fixture.cpp")
target_link_libraries(vc_train_entry_exception_fixture PRIVATE host_transport)
add_executable(test_vc_train_entry "${RESONANTIA_ROOT}/tests/test_vc_train_entry.cpp")
target_link_libraries(test_vc_train_entry PRIVATE host_workflow)
add_dependencies(test_vc_train_entry vc_train_entry_exception_fixture)
register_host_contract(test_vc_train_entry 10 "$<TARGET_FILE:vc_train_entry_exception_fixture>")
