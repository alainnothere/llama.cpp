set(TARGET llama-mtp-replay)
add_executable(${TARGET} mtp-replay.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama-common llama ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
# the MTP circuit engine (llama_mtp_set_circuit, llama_context_params.mtp_circuit_max_passes) is new on 2026-11-07;
# without it the tool builds and replays the default circuit M0 only
if (EXISTS ${CMAKE_SOURCE_DIR}/src/llama-mtp-circuit.h)
    target_compile_definitions(${TARGET} PRIVATE MTP_REPLAY_CIRCUIT_ENGINE=1)
endif()
