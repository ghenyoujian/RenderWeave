find_package(OpenSSL REQUIRED COMPONENTS Crypto)

add_library(rw_sr_source_backbone STATIC backbone.cu prepare.cu weights.cpp)
target_compile_features(rw_sr_source_backbone PUBLIC cxx_std_20 cuda_std_20)
target_include_directories(rw_sr_source_backbone PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_options(rw_sr_source_backbone PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=true;--prec-sqrt=true>")
set_source_files_properties(backbone.cu prepare.cu PROPERTIES COMPILE_OPTIONS "--ftz=false")
target_link_libraries(rw_sr_source_backbone PUBLIC CUDA::cudart PRIVATE OpenSSL::Crypto)

add_executable(rw-sr-source-backbone-probe probe.cu)
target_include_directories(rw-sr-source-backbone-probe PRIVATE "${PROJECT_SOURCE_DIR}/third_party")
target_link_libraries(rw-sr-source-backbone-probe PRIVATE rw_sr_source_backbone OpenSSL::Crypto)

set(RW_SR_K_RECONSTRUCTION_PTX "" CACHE FILEPATH "Extracted source for the original SR K reconstruction entry")
if(RW_SR_K_RECONSTRUCTION_PTX)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    set(generator "${PROJECT_SOURCE_DIR}/tools/models/sr/lower_k_reconstruction.py")
    set(converter "${PROJECT_SOURCE_DIR}/third_party/mlx_dlss/ptx_metal.py")
    set(body "${CMAKE_CURRENT_BINARY_DIR}/rw_sr_k_reconstruction_body.inc")
    add_custom_command(OUTPUT "${body}"
        COMMAND "${Python3_EXECUTABLE}" -B "${generator}"
            --source "${RW_SR_K_RECONSTRUCTION_PTX}" --output "${body}"
        DEPENDS "${RW_SR_K_RECONSTRUCTION_PTX}" "${generator}" "${converter}"
        VERBATIM)
    target_sources(rw_sr_source_backbone PRIVATE reconstruction.cu "${body}")
    set_source_files_properties(reconstruction.cu PROPERTIES COMPILE_OPTIONS "--ftz=true")
    target_include_directories(rw_sr_source_backbone PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
    target_compile_definitions(rw-sr-source-backbone-probe PRIVATE RW_SR_K_RECONSTRUCTION=1)

    set_target_properties(rw_sr_source_backbone PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_library(rw_sr_sdk SHARED sdk.cu display.cpp)
    add_library(rw::sr_sdk ALIAS rw_sr_sdk)
    target_compile_features(rw_sr_sdk PRIVATE cxx_std_20 cuda_std_20)
    target_compile_definitions(rw_sr_sdk PRIVATE RW_BUILDING_SR)
    target_include_directories(rw_sr_sdk PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
    target_link_libraries(rw_sr_sdk PRIVATE rw_sr_source_backbone rw::core Threads::Threads)
    set_target_properties(rw_sr_sdk PROPERTIES CUDA_RUNTIME_LIBRARY Shared
        EXPORT_NAME sr_sdk INSTALL_RPATH "$ORIGIN" CXX_VISIBILITY_PRESET hidden
        CUDA_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
    install(TARGETS rw_sr_sdk EXPORT RenderWeaveTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
