add_library(rw_d3d12_sr_host STATIC sr_bridge.cpp)
target_compile_features(rw_d3d12_sr_host PUBLIC cxx_std_20)
target_link_libraries(rw_d3d12_sr_host PUBLIC rw::sr_sdk CUDA::cudart d3d12)
target_compile_definitions(rw_d3d12_sr_host PUBLIC WIN32_LEAN_AND_MEAN NOMINMAX)

add_executable(rw-d3d12-sr-replay sr_replay.cpp)
target_link_libraries(rw-d3d12-sr-replay PRIVATE rw_d3d12_sr_host dxgi bcrypt psapi)
install(TARGETS rw-d3d12-sr-replay RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
