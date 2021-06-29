cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(dermatologist)
set(CMAKE_CXX_STANDARD 14)

find_package(Torch REQUIRED)

option(DOWNLOAD_DERMATOLOGIST "Download the dermatologist dataset from the interweb" ON)
if (DOWNLOAD_DERMATOLOGIST)
  message(STATUS "Downloading dermatologist dataset")
  execute_process(
    COMMAND python ${CMAKE_CURRENT_LIST_DIR}/download_dataset.py
      -d ${CMAKE_BINARY_DIR}/data
    ERROR_VARIABLE DOWNLOAD_ERROR)
  if (DOWNLOAD_ERROR)
    message(FATAL_ERROR "Error downloading dermatologist dataset: ${DOWNLOAD_ERROR}")
  endif()
endif()

add_executable(dermatologist dermatologist.cpp)
target_compile_features(dermatologist PUBLIC cxx_range_for)
target_link_libraries(dermatologist ${TORCH_LIBRARIES})

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET dermatologist
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:dermatologist>)
endif (MSVC)