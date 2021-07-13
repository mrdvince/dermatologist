cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(derm VERSION 1.0.0 LANGUAGES CXX)

set(EXECUTABLE_NAME derm)
set(CMAKE_CXX_STANDARD 14)

find_package(Torch REQUIRED PATHS ~/libtorch)
find_package(OpenCV REQUIRED PATHS)

include(${CMAKE_CURRENT_SOURCE_DIR}/CMake/check_files.cmake)

add_executable(${EXECUTABLE_NAME})
target_sources(${EXECUTABLE_NAME} PRIVATE main.cpp
                                          src/model.cpp
                                          src/trainer.cpp
                                          src/dataset.cpp
                                          include/trainer.h
                                          include/model.h
                                          include/dataset.h
)

target_include_directories(${EXECUTABLE_NAME} PRIVATE include)

target_link_libraries(${EXECUTABLE_NAME} ${TORCH_LIBRARIES} ${OpenCV_LIBS})

set_target_properties(${EXECUTABLE_NAME} PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

execute_process(COMMAND python ${CMAKE_CURRENT_LIST_DIR}/convert.py)

option(DOWNLOAD_DATASET "Download the dataset from the interwebs" OFF)

if(DOWNLOAD_DATASET)
    file(MAKE_DIRECTORY ${DATA_DIR})
    message(STATUS "Fetching dataset...")
    execute_process( COMMAND python ${CMAKE_CURRENT_LIST_DIR}/download_dataset.py -d ${CMAKE_BINARY_DIR}/data ERROR_VARIABLE DOWNLOAD_ERROR)  
    if (DOWNLOAD_ERROR)
        message(FATAL_ERROR "Error downloading dataset: ${DOWNLOAD_ERROR}")
    endif()
endif()

if(MSVC)
    set_target_properties(${EXECUTABLE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_BINARY_DIR})
    set_target_properties(${EXECUTABLE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_BINARY_DIR})
    include(copy_torch_dlls)
    copy_torch_dlls(${EXECUTABLE_NAME})
endif(MSVC)