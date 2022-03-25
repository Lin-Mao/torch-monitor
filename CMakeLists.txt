cmake_minimum_required (VERSION 3.12)
project (torch_monitor)

# The version number.
set(GBOLT_VERSION_MAJOR 1)
set(GBOLT_VERSION_MINOR 0)

set(TORCH_DIR "" CACHE STRING "PyTorch install path")
if ("${TORCH_DIR}" STREQUAL "")
	message(SEND_ERROR "TORCH_DIR must be specified")
endif()

find_package(Python COMPONENTS Interpreter Development)
if (NOT ${Python_FOUND})
  message(SEND_ERROR "PYTHON not found")
endif()

set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(TORCH_LINKER_FLAGS "-L${TORCH_DIR}/lib -Wl,-rpath=${TORCH_DIR}/lib -lc10 -ltorch -ltorch_cpu")
set(PYTHON_LINKER_FLAGS "-L${Python_LIBRARY_DIRS} -Wl,-rpath=${Python_LIBRARY_DIRS} -lpython3")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${TORCH_LINKER_FLAGS} ${PYTHON_LINKER_FLAGS}")
set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "")
set(CMAKE_CXX_STANDARD 17)

set(INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")
set(TORCH_INCLUDE_DIR "${TORCH_DIR}/include")
set(TORCH_C_INCLUDE_DIR "${TORCH_DIR}/include/torch/csrc/api/include")
set(SOURCES_DIR "${PROJECT_SOURCE_DIR}/src")

include_directories(${INCLUDE_DIR} ${TORCH_INCLUDE_DIR} ${TORCH_C_INCLUDE_DIR} ${Python_INCLUDE_DIRS})

file(GLOB SOURCES "${SOURCES_DIR}/*.cc")

add_library(${CMAKE_PROJECT_NAME} SHARED ${SOURCES})
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${INCLUDE_DIR}/${CMAKE_PROJECT_NAME}.h")

install(TARGETS ${CMAKE_PROJECT_NAME}
        ARCHIVE DESTINATION lib
				LIBRARY DESTINATION lib
				COMPONENT library
				PUBLIC_HEADER DESTINATION include)
