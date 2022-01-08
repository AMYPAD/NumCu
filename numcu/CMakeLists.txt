cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
if("${NUMCU_BUILD_VERSION}" STREQUAL "")
  set(NUMCU_BUILD_VERSION 0 CACHE STRING "version" FORCE)
endif()
project(numcu LANGUAGES C CXX VERSION "${NUMCU_BUILD_VERSION}")

option(NUMCU_CUDA_OPTIONAL "Make CUDA optional rather than forced" ON)

cmake_policy(PUSH)
cmake_policy(SET CMP0074 NEW)  # <PackageName>_ROOT hints for find_package
cmake_policy(SET CMP0104 NEW)  # CMAKE_CUDA_ARCHITECTURES
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
if(NOT NUMCU_CUDA_OPTIONAL)
  find_package(CUDAToolkit REQUIRED)
  enable_language(CUDA)
else()
  find_package(CUDAToolkit)
  if(CUDAToolkit_FOUND)
    enable_language(CUDA)
  else()
    message(WARNING "Could not find CUDA: using CPU")
    add_compile_definitions(CUVEC_DISABLE_CUDA)
    #list(APPEND CMAKE_CXX_SOURCE_FILE_EXTENSIONS cu)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      add_definitions("-x c++")
    endif()
  endif()
endif()
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import cuvec; print(cuvec.include_path)"
  OUTPUT_VARIABLE CUVEC_INCLUDE_DIRS
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if("${CUVEC_INCLUDE_DIRS}" STREQUAL "")
  message(WARNING "Could not find cuvec includes")
else()
  message(STATUS "Found cuvec includes: ${CUVEC_INCLUDE_DIRS}")
endif()
if(SKBUILD)
  find_package(PythonExtensions REQUIRED)
  set(LIB_TYPE "MODULE")
else()
  set(LIB_TYPE "SHARED")
endif()
cmake_policy(POP)

add_compile_definitions(PY_SSIZE_T_CLEAN)

message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
if("${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

option(CUVEC_DEBUG "Print out CUDA malloc & free operations" OFF)
if(CUVEC_DEBUG)
  add_compile_definitions(CUVEC_DEBUG)
endif(CUVEC_DEBUG)
message(STATUS "CuVec debugging: ${CUVEC_DEBUG}")

if("${NUMCU_THREADS}" STREQUAL "")
  set(NUMCU_THREADS 1024 CACHE STRING
    "Maximum number of CUDA threads per block (should be less than cudaDeviceProp::maxThreadsDim)" FORCE)
endif()
add_compile_definitions(NUMCU_THREADS=${NUMCU_THREADS})

set(${CMAKE_PROJECT_NAME}_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/include/")  # / suffix important
install(DIRECTORY "${${CMAKE_PROJECT_NAME}_INCLUDE_DIRS}" DESTINATION ${CMAKE_PROJECT_NAME}/include)

# main project

file(GLOB SRC LIST_DIRECTORIES false "src/*.cu")
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${CUVEC_INCLUDE_DIRS})

add_library(${PROJECT_NAME} SHARED ${SRC})
add_library(AMYPAD::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME} PUBLIC
  "$<BUILD_INTERFACE:${${CMAKE_PROJECT_NAME}_INCLUDE_DIRS}>"
  "$<INSTALL_INTERFACE:${CMAKE_PROJECT_NAME}/include>")
if(CUDAToolkit_FOUND)
  target_link_libraries(${PROJECT_NAME} ${Python3_LIBRARIES} CUDA::cudart_static)
else()
  set_source_files_properties(${SRC} PROPERTIES LANGUAGE CXX)
  target_link_libraries(${PROJECT_NAME} ${Python3_LIBRARIES})
endif()

if(SKBUILD)
  python_extension_module(${PROJECT_NAME})
endif()
set_target_properties(${PROJECT_NAME} PROPERTIES
  CXX_STANDARD 11
  #VERSION ${CMAKE_PROJECT_VERSION} SOVERSION ${CMAKE_PROJECT_VERSION_MAJOR}
  PREFIX ""  # remove shared lib prefix to make importable
  INTERFACE_${PROJECT_NAME}_MAJOR_VERSION ${CMAKE_PROJECT_VERSION_MAJOR})
set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY COMPATIBLE_INTERFACE_STRING ${PROJECT_NAME}_MAJOR_VERSION)
install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}Targets
  INCLUDES DESTINATION ${CMAKE_PROJECT_NAME}/include
  LIBRARY DESTINATION ${CMAKE_PROJECT_NAME})
install(EXPORT ${PROJECT_NAME}Targets FILE AMYPAD${PROJECT_NAME}Targets.cmake
  NAMESPACE AMYPAD:: DESTINATION ${CMAKE_PROJECT_NAME}/cmake)

# install project

include(CMakePackageConfigHelpers)
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/AMYPAD${CMAKE_PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION ${CMAKE_PROJECT_NAME}/cmake)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/AMYPAD${CMAKE_PROJECT_NAME}ConfigVersion.cmake"
  VERSION "${CMAKE_PROJECT_VERSION}" COMPATIBILITY AnyNewerVersion)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/AMYPAD${CMAKE_PROJECT_NAME}Config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/AMYPAD${CMAKE_PROJECT_NAME}ConfigVersion.cmake"
  DESTINATION ${CMAKE_PROJECT_NAME}/cmake)
