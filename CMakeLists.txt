cmake_minimum_required(VERSION 3.2)
set(CMAKE_CXX_STANDARD 11)

project(Nature_Atari)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin/${CMAKE_BUILD_TYPE})

#Add Caffe
find_package(OpenCV)
find_package(Caffe)

include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})

#Add Arcade Learning Environment
set(ALE_INCLUDE_DIR "ALE_INCLUDE_DIR_NOT_FOUND" CACHE PATH "Include directory for Arcade Learning Environment")
set(ALE_DEBUG_LIB "ALE_DEBUG_LIB_NOT_FOUND" CACHE FILEPATH "Debug lib for Arcade Learning Environment")
set(ALE_RELEASE_LIB "ALE_RELEASE_LIB_NOT_FOUND" CACHE FILEPATH "Release lib for Arcade Learning Environment")
set(ALE_SDL_LIB "ALE_SDL_LIB_NOT_FOUND" CACHE FILEPATH "SDL lib for Arcade Learning Environment")
set(ALE_ZDLL_LIB "ALE_ZDLL_LIB_NOT_FOUND" CACHE FILEPATH "Zdll lib for Arcade Learning Environment")
set(ALE_LIBRARIES "ALE")

include_directories(${ALE_INCLUDE_DIR})

#Set src files
file(GLOB src_files "src/*.cpp")


file(GLOB header_files "include/*.h")

include_directories(include)


add_executable(Nature_Atari ${src_files} ${header_files})

if(MSVC)
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /STACK:8000000")
endif()


target_link_libraries(Nature_Atari ${Caffe_LIBRARIES})
target_link_libraries(Nature_Atari ${ALE_SDL_LIB})
target_link_libraries(Nature_Atari ${ALE_ZDLL_LIB})
target_link_libraries(Nature_Atari debug ${ALE_DEBUG_LIB} optimized ${ALE_RELEASE_LIB})
