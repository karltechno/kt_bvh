project(kt_bvh)
cmake_minimum_required(VERSION 3.3)

add_library(kt_bvh
    kt_bvh.h
    kt_bvh.cpp
)

target_include_directories(kt_bvh PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(pathtracer_test 
    pathtracer_test.cpp
    fast_obj.h
)

target_link_libraries(pathtracer_test kt_bvh)
set_property(TARGET pathtracer_test PROPERTY VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# pdbs in debug
if(MSVC)
    add_definitions(/Zi -D_CRT_SECURE_NO_WARNINGS)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /DEBUG")
endif()

if("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_C_COMPILER_ID}" STREQUAL "Clang")
    set_target_properties(kt_bvh pathtracer_test PROPERTIES
        COMPILE_FLAGS "-Wall -Werror"
)
elseif("${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC")
    set_target_properties(kt_bvh pathtracer_test PROPERTIES
        COMPILE_FLAGS "/W3 /WX -DNOMINMAX"
)

endif()