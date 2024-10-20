cmake_minimum_required(VERSION 3.28.3 FATAL_ERROR)
project(smartDIR VERSION 1.0 LANGUAGES C)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


include_directories(extern/${SDL2_INCLUDE_DIRS} extern/${SDL2_INCLUDE_DIRS}/include)

add_executable(smartDIR main.c
        collection/KeyboardActivityLogger.c
        include/KeyboardActivityLogger.h
        collection/MouseActivityLogger.c
        include/MouseActivityLogger.h
        )

target_link_libraries(smartDIR ${SDL2_LIBRARIES})

set_property(TARGET smartDIR PROPERTY C_STANDARD 23)
