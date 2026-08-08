cmake_minimum_required(VERSION 3.25)

project(neurosim VERSION 0.1.0 LANGUAGES C CXX)

option(NEUROSIM_BUILD_VIEWER "Build the interactive raylib viewer" ON)

add_library(neurosim_core
    src/dungeon.cpp
    src/environment.cpp
)

target_include_directories(neurosim_core PUBLIC include)
target_compile_features(neurosim_core PUBLIC cxx_std_20)
set_target_properties(neurosim_core PROPERTIES CXX_EXTENSIONS OFF)

if(MSVC)
    target_compile_options(neurosim_core PRIVATE /W4 /permissive- /EHsc)
else()
    target_compile_options(neurosim_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(neurosim_headless apps/headless_main.cpp)
target_link_libraries(neurosim_headless PRIVATE neurosim_core)

if(MSVC)
    target_compile_options(neurosim_headless PRIVATE /W4 /permissive- /EHsc)
else()
    target_compile_options(neurosim_headless PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(NEUROSIM_BUILD_VIEWER)
    include(FetchContent)

    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(BUILD_GAMES OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(
        raylib
        GIT_REPOSITORY https://github.com/raysan5/raylib.git
        GIT_TAG dbc56a87da87d973a9c5baa4e7438a9d20121d28
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(raylib)

    add_executable(neurosim_game apps/game_main.cpp)
    target_link_libraries(neurosim_game PRIVATE neurosim_core raylib)

    if(MSVC)
        target_compile_options(neurosim_game PRIVATE /W4 /permissive- /EHsc)
    else()
        target_compile_options(neurosim_game PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endif()

include(CTest)

if(BUILD_TESTING)
    add_executable(neurosim_tests tests/environment_tests.cpp)
    target_link_libraries(neurosim_tests PRIVATE neurosim_core)

    if(MSVC)
        target_compile_options(neurosim_tests PRIVATE /W4 /permissive- /EHsc)
    else()
        target_compile_options(neurosim_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_test(NAME neurosim_tests COMMAND neurosim_tests)
endif()
