cmake_minimum_required(VERSION 3.25)

project(neurosim VERSION 0.1.0 LANGUAGES C CXX)

option(NEUROSIM_BUILD_VIEWER "Build the interactive raylib viewer" ON)
option(NEUROSIM_BUILD_CUDA "Build the explicit CUDA brain backend" OFF)

set(NEUROSIM_CUDA_ARCHITECTURES "all-major" CACHE STRING
    "CUDA architectures for neurosim_cuda, such as all-major or 86")

add_library(neurosim_core
    src/adaptive_controller.cpp
    src/brain.cpp
    src/brain_agent.cpp
    src/brain_checkpoint.cpp
    src/brain_genome.cpp
    src/brain_simulator.cpp
    src/cpu_brain.cpp
    src/dungeon.cpp
    src/environment.cpp
    src/evolution.cpp
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

add_executable(neurosim_train apps/train_main.cpp)
target_link_libraries(neurosim_train PRIVATE neurosim_core)
target_compile_definitions(neurosim_train PRIVATE NEUROSIM_VERSION="${PROJECT_VERSION}")

add_executable(neurosim_evolve apps/evolve_main.cpp)
target_link_libraries(neurosim_evolve PRIVATE neurosim_core)
target_compile_definitions(neurosim_evolve PRIVATE NEUROSIM_VERSION="${PROJECT_VERSION}")

if(MSVC)
    target_compile_options(neurosim_headless PRIVATE /W4 /permissive- /EHsc)
    target_compile_options(neurosim_train PRIVATE /W4 /permissive- /EHsc)
    target_compile_options(neurosim_evolve PRIVATE /W4 /permissive- /EHsc)
else()
    target_compile_options(neurosim_headless PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(neurosim_train PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(neurosim_evolve PRIVATE -Wall -Wextra -Wpedantic)
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

if(NEUROSIM_BUILD_CUDA)
    include(CheckLanguage)
    check_language(CUDA)

    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR
            "NEUROSIM_BUILD_CUDA=ON requires an installed CUDA Toolkit with nvcc")
    endif()

    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    add_library(neurosim_cuda src/cuda_brain.cu)
    target_include_directories(neurosim_cuda PUBLIC include)
    target_link_libraries(neurosim_cuda PUBLIC neurosim_core CUDA::cudart)
    target_compile_features(neurosim_cuda PUBLIC cxx_std_20)
    target_compile_definitions(neurosim_cuda PUBLIC NEUROSIM_HAS_CUDA=1)
    set_target_properties(neurosim_cuda PROPERTIES
        CUDA_ARCHITECTURES "${NEUROSIM_CUDA_ARCHITECTURES}"
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF
        CUDA_SEPARABLE_COMPILATION OFF
    )

    if(MSVC)
        target_compile_options(neurosim_cuda PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/permissive->
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc>
        )
    else()
        target_compile_options(neurosim_cuda PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wextra>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wpedantic>
        )
    endif()

    target_link_libraries(neurosim_train PRIVATE neurosim_cuda)
    target_link_libraries(neurosim_evolve PRIVATE neurosim_cuda)

    if(TARGET neurosim_game)
        target_link_libraries(neurosim_game PRIVATE neurosim_cuda)
    endif()
endif()

include(CTest)

if(BUILD_TESTING)
    add_executable(neurosim_tests tests/environment_tests.cpp)
    target_link_libraries(neurosim_tests PRIVATE neurosim_core)

    add_executable(neurosim_brain_tests tests/brain_tests.cpp)
    target_link_libraries(neurosim_brain_tests PRIVATE neurosim_core)

    add_executable(neurosim_agent_tests tests/agent_tests.cpp)
    target_link_libraries(neurosim_agent_tests PRIVATE neurosim_core)

    add_executable(neurosim_checkpoint_tests tests/checkpoint_tests.cpp)
    target_link_libraries(neurosim_checkpoint_tests PRIVATE neurosim_core)

    add_executable(neurosim_genome_tests tests/genome_tests.cpp)
    target_link_libraries(neurosim_genome_tests PRIVATE neurosim_core)

    add_executable(neurosim_evolution_tests tests/evolution_tests.cpp)
    target_link_libraries(neurosim_evolution_tests PRIVATE neurosim_core)

    if(MSVC)
        target_compile_options(neurosim_tests PRIVATE /W4 /permissive- /EHsc)
        target_compile_options(neurosim_brain_tests PRIVATE /W4 /permissive- /EHsc)
        target_compile_options(neurosim_agent_tests PRIVATE /W4 /permissive- /EHsc)
        target_compile_options(neurosim_checkpoint_tests PRIVATE /W4 /permissive- /EHsc)
        target_compile_options(neurosim_genome_tests PRIVATE /W4 /permissive- /EHsc)
        target_compile_options(neurosim_evolution_tests PRIVATE /W4 /permissive- /EHsc)
    else()
        target_compile_options(neurosim_tests PRIVATE -Wall -Wextra -Wpedantic)
        target_compile_options(neurosim_brain_tests PRIVATE -Wall -Wextra -Wpedantic)
        target_compile_options(neurosim_agent_tests PRIVATE -Wall -Wextra -Wpedantic)
        target_compile_options(neurosim_checkpoint_tests PRIVATE -Wall -Wextra -Wpedantic)
        target_compile_options(neurosim_genome_tests PRIVATE -Wall -Wextra -Wpedantic)
        target_compile_options(neurosim_evolution_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_test(NAME neurosim_tests COMMAND neurosim_tests)
    add_test(NAME neurosim_brain_tests COMMAND neurosim_brain_tests)
    add_test(NAME neurosim_agent_tests COMMAND neurosim_agent_tests)
    add_test(NAME neurosim_checkpoint_tests COMMAND neurosim_checkpoint_tests)
    add_test(NAME neurosim_genome_tests COMMAND neurosim_genome_tests)
    add_test(NAME neurosim_evolution_tests COMMAND neurosim_evolution_tests)

    if(NEUROSIM_BUILD_CUDA)
        add_executable(neurosim_cuda_tests tests/cuda_brain_tests.cpp)
        target_link_libraries(neurosim_cuda_tests PRIVATE neurosim_cuda)

        if(MSVC)
            target_compile_options(neurosim_cuda_tests PRIVATE /W4 /permissive- /EHsc)
        else()
            target_compile_options(neurosim_cuda_tests PRIVATE -Wall -Wextra -Wpedantic)
        endif()

        add_test(NAME neurosim_cuda_tests COMMAND neurosim_cuda_tests)
        set_tests_properties(neurosim_cuda_tests PROPERTIES SKIP_RETURN_CODE 77)
    endif()
endif()
