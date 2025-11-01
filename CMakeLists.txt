cmake_minimum_required(VERSION 3.18)
project(Intervalix LANGUAGES CXX CUDA)

# Spécifie le standard C++
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  # Default to a reasonable set; override with -DCMAKE_CUDA_ARCHITECTURES=...
  set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(GTest REQUIRED)


# Ajoute l'exécutable
add_library(interval_intersection src/interval_intersection.cu)
target_include_directories(interval_intersection PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(interval_intersection PUBLIC CUDA::cudart)


add_executable(exemple_main src/main.cu)
target_link_libraries(exemple_main PRIVATE
	interval_intersection
	PUBLIC CUDA::cudart
)



add_executable(intersection_test src/test_intersection.cu)
## Lie les bibliothèques CUDA
target_link_libraries(intersection_test CUDA::cudart interval_intersection GTest::GTest GTest::Main)

add_executable(surface_demo src/surface_demo.cu src/surface_generator.cpp)
target_include_directories(surface_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(surface_demo PRIVATE interval_intersection CUDA::cudart)

add_executable(volume_demo src/volume_demo.cu src/volume_generator.cpp)
target_include_directories(volume_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(volume_demo PRIVATE interval_intersection CUDA::cudart)

add_executable(workspace_benchmark
    bench/workspace_benchmark.cu
    src/surface_generator.cpp
    src/volume_generator.cpp)
target_include_directories(workspace_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(workspace_benchmark PRIVATE interval_intersection CUDA::cudart)

add_executable(count_multishape bench/count_multishape.cpp src/surface_generator.cpp)
target_include_directories(count_multishape PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(count_multishape PRIVATE CUDA::cudart)

# Spécifie que le fichier est en CUDA
#set_source_files_properties(main3.cu PROPERTIES LANGUAGE CUDA)

enable_testing()
add_test(NAME IntervalTests COMMAND intersection_test)
