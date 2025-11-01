cmake_minimum_required(VERSION 3.10)
project(Intervalix LANGUAGES CXX CUDA)

# Spécifie le standard C++
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 89)

# Trouve CUDA
find_package(CUDA REQUIRED)
find_package(GTest REQUIRED)


# Ajoute l'exécutable
add_library(interval_intersection src/interval_intersection.cu)
target_include_directories(interval_intersection PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
# Macro to check for kernel launch errors and synchronize.
target_link_libraries(interval_intersection PUBLIC ${CUDA_LIBRARIES})


add_executable(exemple_main src/main.cu)
target_link_libraries(exemple_main PRIVATE
	interval_intersection
	PUBLIC ${CUDA_LIBRARIES}
)



add_executable(intersection_test src/test_intersection.cu)
## Lie les bibliothèques CUDA
target_link_libraries(intersection_test ${CUDA_LIBRARIES} interval_intersection GTest::GTest GTest::Main)

add_executable(surface_demo src/surface_demo.cu src/volume_generator.cpp)
target_include_directories(surface_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(surface_demo PRIVATE interval_intersection ${CUDA_LIBRARIES})

# Spécifie que le fichier est en CUDA
#set_source_files_properties(main3.cu PROPERTIES LANGUAGE CUDA)

#enable_testing()
#add_test(NAME IntervalTests COMMAND intersection_test)
