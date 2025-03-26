cmake_minimum_required(VERSION 3.10)
project(Intervalix LANGUAGES CXX CUDA)

# Spécifie le standard C++
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_FLAGS "-arch=sm_86") 

# Trouve CUDA
find_package(CUDA REQUIRED)

# Ajoute l'exécutable
add_executable(intervalix main6.cu)

# Lie les bibliothèques CUDA
target_link_libraries(intervalix ${CUDA_LIBRARIES})

# Spécifie que le fichier est en CUDA

set_source_files_properties(main3.cu PROPERTIES LANGUAGE CUDA)
