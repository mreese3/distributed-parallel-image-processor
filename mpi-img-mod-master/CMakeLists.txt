project(mpiimg)
cmake_minimum_required(VERSION 2.8)
find_package(OpenCV REQUIRED)
set(MPI_COMPILER /usr/lib64/mpich/bin/mpicc)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_C_FLAGS -O3)
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH} ${OpenCV_INCLUDE_PATH})
aux_source_directory(. SRC_LIST)
add_executable(${PROJECT_NAME} ${SRC_LIST})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES} ${MPI_LIBRARIES})
