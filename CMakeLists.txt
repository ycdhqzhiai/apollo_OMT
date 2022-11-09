cmake_minimum_required(VERSION 3.6)

project(apollo_trakcer)

SET(CMAKE_BUILD_TYPE "Debug")
add_definitions(-w)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -mavx -msse4.1 -mavx2 -D_GLIBCXX_USE_CXX11_ABI=1")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall  -fPIC -fopenmp -Ofast -g -Wfatal-errors")
option (USE_GPU "Use tutorial provided math implementation" ON)


find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(absl REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(fastrtps REQUIRED)
find_package(fastcdr REQUIRED)

set(SELF_DRIVING_ENV_DIR $ENV{SELF_DRIVING_ENV})

include_directories(
    ./
    ${EIGEN3_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)
get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)

message(${TRT_DIR}/lib)

link_directories(
    ${SELF_DRIVING_ENV_DIR}/lib
    /usr/local/cuda-11.1/targets/x86_64-linux/lib
)

file(GLOB_RECURSE SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} FOLLOW_SYMLINKS
    ./common/*.h
    ./common/*.cc
    ./base/*.h
    ./base/*.cc
    ./lib/*.h
    ./lib/*.cc    
    ./camera/*.cc
    ./camera/*.cu
    ./camera/*.h
    ./proto/*.h
    ./proto/*.cc   
    ./inference/*.cc
    ./inference/*.cu
    ./inference/*.h
    main.cc
)

cuda_add_executable(${PROJECT_NAME}
    ${SRC_LIST}
)

target_link_libraries(apollo_trakcer ${OpenCV_LIBS} ${CUDA_LIBRARIES} 
	${YAML_CPP_LIBRARIES} absl::strings fastcdr fastrtps
	pthread protobuf glog gflags boost_filesystem boost_thread
	cudnn cublas nppidei nppicc nppig cyber gtest)
add_definitions(-O0 -pthread)
add_definitions(-DUSE_GPU)

