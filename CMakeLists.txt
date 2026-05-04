cmake_minimum_required(VERSION 3.15)
project(rinha CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(preprocess tools/preprocess.cpp)
target_compile_options(preprocess PRIVATE -O2)

add_executable(rinha_api src/main.cpp)
target_compile_options(rinha_api PRIVATE -O3 -march=x86-64-v2 -flto -DNDEBUG -funroll-loops)
target_link_options(rinha_api PRIVATE -flto)
