cmake_minimum_required(VERSION 3.15)
project(rinha_api CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Otimizações de velocidade agressivas
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -flto -DNDEBUG")

add_executable(rinha_api src/main.cpp)

# Opcional: linkar com pthread se for usar threads
target_link_libraries(rinha_api pthread)
