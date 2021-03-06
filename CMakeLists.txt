cmake_minimum_required(VERSION 3.5)

project(hpc)

set(CMAKE_CXX_FLAGS "-fPIC -std=c++11 -O3 -Wall -msse4.2 -mavx2 -funroll-loops")
#add_executable(mysse "mysse.cpp")
#add_executable(myavx "myavx.cpp")
add_executable(mydist "mydist.cpp")

