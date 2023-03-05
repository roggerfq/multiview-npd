cmake_minimum_required(VERSION 2.8)  
project(two-Nested-NPD)    

#find_package( OpenCV REQUIRED )
#set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
add_definitions(-DFOO)
add_library(mylib STATIC detector.cpp)
add_executable(main main.cpp)

target_link_libraries(main mylib ${OpenCV_LIBS})
