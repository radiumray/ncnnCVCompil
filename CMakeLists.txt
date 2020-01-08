# 最低版本要求
cmake_minimum_required(VERSION 3.4.1)

project(ncnnOpencv)

# 设置C++编译版本
set(CMAKE_CXX_STANDARD 11)

# ncnn项目所在路径，需要替换
set(NCNN_DIR /home/ray/ncnn)


# 分别设置ncnn的链接库和头文件
set(NCNN_LIBS ${NCNN_DIR}/build/install/lib/libncnn.a)
set(NCNN_INCLUDE_DIRS ${NCNN_DIR}/build/install/include/ncnn)


# 配置OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${NCNN_INCLUDE_DIRS})

# 配置OpenMP
FIND_PACKAGE( OpenMP REQUIRED)  
if(OPENMP_FOUND)  
    message("OPENMP FOUND")  
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")  
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")  
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")  
endif()  


# 建立链接依赖
add_executable(ncnnOpencv Main.cpp)
target_link_libraries(ncnnOpencv ${NCNN_LIBS})
target_link_libraries(ncnnOpencv ${OpenCV_LIBS})