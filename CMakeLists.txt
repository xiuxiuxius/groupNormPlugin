#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

cmake_minimum_required(VERSION 3.10)

project(gnTest)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

# TODO(Call for PR): make cmake compatible with Windows
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
# TODO(Call for PR): make TRT path configurable from command line
include_directories(/data/TensorRT-8.5.1.7/include/)
link_directories(/data/TensorRT-8.5.1.7/lib/)

# opencv
include_directories(/data/limm/opencv-4.5.0/include/)
link_directories(/data/limm/opencv-4.5.0/lib/)


include_directories(${PROJECT_SOURCE_DIR}/src/)
include_directories(${PROJECT_SOURCE_DIR}/plugin/)
file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cu)
file(GLOB_RECURSE PLUGIN_SRCS ${PROJECT_SOURCE_DIR}/plugin/*.cu ${PROJECT_SOURCE_DIR}/plugin/*.cpp)

message(${PLUGIN_SRCS})
add_library(myplugins SHARED ${PLUGIN_SRCS})
target_link_libraries(myplugins nvinfer cudart)


add_executable(test_groupNorm test_groupNorm.cpp ${SRCS})
target_link_libraries(test_groupNorm nvinfer)
target_link_libraries(test_groupNorm cudart)
target_link_libraries(test_groupNorm myplugins)
target_link_libraries(test_groupNorm opencv_core)
target_link_libraries(test_groupNorm opencv_highgui)
target_link_libraries(test_groupNorm opencv_imgproc)
target_link_libraries(test_groupNorm opencv_imgcodecs)
target_link_libraries(test_groupNorm opencv_dnn)
