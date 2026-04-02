# Minimum CMake version required to process this file
cmake_minimum_required(VERSION 3.22.1)

# Declares the project name and version - VERSION keyword must be uppercase
project(navio VERSION 1.0)

# Tell CMake to compile all targets using the C++17 standard
set(CMAKE_CXX_STANDARD 17)

# Make C++17 a hard requirement - do not silently fall back to older standard
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Disable compiler-specific extensions - enforces strict portable C++17
set(CMAKE_CXX_EXTENSIONS OFF)

# Locate Eigen3 on the system - REQUIRED means CMake will error if not found
find_package(Eigen3 REQUIRED)

# Locate OpenCV on the system - REQUIRED means CMake will error if not found
find_package(OpenCV REQUIRED)
#Locate the yaml-cpp
find_package(yaml-cpp REQUIRED)


# Define the executable target and list every source file that compiles into it
# ${PROJECT_NAME} expands to "navio" - keeps target name consistent with project
add_executable(${PROJECT_NAME}
    src/main.cpp
    src/camera.cpp
    src/feature_manager.cpp
    src/frame.cpp
    src/trajectory.cpp
    src/motion_estimator.cpp
    src/visualiser.cpp
    src/landmark_map.cpp
)

# Tell the compiler where to find header files in include/
# PRIVATE means only this target needs these include paths
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Link Eigen and OpenCV libraries to your executable
# Eigen3::Eigen is a modern CMake target that also carries include paths
# ${OpenCV_LIBS} is the variable populated by find_package(OpenCV)
target_link_libraries(${PROJECT_NAME} PRIVATE
    Eigen3::Eigen
    ${OpenCV_LIBS}
    yaml-cpp
)
