cmake_minimum_required(VERSION 3.10)

# Project name and version
project(SampleProject VERSION 1.0)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find Python3
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import sys; print(sys.version)"
    OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(STATUS "PYTHON_INCLUDE_DIR: ${PYTHON_INCLUDE_DIR}")
    
# Add an executable
add_executable(SampleExecutable main.cpp)
target_link_libraries(SampleExecutable PRIVATE python4.12)
message(STATUS "Python version: ${Python3_VERSION}")
message(STATUS "Python executable: ${Python3_EXECUTABLE}")
message(STATUS "Python lib: ${Python3_LIBRARIES}")
message(STATUS "Python header: ${Python3_INCLUDE_DIRS}")
    
    # # Locate NumPy headers
    # execute_process(
        #     COMMAND "${Python3_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
        #     OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
        #     OUTPUT_STRIP_TRAILING_WHITESPACE
        # )
        
        # # message(STATUS "Python3_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")
message(STATUS "NUMPY_INCLUDE_DIR: ${NUMPY_INCLUDE_DIR}")