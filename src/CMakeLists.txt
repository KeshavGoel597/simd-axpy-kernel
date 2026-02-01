add_library(studentlib main.cpp)
target_include_directories(studentlib PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_features(studentlib PUBLIC cxx_std_17)

# More aggressive flags but not crazy
target_compile_options(studentlib PRIVATE 
    -O3
    -march=native
    -mtune=native
    -mavx2
    -mfma
    -funroll-loops
    -ffast-math
    -ftree-vectorize
)

find_package(OpenMP)
if (OPENMP_FOUND)
    target_link_libraries(studentlib PUBLIC OpenMP::OpenMP_CXX)
endif()