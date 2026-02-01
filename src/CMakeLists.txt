add_library(studentlib main.cpp)
target_include_directories(studentlib PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_features(studentlib PUBLIC cxx_std_17)

# NUCLEAR optimization flags
target_compile_options(studentlib PRIVATE 
    -Ofast
    -march=native
    -mtune=native
    -mavx2
    -mfma
    -funroll-loops
    -funroll-all-loops
    -ffast-math
    -fno-math-errno
    -ffinite-math-only
    -fno-trapping-math
    -fassociative-math
    -freciprocal-math
    -fno-signed-zeros
    -ftree-vectorize
    -ftree-loop-vectorize
    -ftree-slp-vectorize
    -fvect-cost-model=unlimited
    -flto
    -fomit-frame-pointer
    -fno-stack-protector
    -fprefetch-loop-arrays
)

find_package(OpenMP REQUIRED)
target_link_libraries(studentlib PUBLIC OpenMP::OpenMP_CXX)

set_target_properties(studentlib PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
target_link_options(studentlib PRIVATE -flto)
