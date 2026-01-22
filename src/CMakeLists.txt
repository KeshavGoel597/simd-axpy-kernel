add_library(studentlib main.cpp)
target_include_directories(studentlib PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_features(studentlib PUBLIC cxx_std_17)
target_compile_options(studentlib PRIVATE -O3 -ffast-math -march=native -funroll-loops)

find_package(OpenMP)
if (OPENMP_FOUND)
    target_link_libraries(studentlib PUBLIC OpenMP::OpenMP_CXX)
endif()
