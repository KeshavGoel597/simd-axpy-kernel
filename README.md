# AXPY Template

Template aligned with `gemm_template` for the BLAS1 operation `y = alpha * x + y`.

## Build & run
```bash
./runner_script.sh
# manual
mkdir -p build && cd build
cmake .. && make -j
./bin/tester 1048576 2.5 42
```

`solution::compute` takes paths to `x`, `y`, scalar `alpha`, and vector length `n`, and returns a path to the output vector file.
