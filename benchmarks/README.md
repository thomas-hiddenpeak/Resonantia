# R0 realtime index diagnostic

`realtime_index_benchmark.cu` is a standalone feasibility probe for a persistent
CUDA exact Flat Index. It measures FP32 SGEMM distance calculation, deterministic
exact top-k selection, RVC inverse-square weighted reconstruction, and
`index_rate` blending. The database, row norms, query batch, distance/top-k/weight,
retrieved/blended output buffers, and an explicit cuBLAS workspace remain resident;
the timed call performs no runner-owned allocation or host/device transfer.

This executable is deliberately not wired into CMake, installation, or CTest. Its
output is diagnostic R0 evidence, **not** product acceptance, audio-quality
evidence, or an end-to-end realtime SLO result. Do not commit generated indexes,
query fixtures, binaries, or result archives to this directory.

## Manual build

The following command is the reproducible Jetson Orin (`sm_87`) build used for
the R0 characterization. Adjust the CUDA root and architecture explicitly when
running on another target.

```sh
PATH=/usr/local/cuda-13.2/bin:/usr/local/bin:/usr/bin:/bin \
/usr/local/cuda-13.2/bin/nvcc -O3 -std=c++20 -arch=sm_87 \
  -isystem /usr/local/cuda-13.2/targets/sbsa-linux/include \
  benchmarks/realtime_index_benchmark.cu \
  -L/usr/local/cuda-13.2/targets/sbsa-linux/lib -lcublas -lcudart \
  -o /tmp/realtime_index_benchmark
```

Synthetic 10k-row smoke, with one query and the current offline pipeline's
top-8 retrieval shape:

```sh
/tmp/realtime_index_benchmark \
  --rows 10000 --dim 768 --nq 1 --k 8 --index-rate 0.75 \
  --seed 305419896 --warmup 10 --reps 100
```

Existing Flat Index plus queries taken from a second Flat Index:

```sh
/tmp/realtime_index_benchmark \
  --index /path/to/database.index --query-index /path/to/queries.index \
  --nq 8 --k 8 --index-rate 0.75 --warmup 10 --reps 100
```

`--queries` is accepted as an alias for `--nq`. Exactly one of `--rows` and
`--index` is required. When a file-backed database has no `--query-index`, the
probe deterministically uses the first `nq` database rows with a small perturbation
to the first component. Synthetic queries use a seed derived from `--seed`.

## Flat Index and safety envelope

The accepted file layout is exactly `[uint64 rows][uint64 dimension][rows *
dimension float32]`, little-endian and row-major, with no trailing bytes. The
probe hashes the complete file with SHA-256 and rejects malformed size, non-finite
or FP32-squared-L2-unsafe payloads, and query-file dimension mismatches.

The command line is strict: unknown and duplicate options, partially parsed
numbers, non-finite rates, and out-of-envelope values fail before allocation.

| Parameter or resource | Limit |
| --- | ---: |
| Database rows | 1,000,000 |
| Dimension | 4,096 |
| Query batch (`nq`) | 16 |
| Exact `k` | 16 and no larger than `rows` |
| Warmup / measured calls | 10,000 each; measured calls must be positive |
| Database payload | 2 GiB |
| Distance workspace | 512 MiB |
| Runner-owned persistent device buffers | 3 GiB |

`index_rate` must be in `(0, 1]`: this is an indexed-path probe, so the no-index
case is intentionally outside its scope. Device allocation errors can still occur
below these caps when other workloads consume GPU memory.

## Result contract

A successful run writes exactly one JSON object to standard output with schema
`resonantia.realtime_index_benchmark.v1`. It includes:

- OS, CUDA driver/runtime, GPU identity and compute capability;
- profile/backend and explicit diagnostic/product-acceptance flags;
- index shape, `k`, rate, full file hash or synthetic seed, and query identity;
- batch, warmup/repetition counts and initialization time;
- CUDA-event and synchronized wall-clock p50/p95/p99/max latency;
- database, distance-workspace, explicit cuBLAS-workspace, and total runner-owned
  persistent device bytes; and
- CPU/GPU exact top-k matches, mixed absolute-or-relative distance/weight gates,
  pure retrieved-vector gates independent of `index_rate`, and blended-output gates.

The emitted threshold object is part of the JSON evidence: distance tolerance is
`abs <= 5e-4 OR rel <= 5e-5`, weight tolerance is `abs <= 1e-5 OR rel <= 1e-4`,
and both pure retrieved and blended vectors require max absolute error `<= 1e-4`
and cosine `>= 0.999999`. Consequently, a small `index_rate` cannot hide a broken
retrieval or weighting implementation.

Exit status is `0` only when the CPU/GPU comparison passes, `2` for a completed
comparison mismatch, and `1` for CLI, input, CUDA, or runtime failure. Human-readable
errors go to standard error. The CPU reference runs after timing and is reported as
a total validation cost; it is not a competing steady-state latency sample.

For decision-grade measurements, retain the one-line JSON together with the exact
binary/source revision, command, power mode, clocks, thermal state, and workload
isolation. This probe does not model pipeline contention, per-hop scheduling,
long-run thermal behavior, drop/xrun behavior, or model/audio quality. Synthetic
results are low-level feasibility evidence only; product characterization requires
manifest-pinned real index, model, and audio assets plus the full pipeline gates.
