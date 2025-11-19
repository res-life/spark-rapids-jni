
# regular

# Devices

## [0] `Quadro RTX 8000`
* SM Version: 750 (PTX Version: 750)
* Number of SMs: 72
* SM Default Clock Rate: 1770 MHz
* Global Memory: 46555 MiB Free / 48593 MiB Total
* Global Memory Bus Peak: 672 GB/sec (384-bit DDR @7001MHz)
* Max Shared Memory: 64 KiB/SM, 48 KiB/Block
* L2 Cache Size: 6144 KiB
* Maximum Active Blocks: 16/SM
* Maximum Active Threads: 1024/SM, 1024/Block
* Available Registers: 65536/SM, 65536/Block
* ECC Enabled: No

# Log

```
Run:  [1/1] projects_baseline [Device=0 num_rows=1048576 num_cols=400]
Pass: Cold: 7.296161ms GPU, 7.299951ms CPU, 8.87s total GPU, 8.91s total wall, 1216x 
```

# Benchmark Results

## projects_baseline

### [0] Quadro RTX 8000

| num_rows | num_cols | Samples | CPU Time | Noise | GPU Time | Noise |
|----------|----------|---------|----------|-------|----------|-------|
|  1048576 |      400 |   1216x | 7.300 ms | 6.80% | 7.296 ms | 6.81% |


# fused

# Devices

## [0] `Quadro RTX 8000`
* SM Version: 750 (PTX Version: 750)
* Number of SMs: 72
* SM Default Clock Rate: 1770 MHz
* Global Memory: 46555 MiB Free / 48593 MiB Total
* Global Memory Bus Peak: 672 GB/sec (384-bit DDR @7001MHz)
* Max Shared Memory: 64 KiB/SM, 48 KiB/Block
* L2 Cache Size: 6144 KiB
* Maximum Active Blocks: 16/SM
* Maximum Active Threads: 1024/SM, 1024/Block
* Available Registers: 65536/SM, 65536/Block
* ECC Enabled: No

# Log

```
Run:  [1/1] projects_fused [Device=0 num_rows=1048576 num_cols=400]
Pass: Cold: 7.377296ms GPU, 7.385983ms CPU, 5.31s total GPU, 5.34s total wall, 720x 
```

# Benchmark Results

## projects_fused

### [0] Quadro RTX 8000

| num_rows | num_cols | Samples | CPU Time | Noise | GPU Time | Noise |
|----------|----------|---------|----------|-------|----------|-------|
|  1048576 |      400 |    720x | 7.386 ms | 5.99% | 7.377 ms | 5.80% |
