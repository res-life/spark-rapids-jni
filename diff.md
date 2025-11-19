
# regular

## Devices

### [0] `NVIDIA RTX 3500 Ada Generation Laptop GPU`
* SM Version: 890 (PTX Version: 890)
* Number of SMs: 40
* SM Default Clock Rate: 1545 MHz
* Global Memory: 11854 MiB Free / 12002 MiB Total
* Global Memory Bus Peak: 432 GB/sec (192-bit DDR @9001MHz)
* Max Shared Memory: 100 KiB/SM, 48 KiB/Block
* L2 Cache Size: 49152 KiB
* Maximum Active Blocks: 24/SM
* Maximum Active Threads: 1536/SM, 1024/Block
* Available Registers: 65536/SM, 65536/Block
* ECC Enabled: No

## Log

```
Run:  [1/1] projects_baseline [Device=0 num_rows=1048576 num_cols=400]
Warn: Current measurement timed out (15.00s) while over noise threshold (73.08% > 0.50%)
Pass: Cold: 24.060566ms GPU, 24.070650ms CPU, 14.82s total GPU, 15.00s total wall, 616x 
```

## Benchmark Results

### projects_baseline

#### [0] NVIDIA RTX 3500 Ada Generation Laptop GPU

| num_rows | num_cols | Samples | CPU Time  | Noise  | GPU Time  | Noise  |
|----------|----------|---------|-----------|--------|-----------|--------|
|  1048576 |      400 |    616x | 24.071 ms | 73.08% | 24.061 ms | 73.08% |

# fused

## Devices

### [0] `NVIDIA RTX 3500 Ada Generation Laptop GPU`
* SM Version: 890 (PTX Version: 890)
* Number of SMs: 40
* SM Default Clock Rate: 1545 MHz
* Global Memory: 11854 MiB Free / 12002 MiB Total
* Global Memory Bus Peak: 432 GB/sec (192-bit DDR @9001MHz)
* Max Shared Memory: 100 KiB/SM, 48 KiB/Block
* L2 Cache Size: 49152 KiB
* Maximum Active Blocks: 24/SM
* Maximum Active Threads: 1536/SM, 1024/Block
* Available Registers: 65536/SM, 65536/Block
* ECC Enabled: No

## Log

```
Run:  [1/1] projects_fused [Device=0 num_rows=1048576 num_cols=400]
Warn: Current measurement timed out (15.01s) while over noise threshold (5.39% > 0.50%)
Pass: Cold: 24.742801ms GPU, 24.750920ms CPU, 14.87s total GPU, 15.01s total wall, 601x 
```

## Benchmark Results

### projects_fused

#### [0] NVIDIA RTX 3500 Ada Generation Laptop GPU

| num_rows | num_cols | Samples | CPU Time  | Noise | GPU Time  | Noise |
|----------|----------|---------|-----------|-------|-----------|-------|
|  1048576 |      400 |    601x | 24.751 ms | 5.39% | 24.743 ms | 5.39% |
