# HangDetect

A CUDA kernel hang detection and monitoring library that provides detailed execution time logging for CUDA kernels.

## Features

- **Kernel Execution Monitoring**: Monitors CUDA kernel launches and tracks execution time
- **Hang Detection**: Detects potential kernel hangs using timeout mechanisms
- **Detailed Logging**: Provides structured JSON logs with kernel information and execution metrics
- **Runtime & Driver API Support**: Supports both CUDA Runtime (`cudaLaunchKernel`) and Driver API (`cuLaunchKernel`) functions
- **User Labels**: Allows custom labeling of kernel executions for better identification

## Current Usage

Currently, HangDetect requires using `LD_PRELOAD` to intercept CUDA kernel launch functions and enable monitoring.

### Building

```bash
cargo build --release
```

### Usage with LD_PRELOAD

1. Build the library:
   ```bash
   cargo build --release
   ```

2. Run your CUDA application with the library preloaded:
   ```bash
   LD_PRELOAD=/path/to/target/release/libhangdetect.so ./your_cuda_application
   ```

### Configuration

The library provides C APIs for configuration:

```c
// Enable or disable hang detection
void hangdetect_set_enable(bool enabled);

// Set a custom label for kernel execution logging
void hangdetect_set_kernel_exec_label(const char* label);
```

## Log Output

The library outputs structured JSON logs containing:

- **Start Events**: When kernels begin execution
- **Complete Events**: When kernels finish execution with duration
- **User Labels**: Custom labels for identification

Example log format:
```json
{"type":"Start","data":{"kern_label":"kernel_name","user_label":"custom_label"}}
{"type":"Complete","data":{"kern_label":"kernel_name","user_label":"custom_label","duration_ms":12.34}}
```

## TODO

### Python API
- [ ] Add Python APIs
- [ ] Add PyTorch example


## License

This project is licensed under the MIT License - see the LICENSE file for details.