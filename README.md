# Intra-Node/Inter-Node Communication on the Leonardo CINECA Cluster - Booster Partition

This [project](https://github.com/y3rbiadit0/hpc-final-project/blob/main/report/intra_inter_node_communication_leonardo_cineca_cluster.pdf) explores the communication capabilities of the Leonardo CINECA Cluster - Booster Partition, specifically focusing on intra-node and inter-node communication in a multi-GPU environment. The experiments were conducted using technologies like NVLink, PCIe, MPI, CUDA-Aware MPI, SYCL, and CUDA to measure the bandwidth and latency of data transfers across CPUs and GPUs within and between nodes.
<div align="center">
  <p float="left">
    <img src="https://github.com/user-attachments/assets/ec353395-c97c-443c-bd80-bc4210400ee7" height="300" />
    <img src="https://github.com/user-attachments/assets/97377bdd-012c-4537-a478-ad985721e66f" height="300" />
  </p>
</div>

**Key Findings**

- Intra-node communication: Using NVLink instead of PCIe significantly boosts data transfer speeds by up to 350.46%, making it ideal for multi-GPU configurations.
- Inter-node communication: MPI without CUDA achieves near-theoretical maximum bandwidth for large data transfers, while CUDA-Aware MPI underperforms due to missing GDRCopy support, limiting device-to-device communication efficiency to around 49.34%.
- SYCL vs CUDA: SYCL provides a simpler and more portable programming model with minimal performance trade-offs (0.12% to 2.37%) compared to CUDA, making it a promising option for future hardware portability.


## Libraries Versions
- CUDA: 12.1
- [Intel® oneAPI tools](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html): v2024.2.0
- [Codeplay plugin oneAPI for NVIDIA® GPUs]([https://developer.codeplay.com/products/oneapi/nvidia/2024.2.1/guides/index](https://developer.codeplay.com/products/oneapi/nvidia/2025.0.0/guides/)): v2025.0.0
- NVIDIA Driver: 530.30.02
- Open MPI: 4.1.6 - Compiled with
  – UCX: 1.13.0
  – CUDA: 12.3
- [OSU Benchmarks v7.3](https://mvapich.cse.ohio-state.edu/benchmarks/)

## Compile and Run Project

1. Configure environment
```bash
source config_environment.sh
```

2. Run CMake to build build folder
```bash
cmake -B build
```

3. Build Binaries
```bash
make -C build/
```

4. Batch jobs
```bash
sbatch scripts/run-inter-node-experiments.sh
sbatch scripts/run-intra-node-experiments.sh
sbatch scripts/run-osu-benchmarks.sh
```


### Updates! - GDRCopy Support
You can run osu-benchmarks to compare the results using gdrcopy, rdma and without it.
