/usr/local/NVIDIA-Nsight-Compute-2021.1/ncu \
    -f \
    -o transpose2d \
    --section ComputeWorkloadAnalysis \
    --section InstructionStats \
    --section LaunchStats \
    --section Occupancy \
    --section SpeedOfLight \
    --section SpeedOfLight_RooflineChart \
    --section MemoryWorkloadAnalysis \
    --section MemoryWorkloadAnalysis_Chart \
    --section SchedulerStats \
    --section WarpStateStats \
    build/bin/transpose2d 1024 1024 1024 1024 1 5