import numpy as np
import time
import os

# Disk I/O
# write data of size 12288 fp32 to disk
hidden_dim = 12288
num_iters = 1e3
all_bw = []

curr_path = os.path.dirname(os.path.realpath(__file__))

for i in range(int(num_iters)):
    start = time.time_ns()
    data = np.load(f"{curr_path}/tmp/data_{i}.npy")
    end = time.time_ns()
    latency = end - start
    bw = (hidden_dim * 4 / latency) * 1e3 # in MB/s
    all_bw.append(bw)

print(f"Disk read time mean: {np.mean(all_bw)} MB/s")
print(f"Disk read time std: {np.std(all_bw)} MB/s")
