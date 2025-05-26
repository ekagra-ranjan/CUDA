import numpy as np
import time
import os

# Disk I/O
# write data of size 12288 fp32 to disk
hidden_dim = 12288
num_iters = 1e4
all_bw = []

curr_path = os.path.dirname(os.path.realpath(__file__))

for i in range(int(num_iters)):
    data = np.random.rand(hidden_dim).astype(np.float32)
    # import pdb; pdb.set_trace()
    start = time.time_ns()
    np.save(f"{curr_path}/tmp/data_{i}.npy", data)
    end = time.time_ns()
    latency = end - start
    bw = (hidden_dim * 4 / latency) * 1e3 # in MB/s
    all_bw.append(bw)

print(f"Disk write time mean: {np.mean(all_bw)} MB/s")
print(f"Disk write time std: {np.std(all_bw)} MB/s")
