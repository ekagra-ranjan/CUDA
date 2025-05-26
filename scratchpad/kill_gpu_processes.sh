#!/bin/bash

# Check if at least one GPU number is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <GPU_ID> [<GPU_ID> ...]"
    exit 1
fi

echo "Starting process kill script for GPUs: $@"

# Step 1: Retrieve UUIDs for the specified GPU IDs
declare -A gpu_uuid_map
while IFS=, read -r index uuid; do
    gpu_uuid_map["$index"]="$uuid"
done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)

# Display the mapping of GPU numbers to UUIDs
echo "Retrieved GPU UUID mapping:"
for gpu_id in "${!gpu_uuid_map[@]}"; do
    echo "  GPU $gpu_id -> UUID ${gpu_uuid_map[$gpu_id]}"
done

# Step 2: Gather the UUIDs based on input GPU IDs
target_uuids=()
for gpu_id in "$@"; do
    uuid="${gpu_uuid_map[$gpu_id]}"
    if [ -n "$uuid" ]; then
        echo "Target GPU $gpu_id has UUID $uuid"
        target_uuids+=("$uuid")
    else
        echo "Warning: GPU $gpu_id not found."
    fi
done

# Step 3: Find and list processes to be killed
if [ ${#target_uuids[@]} -gt 0 ]; then
    echo "Identifying processes to be killed on target GPUs..."

    # Find all processes associated with the target UUIDs and store in a list
    processes_to_kill=()
    nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | \
    awk -v uuids="${target_uuids[*]}" '
    BEGIN {
        split(uuids, uuid_arr, " ")
        for (i in uuid_arr) uuid_map[uuid_arr[i]] = 1
    }
    uuid_map[$1] { print $1, $2 }
    ' | while read -r gpu_uuid pid; do
        echo "Process with PID $pid on GPU UUID $gpu_uuid is marked for termination."
        processes_to_kill+=("$pid")
    done

    # Confirm and kill processes
    if [ ${#processes_to_kill[@]} -gt 0 ]; then
        echo "The following processes will be killed:"
        for pid in "${processes_to_kill[@]}"; do
            echo "  PID: $pid"
        done

        # Kill the processes
        for pid in "${processes_to_kill[@]}"; do
            echo "Attempting to kill PID $pid..."
            kill -9 "$pid" && echo "Successfully killed PID $pid" || echo "Failed to kill PID $pid"
        done
    else
        echo "No processes found to kill on the specified GPUs."
    fi

    echo "Process termination complete for GPUs: $@"
else
    echo "No target UUIDs found for the specified GPUs."
fi
