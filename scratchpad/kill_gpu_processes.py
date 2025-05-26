import subprocess
import os
import signal
import sys

log=False
def logger(string):
    if log:
        print(string)


def get_gpu_uuid_map():
    """Retrieve the UUID for each GPU index."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        stdout=subprocess.PIPE,
        text=True,
    )
    gpu_uuid_map = {}
    for line in result.stdout.strip().splitlines():
        index, uuid = line.split(", ")
        gpu_uuid_map[index] = uuid
    return gpu_uuid_map

def get_processes_by_uuid(target_uuids):
    """Retrieve processes running on specified GPU UUIDs."""
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        stdout=subprocess.PIPE,
        text=True,
    )
    processes = []
    for line in result.stdout.strip().splitlines():
        gpu_uuid, pid = line.split(", ")
        if gpu_uuid in target_uuids:
            processes.append((gpu_uuid, int(pid)))
    return processes

def main(gpu_ids):
    # Step 1: Get the UUIDs for the specified GPU IDs
    gpu_uuid_map = get_gpu_uuid_map()
    logger("Retrieved GPU UUID mapping:")
    for gpu_id, uuid in gpu_uuid_map.items():
        logger(f"  GPU {gpu_id} -> UUID {uuid}")

    # Step 2: Find target UUIDs based on the specified GPU IDs
    target_uuids = [gpu_uuid_map[gpu_id] for gpu_id in gpu_ids if gpu_id in gpu_uuid_map]
    if not target_uuids:
        logger("No valid UUIDs found for the specified GPUs.")
        return

    logger(f"\nTarget UUIDs for GPUs: {target_uuids}")

    # Step 3: Get and list processes on the target UUIDs
    processes_to_kill = get_processes_by_uuid(target_uuids)
    if not processes_to_kill:
        logger("No processes found to kill on the specified GPUs.")
        return

    logger("\nThe following processes will be killed:")
    for gpu_uuid, pid in processes_to_kill:
        logger(f"  GPU UUID: {gpu_uuid}, PID: {pid}")

    # Confirm before killing
    if log:
        confirm = input("\nAre you sure you want to kill these processes? (y/n): ")
        if confirm.lower() != 'y':
            logger("Process termination canceled.")
            return

    # Step 4: Kill each process
    for gpu_uuid, pid in processes_to_kill:
        try:
            logger(f"Killing process PID {pid} on GPU UUID {gpu_uuid}...")
            os.kill(pid, signal.SIGKILL)
            logger(f"Successfully killed PID {pid}")
        except ProcessLookupError:
            logger(f"Process PID {pid} does not exist.")
        except PermissionError:
            logger(f"Permission denied to kill PID {pid}. Run the script as root.")
        except Exception as e:
            logger(f"Failed to kill PID {pid}: {e}")

    logger("Process termination complete.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger("Usage: python kill_gpu_processes.py <GPU_ID> [<GPU_ID> ...]")
        sys.exit(1)

    gpu_ids = sys.argv[1:]
    print(f"gpu_ids: {gpu_ids}")
    main(gpu_ids)
