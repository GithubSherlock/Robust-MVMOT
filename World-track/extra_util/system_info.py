import platform
import psutil
import sys

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class SystemInfo:
    @staticmethod
    def get_os_info():
        """Getting Operating System Info"""
        return {
            "Operating system": platform.system(),
            "System version": platform.version(),
            "System architecture": platform.machine(),
            "Processor": platform.processor(),
            "Python version": sys.version
        }

    @staticmethod
    def get_cpu_info():
        """Getting CPU Info"""
        cpu_info = {
            "CPU Physical Cores": psutil.cpu_count(logical=False),
            "CPU Logical Cores": psutil.cpu_count(logical=True),
            "CPU usage ratio": f"{psutil.cpu_percent()}%"
        }
        return cpu_info

    @staticmethod
    def get_memory_info():
        """Getting Memory Info"""
        memory = psutil.virtual_memory()
        return {
            "Total memory": f"{memory.total / (1024**3):.2f} GB",
            "Used memory": f"{memory.used / (1024**3):.2f} GB",
            "memory usage ratio": f"{memory.percent}%"
        }

    @staticmethod
    def get_disk_info():
        """Getting Disk Info"""
        disk_info = {}
        for partition in psutil.disk_partitions():
            try:
                partition_usage = psutil.disk_usage(partition.mountpoint)
                disk_info[partition.mountpoint] = {
                    "Total capacity": f"{partition_usage.total / (1024**3):.2f} GB",
                    "Used capacity": f"{partition_usage.used / (1024**3):.2f} GB",
                    "Usage ratio": f"{partition_usage.percent}%"
                }
            except:
                continue
        return disk_info

    @staticmethod
    def get_gpu_info():
        """Getting GPU Info"""
        if not TORCH_AVAILABLE:
            return {"GPU Info": "PyTorch not installed, can't get GPU information"}

        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info["Is CUDA available"] = "Yes"
            gpu_info["CUDA version"] = torch.version.cuda
            gpu_info["Number of GPUs"] = torch.cuda.device_count()
            for i in range(torch.cuda.device_count()):
                gpu_info[f"GPU {i}"] = {
                    "Name": torch.cuda.get_device_name(i),
                    "Total graphics memory": f"{torch.cuda.get_device_properties(i).total_memory / (1024**2):.2f} MB"
                }
        else:
            gpu_info["Is CUDA available"] = "No"
        return gpu_info

    @staticmethod
    def print_info():
        """Print all system information"""
        print("\n=== System Info ===")
        for k, v in SystemInfo.get_os_info().items():
            print(f"{k}: {v}")

        print("\n=== CPU Info ===")
        for k, v in SystemInfo.get_cpu_info().items():
            print(f"{k}: {v}")

        print("\n=== Memory Info ===")
        for k, v in SystemInfo.get_memory_info().items():
            print(f"{k}: {v}")

        print("\n=== Disk Info ===")
        for mount, info in SystemInfo.get_disk_info().items():
            print(f"Disk {mount}:")
            for k, v in info.items():
                print(f"{k}: {v}")

        print("\n=== GPU Info ===")
        for k, v in SystemInfo.get_gpu_info().items():
            if isinstance(v, dict):
                print(f"\n{k}:")
                for sub_k, sub_v in v.items():
                    print(f"{sub_k}: {sub_v}")
            else:
                print(f"{k}: {v}")

if __name__ == "__main__":
    SystemInfo.print_info()
    # os_info = SystemInfo.get_os_info()['Operating system']
    # print(os_info)
