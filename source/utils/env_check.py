"""Environment inspection and version locking utility."""

import json
import os
import platform
import sys
from datetime import datetime, timezone
from typing import Any, Dict


def inspect_environment() -> Dict[str, Any]:
    """Collect runtime environment metadata and package versions."""
    packages = [
        "python",
        "torch",
        "torch_geometric",
        "toponetx",
        "gudhi",
        "networkx",
        "numpy",
        "scipy",
        "sklearn",
        "yaml",
        "pytest",
    ]

    package_versions: Dict[str, Any] = {}
    for pkg in packages:
        if pkg == "python":
            package_versions["python"] = sys.version.split()[0]
            continue
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "installed")
            package_versions[pkg] = {
                "installed": True,
                "version": str(version),
            }
        except ImportError:
            package_versions[pkg] = {
                "installed": False,
                "version": None,
            }

    # Hardware acceleration detection
    hardware_accel: Dict[str, Any] = {
        "cuda_available": False,
        "cuda_device_count": 0,
        "mps_available": False,
        "device": "cpu",
    }

    if package_versions.get("torch", {}).get("installed"):
        try:
            import torch
            if torch.cuda.is_available():
                hardware_accel["cuda_available"] = True
                hardware_accel["cuda_device_count"] = torch.cuda.device_count()
                hardware_accel["device"] = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                hardware_accel["mps_available"] = True
                hardware_accel["device"] = "mps"
        except Exception:
            pass

    env_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_implementation": platform.python_implementation(),
            "python_version": sys.version,
        },
        "packages": package_versions,
        "hardware": hardware_accel,
    }
    return env_data


def write_environment_lock(output_path: str = "project/environment_lock.json") -> Dict[str, Any]:
    """Inspect environment and write lock file."""
    data = inspect_environment()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return data


if __name__ == "__main__":
    lock = write_environment_lock()
    print(f"Environment lock written to project/environment_lock.json")
    print(f"Python: {lock['packages']['python']}")
    print(f"Device: {lock['hardware']['device']}")
    for k, v in lock["packages"].items():
        if k != "python":
            status = v["version"] if v["installed"] else "NOT INSTALLED"
            print(f"  {k}: {status}")
