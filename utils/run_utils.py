
import os
from datetime import datetime
from typing import Optional, Union



def ensure_dir(path: str) -> None:
    #Create directory if it does not exist.
    os.makedirs(path, exist_ok=True)


def make_run_dir(
    base_dir: str,
    run_name: str,
    date_fmt: str = "%Y%m%d_%H%M%S",
) -> str:
    #Create and return a directory for a specific run.

    timestamp = datetime.now().strftime(date_fmt)
    run_dir = os.path.join(base_dir, run_name, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def make_run_name(
    model: str,
    digit: Union[int, str],
    nu: float,
    date_fmt: str = "%Y%m%d_%H%M%S",
) -> str:
    timestamp = datetime.now().strftime(date_fmt)
    return f"{model}_digit{digit}_nu{nu}_{timestamp}"
