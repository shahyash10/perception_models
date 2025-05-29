import grp
import itertools
import json
import os
import pwd
import re
import shutil
import subprocess
from datetime import UTC, datetime
from functools import cache
from itertools import product
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def get_launch_date_str() -> str:
    return f"{datetime.now(tz=UTC):%Y_%m_%d_%H_%M_%S}"


def check_is_slurm_job() -> bool:
    return "SLURM_JOBID" in os.environ


def get_cwd_git_commit_hash() -> str | None:
    try:
        output = subprocess.check_output(
            ["git", "describe", "--match=''", "--always", "--abbrev=7", "--dirty"]
        )
        commit_hash = output.decode("utf-8").strip()
        return commit_hash
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def read_exclude_from_path(exclude_path: str) -> str:
    """Read exclude nodes from a list of comma-separated paths."""
    excludes = []
    for exclude_file in exclude_path.split(","):
        exclude_file_path = Path(exclude_file)
        assert exclude_file_path.exists(), (exclude_file, Path.cwd())
        with Path.open(exclude_file_path) as fp:
            excludes.append(fp.readlines()[0].strip())
    host_prefix = None
    nodes = []
    for exclude in excludes:
        assert exclude.endswith("]")
        exclude = exclude[:-1]
        this_host_prefix, these_nodes = exclude.split("[")
        host_prefix = this_host_prefix or host_prefix
        assert this_host_prefix == host_prefix, (this_host_prefix, host_prefix)
        nodes.append(these_nodes)
    merged_nodes = ",".join(nodes)
    return f"{host_prefix}[{merged_nodes}]"


def retrieve_max_time_per_partition() -> dict[str, int]:
    # retrieve partition max times (a bit slow)
    sinfo = json.loads(subprocess.check_output(["sinfo", "--json"]))["sinfo"]
    max_times: dict[str, int] = {}

    for info in sinfo:
        if info["partition"]["maximums"]["time"]["infinite"]:
            max_times[info["partition"]["name"]] = 14 * 24 * 60  # 14 days
        else:
            max_times[info["partition"]["name"]] = info["partition"]["maximums"][
                "time"
            ]["number"]  # in minutes

    return max_times


def get_slurm_job_names(user: str, *, return_job_id: bool = False) -> list[str]:
    """Return SLURM job names for a given user."""
    assert len(user) > 0
    squeue_cmd = ["squeue", "-a", "--noheader", "-u", user]
    if return_job_id:
        squeue_cmd += ["-o", "%j %i"]
    else:
        squeue_cmd += ["-o", "%j"]
    job_names = subprocess.check_output(squeue_cmd, text=True).strip()
    return [x for x in job_names.split("\n") if x != ""]


def get_job_names_and_ids(user: str) -> list[tuple[str, int]]:
    # retrieve list of jobs with their names and IDs
    job_names: list[tuple[str, int]] = []
    for s in get_slurm_job_names(user=user, return_job_id=True):
        x = s.split()
        assert len(x) == 2
        job_name = x[0]
        job_id_str = re.sub(r"_\[[0-9%-]+]$", "", x[1])  # remove job array element IDs
        job_id_str = re.sub(r"_[0-9]+$", "", job_id_str)
        assert job_id_str.isdigit(), (x[1], job_id_str)
        job_id = int(job_id_str)
        job_names.append((job_name, job_id))
    return job_names


def get_conda_python_bin(anaconda: str) -> Path:
    return Path(anaconda) / "bin/python"


def get_current_conda() -> Path:
    if "CONDA_PREFIX" in os.environ:
        return os.environ["CONDA_PREFIX"]
    else:
        conda_python_bin = shutil.which("python")
        conda_env = Path(conda_python_bin).parent.parent
        return str(conda_env)


def check_conda(anaconda: Path) -> None:
    assert Path(anaconda).is_dir()
    assert get_conda_python_bin(anaconda).is_file()


def get_user_group(path: str) -> tuple[str, str]:
    stat_info = os.stat(path)
    user_id = stat_info.st_uid
    group_id = stat_info.st_gid
    user_name = pwd.getpwuid(user_id).pw_name
    group_name = grp.getgrgid(group_id).gr_name
    return user_name, group_name


def copy_dir(input_dir: str, output_dir: str) -> None:
    print(f"Copying : {input_dir}\nto      : {output_dir}...")
    assert Path(input_dir).is_dir(), f"{input_dir} is not a directory"
    assert Path(output_dir).is_dir(), f"{output_dir} is not a directory"
    user, group = get_user_group(Path(output_dir).parent)
    rsync_cmd = (
        f"rsync -ar --chown={user}:{group} --copy-links "
        f"--exclude='.git/' "
        f"--exclude checkpoints "
        f"--exclude notebooks "
        f"--exclude '*notebooks' "
        f"--exclude '*.ipynb' "
        f"--exclude '*.stderr' "
        f"--exclude '*.stdout' "
        f"--exclude '*.out' "
        f"--exclude '*.log' "
        f"--exclude '*.pth' "
        f"{input_dir}/ {output_dir}"
    )
    print(f"Copying command: {rsync_cmd}")
    subprocess.call([rsync_cmd], shell=True)
    print("Copy done.")


def get_local_diff() -> str:
    # Check if the current directory is a git repository
    is_git_repo = subprocess.check_output(
        ["git", "rev-parse", "--is-inside-work-tree"],
        encoding="utf-8",
        stderr=subprocess.DEVNULL,
    ).strip()
    if is_git_repo != "true":
        raise RuntimeError("Not inside a git repo")
    # Get the current branch name
    current_branch = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        encoding="utf-8",
        stderr=subprocess.DEVNULL,
    ).strip()
    try:
        # Try to get the upstream branch
        upstream_branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", f"{current_branch}@{{upstream}}"],
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        # No upstream branch set. Falling back to 'main'.
        upstream_branch = "main"
    diff = subprocess.check_output(
        ["git", "diff", upstream_branch],
        encoding="utf-8",
        stderr=subprocess.DEVNULL,
    )
    return diff


def dict_merge(tgt: dict, src: dict, *, override: bool = False) -> None:
    for k in src:
        if k in tgt and isinstance(tgt[k], dict) and isinstance(src[k], dict):
            dict_merge(tgt[k], src[k])
        else:
            if k in tgt and not override:
                continue
            tgt[k] = src[k]


def expand_dict(d: dict) -> list[dict]:
    v_prod = product(*d.values())
    return [dict(zip(d.keys(), v, strict=True)) for v in v_prod]


def recursive_expand_dict(d: dict) -> list[dict]:
    result = {}
    for key, value in d.items():
        if isinstance(value, dict):
            result[key] = recursive_expand_dict(value)
        else:
            result[key] = value
    return expand_dict(result)


def compute_grid(grid: dict) -> list[dict[str, Any]]:
    sub_keys = [k for k in grid if k.startswith("SUBSWEEP")]
    if sub_keys:
        sub_grids = itertools.product(*[grid.pop(k).values() for k in sub_keys])
        results = []
        for subsweep in sub_grids:
            combined_grid = {**grid}
            for params in subsweep:
                combined_grid.update(params)
            results.extend(compute_grid(combined_grid))
        return results

    return recursive_expand_dict(grid)


def get_swept_params(grid: dict | DictConfig) -> list[dict[str, Any]]:
    grid = OmegaConf.to_container(grid)
    sweep_grid = grid.pop("sweep", {})
    params = compute_grid(sweep_grid)
    for p in params:
        dict_merge(p, grid, override=False)
    return params


def get_job_ping(path: Path) -> str | None:
    minute, hour = 60, 3600
    files = list(path.glob("*"))
    if not files:
        return None
    newest_file = max(files, key=lambda f: f.stat().st_mtime)
    newest_file_mtime = newest_file.stat().st_mtime
    time_passed = datetime.now(tz=UTC).timestamp() - newest_file_mtime
    if time_passed < minute:
        return f"{int(time_passed)}s"
    if minute <= time_passed < hour:
        return f"{int(time_passed / minute)}m"
    return f"{int(time_passed / hour)}h"


@cache
def get_slurm_job_status(job_id: str) -> str:
    result = subprocess.run(
        ["sacct", "-j", job_id, "--format=State", "--noheader"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.split("\n")[0].strip()


def is_completed(job_id: str) -> bool:
    return "completed" in get_slurm_job_status(job_id).lower()


def is_running(job_id: str) -> bool:
    return "run" in get_slurm_job_status(job_id).lower()


def is_failed(job_id: str) -> bool:
    return "fail" in get_slurm_job_status(job_id).lower()


def is_pending(job_id: str) -> bool:
    return "pending" in get_slurm_job_status(job_id).lower()


def is_cancelled(job_id: str) -> bool:
    return "cancel" in get_slurm_job_status(job_id).lower()


def print_table(table: list[tuple]) -> None:
    column_widths = [
        max(len(str(item)) for item in column) for column in zip(*table, strict=True)
    ]
    row_format = " | ".join(f"{{:<{width}}}" for width in column_widths)
    for row in table:
        print(row_format.format(*row))


def confirm(msg: str) -> bool:
    return input(msg).lower() == "yes"


def confirm_rmtree(msg: str, path: Path) -> bool:
    if confirm(msg):
        shutil.rmtree(str(path))
        print(f"Directory '{path}' has been deleted.")
        return True
    print("Operation cancelled.")
    return False


def is_sweep_dir(path: Path) -> bool:
    return (path / "sweep.yaml").is_file()


def get_sweep_dirs(path: Path | str) -> list[Path]:
    return [d for d in Path(path).iterdir() if is_sweep_dir(d)]


def is_run_dir(path: Path) -> bool:
    return (path / "base_config.yaml").is_file()


def resolve_active_job_id(path: Path) -> tuple[str | None, list[str]]:
    """Find the most recent job id in a run eligible for relaunch

    Args:
        path (Path): a path to a stool run

    Logic: say we have the following run,
    .../sweep_name/run_001/
    PENDING
    PENDING
    PENDING              <-- oldest pending job
    COMPLETED/CANCELLED  <-- first non-pending job
    COMPLETED/CANCELLED
    ...

    we iterate over job_id from newest to oldest:
    1. if we find a running job_id -- relaunch that.
    2. otherwise we find the oldest pending job. if it exists -- launch that.
    3. if there are no running jobs nor pending jobs -- relaunch the last cancelled/failed.

    after we find the job_id the should be relaunched, we designate all
    pending jobs newer that this job to be cancelled

    Returns:
        str | None - the job_id that should be relaunched (None if we couldn't find any)
        list[str] - a list of pending jobs that should be cancelled
    """
    pending = []
    job_dirs = [p.name for p in path.iterdir() if p.name.isdigit()]
    job_dirs = sorted(job_dirs, reverse=True, key=int)

    # if we couldn't find any job_ids we return None
    if not job_dirs:
        return None, []

    for job in job_dirs:
        # we keep track if existing pending job_ids
        if is_pending(job):
            pending.append(job)
        # if we find a running job_id, we return it and designate
        # all existing pending jobs to be cancelled
        elif is_running(job):
            return job, pending

    # if there are no pending jobs, nor running jobs, we return
    # the newest job_id as relaunch target
    if not pending:
        return job_dirs[0], []

    # if there are only pending jobs, we designate the oldest
    # pending job_id for relaunch, and the rest for cancellation
    oldest_pending = pending.pop(0)
    return oldest_pending, pending
