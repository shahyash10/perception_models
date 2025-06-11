"""
python -m launchers.stool run_sweep  config=apps/plm/sweeps/plmt_1b_data.yaml name=datamix group=a100-comem account=a100-comem nodes=16 ncpu=10 ngpu=8 anaconda=$CONDA_PREFIX
"""

import getpass
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field, fields, replace
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

from omegaconf import MISSING, OmegaConf

from launchers.cluster import (
    get_slurm_account,
    get_slurm_mem,
    get_slurm_partition,
    get_slurm_qos,
)
from launchers.detect import ClusterType, detect_amaia_group, detect_cluster
from launchers.utils import (
    check_conda,
    check_is_slurm_job,
    confirm,
    confirm_rmtree,
    copy_dir,
    get_conda_python_bin,
    get_current_conda,
    get_cwd_git_commit_hash,
    get_job_names_and_ids,
    get_job_ping,
    get_launch_date_str,
    get_local_diff,
    get_slurm_job_status,
    get_sweep_dirs,
    get_swept_params,
    is_cancelled,
    is_failed,
    is_run_dir,
    print_table,
    read_exclude_from_path,
    resolve_active_job_id,
    retrieve_max_time_per_partition,
)

T = TypeVar("T")

CLUSTER = detect_cluster()
JOB_ID_REGEX = re.compile(r"Submitted batch job (\d+)", flags=re.IGNORECASE)


def dataclass_from_dict(cls: type[T], data: dict) -> T:
    """Converts a dictionary to a dataclass instance, recursively for nested structures."""
    base = OmegaConf.structured(cls)
    override = OmegaConf.create(data)
    return OmegaConf.to_object(OmegaConf.merge(base, override))


def get_root_dump_dir(cluster_type: ClusterType, group: str) -> str:
    assert group != ""
    user = getpass.getuser()
    default_dump_dir_map: dict[ClusterType, Path] = {
        ClusterType.RSC: Path(f"/checkpoint/{group}/{user}/amaia_dumps"),
        ClusterType.COREWEAVE: Path(f"/checkpoint/amaia/{group}/{user}/amaia_dumps"),
        ClusterType.AWS_FAIR_A100: Path(f"/fsx-checkpoints/{user}/dumps"),
        ClusterType.FAIR_CLUSTER_H2: Path(f"/checkpoint/{user}/amaia_dumps"),
    }
    return str(default_dump_dir_map[cluster_type])


@dataclass
class StoolCLIArgs:
    root_dump_dir: Path | None = None
    # If specified, fills defaults for slurm qos, account and partition.
    # Overrides AMAIA_GROUP environment variable for the launched job.
    # The provided qos, account and partition still have precedence over this parameter.
    group: str = ""

    # SLURM parameters
    account: str = ""
    partition: str = ""
    qos: str = ""

    def __post_init__(self):
        # Instantiate AMAIA group
        if self.group == "":
            self.group = detect_amaia_group() or ""
        # Instantiate root dump dir and current dump_dir
        self.root_dump_dir = self.root_dump_dir or get_root_dump_dir(
            CLUSTER.cluster_type, self.group
        )

    def init_slurm_defaults(self) -> None:
        if self.account == "":
            self.account = get_slurm_account(self.group) or ""
        if self.qos == "":
            self.qos = get_slurm_qos(self.group) or ""
        if self.partition == "":
            self.partition = get_slurm_partition()

    @classmethod
    def from_cli(cls, argv: list):
        args = OmegaConf.from_cli(argv)
        return dataclass_from_dict(cls, args)


@dataclass
class StoolRunArgs(StoolCLIArgs):
    name: str = MISSING
    config: str = MISSING

    launcher: str = "sbatch"  # Can be sbatch or bash if already in salloc
    script: str = "apps.plm.train"  # The script to run
    dirs_exists_ok: bool = (
        False  # Whether to copy new code and config and run regardless that dir exists
    )
    override: bool = False  # Whether to delete dump dir and restart
    anaconda: str = "default"  # The path to the anaconda environment
    launch_restart_dependencies: int = -1  # Restart dependencies for the job
    no_requeue: bool = False  # If false, no automatic requeue
    preview: bool = (
        False  # Display the SLURM command to run the job (useful for debugging)
    )

    # SLURM parameters
    nodes: int = -1  # The number of nodes to run the job on
    ngpu: int = 8  # The number of GPUs required per node
    ncpu: int = 16  # The number of CPUs allocated per GPU
    mem: str = ""  # The amount of memory to allocate
    exclude: str = ""  # The list of nodes to exclude
    exclude_path: str = ""  # Path to a file containing exclude nodes list
    constraint: str = ""  # The constraint on the nodes
    dependency: str = ""  # Job id on which this run will depend
    reservation: str = ""
    time: int = -1  # The time limit of the job (in minutes)
    stdout: bool = False

    dump_dir: Path = field(init=False)  # The directory of job outputs and logs
    launch_date_str: str = field(init=False)
    run_dir: Path | None = None  # The directory of the job's code
    sbatch_dir: Path | None = None

    git_commit_hash: str | None = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.init_slurm_defaults()

        # check that the command is not run from within another SLURM job
        # (in which case the `srun` commands would be run on the allocation,
        # and their resource requests would likely conflict)
        if not self.preview and self.launcher == "sbatch" and check_is_slurm_job():
            print("Already within a SLURM job (salloc shell?), terminating!")
            sys.exit(0)

        # set dump dir
        self.dump_dir = Path(self.root_dump_dir) / f"{self.name}"
        # set anaconda
        if self.anaconda == "default":
            self.anaconda = get_current_conda()
        check_conda(self.anaconda)

        self.launch_date_str = get_launch_date_str()
        self.git_commit_hash = get_cwd_git_commit_hash()

        self.sbatch_dir = self.dump_dir / ".stool" / self.launch_date_str

    # dedicated init logic for run dir as we may relaunch with previous run_dir
    # when relaunching existing experiments
    def init_run_dir(self, code_path: Path) -> None:
        "Make a code directory versioned by date and copy the code from `code_path` there."
        self.run_dir = self.dump_dir / "code" / self.launch_date_str
        self.run_dir.mkdir(parents=True, exist_ok=self.dirs_exists_ok)
        print("Copying code...")
        copy_dir(str(code_path), f"{self.run_dir!s}")


@dataclass
class StoolCancelArgs(StoolCLIArgs):
    pattern: str = ""  # pattern on SLURM job name
    substr: bool = False  # allow substring


@dataclass
class StoolMonitorArgs(StoolCLIArgs):
    name: str = MISSING
    group: str = ""
    root_dump_dir: str = ""
    dump_dir: Path = field(init=False)  # The directory of job outputs and logs

    def __post_init__(self):
        super().__post_init__()
        self.dump_dir = Path(self.root_dump_dir) / self.name


@dataclass
class StoolRelaunchArgs(StoolCLIArgs):
    use_group_defaults: bool = False  # use from reloaded args by default
    # SLURM overridden args
    dependency: str = ""
    exclude: str = ""
    account: str = ""
    partition: str = ""
    qos: str = ""

    job_dir: str | Path = ""
    name: str = ""
    job_id: str = ""

    # Relaunch-specific args
    use_new_code: bool = False  # Whether to use new code when relaunching a job
    anaconda: str = (
        ""  # Not set by default, will resume with the one used in the original job
    )
    launch_restart_dependencies: int = -1  # Restart dependencies for the job
    dump_dir: Path = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        if self.use_group_defaults:
            self.init_slurm_defaults()
        if self.job_dir == MISSING:
            return  # call _set_job_dir later, used in sweep relaunching
        elif self.job_dir != "":
            self._set_job_dir(self.job_dir)
        else:
            assert len(self.name) > 0
            assert len(self.job_id) > 0
            self.job_dir = Path(self.root_dump_dir) / self.name / self.job_id

        assert self.name != MISSING
        assert self.job_id != 0
        assert self.job_dir != ""
        self.dump_dir = Path(self.root_dump_dir) / self.name

    def _set_job_dir(self, job_dir: str | Path):
        self.job_dir = Path(job_dir)
        assert self.job_dir.exists()
        assert self.job_dir.is_relative_to(self.root_dump_dir), (
            f"Relaunch experiments that are under your root_dump_dir only! root_dump_dir={self.root_dump_dir}, job_dir={self.job_dir}"
        )
        self.name = str(self.job_dir.parent.relative_to(self.root_dump_dir))
        self.dump_dir = Path(self.root_dump_dir) / self.name
        self.job_id = self.job_dir.name

    def reload_run_args(self) -> StoolRunArgs:
        return dataclass_from_dict(
            StoolRunArgs, OmegaConf.load(Path(self.job_dir) / "stool_args.yaml")
        )

    def get_updated_run_args(self) -> StoolRunArgs:
        # post-init called with replace, updating launch_date_str and sbatch_dir
        run_args = replace(self.reload_run_args())
        if self.launch_restart_dependencies != run_args.launch_restart_dependencies:
            run_args.launch_restart_dependencies = self.launch_restart_dependencies

        if self.dependency != "":
            print(f"Setting relaunch dependency to {self.dependency!r}")
            run_args.dependency = self.dependency

        if self.use_new_code:
            # update run_dir with new code
            run_args.init_run_dir(code_path=Path.cwd())
        else:
            print(f"Continuing with previously copied code: {run_args.run_dir}")

        if self.launch_restart_dependencies != run_args.launch_restart_dependencies:
            run_args.launch_restart_dependencies = self.launch_restart_dependencies

        if self.anaconda not in {"", run_args.anaconda}:
            run_args.anaconda = self.anaconda

        # if job to relaunch already had 'exclude' and new args have exclude = "", we keep old exclude
        if self.exclude not in {"", run_args.exclude}:
            run_args.exclude = self.exclude

        # if job to relaunch already had 'qos' and new args have qos = "", we keep old qos
        if self.qos not in {"", run_args.qos}:
            run_args.qos = self.qos

        # if job to relaunch already had 'qos' and new args have qos = "", we keep old qos
        if self.partition not in {"", run_args.partition}:
            run_args.partition = self.partition

        # if job to relaunch already had 'qos' and new args have qos = "", we keep old qos
        if self.account not in {"", run_args.account}:
            run_args.account = self.account

        # use provided group if specified, otherwise keep old one
        if self.group not in {"", run_args.group}:
            run_args.group = self.group

        return run_args


@dataclass
class StoolSweepRelaunchArgs(StoolRelaunchArgs):
    sweep_dir: str | Path = ""
    force: bool = (
        False  # relaunch running jobs too in addition to failed/cancelled ones
    )
    indices: str = ""  # a comma-separated list of indices that will be relaunched, leave empty to relaunch all

    def __post_init__(self):
        StoolCLIArgs.__post_init__(self)
        if self.use_group_defaults:
            self.init_slurm_defaults()

        assert self.job_dir == "", (
            "sweep relaunch only uses either name or sweep_dir argument"
        )
        assert self.job_id == "", (
            "sweep relaunch only uses either name or sweep_dir argument"
        )
        assert (self.sweep_dir == "") != (self.name == ""), (
            "sweep relaunch only uses either name or sweep_dir argument"
        )

        if self.name != "":
            self.sweep_dir = Path(self.root_dump_dir) / self.name

        self.sweep_dir = Path(self.sweep_dir)
        assert self.sweep_dir.exists()

        # set later when launching the individual jobs
        self.dump_dir = MISSING
        self.job_id = MISSING
        self.job_dir = MISSING


SBATCH_COMMAND = """#!/bin/bash

{constraint}
{dependency}
{reservation}
{no_requeue}
{exclude}
{qos}
{account}
#SBATCH --job-name={name}
#SBATCH --nodes={nodes}
#SBATCH --gres=gpu:{ngpus}
#SBATCH --cpus-per-gpu={ncpu}
#SBATCH --time={time}
#SBATCH --partition={partition}
#SBATCH --mem={mem}

#SBATCH --output={dump_dir}/%j/%j.stdout
#SBATCH --error={dump_dir}/%j/%j.stderr

#SBATCH --open-mode=append
#SBATCH --signal=USR2@120
#SBATCH --distribution=block

{group_env}
{cluster_specific_envs}

# Mimic the effect of "conda init", which doesn't work for scripts
eval "$({conda_bash_hook_cmd})"
{conda_activate_cmd}

cd {run_dir}

export TMPDIR="/fsx-checkpoints/ngjhn/tmp/slurm_tmpdir/$SLURM_JOB_ID"
export TMP_DIR="/fsx-checkpoints/ngjhn/tmp/slurm_tmpdir/$SLURM_JOB_ID"
export DUMP_DIR={dump_dir}
export PYTHONPATH=$PYTHONPATH:{run_dir}
srun {log_output} --label -n {tasks} -N {nodes_per_run} python -u -m {script} config=$DUMP_DIR/base_config.yaml
"""


def wrap_sbatch_str(name: str, value: Any) -> str | None:
    if isinstance(value, str) and value.startswith("#SBATCH"):
        return value
    return f"#SBATCH --{name}={value!s}"


def build_sbatch_command(args: StoolRunArgs) -> str:
    # avoid overriding StoolRunArgs
    args = replace(args)
    # Set maximum time limit if not specified
    if args.time == -1:
        max_times = retrieve_max_time_per_partition()
        args.time = max_times.get(
            args.partition, 3 * 24 * 60
        )  # Default to 3 days if not found
        print(
            f"No time limit specified, using max time for partitions: {args.time} minutes"
        )

    if args.constraint:
        args.constraint = wrap_sbatch_str("constraint", args.constraint)

    if args.account:
        args.account = wrap_sbatch_str("account", args.account)

    if args.qos:
        args.qos = wrap_sbatch_str("qos", args.qos)

    if getattr(args, "exclude_path", ""):
        assert args.exclude == "", (
            "Exclude node should be specified either with 'exclude' or 'exclude_path' but not both"
        )
        args.exclude = read_exclude_from_path(args.exclude_path)

    if getattr(args, "exclude", ""):
        args.exclude = wrap_sbatch_str("exclude", args.exclude)

    if getattr(args, "dependency", ""):
        args.dependency = wrap_sbatch_str("dependency", args.dependency)

    if getattr(args, "reservation", ""):
        args.reservation = wrap_sbatch_str("reservation", args.reservation)

    assert hasattr(args, "anaconda")
    assert args.anaconda is not None
    conda_python_bin = get_conda_python_bin(args.anaconda)
    assert conda_python_bin.is_file(), f"Invalid conda python bin: {conda_python_bin}"

    if args.mem == "":
        args.mem = f"{get_slurm_mem()}G"

    if not args.partition:
        args.partition = get_slurm_partition()

    assert args.ngpu > 0
    assert args.ncpu > 0
    assert args.nodes > 0
    assert args.time > 0
    assert args.partition

    if args.no_requeue:
        print("This job will not be requeued. Are you sure? [yN]")
        if input().lower() != "y":
            sys.exit(0)
        no_requeue = "#SBATCH --no-requeue"
    else:
        no_requeue = ""

    # cluster specific flags and environment variables
    # => https://fb.workplace.com/groups/aws.fair.discuss/permalink/1462900264628320/
    cluster_specific_env_vars: tuple[str, ...] = ()
    if CLUSTER.cluster_type == ClusterType.AWS_FAIR_A100:
        cluster_specific_env_vars = (
            "export FI_EFA_SET_CUDA_SYNC_MEMOPS=0",
            "export NCCL_P2P_NET_CHUNKSIZE=524288",
        )
    cluster_specific_envs = "\n".join(cluster_specific_env_vars)
    group_env = "" if args.group == "" else f"export AMAIA_GROUP={args.group}"

    conda_env_path = args.anaconda
    # if mamba_exe := os.environ.get("MAMBA_EXE"):
    #     conda_bash_hook_cmd = f"{mamba_exe} shell hook -s bash"
    #     conda_activate_cmd = f"micromamba activate {conda_env_path}"
    # else:
    conda_exe = os.environ.get("CONDA_EXE", "conda")
    conda_bash_hook_cmd = f"{conda_exe} shell.bash hook"
    conda_activate_cmd = f"source activate {conda_env_path}"

    log_output = (
        "-o $DUMP_DIR/%j/%j_%t_log.out -e $DUMP_DIR/%j/%j_%t_log.err"
        if not args.stdout
        else ""
    )
    return SBATCH_COMMAND.format(
        name=args.name,
        script=args.script,
        dump_dir=str(args.dump_dir),
        run_dir=str(args.run_dir),
        nodes=args.nodes,
        tasks=args.nodes * args.ngpu,
        nodes_per_run=args.nodes,
        ngpus=args.ngpu,
        ncpu=args.ncpu,
        mem=args.mem,
        qos=args.qos,
        account=args.account,
        constraint=args.constraint,
        exclude=args.exclude,
        dependency=args.dependency,
        reservation=args.reservation,
        time=args.time,
        partition=args.partition,
        conda_bash_hook_cmd=conda_bash_hook_cmd,
        conda_activate_cmd=conda_activate_cmd,
        group_env=group_env,
        cluster_specific_envs=cluster_specific_envs,
        no_requeue=no_requeue,
        log_output=log_output,
    )


def submit_slurm_job(args: StoolRunArgs) -> str | None:
    """Submit a SLURM job with stool, saving the stool launch args along with the job."""
    assert args.config != MISSING, (
        "A config must be provided using stool [cmd] config=<config_file>"
    )
    assert args.name != MISSING, "A name must be provided using stool [cmd] name=<name>"
    assert args.dump_dir.exists()
    assert args.run_dir.exists()

    # define experiment ID / bash filepath to run the experiment
    run_uuid = str(uuid.uuid4())
    # tmp folder that will be copied with good submission id after
    tmp_stool_dir_for_run = args.sbatch_dir / run_uuid

    sbatch = build_sbatch_command(args)
    if args.preview:
        print("Preview of sbatch command")
        print(sbatch)
        return None
    print("Writing sbatch command...")
    args.sbatch_dir.mkdir(parents=True, exist_ok=True)
    tmp_stool_dir_for_run.mkdir()
    tmp_stool_submit_script = tmp_stool_dir_for_run / "submit.slurm"
    tmp_stool_args_file = tmp_stool_dir_for_run / "stool_args.yaml"
    tmp_stool_submit_script.write_text(sbatch)
    with Path.open(tmp_stool_args_file, "w") as f:
        OmegaConf.save(config=args, f=f)

    print("Submitting job...")
    submit_cmd = f"{args.launcher} {tmp_stool_submit_script!s}"
    if args.launcher == "bash":
        assert "SLURM_JOBID" in os.environ, "SLURM_JOBID should be set in salloc shell"
        job_id = os.environ["SLURM_JOBID"]
        os.makedirs(args.dump_dir / job_id)
        output = subprocess.check_output(submit_cmd, shell=True, encoding="utf-8")
    elif args.launcher == "sbatch":
        output = subprocess.check_output(submit_cmd, shell=True, encoding="utf-8")
        print(output.strip())
        job_id_match = JOB_ID_REGEX.search(output)
        assert job_id_match is not None, output
        job_id = job_id_match.groups()[0]
        assert job_id.isdigit()
        print(f"Job {job_id} submitted.")
    else:
        raise ValueError(f"Invalid launcher: {args.launcher}")

    stool_dir_for_run = str(args.dump_dir / job_id)
    shutil.copytree(tmp_stool_dir_for_run, stool_dir_for_run)
    print(f"Job infos in {stool_dir_for_run}")

    print("Done.")
    return job_id


def run_launch_restart_dependent_jobs(
    args: StoolRunArgs, job_id: str
) -> list[str] | None:
    """Submit restart dependencies jobs."""
    dependent_jobs = []
    for restart in range(args.launch_restart_dependencies):
        print()
        time.sleep(1)
        print(f"======= DEPENDENCY {restart} =======")
        relaunch_args = StoolRelaunchArgs(
            job_dir=Path(args.dump_dir) / job_id,
            dependency=str(job_id),
            use_new_code=False,
            launch_restart_dependencies=-1,
            # it will keep the same values that job to relaunch
            group=args.group,
            anaconda="",
            account="",
            partition="",
            qos="",
            exclude="",
        )
        relaunch_job_id = run_relaunch_job(relaunch_args)
        assert relaunch_job_id is not None
        job_id = relaunch_job_id
        dependent_jobs.append(job_id)
    return dependent_jobs


def run_launch_job(args: StoolRunArgs, *, sweep: bool = False) -> str | None:
    """Launch a SLURM job."""
    # Set up args default and validate them depending on the cluster or partition requested
    print("Creating directories...")

    if sweep:
        # if we are under a sweep, two things should already have happened:
        # 1. a dump dir for the individual run should have been created in `run_launch_sweep`
        # 2. a symlink to the shared code dir should have been created in the dump dir
        assert args.run_dir.exists()
        assert args.dump_dir.exists()
    else:
        if args.dump_dir.exists() and args.override:
            # TODO: Ensure we don't have a running job
            msg = (
                f"Are you sure you want to delete the directory '{args.dump_dir!s}'? "
                "Make sure that no job remaining is running. Use at your own risks. "
                "This action cannot be undone. (yes/no): "
            )
            if not confirm_rmtree(msg, args.dump_dir):
                return None

    print(f"Loading config from: {args.config}")
    config = OmegaConf.load(str(args.config))

    if not sweep:
        print(f"Creating dump_dir: {args.dump_dir}")
        args.dump_dir.mkdir(parents=True, exist_ok=args.dirs_exists_ok)

    print("Setting config dump_dir")
    config["dump_dir"] = str(args.dump_dir)

    if not sweep:
        args.init_run_dir(code_path=Path.cwd())

    # Writing diff between current state and upstream branch (or main)
    try:
        local_diff = get_local_diff()
        diff_file_path = args.run_dir / "local_diff.patch"
        with open(diff_file_path, "w") as diff_file:
            diff_file.write(local_diff)
        print(f"Saving local diff to {diff_file_path}")
    except Exception as e:
        print(f"Skipping local diff generation with reason: {str(e)}")

    print("Saving config file...")
    OmegaConf.save(config, f=args.dump_dir / "base_config.yaml")
    # updating config in stool args to point to dump dir config
    args.config = str(args.dump_dir / "base_config.yaml")

    job_id = submit_slurm_job(args)
    if job_id:
        run_launch_restart_dependent_jobs(args, job_id)
    return job_id


def run_relaunch_job(args: StoolRelaunchArgs) -> str | None:
    """Relaunch SLURM jobs."""
    print(f"Relaunching job: {args.job_id} from {args.dump_dir}")
    # update arguments from relaunch args
    run_args = args.get_updated_run_args()
    job_id = submit_slurm_job(run_args)
    if job_id:
        run_launch_restart_dependent_jobs(run_args, job_id)
    return job_id


def run_cancel_job(args: StoolCancelArgs) -> None:
    """Cancel SLURM jobs from a pattern on their names, possibly matching only a substring."""
    # retrieve list of jobs with their names and IDs
    job_names: list[tuple[str, int]] = get_job_names_and_ids(getpass.getuser())
    # select jobs to kill
    match_: dict[str, int] = defaultdict(int)
    kill_ids: set[int] = set()
    for job_name, job_id in job_names:
        if job_name == args.pattern or args.substr and args.pattern in job_name:
            match_[job_name] += 1
            kill_ids.add(job_id)
    match = sorted(match_.items(), key=lambda nc: (nc[1], nc[0]), reverse=True)

    # nothing to do
    if len(match) == 0:
        hint = (
            ""
            if args.substr
            else " Try with the flag substring to match job name substrings."
        )
        print(f"Did not find any job with this name. Aborting.{hint}")
        sys.exit(0)

    # preview jobs to kill
    if args.substr:
        to_show = 50
        for name, count in match[:to_show]:
            print(f"\t{count:>3}: {name}")
        if len(match) > to_show:
            print(f"+ {len(match) - to_show} others")
    else:
        assert len(match) == 1
    print(
        f"About to kill {sum(match_.values())} jobs ({len(match)} different job names)."
    )

    # confirm & kill
    confirm = input('Enter "yes" to confirm kill: ')
    if confirm == "yes":
        ids = " ".join(str(job_id) for job_id in kill_ids)
        subprocess.call(f"scancel {ids}", shell=True)
    else:
        print("Aborting.")


def run_launch_sweep(args: StoolRunArgs) -> list[str] | None:
    sweep_name = args.name
    conf_path = args.config
    conf = OmegaConf.load(conf_path)
    grid = get_swept_params(conf)  # compute cartesian product

    msg = f"Launching a sweep with {len(grid)} runs\n"
    msg = msg + "-" * len(msg)
    print(msg)

    sweep_dump_dir = args.dump_dir
    if sweep_dump_dir.exists() and args.override:
        msg = (
            f"Are you sure you want to delete the sweep directory '{sweep_dump_dir!s}'? "
            "Make sure that no job remaining is running. Use at your own risks. "
            "This action cannot be undone. (yes/no): "
        )
        if not confirm_rmtree(msg, sweep_dump_dir):
            return None

    print(f"Creating sweep dump_dir: {args.dump_dir}")
    args.dump_dir.mkdir(parents=True, exist_ok=args.dirs_exists_ok)

    # setup the sweep directory: copy sweep.yaml
    shutil.copyfile(src=conf_path, dst=(sweep_dump_dir / "sweep.yaml"))

    # copy code
    args.init_run_dir(code_path=Path.cwd())
    sweep_run_dir = args.run_dir

    # use run_launch_job to launch grid
    jobs = []
    args.root_dump_dir = str(sweep_dump_dir)
    for i, conf in enumerate(grid):
        args.name = f"{sweep_name}_run{i:03}"
        args.dump_dir = sweep_dump_dir / args.name
        args.dump_dir.mkdir(parents=True, exist_ok=args.dirs_exists_ok)
        # make a code directory symlink to the sweep's code directory
        os.symlink(
            sweep_run_dir.parent,
            args.dump_dir / "code",
            target_is_directory=True,
        )
        # creating a the yaml config in a temp file for run_launch_job
        # run_launch_job will then dump the config in the run folder
        # as "base_config.yaml"
        with tempfile.NamedTemporaryFile() as f:
            OmegaConf.save(config=conf, f=f.name)
            args.config = f.name  # set config from grid
            args.dirs_exists_ok = True
            job_id = run_launch_job(args, sweep=True)
            jobs.append(job_id)
    return jobs


def run_relaunch_sweep(args: StoolSweepRelaunchArgs) -> None:
    relaunch_args = {
        k: v
        for k, v in asdict(args).items()
        if k in [x.name for x in fields(StoolRelaunchArgs) if x.init]
    }
    relaunch_args = StoolRelaunchArgs(**relaunch_args)

    to_cancel: list[tuple[str, str]] = []
    to_relaunch: list[Path] = []
    runs = [p for p in args.sweep_dir.iterdir() if is_run_dir(p)]
    indices = (
        list(range(len(runs)))
        if args.indices == ""
        else [int(x) for x in args.indices.split(",")]
    )
    for i, run in enumerate(runs):
        if i not in indices:
            continue
        job_id, pending_jobs = resolve_active_job_id(run)
        if job_id is None:
            print(
                f"warning! could not find a job_id for run: {run!s} (if PENDING consider using `filter_active=false`)"
            )
            continue
        status = get_slurm_job_status(job_id)
        cancel_or_fail = is_cancelled(job_id) or is_failed(job_id)
        # job failed or cancelled - relaunch it
        if cancel_or_fail:
            to_relaunch.append(run / job_id)
        # job is running (and force=true) - cancel it and its dependencies, and relaunch it
        if not cancel_or_fail and args.force:
            to_relaunch.append(run / job_id)
            to_cancel.append((f"{args.name}_{run.name}", job_id))
            to_cancel.extend([(f"{args.name}_{run.name}", x) for x in pending_jobs])
        # job is running (and force=false) - nothing to do
        if not cancel_or_fail and not args.force:
            print(f"skipping job {job_id} as it is in {status} status")

    print()
    if to_cancel:
        msg = "warning! the following jobs are active and will be cancelled before relaunch:\n"
        msg += "\n".join(f"{job_id}  {name}" for (name, job_id) in to_cancel)
        msg += '\ntype "yes" to confirm: '
        if confirm(msg):
            for _, job_id in to_cancel:
                print(f"Cancelling: {job_id}")
                subprocess.call(f"scancel {job_id}", shell=True)
        else:
            print("aborting...")
            return

    if len(to_relaunch) == 0:
        print("nothing to relaunch.")

    print("")
    for run in to_relaunch:
        msg = f"Relaunching: {'/'.join(run.parts[-3:-1])}\n"
        msg = msg + "-" * len(msg)
        print(msg)
        relaunch_args._set_job_dir(run)
        run_relaunch_job(relaunch_args)


def run_monitor_sweep(args: StoolMonitorArgs) -> None:
    table = [("run", "slurm id", "status", "ping")]
    avail_sweeps = get_sweep_dirs(args.root_dump_dir)
    sweep_dump_dir = args.dump_dir
    if sweep_dump_dir not in avail_sweeps:
        raise ValueError(
            "invalid sweep name! available sweeps: ",
            f"{[p.name for p in avail_sweeps]}",
        )
    runs = [p for p in sweep_dump_dir.iterdir() if is_run_dir(p)]
    for run in runs:
        job_id, _ = resolve_active_job_id(run)
        if job_id is None:
            continue
        status = get_slurm_job_status(job_id)
        ping = get_job_ping(run / job_id)
        if ping is None:
            ping = "n/a"
        table.append((run.name, job_id, status, ping))
    print_table(table)


class StoolCommand(Enum):
    RUN: str = "run"
    RELAUNCH: str = "relaunch"
    CANCEL: str = "cancel"
    SWEEP: str = "run_sweep"
    MONITOR: str = "monitor_sweep"
    RELAUNCH_SWEEP: str = "relaunch_sweep"

    def run(self, argv: list[str]) -> None:
        match self:
            case StoolCommand.RUN:
                args = StoolRunArgs.from_cli(argv)
                run_launch_job(args)
            case StoolCommand.RELAUNCH:
                args = StoolRelaunchArgs.from_cli(argv)
                run_relaunch_job(args)
            case StoolCommand.CANCEL:
                args = StoolCancelArgs.from_cli(argv)
                run_cancel_job(args)
            case StoolCommand.SWEEP:
                args = StoolRunArgs.from_cli(argv)
                run_launch_sweep(args)
            case StoolCommand.MONITOR:
                args = StoolMonitorArgs.from_cli(argv)
                run_monitor_sweep(args)
            case StoolCommand.RELAUNCH_SWEEP:
                args = StoolSweepRelaunchArgs.from_cli(argv)
                run_relaunch_sweep(args)
            case _:
                raise ValueError(f"Unrecognized stool command: {self}")


# TODO: Provide helper usage
def stool_cli() -> None:
    argv = sys.argv
    available_commands = [e.value for e in StoolCommand]
    if len(argv) <= 1:
        print(
            f"Usage: python -m launchers.stool [{'|'.join(available_commands)}] arg1=value1 arg2=value2"
        )
        sys.exit(1)

    command = StoolCommand(argv[1])
    command.run(argv[2:])


if __name__ == "__main__":
    stool_cli()
