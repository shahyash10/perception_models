import logging
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar

import submitit
from submitit.core.core import Job
from submitit.core.utils import FailedJobError

from amaia.cluster.detect import ClusterType, detect_amaia_group, detect_cluster

logger = logging.getLogger()


def get_slurm_account(group: str | None = None) -> str:
    cluster = detect_cluster()
    cluster_type = cluster.cluster_type
    group = group or detect_amaia_group()

    if "SLURM_JOB_ACCOUNT" in os.environ:
        return os.environ["SLURM_JOB_ACCOUNT"]

    assert group is not None

    match cluster_type:
        case ClusterType.AWS_FAIR_A100:
            return f"fairaws-{group}"
        case ClusterType.COREWEAVE:
            return f"fair_amaia_cw_{group}"
        case ClusterType.FAIR_CLUSTER_H2:
            return group
        case ClusterType.RSC:
            return group
        case _:
            raise NotImplementedError(f"No account mapping defined for {cluster_type}")


def get_slurm_partition() -> str:
    cluster = detect_cluster()
    cluster_type = cluster.cluster_type

    if "SLURM_JOB_PARTITION" in os.environ:
        return os.environ["SLURM_JOB_PARTITION"]

    match cluster_type:
        case ClusterType.FAIR_CLUSTER_H2:
            return "learnfair"
        case _:
            return "learn"


def get_slurm_qos(
    group: str | None = None,
    *,
    lowest_qos: bool = False,
) -> str | None:
    cluster = detect_cluster()
    cluster_type = cluster.cluster_type
    group = group or detect_amaia_group()

    if lowest_qos and cluster_type in {
        ClusterType.COREWEAVE,
        ClusterType.RSC,
        ClusterType.AWS_FAIR_A100,
    }:
        return "lowest"

    if "SLURM_JOB_QOS" in os.environ:
        return os.environ.get("SLURM_JOB_QOS")

    match (cluster_type, group):
        case (ClusterType.COREWEAVE, "scale" | "bigjob"):
            return group
        case (ClusterType.COREWEAVE, _):
            return "lowest"
        case (ClusterType.RSC, _):
            return group
        case _:
            return None


def get_slurm_mem() -> int:
    cluster = detect_cluster()
    cluster_type = cluster.cluster_type

    match cluster_type:
        case ClusterType.AWS_FAIR_A100:
            return 1600
        case ClusterType.COREWEAVE:
            return 1760
        case ClusterType.FAIR_CLUSTER_H2:
            return 480
        case ClusterType.RSC:
            return 1760

    return 64


def get_slurm_extra_params() -> dict:
    cluster = detect_cluster()
    cluster_type = cluster.cluster_type

    match cluster_type:
        # aws cluster use mem_per_gpu to set the memory instead of mem_gb, setting both
        # of them will cause a conflict
        case ClusterType.AWS_FAIR_A100:
            return {"mem_per_gpu": "80g"}
        case ClusterType.FAIR_CLUSTER_H2:
            return {"constraint": "volta32gb"}

    return {}


def init_signal_handler(handler: Callable) -> None:
    """Handle signals sent by SLURM for time limit / pre-emption."""
    signal.signal(signal.SIGUSR2, handler)
    logger.warning("Signal handler installed.")


def requeue_slurm_job() -> None:
    proc_id = int(os.environ["SLURM_PROCID"])
    logger.warning(f"Host: {socket.gethostname()} - Global rank: {proc_id}")
    if proc_id == 0:
        logger.warning("Requeuing job %s", os.environ["SLURM_JOB_ID"])
        subprocess.run(["scontrol", "requeue", os.environ["SLURM_JOB_ID"]], check=False)
    else:
        logger.warning("Not the master process, no need to requeue.")
    sys.exit(0)


@contextmanager
def clean_env() -> Iterator[None]:
    # Solves issues when launching wandb in async evals
    # See: https://github.com/wandb/wandb/issues/5272#issuecomment-2041556476
    os.environ["WANDB_DISABLE_SERVICE"] = "True"

    distrib_names = (
        "MASTER_ADDR",
        "MASTER_PORT",
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
    )
    other_names = (
        # Variables below are set by setup_env() and point to local directories
        "TMPDIR",
        "TMP_DIR",
        "TRITON_CACHE_DIR",
        # Purge Torch inductor location
        "TORCHINDUCTOR_CACHE_DIR",
    )
    cluster_env = {
        x: os.environ.pop(x)
        for x in os.environ
        if x.startswith(
            ("SLURM_", "SLURMD_", "SRUN_", "SBATCH_", "SUBMITIT_", "TORCHELASTIC_")
        )
        or x in distrib_names
        or x in other_names
    }
    try:
        yield
    finally:
        os.environ.update(cluster_env)


C = TypeVar("C")


def launch_with_submitit(
    fn: Callable[[C], Any],
    cfg: C,
    n_gpus: int,
    folder: Path,
    job_name: str,
    *,
    group: str | None = None,
    partition: str | None = None,
    account: str | None = None,
    qos: str | None = None,
    lowest_qos: bool = False,
    exclude_nodes: str | None = None,
) -> Job | None:
    """Launch a job with Submitit to execute the provided fn passing the config cfg as argument.

    Args:
        fn (callable): The function to execute in Submitit.
        cfg: Configuration passed as arguments to the function fn.
        n_gpus (int): Number of GPUs used for the Submitit job.
        folder (Path): Log folder for Submitit.
        job_name (str): Name of the Submitit job.
        group (str): Group or project name associated with cluster resources.
        partition (str, optional): SLURM partition for submitit.
        account (str, optional): SLURM account for submitit.
        qos (str, optional): SLURM QoS for submitit.
        lowest_qos (bool, optional): Whether to try launching on the lowest QoS available.
        exclude_nodes (str): List of nodes to exclude when launching with submitit.
    """
    assert n_gpus % 8 == 0, n_gpus
    assert n_gpus > 0

    cluster = detect_cluster()
    cluster_type = cluster.cluster_type

    account = account or get_slurm_account(group)
    partition = partition or get_slurm_partition()
    qos = qos or get_slurm_qos(group, lowest_qos=lowest_qos)

    extra_params = {
        "slurm_account": account,
        "slurm_partition": partition,
        **get_slurm_extra_params(),
    }
    if qos is not None:
        extra_params["slurm_qos"] = qos

    mem_gb = get_slurm_mem()

    executor = submitit.AutoExecutor(folder=folder, slurm_max_num_timeout=-1)
    executor.update_parameters(
        slurm_timeout_min=3 * 24 * 60,  # days for the moment
        slurm_gpus_per_node=8,
        slurm_ntasks_per_node=8,
        slurm_cpus_per_task=2,
        nodes=int(n_gpus // 8),
        mem_gb=mem_gb,
        slurm_job_name=job_name,
        slurm_srun_args=["-vv"],
        slurm_exclude=exclude_nodes,
        **extra_params,
    )
    logger.debug(f"SubmitIt job {job_name} with params={executor.parameters}")
    # a hack to remove the mem parameter since mem_gb is mandatory for executor.update_parameters
    if cluster_type | ClusterType.AWS:
        executor.update_parameters(slurm_account=account)
        del executor.parameters["mem"]

    # submit job
    with clean_env():
        backoff = 60 * 3  # wait 5 minutes if the slurm scheduler times out
        max_attempts = 3
        for attempt_id in range(max_attempts):
            try:
                return executor.submit(fn, cfg)
            except FailedJobError as e:
                logger.warning(
                    f"Error submitting submit job {job_name}"
                    f"({attempt_id + 1}/{max_attempts}): {e}",
                )
                time.sleep(backoff)
        logger.warning(f"Didn't managed to submit {job_name}")
        return None

def dataclass_from_dict(cls: type[T], data: dict) -> T:
    """Converts a dictionary to a dataclass instance, recursively for nested structures."""
    base = OmegaConf.structured(cls)
    override = OmegaConf.create(data)
    return OmegaConf.to_object(OmegaConf.merge(base, override))