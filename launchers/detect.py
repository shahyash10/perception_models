"""Module for detecting cluster type and properties."""

import logging
import os
import platform
from dataclasses import dataclass
from enum import Flag, auto
from functools import cache

logger = logging.getLogger()


class ClusterType(Flag):
    """Enum representing FAIR clusters."""

    UNKNOWN = auto()

    MAC = auto()

    FAIR_CLUSTER = auto()
    FAIR_CLUSTER_H1 = auto()
    FAIR_CLUSTER_H2 = auto()
    DEVFAIR = auto()
    LEARNFAIR = auto()

    AWS = auto()
    FAIR_SC = auto()
    COREWEAVE = auto()
    RSC = auto()

    META_DEVVM = auto()
    META_OD = auto()

    AWS_FAIR_A100 = AWS

    FAIR_CLUSTER_H1_DEVFAIR = FAIR_CLUSTER | FAIR_CLUSTER_H1 | DEVFAIR
    FAIR_CLUSTER_H2_DEVFAIR = FAIR_CLUSTER | FAIR_CLUSTER_H2 | DEVFAIR
    FAIR_CLUSTER_H2_LEARNFAIR = FAIR_CLUSTER | FAIR_CLUSTER_H2 | LEARNFAIR


@dataclass
class Cluster:
    """Dataclass representing a cluster with its type and properties."""

    cluster_type: ClusterType
    is_login: bool = False
    is_transfer: bool = False


@cache
def detect_cluster() -> Cluster:
    """Detect the cluster type and properties of the current node."""
    cluster_name = os.environ.get("SLURM_CLUSTER_NAME", None)
    node = platform.node()
    os_name = platform.system()
    release = platform.release()

    if os_name == "Darwin":
        return Cluster(ClusterType.MAC)

    is_login = "login" in node or "submit" in node

    if cluster_name == "fair-sc":
        return Cluster(ClusterType.FAIR_SC, is_login=is_login)

    if "coreweave" in release:
        is_transfer = "transfer" in node
        return Cluster(
            ClusterType.COREWEAVE,
            is_login=is_login,
            is_transfer=is_transfer,
        )

    cluster_node_substr = {
        "devvm": Cluster(ClusterType.META_DEVVM, is_login=False),
        "fair-a100": Cluster(ClusterType.AWS_FAIR_A100, is_login=is_login),
        "submit": Cluster(ClusterType.AWS_FAIR_A100, is_login=is_login),
        "fairjmp": Cluster(ClusterType.FAIR_CLUSTER, is_login=True),
        "h1.fair": Cluster(ClusterType.FAIR_CLUSTER_H1_DEVFAIR, is_login=False),
        "h2.fair": Cluster(ClusterType.FAIR_CLUSTER_H2_DEVFAIR, is_login=False),
        "learnfair": Cluster(ClusterType.FAIR_CLUSTER_H2_LEARNFAIR, is_login=False),
        "od.fbinfra": Cluster(ClusterType.META_OD, is_login=False),
        "rsccpu": Cluster(ClusterType.RSC, is_login=True),
        "rsclearn": Cluster(ClusterType.RSC, is_login=False),
    }

    for substr, cluster in cluster_node_substr.items():
        if substr in node:
            return cluster

    return (
        Cluster(ClusterType.AWS_FAIR_A100, is_login=is_login)
        if os.path.exists("/fsx-checkpoints/")
        else Cluster(ClusterType.UNKNOWN)
    )


@cache
def detect_amaia_group() -> str | None:
    """Detect amaia group from environment variable."""
    amaia_group = os.environ.get("AMAIA_GROUP", None)
    logger.debug(f"Detected AMAIA_GROUP variable set to {amaia_group}")
    return amaia_group
