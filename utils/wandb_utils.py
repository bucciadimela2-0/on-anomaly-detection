from __future__ import annotations
from typing import Any, Dict, Optional


def wandb_init_if_enabled(enabled: bool, project: str, name: str, config: Dict[str, Any]):
    """
    Initializes Weights & Biases only if enabled=True.
    Returns the wandb module if active, else None.
    """
    if not enabled:
        return None

    import wandb  # lazy import

    wandb.init(project=project, name=name, config=config)
    return wandb


def wandb_log(wandb_mod, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    if wandb_mod is None:
        return
    wandb_mod.log(metrics, step=step)


def wandb_finish(wandb_mod) -> None:
    if wandb_mod is None:
        return
    wandb_mod.finish()


def wandb_log_artifact_dir(wandb_mod, run_dir: str, artifact_name: str = "run_artifacts") -> None:
    """
    Uploads the entire run directory as an artifact (plots + checkpoints).
    """
    if wandb_mod is None:
        return
    import wandb
    artifact = wandb.Artifact(artifact_name, type="run")
    artifact.add_dir(run_dir)
    wandb_mod.log_artifact(artifact)
