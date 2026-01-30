from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn

from utils.models_io import AE_REGISTRY, OCNN_REGISTRY


# Save a generic checkpoint payload
def save_checkpoint(path: str, payload: Dict[str, Any]) -> None:
    torch.save(payload, path)


# Load a generic checkpoint payload
def load_checkpoint(path: str, device: torch.device) -> Dict[str, Any]:
    return torch.load(path, map_location=device)


# Save an autoencoder checkpoint with architecture metadata
def save_ae(
    path: str,
    ae: nn.Module,
    ae_name: str,
    ae_kwargs: Optional[Dict[str, Any]] = None,
    run_name: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    # Collect optional metadata
    extra_dict = dict(extra) if extra is not None else {}
    if run_name is not None:
        extra_dict["run_name"] = run_name

    # Build checkpoint payload
    payload = {
        "type": "ae",
        "run_name": run_name,          # duplicated at top-level for quick inspection
        "ae_name": ae_name,
        "ae_kwargs": ae_kwargs or {},
        "ae_state": ae.state_dict(),
        "extra": extra_dict,
    }

    save_checkpoint(path, payload)


# Load an autoencoder checkpoint and instantiate the correct class
def load_ae(
    path: str,
    device: torch.device,
    strict: bool = True,
) -> Tuple[nn.Module, Dict[str, Any]]:
    payload = load_checkpoint(path, device)

    ae_name = payload["ae_name"]
    ae_kwargs = payload.get("ae_kwargs", {})

    # Instantiate AE via registry
    if ae_name not in AE_REGISTRY:
        raise KeyError(
            f"Unknown ae_name '{ae_name}'. Available: {list(AE_REGISTRY.keys())}"
        )

    ae = AE_REGISTRY[ae_name](**ae_kwargs)
    ae.load_state_dict(payload["ae_state"], strict=strict)
    ae.to(device)

    # Recover optional metadata
    extra = payload.get("extra", {}) or {}
    if "run_name" not in extra and payload.get("run_name") is not None:
        extra["run_name"] = payload["run_name"]

    return ae, extra


# Save a joint AE + OCNN checkpoint (single bundle)
def save_ae_ocnn(
    path: str,
    ae: nn.Module,
    ae_name: str,
    ae_kwargs: Dict[str, Any],
    ocnn: nn.Module,
    ocnn_name: str,
    ocnn_kwargs: Dict[str, Any],
    r: float,
    nu: float,
    run_name: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    # Collect optional metadata
    extra_dict = dict(extra) if extra is not None else {}
    if run_name is not None:
        extra_dict["run_name"] = run_name

    # Build checkpoint payload
    payload = {
        "type": "ae+ocnn",
        "run_name": run_name,
        "ae_name": ae_name,
        "ae_kwargs": ae_kwargs or {},
        "ae_state": ae.state_dict(),
        "ocnn_name": ocnn_name,
        "ocnn_kwargs": ocnn_kwargs or {},
        "ocnn_state": ocnn.state_dict(),
        "r": float(r),
        "nu": float(nu),
        "extra": extra_dict,
    }

    save_checkpoint(path, payload)


# Load a joint AE + OCNN checkpoint and instantiate both modules
def load_ae_ocnn(
    path: str,
    device: torch.device,
    strict: bool = True,
) -> Tuple[nn.Module, nn.Module, Dict[str, Any]]:
    payload = load_checkpoint(path, device)

    # Instantiate AE
    ae_name = payload["ae_name"]
    ae_kwargs = payload.get("ae_kwargs", {})
    if ae_name not in AE_REGISTRY:
        raise KeyError(
            f"Unknown ae_name '{ae_name}'. Available: {list(AE_REGISTRY.keys())}"
        )

    ae = AE_REGISTRY[ae_name](**ae_kwargs)
    ae.load_state_dict(payload["ae_state"], strict=strict)
    ae.to(device)

    # Instantiate OCNN
    ocnn_name = payload["ocnn_name"]
    ocnn_kwargs = payload.get("ocnn_kwargs", {})
    if ocnn_name not in OCNN_REGISTRY:
        raise KeyError(
            f"Unknown ocnn_name '{ocnn_name}'. Available: {list(OCNN_REGISTRY.keys())}"
        )

    ocnn = OCNN_REGISTRY[ocnn_name](**ocnn_kwargs)
    ocnn.load_state_dict(payload["ocnn_state"], strict=strict)
    ocnn.to(device)

    # Recover auxiliary values and metadata
    extra = payload.get("extra", {}) or {}
    extra["r"] = float(payload["r"])
    extra["nu"] = float(payload["nu"])

    if "run_name" not in extra and payload.get("run_name") is not None:
        extra["run_name"] = payload["run_name"]

    return ae, ocnn, extra
