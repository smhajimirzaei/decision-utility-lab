"""Pinned upstream JEPA-WM loader and raw-action adapter for M1 diagnostics.

No datasets, planner evaluation module, or optional image decoder are loaded.
Upstream per-file attribution and licenses remain in the isolated source tree.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from .action_groups import pack_actions


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_metaworld(source, manifest_path, cache_dir, device):
    import torch
    import yaml
    from huggingface_hub import hf_hub_download

    source = Path(source).resolve()
    manifest = json.loads(Path(manifest_path).read_text())
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != manifest["source_commit"]:
        raise RuntimeError(f"Source revision mismatch: {revision}")
    if subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"], text=True
    ).strip():
        raise RuntimeError("Upstream source has unrecorded changes")
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA but no CUDA device is available")
    checkpoint = hf_hub_download(
        repo_id=manifest["hf_repo"],
        filename=manifest["checkpoint_file"],
        revision=manifest["hf_revision"],
        cache_dir=str(cache_dir),
    )
    actual_digest = sha256(checkpoint)
    if (
        Path(checkpoint).stat().st_size != manifest["checkpoint_bytes"]
        or actual_digest != manifest["checkpoint_sha256"]
    ):
        raise RuntimeError("Checkpoint size/digest mismatch; refusing to load")

    # Upstream uses top-level src/app imports. Prepend its isolated source root.
    sys.path.insert(0, str(source))
    from app.plan_common.datasets import get_data_stats
    from app.plan_common.datasets.preprocessor import Preprocessor
    from app.plan_common.datasets.transforms import (
        make_inverse_transforms,
        make_transforms,
    )

    config = yaml.safe_load((source / manifest["source_config"]).read_text())
    kwargs = config["model_kwargs"]
    data = copy.deepcopy(kwargs["data"])
    augmentation = kwargs["data_aug"]
    pretrain = copy.deepcopy(kwargs["pretrain_kwargs"])
    # Latent control does not use the 24-layer image decoder.
    pretrain["heads_cfg"] = {"architectures": {}, "pretrain_dec_path": None}
    if (
        data["img_size"],
        data["custom"]["frameskip"],
        data["custom"]["action_skip"],
        kwargs["wrapper_kwargs"]["ctxt_window"],
    ) != (224, 5, 1, 2):
        raise RuntimeError("Unexpected upstream observation/action configuration")
    if (
        config["task_specification"]["num_frames"],
        config["task_specification"]["num_proprios"],
    ) != (1, 1):
        raise RuntimeError("Unexpected initial observed-frame count")
    stats = get_data_stats("metaworld")
    transform = make_transforms(
        img_size=data["img_size"],
        normalize=augmentation["normalize"],
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1.0, 1.0),
        random_resize_scale=(1.0, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
    )
    preprocessor = Preprocessor(
        **{
            key: torch.tensor(stats[key])
            for key in (
                "action_mean",
                "action_std",
                "state_mean",
                "state_std",
                "proprio_mean",
                "proprio_std",
            )
        },
        transform=transform,
        inverse_transform=make_inverse_transforms(
            img_size=data["img_size"], **augmentation
        ),
    )

    original_hub_load = torch.hub.load
    original_validator = torch.hub._validate_not_a_forked_repo
    original_state_load = torch.nn.Module.load_state_dict
    load_reports = []

    def pinned_hub_load(repo, entrypoint, *args, **hub_kwargs):
        if (
            repo != "facebookresearch/dinov2"
            or entrypoint != manifest["dinov2_entrypoint"]
        ):
            raise RuntimeError(f"Unmanifested torch.hub request: {repo}/{entrypoint}")
        hub_kwargs.update(trust_repo=True, skip_validation=True)
        return original_hub_load(
            f"facebookresearch/dinov2:{manifest['dinov2_source_commit']}",
            entrypoint,
            *args,
            **hub_kwargs,
        )

    def checked_state_load(module, state_dict, *args, **state_kwargs):
        result = original_state_load(module, state_dict, *args, **state_kwargs)
        missing_parameters = set(result.missing_keys).intersection(
            dict(module.named_parameters())
        )
        load_reports.append(
            {
                "module": type(module).__name__,
                "missing_keys": result.missing_keys,
                "unexpected_keys": result.unexpected_keys,
            }
        )
        if missing_parameters or result.unexpected_keys:
            raise RuntimeError(f"Checkpoint incompatibility: {load_reports[-1]}")
        return result

    original_fetch = None
    try:
        torch.hub.load = pinned_hub_load
        torch.nn.Module.load_state_dict = checked_state_load
        from app.vjepa_wm.modelcustom.simu_env_planning import vit_enc_preds

        original_fetch = vit_enc_preds.fetch_checkpoint

        def verified_fetch(path, device="cpu"):
            if Path(path).resolve() != Path(checkpoint).resolve():
                raise RuntimeError("Refusing an unmanifested checkpoint")
            # Official legacy checkpoint, verified above; training metadata may
            # require pickle types outside torch's weights-only allowlist.
            return torch.load(path, map_location=device, weights_only=False)

        vit_enc_preds.fetch_checkpoint = verified_fetch
        model = vit_enc_preds.init_module(
            folder=str(Path(checkpoint).parent),
            checkpoint=Path(checkpoint).name,
            model_kwargs=pretrain,
            device=target_device,
            action_dim=stats["action_dim"],
            proprio_dim=stats["proprio_dim"],
            preprocessor=preprocessor,
            cfgs_data=data,
            wrapper_kwargs=kwargs["wrapper_kwargs"],
        )
    finally:
        torch.hub.load = original_hub_load
        torch.hub._validate_not_a_forked_repo = original_validator
        torch.nn.Module.load_state_dict = original_state_load
        if original_fetch is not None:
            vit_enc_preds.fetch_checkpoint = original_fetch
    model.eval().requires_grad_(False)
    encoder_weight = (
        Path(torch.hub.get_dir()) / "checkpoints" / "dinov2_vits14_pretrain.pth"
    )
    if not encoder_weight.is_file():
        raise RuntimeError("Cannot record the DINOv2 encoder weight digest")
    if sha256(encoder_weight) != manifest["dinov2_weights_sha256"]:
        raise RuntimeError("DINOv2 encoder weight digest mismatch")
    report = {
        "manifest": manifest,
        "checkpoint_actual_sha256": actual_digest,
        "encoder_actual_sha256": sha256(encoder_weight),
        "load_reports": load_reports,
        "effective_pretrain_config": pretrain,
        "data": data,
        "normalization_stats": stats,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "device": str(target_device),
    }
    return model, preprocessor, report


def grouped_actions(raw, preprocessor, device):
    import torch

    packed = pack_actions(
        raw,
        preprocessor.action_mean.cpu().numpy(),
        preprocessor.action_std.cpu().numpy(),
    )
    return torch.from_numpy(packed).to(device)
