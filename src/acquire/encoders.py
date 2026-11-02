"""Download encoder weights to data/weights/ and verify each loads offline.

Network-permitted, local-only. The compute path never
names a hub identifier: src/embed.py loads from the local directory recorded in
data/weights/manifest.json.

UNI2-h and Virchow2 are gated. Authenticate once with `huggingface-cli login`
before running; the token is read from the CLI credential store and is never
recorded in this repo.

Both histopathology encoders build with non-default timm kwargs (SwiGLU packed
MLP, SiLU activation, register tokens). The kwargs below come from each repo's
own model card, not from the timm defaults for the named architecture, and a
mismatch shows up here as missing/unexpected state-dict keys rather than
silently as garbage embeddings.
"""

from __future__ import annotations

import json
from pathlib import Path

import timm
import torch
from huggingface_hub import snapshot_download
from timm.layers import SwiGLUPacked

from src.acquire.common import ROOT

WEIGHTS = ROOT / "data" / "weights"

# repo -> (local dir, timm architecture, model kwargs, expected feature dim)
GATED = {
    "MahmoodLab/UNI2-h": (
        "uni2_h",
        "vit_giant_patch14_224",
        dict(
            img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5,
            embed_dim=1536, mlp_ratio=2.66667 * 2, num_classes=0,
            no_embed_class=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU,
            reg_tokens=8, dynamic_img_size=True,
        ),
        1536,
    ),
    "paige-ai/Virchow2": (
        "virchow2",
        "vit_huge_patch14_224",
        dict(
            img_size=224, init_values=1e-5, num_classes=0, reg_tokens=4,
            mlp_ratio=5.3375, global_pool="", dynamic_img_size=True,
            mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU,
        ),
        1280,
    ),
}


def fetch(repo: str) -> Path:
    """Download a snapshot into data/weights/<local>. Idempotent."""
    local, _, _, _ = GATED[repo]
    dest = WEIGHTS / local
    snapshot_download(
        repo_id=repo,
        local_dir=dest,
        allow_patterns=["*.safetensors", "*.bin", "*.json", "*.md", "LICENSE*"],
    )
    print(f"[encoders] {repo} -> {dest}")
    return dest


def verify(repo: str) -> dict:
    """Build the model from the local snapshot and confirm the state dict fits."""
    local, arch, kwargs, dim = GATED[repo]
    dest = WEIGHTS / local

    src = dest / "model.safetensors"
    if src.exists():
        from safetensors.torch import load_file
        state = load_file(str(src))
    else:
        state = torch.load(dest / "pytorch_model.bin", map_location="cpu",
                           weights_only=True)

    model = timm.create_model(arch, pretrained=False, **kwargs)
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"{repo}: state dict does not fit {arch} with the model-card kwargs "
            f"({len(result.missing_keys)} missing, "
            f"{len(result.unexpected_keys)} unexpected). "
            f"missing[:5]={result.missing_keys[:5]} "
            f"unexpected[:5]={result.unexpected_keys[:5]}"
        )

    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 224, 224))
    # Virchow2 uses global_pool="" and so returns tokens, not a pooled vector.
    tokens = out.shape[-1]
    if tokens != dim:
        raise RuntimeError(f"{repo}: got feature dim {tokens}, expected {dim}")

    cfg = json.loads((dest / "config.json").read_text())
    info = {
        "repo": repo,
        "local_dir": str(dest.relative_to(ROOT)),
        "architecture": arch,
        "feature_dim": dim,
        "output_shape": list(out.shape),
        "n_params": sum(p.numel() for p in model.parameters()),
        "license": cfg.get("pretrained_cfg", {}).get("license"),
        "mean": cfg.get("pretrained_cfg", {}).get("mean"),
        "std": cfg.get("pretrained_cfg", {}).get("std"),
        "interpolation": cfg.get("pretrained_cfg", {}).get("interpolation"),
        "state_dict_keys": len(state),
    }
    print(f"[encoders] verified {repo}: {info['output_shape']} "
          f"{info['n_params']/1e6:.0f}M params, license {info['license']}")
    return info


def main() -> None:
    manifest_path = WEIGHTS / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for repo in GATED:
        fetch(repo)
        manifest[repo] = verify(repo)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[encoders] wrote {manifest_path}")


if __name__ == "__main__":
    main()
