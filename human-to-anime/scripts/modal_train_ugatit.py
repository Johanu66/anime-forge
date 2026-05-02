from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import modal

APP_NAME = "ugatit-selfie2anime-train"
VOLUME_NAME = os.environ.get("UGATIT_MODAL_VOLUME", "anime-i2i-selfie2anime")
GPU_TYPE = os.environ.get("UGATIT_MODAL_GPU", "B200")
TORCH_VERSION = os.environ.get("UGATIT_TORCH_VERSION", "2.7.1")
TORCHVISION_VERSION = os.environ.get("UGATIT_TORCHVISION_VERSION", "0.22.1")
TORCHAUDIO_VERSION = os.environ.get("UGATIT_TORCHAUDIO_VERSION", "2.7.1")
TORCH_INDEX_URL = os.environ.get("UGATIT_TORCH_INDEX_URL", "https://download.pytorch.org/whl/cu128")
REMOTE_ROOT = Path("/workspace")
REMOTE_PROJECT = REMOTE_ROOT / "project"

app = modal.App(APP_NAME)
project_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        f"torch=={TORCH_VERSION}",
        f"torchvision=={TORCHVISION_VERSION}",
        f"torchaudio=={TORCHAUDIO_VERSION}",
        index_url=TORCH_INDEX_URL,
    )
    .pip_install(
        "PyYAML>=6.0.1",
        "tqdm>=4.66.0",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "matplotlib>=3.8.0",
        "scikit-image>=0.22.0",
        "transformers>=4.39.0",
        "datasets",
        "Pillow>=10.0.0",
        # NOTE: facenet-pytorch is intentionally omitted here.
        # It may pull older torch builds and break Blackwell (sm_100) compatibility.
    )
)


def _safe_volume_reload() -> None:
    """Reload volume metadata when possible, skip known Modal cwd/handle edge case."""
    try:
        os.chdir("/tmp")
        project_volume.reload()
    except RuntimeError as exc:
        if "open files preventing the operation" in str(exc):
            print(f"[modal] volume.reload skipped: {exc}")
        else:
            raise


def _safe_volume_commit() -> None:
    """Commit writes back to the volume, tolerating the same handle edge case."""
    try:
        os.chdir("/tmp")
        project_volume.commit()
    except RuntimeError as exc:
        if "open files preventing the operation" in str(exc):
            print(f"[modal] volume.commit skipped: {exc}")
        else:
            raise


@app.function(
    image=image,
    gpu=GPU_TYPE,
    cpu=8,
    memory=65536,
    timeout=60 * 60 * 24,
    volumes={str(REMOTE_ROOT): project_volume},
)
def train_remote(config_rel_path: str, resume_checkpoint_rel_path: str = ""):
    _safe_volume_reload()

    project_root = REMOTE_PROJECT
    config_path = project_root / config_rel_path
    resume_path = project_root / resume_checkpoint_rel_path if resume_checkpoint_rel_path else None

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config introuvable dans le volume: {config_path}. "
            "Assure-toi d'avoir fait le sync avec `modal volume put`."
        )

    env = os.environ.copy()
    src_path = project_root / "src"
    env["PYTHONPATH"] = f"{src_path}:{env.get('PYTHONPATH', '')}" if env.get("PYTHONPATH") else str(src_path)
    env["CUDA_LAUNCH_BLOCKING"] = env.get("CUDA_LAUNCH_BLOCKING", "1")

    diag_cmd = [
        sys.executable,
        "-c",
        (
            "import torch; "
            "print('[modal] torch', torch.__version__); "
            "print('[modal] cuda', torch.version.cuda); "
            "print('[modal] cuda_available', torch.cuda.is_available()); "
            "print('[modal] arch_list', torch.cuda.get_arch_list() if torch.cuda.is_available() else []); "
            "print('[modal] device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
        ),
    ]
    subprocess.run(diag_cmd, cwd=str(project_root), env=env, check=True)

    cmd = [sys.executable, "-m", "anime_gan_i2i.train", "--config", str(config_path)]
    if resume_path is not None:
        cmd.extend(["--resume-checkpoint", str(resume_path)])

    print("[modal] Running:", " ".join(shlex.quote(part) for part in cmd))
    subprocess.run(cmd, cwd=str(project_root), env=env, check=True)

    _safe_volume_commit()
    return {
        "status": "ok",
        "config": str(config_path),
    }


@app.local_entrypoint()
def run(
    config_rel_path: str = "configs/ugatit_selfie2anime_modal_b200.yaml",
    resume_checkpoint_rel_path: str = "",
):
    result = train_remote.remote(
        config_rel_path=config_rel_path,
        resume_checkpoint_rel_path=resume_checkpoint_rel_path,
    )
    print(result)
