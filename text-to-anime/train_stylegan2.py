import modal
import os

app = modal.App("stylegan2-anime")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "libgl1", "libglib2.0-0", "unzip")
    .pip_install(
        "torch==2.1.0",
        "torchvision==0.16.0",
        "numpy<2",
        "pillow",
        "scipy",
        "requests",
        "tqdm",
        "click",
        "psutil",
        "huggingface_hub"
    )
    .run_commands(
        "git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git /stylegan2"
    )
)

volume = modal.Volume.from_name("stylegan2-anime-vol", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=60 * 30
)
def prepare_10k():
    import os, random, shutil
    volume.reload()

    all_images = [f for f in os.listdir('/data')
                  if f.endswith('.jpg') or f.endswith('.png')]
    print(f"✅ Total images : {len(all_images)}")

    random.seed(42)
    selected = random.sample(all_images, 10000)

    # symlinks au lieu de copie — beaucoup plus rapide
    os.makedirs('/data/images_10k', exist_ok=True)
    for fname in selected:
        src = f'/data/{fname}'
        dst = f'/data/images_10k/{fname}'
        if not os.path.exists(dst):
            os.symlink(src, dst)

    print(f" {len(os.listdir('/data/images_10k'))} liens créés")
    volume.commit()

@app.function(
    gpu="a100-80gb",
    image=image,
    volumes={"/data": volume},
    timeout=60 * 90  # ← 90 min au lieu de 60
)
def prepare_dataset():
    import subprocess, sys, os
    volume.reload()
    sys.path.insert(0, '/stylegan2')

    print(f"Images 10k : {len(os.listdir('/data/images_10k'))}")

    result = subprocess.run([
        "python", "/stylegan2/dataset_tool.py",
        "--source=/data/images_10k",   # ← 10k seulement
        "--dest=/data/anime_dataset.zip",
        "--width=512",
        "--height=512"
    ], capture_output=True, text=True)

    print(result.stdout[-2000:])
    if result.returncode != 0:
        print(f"Erreur : {result.stderr[-1000:]}")
        return

    volume.commit()
    print("✅ Dataset préparé !")

@app.function(
    gpu="a100-80gb",
    image=image,
    volumes={"/data": volume},
    timeout=60 * 60 * 6,
)
def train():
    import subprocess, sys, requests, os
    volume.reload()
    sys.path.insert(0, '/stylegan2')

    if not os.path.exists("/data/anime_dataset.zip"):
        print("anime_dataset.zip introuvable")
        return

    model_path = "/data/ffhq.pkl"
    if not os.path.exists(model_path):
        print("⬇️ Téléchargement modèle FFHQ...")
        r = requests.get(
            "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl",
            stream=True
        )
        with open(model_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(" Modèle FFHQ téléchargé !")

    os.makedirs("/data/results", exist_ok=True)

    subprocess.run([
        "python", "/stylegan2/train.py",
        "--outdir=/data/results",
        "--data=/data/anime_dataset.zip",
        "--cfg=paper512",
        "--mirror=1",
        "--resume=/data/ffhq.pkl",
        "--snap=10",
        "--kimg=1000",
        "--aug=ada",
        "--target=0.6",
        "--gpus=1"
    ], capture_output=False, text=True)

    volume.commit()
    print("Fine-tuning terminé !")

@app.function(
    gpu="a100-80gb",
    image=image,
    volumes={"/data": volume},
    timeout=60 * 30
)
def generate(checkpoint: str = "network-snapshot-001000.pkl"):
    import subprocess, os
    volume.reload()

    checkpoint_path = f"/data/results/{checkpoint}"
    if not os.path.exists(checkpoint_path):
        if os.path.exists("/data/results"):
            pkls = sorted([f for f in os.listdir("/data/results") if f.endswith('.pkl')])
            if pkls:
                checkpoint_path = f"/data/results/{pkls[-1]}"
                print(f"Checkpoint : {pkls[-1]}")
            else:
                print("Aucun checkpoint trouvé")
                return
        else:
            print(" Dossier results introuvable")
            return

    os.makedirs("/data/generated", exist_ok=True)
    subprocess.run([
        "python", "/stylegan2/generate.py",
        "--outdir=/data/generated",
        "--trunc=0.7",
        "--seeds=0-15",
        f"--network={checkpoint_path}"
    ])

    volume.commit()
    print(" 16 images générées !")

@app.local_entrypoint()
def main():
    print(" Étape 0 — Sélection 10k images")
    prepare_10k.remote()
    print(" Étape 1 — Préparation dataset")
    prepare_dataset.remote()
    print("Étape 2 — Fine-tuning StyleGAN2")
    train.remote()
    print(" Étape 3 — Génération images")
    generate.remote()