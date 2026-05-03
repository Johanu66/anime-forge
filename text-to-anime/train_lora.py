import modal
import os

app = modal.App("lora-anime")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "libgl1", "libglib2.0-0")
    .run_commands(
        "pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128"
    )
    .pip_install(
        "huggingface_hub==0.23.0",
        "diffusers==0.27.2",
        "transformers==4.40.0",
        "accelerate==0.30.0",
        "peft==0.10.0",
        "pillow",
        "numpy",
    )
)

volume = modal.Volume.from_name("stylegan2-anime-vol", create_if_missing=True)


@app.function(
    gpu="b200",
    image=image,
    volumes={"/data": volume},
    timeout=60 * 60 * 4,
)
def train_lora():
    import torch
    from diffusers import StableDiffusionPipeline, DDPMScheduler
    from peft import LoraConfig, get_peft_model
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    import os, random

    volume.reload()

    print(f"GPU : {torch.cuda.get_device_name(0)}")

    model_id = "runwayml/stable-diffusion-v1-5"

    # float32 pour eviter les problemes de gradients
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        safety_checker=None,
        token=HF_TOKEN
    ).to("cuda")
    print("Modele charge")

    lora_config = LoraConfig(
        r=4,
        lora_alpha=32,
        target_modules=["to_q", "to_v"],
        lora_dropout=0.1,
    )
    pipe.unet = get_peft_model(pipe.unet, lora_config)
    pipe.unet.print_trainable_parameters()

    class AnimeDataset(Dataset):
        def __init__(self, img_dir, n=5000):
            all_imgs = [f for f in os.listdir(img_dir)
                       if f.endswith('.jpg') or f.endswith('.png')]
            random.seed(42)
            self.imgs = random.sample(all_imgs, min(n, len(all_imgs)))
            self.img_dir = img_dir
            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])

        def __len__(self):
            return len(self.imgs)

        def __getitem__(self, idx):
            img = Image.open(os.path.join(self.img_dir, self.imgs[idx])).convert('RGB')
            return self.transform(img), "anime face, high quality, detailed"

    dataset = AnimeDataset('/data', n=5000)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    print(f"Dataset : {len(dataset)} images")

    optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=1e-4)
    noise_scheduler = DDPMScheduler.from_pretrained(
        model_id, subfolder="scheduler", token=HF_TOKEN
    )

    pipe.unet.train()
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)

    EPOCHS = 3
    for epoch in range(EPOCHS):
        for i, (images, prompts) in enumerate(dataloader):
            # float32 partout
            images = images.to("cuda", dtype=torch.float32)

            with torch.no_grad():
                latents = pipe.vae.encode(images).latent_dist.sample()
                latents = latents * pipe.vae.config.scaling_factor

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],), device="cuda"
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with torch.no_grad():
                text_inputs = pipe.tokenizer(
                    list(prompts),
                    padding="max_length",
                    max_length=pipe.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                ).to("cuda")
                text_embeddings = pipe.text_encoder(text_inputs.input_ids)[0]

            noise_pred = pipe.unet(
                noisy_latents, timesteps, text_embeddings
            ).sample
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 50 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} | Step {i} | Loss: {loss.item():.4f}")

        print(f"Epoch {epoch+1} terminee")

    os.makedirs("/data/lora_weights", exist_ok=True)
    pipe.unet.save_pretrained("/data/lora_weights")
    volume.commit()
    print("LoRA sauvegarde")

@app.function(
    gpu="b200",
    image=image,
    volumes={"/data": volume},
    timeout=60 * 30,
)
def generate():
    import torch
    from diffusers import StableDiffusionPipeline
    from peft import PeftModel
    import os

    volume.reload()

    model_id = "runwayml/stable-diffusion-v1-5"

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        token=HF_TOKEN
    ).to("cuda")

    pipe.unet = PeftModel.from_pretrained(pipe.unet, "/data/lora_weights")
    print("LoRA charge")

    prompts = [
        "girl with blue hair, anime style, high quality",
        "girl with red hair and smile, anime style",
        "boy with dark hair, anime style",
        "girl with white hair and sad expression, anime style"
    ]

    os.makedirs("/data/lora_generated", exist_ok=True)

    for prompt in prompts:
        image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
        fname = prompt[:20].replace(" ", "_")
        image.save(f"/data/lora_generated/{fname}.png")
        print(f"Genere : {prompt}")

    volume.commit()
    print("Generation terminee")

@app.local_entrypoint()
def main():
    print("Etape 1 - Fine-tuning LoRA")
    train_lora.remote()
    print("Etape 2 - Generation images")
    generate.remote()