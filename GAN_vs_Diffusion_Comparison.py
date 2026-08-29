# %% [markdown]
# # GANs vs. Diffusion Models -- Head-to-Head Comparison (Hugging Face + from-scratch GAN)
#
# Plain-Python export of `GAN_vs_Diffusion_Comparison.ipynb`, using the 'percent' cell
# format (`# %%` / `# %% [markdown]`) recognized by VS Code, Jupytext, and Spyder.
# Note: the `!pip install ...` line only works inside a Jupyter/Colab cell; if running
# this as a plain script, run that install command in your shell first.

# %% [markdown]
# # GANs vs. Diffusion Models — A Head-to-Head Comparison on CIFAR-10
#
# This notebook puts the two major generative model families side by side, on the **same dataset**:
#
# - **Diffusion model:** a pretrained DDPM loaded directly from the **Hugging Face Hub** (`google/ddpm-cifar10-32`) using the `diffusers` library — no training required, since diffusion models are expensive to train from scratch.
# - **GAN:** a DCGAN trained **from scratch, in this notebook**, on the same CIFAR-10 dataset — so we can directly observe GAN-specific behavior (adversarial loss dynamics, single-shot generation, mode collapse risk) that a pretrained model would hide.
#
# **Goal:** not just "which looks better," but a clear, evidence-based picture of *why* the two families differ — in training objective, training stability, sampling procedure, sampling speed, and output diversity.
#
# **How to run:** Runtime → Change runtime type → GPU, then Runtime → Run all. Total runtime is roughly 10–20 minutes (mostly GAN training; the diffusion model needs no training).
#
# ## Contents
# 1. Setup
# 2. Conceptual differences: GAN vs. Diffusion
# 3. Part A — Loading a pretrained Diffusion model from Hugging Face
# 4. Part B — Training a DCGAN from scratch on CIFAR-10
# 5. Part C — Head-to-head comparison
#    - 5.1 Visual sample quality
#    - 5.2 Training dynamics
#    - 5.3 Sampling speed
#    - 5.4 Output diversity
# 6. Summary table & takeaways
#

# %% [markdown]
# ## 1. Setup
#

# %%
# diffusers/transformers/accelerate are needed to load the pretrained Hugging Face diffusion model
!pip install -q diffusers transformers accelerate


# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm.auto import tqdm

from diffusers import DDPMPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cpu":
    print("WARNING: no GPU detected — both the diffusion sampler and GAN training will be slow. "
          "Go to Runtime -> Change runtime type -> GPU.")

torch.manual_seed(0)
np.random.seed(0)


# %% [markdown]
# ## 2. Conceptual Differences: GAN vs. Diffusion
#
# Before running any code, it's worth being precise about *why* these two families behave differently — everything we measure in Part C is a direct consequence of these design choices.
#
# | | **GAN** | **Diffusion (DDPM)** |
# |---|---|---|
# | **Core idea** | Two networks compete: a **generator** $G$ tries to fool a **discriminator** $D$; $D$ tries to tell real from fake. | A single network learns to reverse a *fixed* noising process, one small denoising step at a time. |
# | **Training objective** | Minimax game: $\min_G \max_D \ \mathbb{E}[\log D(x)] + \mathbb{E}[\log(1-D(G(z)))]$ — no explicit likelihood being optimized. | Evidence Lower Bound (ELBO) on the true log-likelihood, which collapses to a simple regression loss $\mathbb{E}\|\epsilon-\epsilon_\theta(x_t,t)\|^2$ (see our companion DDPM derivation notebook). |
# | **Generation (sampling)** | **One forward pass**: $x = G(z)$, $z\sim\mathcal N(0,I)$. | **Many sequential forward passes** (typically hundreds to a thousand): start at pure noise, iteratively denoise. |
# | **Training stability** | Notoriously unstable — the generator and discriminator can oscillate, and either can "win" too early (vanishing gradients / mode collapse). | Comparatively very stable — it's ordinary supervised regression (predict the noise) with a fixed, well-behaved target at every step. |
# | **Mode collapse risk** | A known, common failure mode: $G$ learns to produce only a few plausible outputs that reliably fool $D$, sacrificing diversity. | Not a typical failure mode of the *training* objective itself — every noise level and every training image contributes an independent regression target. |
# | **Sample quality** | Can be extremely sharp, but historically harder to scale reliably to high resolution/diversity. | State-of-the-art photorealism and diversity at scale (e.g. Stable Diffusion, Imagen), at the cost of slower sampling. |
# | **Likelihood** | No explicit likelihood estimate available. | Provides a (lower bound on the) log-likelihood — a principled measure of how well the model fits the data. |
# | **Inference cost** | Cheap — one network call per sample. | Expensive — $T$ network calls per sample (mitigated by fast samplers like DDIM, but still costlier than a GAN). |
#
# We now make each of these differences concrete by actually running both models.
#

# %% [markdown]
# ## 3. Part A — Loading a Pretrained Diffusion Model from Hugging Face
#
# Training a diffusion model to convergence from scratch is expensive (the original DDPM paper trained for many GPU-days even on CIFAR-10). Rather than retraining one badly in a few minutes, we load a **fully-trained checkpoint directly from the Hugging Face Hub** — this also demonstrates the standard, practical way most people actually *use* diffusion models (via `diffusers`, not by rederiving them).
#
# We use [`google/ddpm-cifar10-32`](https://huggingface.co/google/ddpm-cifar10-32) — a DDPM trained on CIFAR-10 at the native $32\times32$ resolution, so it's directly comparable to the GAN we train on the same dataset in Part B.
#

# %%
# Downloads the pretrained weights + noise-schedule config from the Hugging Face Hub.
diffusion_pipeline = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32").to(device)
print(f"Loaded pretrained DDPM. Scheduler timesteps: {diffusion_pipeline.scheduler.config.num_train_timesteps}")


# %%
# --- Generate a batch of images and time it ---------------------------------
N_SAMPLES = 16

torch.manual_seed(0)
t0 = time.time()
diffusion_output = diffusion_pipeline(
    batch_size=N_SAMPLES,
    num_inference_steps=1000,   # full ancestral sampling, matching training-time T
    output_type="numpy",
)
diffusion_time = time.time() - t0
diffusion_images = diffusion_output.images  # (N, 32, 32, 3) in [0, 1]

print(f"Generated {N_SAMPLES} diffusion samples in {diffusion_time:.1f}s "
      f"({diffusion_time / N_SAMPLES:.2f}s per sample, 1000 network evaluations each)")

fig, axes = plt.subplots(4, 4, figsize=(6, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(diffusion_images[i])
    ax.axis("off")
plt.suptitle("Pretrained Diffusion Model (google/ddpm-cifar10-32) — Hugging Face checkpoint")
plt.tight_layout()
plt.show()


# %% [markdown]
# **Note:** 1000 inference steps is the setting the model was trained with, and gives the highest-quality samples but is the slowest. `diffusers` schedulers also support faster samplers (e.g. `DDIMScheduler`, `DPMSolverMultistepScheduler`) that reach similar quality in 20–50 steps — conceptually identical to the DDIM approach in our companion DDPM notebook. Let's also time a much faster setting for a fair later comparison against the GAN's single-pass speed.
#

# %%
from diffusers import DDIMScheduler

# Swap in a DDIM scheduler on the SAME trained network -- no retraining needed
# (this mirrors the DDIM section of the from-scratch DDPM notebook).
diffusion_pipeline.scheduler = DDIMScheduler.from_config(diffusion_pipeline.scheduler.config)

torch.manual_seed(0)
t0 = time.time()
fast_diffusion_output = diffusion_pipeline(
    batch_size=N_SAMPLES,
    num_inference_steps=50,     # 20x fewer network evaluations than the full 1000-step sampler
    output_type="numpy",
)
fast_diffusion_time = time.time() - t0
fast_diffusion_images = fast_diffusion_output.images

print(f"Generated {N_SAMPLES} diffusion samples with DDIM (50 steps) in {fast_diffusion_time:.1f}s "
      f"({fast_diffusion_time / N_SAMPLES:.3f}s per sample)")
print(f"Speedup vs. full 1000-step sampling: {diffusion_time / fast_diffusion_time:.1f}x")

fig, axes = plt.subplots(4, 4, figsize=(6, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(fast_diffusion_images[i])
    ax.axis("off")
plt.suptitle("Same pretrained model, DDIM sampler (50 steps instead of 1000)")
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 4. Part B — Training a DCGAN From Scratch on CIFAR-10
#
# To make the comparison fair and informative, we now train a **GAN on the exact same dataset** (CIFAR-10, $32\times32$), in this notebook, so we can directly observe GAN-specific training behavior — something a pretrained checkpoint would hide.
#
# We implement a standard **DCGAN** (Radford et al., 2015): a convolutional generator that upsamples a random latent vector $z$ into a $32\times32\times3$ image, and a convolutional discriminator that classifies images as real or fake.
#
# ### 4.1 The adversarial objective
#
# $$
# \min_G \max_D \ \ \mathbb{E}_{x\sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z\sim\mathcal N(0,I)}[\log(1 - D(G(z)))]
# $$
#
# In practice we train $D$ and $G$ with **alternating gradient steps**:
# - **Discriminator step:** maximize $\log D(x_{\text{real}}) + \log(1 - D(x_{\text{fake}}))$ — i.e., get better at telling real from fake.
# - **Generator step:** maximize $\log D(x_{\text{fake}})$ (the standard "non-saturating" trick, instead of literally minimizing $\log(1-D(G(z)))$, which has weak gradients early in training) — i.e., get better at fooling $D$.
#
# Unlike the diffusion model's single, fixed regression target at every step, **the generator's target here is a moving one** (a discriminator that is itself constantly changing) — this is the root cause of GAN training instability, which we'll observe directly in the loss curves.
#

# %%
# --- CIFAR-10 dataset --------------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # scale to [-1, 1]
])

cifar10 = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

BATCH_SIZE = 128
gan_dataloader = DataLoader(cifar10, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2)

# Show a few real training images for reference
real_batch, _ = next(iter(gan_dataloader))
grid = vutils.make_grid(real_batch[:16], nrow=4, normalize=True)
plt.figure(figsize=(6, 6))
plt.imshow(grid.permute(1, 2, 0).numpy())
plt.axis("off")
plt.title("Real CIFAR-10 training images")
plt.show()


# %%
LATENT_DIM = 100

class Generator(nn.Module):
    '''DCGAN generator: latent vector z (LATENT_DIM,) -> image (3, 32, 32).'''
    def __init__(self, latent_dim=LATENT_DIM, feature_maps=64):
        super().__init__()
        self.net = nn.Sequential(
            # z: (B, latent_dim, 1, 1) -> (B, fm*4, 4, 4)
            nn.ConvTranspose2d(latent_dim, feature_maps * 4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            # -> (B, fm*2, 8, 8)
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            # -> (B, fm, 16, 16)
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            # -> (B, 3, 32, 32)
            nn.ConvTranspose2d(feature_maps, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),   # output in [-1, 1], matching the normalized real images
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    '''DCGAN discriminator: image (3, 32, 32) -> real/fake logit.'''
    def __init__(self, feature_maps=64):
        super().__init__()
        self.net = nn.Sequential(
            # (B, 3, 32, 32) -> (B, fm, 16, 16)
            nn.Conv2d(3, feature_maps, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # -> (B, fm*2, 8, 8)
            nn.Conv2d(feature_maps, feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # -> (B, fm*4, 4, 4)
            nn.Conv2d(feature_maps * 2, feature_maps * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # -> (B, 1, 1, 1)
            nn.Conv2d(feature_maps * 4, 1, kernel_size=4, stride=1, padding=0, bias=False),
        )

    def forward(self, x):
        return self.net(x).view(-1)


def weights_init(m):
    '''DCGAN paper's recommended weight initialization -- meaningfully improves stability.'''
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


netG = Generator().to(device)
netD = Discriminator().to(device)
netG.apply(weights_init)
netD.apply(weights_init)

print(f"Generator parameters:     {sum(p.numel() for p in netG.parameters()):,}")
print(f"Discriminator parameters: {sum(p.numel() for p in netD.parameters()):,}")


# %%
# --- GAN training loop -------------------------------------------------------
GAN_EPOCHS = 15          # increase for sharper samples; DCGAN on CIFAR-10 typically needs 50-100+ for good results
LR = 2e-4
BETA1 = 0.5              # standard DCGAN Adam momentum setting

criterion = nn.BCEWithLogitsLoss()
optimizerD = torch.optim.Adam(netD.parameters(), lr=LR, betas=(BETA1, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=LR, betas=(BETA1, 0.999))

REAL_LABEL, FAKE_LABEL = 1.0, 0.0
fixed_noise = torch.randn(16, LATENT_DIM, 1, 1, device=device)  # for tracking progress across epochs

G_losses, D_losses = [], []
progress_grids = []

for epoch in range(GAN_EPOCHS):
    pbar = tqdm(gan_dataloader, desc=f"Epoch {epoch+1}/{GAN_EPOCHS}")
    for real_imgs, _ in pbar:
        real_imgs = real_imgs.to(device)
        b_size = real_imgs.size(0)

        # ---------------------------------------------------------------
        # (1) Update Discriminator: maximize log(D(real)) + log(1 - D(fake))
        # ---------------------------------------------------------------
        netD.zero_grad()
        label = torch.full((b_size,), REAL_LABEL, device=device)
        output_real = netD(real_imgs)
        lossD_real = criterion(output_real, label)
        lossD_real.backward()

        noise = torch.randn(b_size, LATENT_DIM, 1, 1, device=device)
        fake_imgs = netG(noise)
        label.fill_(FAKE_LABEL)
        output_fake = netD(fake_imgs.detach())
        lossD_fake = criterion(output_fake, label)
        lossD_fake.backward()

        lossD = lossD_real + lossD_fake
        optimizerD.step()

        # ---------------------------------------------------------------
        # (2) Update Generator: maximize log(D(fake))  (non-saturating trick)
        # ---------------------------------------------------------------
        netG.zero_grad()
        label.fill_(REAL_LABEL)   # generator WANTS the discriminator to say "real"
        output = netD(fake_imgs)
        lossG = criterion(output, label)
        lossG.backward()
        optimizerG.step()

        G_losses.append(lossG.item())
        D_losses.append(lossD.item())
        pbar.set_postfix(loss_D=lossD.item(), loss_G=lossG.item())

    # snapshot generated images at this epoch, using the SAME fixed noise each time
    with torch.no_grad():
        netG.eval()
        snapshot = netG(fixed_noise).detach().cpu()
        netG.train()
    progress_grids.append(vutils.make_grid(snapshot, nrow=4, normalize=True))

print("GAN training complete.")


# %%
# --- GAN loss curves ----------------------------------------------------------
plt.figure(figsize=(9, 4))
plt.plot(G_losses, label="Generator loss", alpha=0.8)
plt.plot(D_losses, label="Discriminator loss", alpha=0.8)
plt.xlabel("Training iteration")
plt.ylabel("Loss")
plt.title("GAN adversarial training dynamics")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# %% [markdown]
# **What to look for:** unlike the (from-scratch DDPM notebook's) smoothly, monotonically decreasing MSE loss, GAN losses **oscillate** — $D$ and $G$ are locked in a moving-target game, so neither loss converges to a fixed value the way a normal regression loss does. A loss curve alone isn't a reliable indicator of *sample quality* for GANs (this is itself a well-known practical headache of adversarial training) — visual inspection and downstream metrics (e.g. FID) are typically necessary.
#

# %%
# --- Generated samples from the fully-trained GAN ----------------------------
netG.eval()
torch.manual_seed(1)
with torch.no_grad():
    gan_noise = torch.randn(16, LATENT_DIM, 1, 1, device=device)
    t0 = time.time()
    gan_images = netG(gan_noise)
    gan_time = time.time() - t0
netG.train()

gan_images = (gan_images.clamp(-1, 1) + 1) / 2  # back to [0, 1] for plotting

print(f"Generated 16 GAN samples in {gan_time*1000:.1f}ms "
      f"({gan_time*1000/16:.2f}ms per sample, a SINGLE network evaluation each)")

fig, axes = plt.subplots(4, 4, figsize=(6, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(gan_images[i].permute(1, 2, 0).cpu().numpy())
    ax.axis("off")
plt.suptitle(f"DCGAN trained from scratch for {GAN_EPOCHS} epochs on CIFAR-10")
plt.tight_layout()
plt.show()


# %%
# --- Visualize the GAN's samples improving over training epochs -------------
fig, axes = plt.subplots(1, len(progress_grids), figsize=(3 * len(progress_grids), 3))
for ax, grid, ep in zip(axes, progress_grids, range(1, GAN_EPOCHS + 1)):
    ax.imshow(grid.permute(1, 2, 0).numpy())
    ax.set_title(f"epoch {ep}")
    ax.axis("off")
plt.suptitle("GAN sample quality over training epochs (same fixed noise each time)")
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 5. Part C — Head-to-Head Comparison
#
# ### 5.1 Visual sample quality, side by side
#

# %%
fig, axes = plt.subplots(2, 8, figsize=(16, 4.5))

for i in range(8):
    axes[0, i].imshow(diffusion_images[i])
    axes[0, i].axis("off")
    axes[1, i].imshow(gan_images[i].permute(1, 2, 0).cpu().numpy())
    axes[1, i].axis("off")

axes[0, 0].set_title("Diffusion (pretrained, 1000 steps)", loc="left", fontsize=10)
axes[1, 0].set_title(f"GAN (from scratch, {GAN_EPOCHS} epochs)", loc="left", fontsize=10)
plt.suptitle("Diffusion vs. GAN — CIFAR-10 samples")
plt.tight_layout()
plt.show()


# %% [markdown]
# **Expected observation:** the pretrained diffusion model (trained to convergence, likely for many GPU-days by its original authors) produces noticeably more coherent, diverse CIFAR-10-like images than our GAN, which only had a handful of epochs to train in this notebook. This is **not** a fully fair comparison of the *algorithms* — it's a comparison of a *fully-trained* model against a *lightly-trained* one — but it's an honest and important practical point: **GANs and diffusion models have very different training-cost/quality trade-off curves**, and this notebook lets you feel that difference directly. Try increasing `GAN_EPOCHS` to see the GAN samples improve.
#
# ### 5.2 Training dynamics — stability comparison
#

# %%
# Diffusion models don't have a comparably "live" adversarial loss to show here since we used a
# pretrained checkpoint, but we can illustrate the qualitative difference schematically alongside
# the GAN's actual measured loss curve, and note the characteristic difference from the literature.

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(D_losses, color="tab:red", alpha=0.7, label="Discriminator")
axes[0].plot(G_losses, color="tab:blue", alpha=0.7, label="Generator")
axes[0].set_title("GAN: adversarial losses (measured, this notebook)")
axes[0].set_xlabel("Iteration")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Schematic illustration of typical diffusion training loss shape (monotonic decay),
# based on the loss curve produced when training a DDPM from scratch (see the companion notebook).
example_iters = np.arange(500)
schematic_diffusion_loss = 1.0 * np.exp(-example_iters / 150) + 0.05 + 0.01 * np.random.randn(len(example_iters))
axes[1].plot(example_iters, schematic_diffusion_loss, color="tab:green", alpha=0.8)
axes[1].set_title("Diffusion: noise-prediction MSE loss\n(schematic shape; see companion DDPM notebook for a measured curve)")
axes[1].set_xlabel("Iteration")
axes[1].set_ylabel("Loss")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()


# %% [markdown]
# **Key structural difference:** the GAN's losses oscillate because $G$ and $D$ chase a **moving target** (each other) — there is no fixed point either loss is monotonically approaching. The diffusion model's loss (see the companion from-scratch DDPM notebook for an actually-measured curve on this same kind of data) decreases **smoothly and monotonically**, because it's ordinary supervised regression against a **fixed**, well-defined target ($\epsilon$, the noise that was actually added) at every step — there is no adversary and no moving target.
#
# ### 5.3 Sampling speed — the diffusion model's key weakness
#

# %%
labels = ["GAN\n(1 forward pass)", "Diffusion - DDIM\n(50 steps)", "Diffusion - full\n(1000 steps)"]
times_per_sample_ms = [
    gan_time * 1000 / 16,
    fast_diffusion_time * 1000 / N_SAMPLES,
    diffusion_time * 1000 / N_SAMPLES,
]

plt.figure(figsize=(7, 4))
bars = plt.bar(labels, times_per_sample_ms, color=["tab:blue", "tab:orange", "tab:red"])
plt.ylabel("Time per sample (ms, log scale)")
plt.yscale("log")
plt.title("Sampling speed: GAN vs. Diffusion (measured on this hardware)")
for bar, t in zip(bars, times_per_sample_ms):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{t:.1f}ms",
              ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.show()

print(f"Diffusion (1000 steps) is ~{diffusion_time/gan_time:.0f}x slower per sample than the GAN.")
print(f"Diffusion (DDIM, 50 steps) is ~{fast_diffusion_time/gan_time:.0f}x slower per sample than the GAN.")


# %% [markdown]
# This is the single most consistent, well-documented practical trade-off between the two families: a GAN generates an image with **one** network evaluation; a diffusion model needs **dozens to thousands**, because generation is fundamentally an iterative denoising walk, not a single decode. Techniques like DDIM (used here) and distillation-based methods (e.g. consistency models) exist specifically to close this gap.
#
# ### 5.4 Output diversity — a simple empirical check
#
# Mode collapse (a GAN generating only a narrow slice of the data distribution) is hard to measure rigorously without a proper metric like FID, but a cheap diagnostic is to look at the **pairwise pixel-space distance** between generated samples: a model producing near-duplicate outputs will show unusually small distances between "different" samples.
#

# %%
@torch.no_grad()
def pairwise_diversity(images_tensor):
    '''
    images_tensor: (N, C, H, W) in [0, 1]. Returns the mean pairwise L2 distance
    between all sample pairs, in flattened pixel space -- a simple, fast diversity proxy.
    '''
    N = images_tensor.shape[0]
    flat = images_tensor.reshape(N, -1)
    dists = torch.cdist(flat, flat, p=2)
    # exclude the zero diagonal (distance of a sample to itself)
    mask = ~torch.eye(N, dtype=torch.bool)
    return dists[mask].mean().item()


gan_batch = gan_images.cpu()  # already in [0, 1], shape (16, 3, 32, 32)
diffusion_batch = torch.from_numpy(diffusion_images).permute(0, 3, 1, 2).float()  # (16, 3, 32, 32), [0,1]

gan_diversity = pairwise_diversity(gan_batch)
diffusion_diversity = pairwise_diversity(diffusion_batch)

print(f"Mean pairwise pixel-space L2 distance (higher = more diverse batch):")
print(f"  GAN samples:       {gan_diversity:.3f}")
print(f"  Diffusion samples: {diffusion_diversity:.3f}")

plt.figure(figsize=(4, 4))
plt.bar(["GAN", "Diffusion"], [gan_diversity, diffusion_diversity], color=["tab:blue", "tab:red"])
plt.ylabel("Mean pairwise L2 distance (pixel space)")
plt.title("Simple diversity proxy")
plt.show()


# %% [markdown]
# **Caveat:** this pixel-space L2 metric is a crude proxy, not a substitute for a proper diversity/quality metric like **FID (Fréchet Inception Distance)** or **Inception Score**, which compare distributions of learned *features* rather than raw pixels. It's included here because it's fast and dependency-free; for a rigorous comparison, compute FID between each model's generated batch and a held-out set of real CIFAR-10 images (e.g. using the `torchmetrics.image.fid.FrechetInceptionDistance` implementation, or `clean-fid`).
#

# %% [markdown]
# ## 6. Summary Table & Takeaways
#
# | Dimension | GAN (this notebook) | Diffusion (this notebook) |
# |---|---|---|
# | Training cost to reach these results | ~15 epochs, a few minutes | **0** (pretrained, downloaded from Hugging Face) |
# | Training stability | Oscillating adversarial losses; sensitive to hyperparameters | N/A here (pretrained); characteristically smooth, monotonic MSE loss when trained from scratch |
# | Sampling cost | 1 network evaluation per sample | 50–1000 network evaluations per sample |
# | Measured time / sample | ~few ms | ~10–100x+ slower (see Section 5.3 numbers) |
# | Sample diversity proxy | see Section 5.4 bar chart | see Section 5.4 bar chart |
# | Likelihood estimate available? | No | Yes (ELBO) |
# | Typical failure mode | Mode collapse, training divergence | Slow sampling; less of a stability concern |
#
# **Bottom line:** GANs and diffusion models sit at different points on the training-cost/inference-cost/stability trade-off curve. GANs are cheap to *sample from* but can be finicky to *train* well; diffusion models are comparatively easy and stable to train (a well-behaved regression problem) but expensive to *sample from*, which is precisely why so much recent diffusion research (DDIM, distillation, consistency models, latent diffusion) targets **faster sampling** specifically — the training side was never the bottleneck.
#
# ### Suggested extensions
# - Train the GAN for many more epochs (50–100+) to see how much the gap in Section 5.1 closes.
# - Swap in `torchmetrics`' FID implementation for a rigorous, feature-space diversity/quality comparison instead of the pixel-space proxy in Section 5.4.
# - Try a different Hugging Face checkpoint, e.g. `google/ddpm-celebahq-256`, alongside a GAN trained on CelebA, to see whether the same qualitative conclusions hold at higher resolution.
# - Load a pretrained GAN from Hugging Face too (e.g. a `StyleGAN2` checkpoint via a community pipeline) for an even more apples-to-apples "both pretrained" comparison.
#
