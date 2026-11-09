'''
Implicit Neural Representation (INR) for a single image, using a SIREN network.

Idea:
Image is a discrete grid of pixels H×W×3 (height, width, RGB).
Train a small MLP that maps a continuous 2D coordinate (x, y) -> RGB color.

Training is full-batch Adam on an MSE loss, and PSNR is calculated along process.
After training, the weights of the network are the image.

------------------------------------------

Resolution testing:
Since the network is a continuous function, we can query it at any resolution (e.g. 2x super-resolution)
or at arbitrary points.

After training, the network is queried to produce image with 2x training image resolution.
It saves image files original_vs_recon.png and recon_2x.png and siren model (weights) to /inr_output/siren_fp32.pt

------------------------------------------
Model size and compression:
INR only saves space if it has fewer parameters than the image has pixels.
- If the default hidden layer width is  64, it has about 13k parameters, and about 51 KiB in float32 (26 KiB in float16),
  compared to 192 KiB of raw pixels at 256×256×3 RGB image (256×256×3 / 1024 = 192 KiB).
- If the default hidden layer width is 256, it has about 198k parameters, and about 777 KiB in float32,
  consequently about 4 times (777 KiB / 192 KiB = 4) larger than the raw pixel image.

Increasing hidden layer width improves PSNR and SSIM but makes the model bigger than the image.

Plain INRs are not competitive with modern codecs as JPEG or PNG.
A JPEG or PNG photo is often smaller than a SIREN of comparable quality,

Thus, INRs are not great for compression, but are great for image scaling.

------------------------------------------
Script does:
1. Trains a SIREN network (coordinates (x, y) -> RGB) to represent one image.
2. Evaluates the reconstruction using three metrics:
      * MSE  - Mean Squared Error
      * PSNR - Peak Signal-to-Noise Ratio (dB)
      * SSIM - Structural Similarity Index
3. Saves the trained model to disk (float32).
4. Compares the memory needed to store the model with the memory needed to
   store the image (raw pixels, PNG-encoded and the original file on disk).

------------------------------------------

Requires torch, numpy, pillow:
pip install torch numpy pillow

To run it:
python image_inr.py --image my_photo.jpg --steps 2000

Without --image, it uses a built-in synthetic pattern with gradients and a checkerboard.


If the reconstruction is noisy, try lowering --lr or omega_0.
For sharper results, increase --hidden, --layers or --steps.


------------------------------------------
Why SIREN:
ReLU MLP produces blurry images, since it is biased towards low-frequency functions ("spectral bias").
SIREN uses sine activations sin(w_0 * (Wx + b)), together with a special weight initialization,
which lets the network represent fine details very well.

'''

import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


# ---------------------------------------------------------------------------
# 1. Model
# ---------------------------------------------------------------------------
# Vincent Sitzmann,... "Implicit Neural Representations with Periodic Activation Functions"
#
# SIREN initialisation is designed so that activations keep a stable distribution from layer to layer:
# (1) If x ~ U(-1;1) single input x has uniform distribution
#     y = sin(ax+b) single neuron output;
#     then y ~ arcsine(-1,1), if a > PI/2, and independent of bias b
#
# (2) If x is linear combination of n inputs (x in R^n) and weights (w in R^n)
#     x = (x1,...,xn), w = (w1,...wn)   (columns)
#     x ~ U(-1/n;1/n)
#     y = sin(w^T x + b)  output  (^T transpose)
#
#     If w_i ~ U(-c/sqrt(n), c/sqrt(n)), (c in R)
#     then (w^T x) ~ N(0,c^2/6) normal distribution as n grows
#
#     For convenience let's take c = sqrt(6), then:
#     w_i ~ U(-sqrt(6/n),sqrt(6/n))
#     (w^T x) ~ N(0,1), i.e. normal distribution with standard deviation equal to 1
#     thus the input (w^T x + b) for sin(...) has standard deviation 1
#
# (0) Initialize the first layer with weights so that
#     sin(w0 * Wx + b) span multiple periods over [-1;1]
#     Authors found experimentally that w0 = 30.0 works well using Adam optimizer

# ---------------------------------------------------------------------------
class SineLayer(nn.Module):
    # One Sine layer: y = sin(w0 * (W x + b))

    def __init__(self, in_features, out_features, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)  # linear layer; implements y = Ax+b, takes m-dimensional input x and produces n-dimensional y, weights - values of n x m matrix A, bias - values of n-vector b
        self._init_weights(in_features, is_first)

    def _init_weights(self, in_features, is_first):
        """SIREN initialisation, designed so activations keep a stable
        distribution from layer to layer."""
        with torch.no_grad():
            if is_first:
                # First layer: uniform in [-1/n, 1/n]. Combined with w0 = omega_0
                # sine wraps around the input range several times.
                bound = 1.0 / in_features
            else:
                # Hidden layers: compensates for the w0 = omega_0 multiplication.
                bound = math.sqrt(6.0 / in_features) / self.omega_0   # w_i ~ U(-sqrt(6/n),sqrt(6/n))
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))  # y = sin(w0 * (W x + b))


class Siren(nn.Module):
    # Coordinate MLP: (x, y) in [-1, 1]^2  ->  (r, g, b) in [-1, 1]^3

    def __init__(self, in_features=2, hidden_features=256, hidden_layers=3,
                 out_features=3, first_omega_0=30.0, hidden_omega_0=30.0):
        super().__init__()
        layers = [SineLayer(in_features, hidden_features,
                            is_first=True, omega_0=first_omega_0)]
        for _ in range(hidden_layers):
            layers.append(SineLayer(hidden_features, hidden_features,
                                    is_first=False, omega_0=hidden_omega_0))

        # Final layer is linear (no sine on the output) so outputs aren't squashed by a
        # periodic function; it uses the same init as the hidden layers.
        final = nn.Linear(hidden_features, out_features)  # linear layer; implements y = Ax+b, takes m-dimensional input x and produces n-dimensional y, weights - values of n x m matrix A, bias - values of n-vector b
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden_features) / hidden_omega_0
            final.weight.uniform_(-bound, bound)
        layers.append(final)

        self.net = nn.Sequential(*layers)

    def forward(self, coords):
        return self.net(coords)


# ---------------------------------------------------------------------------
# 2. Data
# ---------------------------------------------------------------------------
def make_coord_grid(height, width, device):
    # Return (H*W, 2) tensor of (y, x) coordinates normalized to [-1, 1].
    # Pixel centers are used, so the grid is resolution-independent:
    # continuous domain [-1, 1]^2 can be sampled at any H, W.

    ys = (torch.arange(height, device=device) + 0.5) / height * 2 - 1  # normalized to [-1, 1]
    xs = (torch.arange(width, device=device) + 0.5) / width * 2 - 1    # normalized to [-1, 1]
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([grid_y, grid_x], dim=-1).reshape(-1, 2)


def load_image(path, size):
    # Load an image from file or build a synthetic.
    # Return tensor arr (H, W, 3) with normalized values in [-1, 1] (to match the range of the network's output)
    # and PIL image img used for training

    if path is not None and os.path.exists(path):
        img = Image.open(path).convert("RGB")             # Python Imaging Library (PIL) # return an Image object
        arr = np.asarray(img, dtype=np.float32) / 255.0   # RGB color values in [0, 255]; after division by 255 values in [0, 1]
    else:
        if path is not None:
            print(f"[warn] '{path}' not found, using a synthetic image instead.")
        arr = synthetic_image(size)
        img = Image.fromarray( (synthetic_image(size) * 255).astype(np.uint8))  # Python Imaging Library (PIL) # return an Image object

    return torch.from_numpy(arr) * 2 - 1 , img      # normalized: [0, 1] -> [-1, 1]


def synthetic_image(size):
    # A test pattern with both smooth gradients and sharp high-frequency detail,
    # so the script works without any external image file.
    y, x = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size), indexing="ij")
    r = 0.5 + 0.5 * np.sin(2 * np.pi * (3 * x + 2 * y))
    g = 0.5 + 0.5 * np.cos(2 * np.pi * 5 * np.hypot(x - 0.5, y - 0.5))
    b = ((np.floor(x * 8) + np.floor(y * 8)) % 2).astype(np.float32)  # checkerboard
    return np.stack([r, g, b], axis=-1).astype(np.float32)

# Convert a (H, W, 3) tensor in [-1, 1] to a PIL image.
def to_pil(tensor_hw3):
    arr = ((tensor_hw3.clamp(-1, 1) + 1) / 2 * 255).byte().cpu().numpy()
    return Image.fromarray(arr)



# ---------------------------------------------------------------------------
# 3. Metrics: MSE, PSNR
# ---------------------------------------------------------------------------
# Mean Squared Error (MSE) (how close are pixel values).
# Lower = Better
def mse(a, b):
    return torch.mean((a - b) ** 2).item()


# Peak signal-to-noise ratio (PSNR), how close are pixel values (the most popular metric for INR)
# Peak signal-to-noise ratio in dB.
# Inputs are in [-1, 1], so the range is 2 (peak^2 = 4).
# 10 * log10(MAX^2 / MSE)
# Higher = Better
def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    data_range = 2.0

    return float("inf") if mse == 0 else (10 * math.log10(data_range ** 2 / mse))

    # return (10 * torch.log10(4.0 / mse)).item()


# ---------------------------------------------------------------------------
# 4. Training
# ---------------------------------------------------------------------------
# Train SIREN so that model reproduces ~= pixel colours."""
def fit_image(image, args, device):

    height, width, _ = image.shape
    coords = make_coord_grid(height, width, device)   # inputs  (N, 2)
    target = image.reshape(-1, 3).to(device)          # targets (N, 3)

    model = Siren(hidden_features=args.hidden, hidden_layers=args.layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}  "
          f"(image has {height * width * 3:,} values)")

    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)   # Adam optimizer - determines the weight updates; learning rate (default: 1e-3)

    for step in range(1, args.steps + 1):  # steps= epochs
        # Full-batch training if the whole image fits in memory at typical size.
        # For large images, sample random mini-batches of coordinates instead.

        pred = model(coords)                     # full-batch forward pass
        loss = torch.mean((pred - target) ** 2)  # MSE reconstruction loss

        optimiser.zero_grad()  # reset gradients
        loss.backward()        # adds grad attribute to original tensor
        optimiser.step()

        if step % args.log_every == 0 or step == 1:
            print(f"step {step:5d} | loss {loss.item():.6f} | "
                  f"PSNR {psnr(pred.detach(), target):.2f} dB")

    return model


@torch.no_grad()
def render(model, height, width, device):
    # Query the trained network on an H x W grid -> (H, W, 3) image tensor.
    # H and W do NOT have to match the training resolution.
    coords = make_coord_grid(height, width, device)
    out = model(coords)
    return out.reshape(height, width, 3)


# ---------------------------------------------------------------------------
# 5. Saving the model to file
# ---------------------------------------------------------------------------

# Save weights "state_dict" + parameters "hidden" and "layers" needed to rebuild the model.
# 4 bytes (float32) per weight
def save_model(model, args, path, half=False):
    torch.save({"state_dict": model.state_dict(),
                "hidden": args.hidden,
                "layers": args.layers}, path)


# Rebuild the model from file.
def load_model(path, device):
    ckpt = torch.load(path, map_location=device)
    model = Siren(ckpt["hidden"], ckpt["layers"]).to(device)
    model.load_state_dict(ckpt['state_dict'])
    return model.eval()



# ---------------------------------------------------------------------------
# 6. Memory comparison
# ---------------------------------------------------------------------------
def fmt_bytes(n):
    # Memory size in Kib
    return f"{n:,} B ({n / 1024:.1f} KiB)"


# Compare model storage against several representations of the image.
def memory_report(model, image_tensor, pil_image, original_path, fp32_path, out_dir):

    h, w, c = image_tensor.shape
    n_params = sum(p.numel() for p in model.parameters())

    # --- Model storage -----------------------------------------------------
    model_fp32_theory = n_params * 4               # 4 bytes per float32 weight
    model_fp32_file = os.path.getsize(fp32_path)

    # --- Image storage -----------------------------------------------------
    raw_bytes = h * w * c                     # uncompressed 8-bit RGB per pixel
    original_bytes = (os.path.getsize(original_path)
                      if original_path and os.path.exists(original_path) else None)

    png_path = os.path.join(out_dir, "training_image.png")
    pil_image.save(png_path, optimize=True)   # lossless compressed version
    png_bytes = os.path.getsize(png_path)

    report = {
        "n of model parameters": n_params,
        "model_fp32_theoretical_bytes": model_fp32_theory,
        "model_fp32_file_bytes": model_fp32_file,
        "image_raw_pixel_bytes": raw_bytes,
        "image_original_file_bytes": original_bytes,
        "image_png_bytes": png_bytes
    }

    print("\n" + "-------------------------------")
    print("------ MEMORY COMPARISON ------")
    print("-------------------------------")

    print(f"Image resolution (w, h, c) : {w} x {h} x {c}")
    print(f"Num of model parameters    : {n_params}")

    print()

    print("------ Model ------")
    print(f"Num of model params * 4 B (float32) : {fmt_bytes(model_fp32_theory)}")
    print(f"Model file on disk (float32)        : {fmt_bytes(model_fp32_file)}")

    print()

    print("-- Image --")
    print(f"Raw 8-bit pixels          : {fmt_bytes(raw_bytes)}")
    if original_bytes is not None:
        print(f"Original file     : {fmt_bytes(original_bytes)}")
    print(f"PNG file(lossless, resized)   : {fmt_bytes(png_bytes)}")

    print()

    print("--- Compression ratio (image memory / model memory; if > 1, then model is smaller) ---")

    print(f"Raw pixel memory / model fp32 file : {(raw_bytes / model_fp32_file):.2f}")
    print(f"Original file    / model fp32 file : {(original_bytes / model_fp32_file):.2f}")
    print(f"PNG file         / model fp32 file : {(png_bytes / model_fp32_file):.2f}")

    print("-------------------------------")
    return report

# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------
def main():
    # python image_inr.py --image photo.jpg --size 256 --steps 2000

    parser = argparse.ArgumentParser(description="Image INR with SIREN")
    parser.add_argument("--image", type=str, default=None,
                        help="path to input image (default: synthetic image)")
    parser.add_argument("--size", type=int, default=256,
                        help="image is resized to size x size")
    parser.add_argument("--hidden", type=int, default=256,
                        help="width of hidden layers")
    parser.add_argument("--layers", type=int, default=3,
                        help="number of hidden sine layers")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--out_dir", type=str, default="inr_output")
    args = parser.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(0)

    # --- Train -------------------------------------------------
    # Load ground-truth image and fit the network to it.
    image, pil_image = load_image(args.image, args.size)
    model = fit_image(image, args, device)

    # --- Save model (float32) to file ---------------------------
    fp32_path = os.path.join(args.out_dir, "siren_fp32.pt")
    torch.save(model.state_dict(), fp32_path)

    print(f"Saved model to '{args.out_dir}/'")

    # --- Image reconstruction at original resolution. -----------------
    h, w, _ = image.shape
    recon = render(model, h, w, device).cpu()

    print(f"Final reconstruction MSE: {mse(recon, image):.2f} (lower = better)")
    print(f"Final reconstruction PSNR: {psnr(recon, image):.2f} dB (higher = better)")


    # --- Image reconstruction at higher resolution. -----------------
    # INR is a continuous function, sample at a higher resolution than it was trained on (super-resolution).
    # 2x original resolution:
    recon_2x = render(model, h * 2, w * 2, device).cpu()


    # --- Save results ---
    # 1) original | reconstruction side by side, file "original_vs_recon.png"
    # 2) 2x original resolution, file "recon_2x.png"
    side_by_side = torch.cat([image, recon], dim=1)
    to_pil(side_by_side).save(os.path.join(args.out_dir, "original_vs_recon.png"))
    to_pil(recon_2x).save(os.path.join(args.out_dir, "recon_2x.png"))

    print(f"Saved images to '{args.out_dir}/'")


    # --- Memory comparison --------------------
    mem = memory_report(model, image, pil_image, args.image, fp32_path, args.out_dir)
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({"quality": "fp32", "memory": mem}, f, indent=2)

    print(f"Saved memory repot to '{args.out_dir}/'")

if __name__ == "__main__":
    main()
    # python image_inr.py --image my_image.jpg --steps 2000
