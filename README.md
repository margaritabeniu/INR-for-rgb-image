Implicit Neural Representation (INR) for RGB Images

A PyTorch implementation of an Implicit Neural Representation (INR) using a SIREN (Sinusoidal Representation Network) to encode and reconstruct RGB images.

Instead of storing an image as a conventional matrix of pixel values, this project trains a neural network to represent the image as a continuous mathematical function.

The trained network learns a mapping:

fθ:(x,y)→(R,G,B)

where:

x,y are normalized spatial coordinates.
R,G,B are the corresponding color values.
θ represents the trainable neural network parameters.

After training, the network can reconstruct the original image or generate images at different resolutions by evaluating the learned function at new coordinates.

1. Main Idea

A conventional RGB image is represented as a discrete tensor:

I∈RH×W×3

where:

H is image height.
W is image width.
3 represents the RGB color channels.

In this project, the image is represented by a neural network instead.

Every pixel is treated as a training example:

Input	Target
Normalized pixel coordinates (x, y)	RGB color (R, G, B)

For example:

text
(x, y) = (-1.0, -1.0)  ->  (R, G, B)
(x, y) = (-0.5,  0.0)  ->  (R, G, B)
(x, y) = ( 0.0,  0.0)  ->  (R, G, B)
(x, y) = ( 1.0,  1.0)  ->  (R, G, B)


The network learns the relationship between spatial coordinates and pixel colors.

Once trained, the network weights contain the information needed to reconstruct the image.

2. Why SIREN?

The project uses a SIREN architecture rather than a conventional MLP with ReLU activation functions.

SIREN stands for Sinusoidal Representation Network.

It was introduced in the paper:

Implicit Neural Representations with Periodic Activation Functions

Vincent Sitzmann et al.

Paper: https://arxiv.org/abs/2006.09661

The problem with conventional MLPs

Standard neural networks using ReLU activations can have difficulty representing high-frequency image details.

This behavior is related to spectral bias: neural networks tend to learn low-frequency components before high-frequency components.

For image reconstruction, this can result in:

Blurred edges.
Loss of fine textures.
Missing high-frequency details.
Poor reconstruction of checkerboards and sharp transitions.
Sinusoidal activation

SIREN uses periodic sine activations:

ϕ(x)=sin⁡(ω0x)

For a linear layer, the activation becomes:

y=sin⁡(ω0(Wx+b))

where:

W is the weight matrix.
b is the bias.
ω0 controls the frequency of the activation.

Sinusoidal functions allow the network to represent complex signals containing both low- and high-frequency components.

This makes SIREN particularly suitable for representing images.

Special weight initialization

SIREN uses a specific weight initialization strategy to maintain stable activation distributions.

For the first layer:

W∼U(−1n,1n)

For subsequent hidden layers:

W∼U(−6/nω0,6/nω0)

where n is the number of input features.

The implementation uses:

First-layer frequency: 30.
Hidden-layer frequency: 30.
Sine activation in hidden layers.
Linear output layer.

The final layer does not use a sine activation because the network should be able to produce unrestricted RGB predictions before conversion to image values.

3. Network Architecture

The network is implemented using two PyTorch classes:

SineLayer
Siren
SineLayer

A single SIREN layer performs the following operation:

h=sin⁡(ω0(Wx+b))

It consists of:

A fully connected linear layer.
Special SIREN weight initialization.
A sinusoidal activation function.
Siren

The complete network maps two-dimensional coordinates to three-dimensional RGB values.

Default architecture:

text
Input
  |
  | 2 features: (x, y)
  |
  v
SineLayer (2 -> 256)
  |
  v
SineLayer (256 -> 256)
  |
  v
SineLayer (256 -> 256)
  |
  v
SineLayer (256 -> 256)
  |
  v
Linear Layer (256 -> 3)
  |
  v
Output
  |
  | 3 features: (R, G, B)
  |
  v
Predicted pixel color


The default configuration uses:

256 neurons per hidden layer.
3 additional hidden sine layers after the first layer.
4 sine layers in total.
1 linear output layer.

The architecture is configurable through command-line arguments.

4. Image Preparation

The input image is loaded using Pillow.

python
img = Image.open(path).convert("RGB")


The image is converted to a NumPy array and normalized.

Original pixel values:

RGB∈[0,255]

First, values are converted to the interval [0, 1]:

Inormalized=I255

Then they are transformed into [-1, 1]:

Itarget=2Inormalized−1

This produces the target range used by the network.

Coordinate generation

The function make_coord_grid() creates a coordinate grid for all image pixels.

Coordinates are normalized to:

(x,y)∈[−1,1]2

Pixel centers are used rather than pixel corners.

For an image of height H and width W, the coordinate tensor has the shape:

(H⋅W,2)

The corresponding RGB target tensor has the shape:

(H⋅W,3)

This converts the original image into a supervised learning dataset.

Synthetic image

If no input image is provided, the script generates a synthetic RGB image.

It contains three components:

Red: sinusoidal gradients.
Green: circular cosine patterns.
Blue: checkerboard pattern.

The synthetic image is useful for testing the network on both smooth and high-frequency visual structures.

5. Training Process

Training is implemented in the fit_image() function.

The objective is to find network parameters that minimize the difference between predicted and original pixel colors.

Step 1: Prepare training data

The coordinate grid is flattened into a list of two-dimensional coordinates.

The image is flattened into corresponding RGB targets.

python
coords = make_coord_grid(height, width, device)

target = image.reshape(-1, 3).to(device)

Step 2: Initialize the network

A SIREN model is created using the configured hidden-layer width and depth.

python
model = Siren(
    hidden_features=args.hidden,
    hidden_layers=args.layers
).to(device)

Step 3: Define the optimizer

The implementation uses Adam:

python
optimiser = torch.optim.Adam(
    model.parameters(),
    lr=args.lr
)


The default learning rate is:

lr=10−4
Step 4: Calculate reconstruction loss

The network predicts RGB values for all image coordinates.

The predictions are compared with the original pixel values using Mean Squared Error (MSE).

MSE=1N∑i=1N(I^i−Ii)2

where:

N is the number of RGB values.
I^i is the predicted value.
Ii is the original value.

The training objective is:

θ∗=arg⁡min⁡θMSE(fθ(x,y),I(x,y))
Step 5: Backpropagation

The network parameters are updated using gradient descent through Adam.

python
optimiser.zero_grad()

loss.backward()

optimiser.step()


The process repeats for the configured number of training steps.

Full-batch training

The current implementation uses full-batch training.

This means that all image coordinates are evaluated during every training step.

For larger images, mini-batch training could be introduced to reduce memory consumption.

6. Reconstruction

After training, the network represents a continuous function:

fθ(x,y)→(R,G,B)

The render() function generates an image by evaluating this function on a coordinate grid.

python
recon = render(model, height, width, device)


The predictions are reshaped into:

H×W×3

The resulting tensor represents the reconstructed RGB image.

The output values are clipped to [-1, 1], converted back to [0, 255], and saved as an image.

Resolution independence

One of the main advantages of INR is that the network does not directly store a fixed pixel grid.

It learns a continuous function that can be evaluated at arbitrary coordinates.

Consequently, the same trained model can generate images at different resolutions.

The implementation demonstrates this by generating a reconstruction at twice the original resolution:

python
recon_2x = render(
    model,
    height * 2,
    width * 2,
    device
)


For example:

text
Training resolution:      256 x 256
Reconstruction:           256 x 256
Higher-resolution output: 512 x 512


This is a form of implicit neural upsampling.

Important: This is not guaranteed to recover details that were absent from the training image. The higher-resolution image is generated by sampling the learned continuous function more densely.

7. Image Quality Metrics

The implementation evaluates reconstruction quality using two metrics.

Mean Squared Error (MSE)

MSE measures the average squared difference between original and reconstructed pixel values.

MSE=1N∑i=1N(Ii−I^i)2

Interpretation:

Lower MSE indicates better pixel-level reconstruction.
Zero MSE indicates a perfect reconstruction.
Peak Signal-to-Noise Ratio (PSNR)

PSNR measures reconstruction quality in decibels.

PSNR=10log⁡10(MAX2MSE)

Since the image values are normalized to [-1, 1]:

MAX=2

Therefore:

PSNR=10log⁡10(4MSE)

Interpretation:

Higher PSNR indicates better reconstruction.
Infinite PSNR indicates zero reconstruction error.

The script calculates PSNR during training and after reconstruction.

Note: Although SSIM is mentioned in the introductory comments, it is not currently implemented in the Python code.

8. Model Storage and Memory Comparison

The project also investigates whether storing an image as a neural network is more memory-efficient than storing its original pixel representation.

The model is saved using PyTorch:

python
torch.save(
    model.state_dict(),
    fp32_path
)


The memory report compares:

The theoretical size of the float32 model.
The actual model file size.
The uncompressed RGB image size.
The original input file size.
The optimized PNG file size.
Model size

Each float32 parameter requires 4 bytes.

Therefore:

Smodel=Nparameters⋅4
Raw RGB image size

For an 8-bit RGB image:

Simage=H⋅W⋅3

For a 256 × 256 RGB image:

256⋅256⋅3=196608 bytes

This is approximately 192 KiB.

Compression ratio

The implementation calculates:

CR=SimageSmodel

Interpretation:

CR > 1: The neural representation is smaller.
CR < 1: The neural representation is larger.
CR = 1: Both representations have approximately the same size.
Important observation

Model size depends strongly on network architecture.

Increasing hidden-layer width and depth generally increases the network's representation capacity, potentially improving reconstruction quality.

However, it also increases the number of parameters.

A larger SIREN can easily become bigger than the original image.

Therefore, this implementation is primarily interesting for continuous image representation and resolution-independent reconstruction rather than conventional image compression.

9. Installation
Requirements
Python 3
PyTorch
NumPy
Pillow

Install dependencies:

bash
pip install torch numpy pillow


For GPU acceleration, install the appropriate PyTorch version for your CUDA environment.

10. Usage
Run with a custom image
bash
python image_inr.py --image my_photo.jpg

Run with a synthetic image
bash
python image_inr.py

Specify training parameters
bash
python image_inr.py \
    --image my_photo.jpg \
    --size 256 \
    --hidden 256 \
    --layers 3 \
    --steps 2000 \
    --lr 0.0001 \
    --out_dir inr_output

Command-line arguments
Argument	Default	Description
--image	None	Input image path
--size	256	Synthetic image size
--hidden	256	Number of neurons per hidden layer
--layers	3	Number of additional hidden sine layers
--steps	2000	Number of optimization steps
--lr	0.0001	Adam learning rate
--log_every	200	Training logging interval
--out_dir	inr_output	Output directory

Note: The current implementation does not resize an existing input image according to --size. That argument controls the size of the generated synthetic image.

11. Output Files

After successful execution, the output directory contains:

text
inr_output/
│
├── siren_fp32.pt
│
├── original_vs_recon.png
│
├── recon_2x.png
│
├── training_image.png
│
└── summary.json

File descriptions
File	Description
siren_fp32.pt	Trained SIREN model weights
original_vs_recon.png	Original and reconstructed images displayed side by side
recon_2x.png	Reconstruction at twice the original resolution
training_image.png	Training image saved as an optimized PNG
summary.json	Reconstruction and memory comparison information

The model is saved in float32 precision.

12. Hardware Acceleration

The script automatically selects an available computation device.

The selection priority is:

CUDA GPU.
Apple Metal Performance Shaders (MPS).
CPU.
python
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


This allows the same implementation to run on different hardware configurations without changing the source code.

13. Practical Considerations
Noisy reconstruction

If the reconstructed image contains noise or unstable patterns:

Reduce the learning rate.
Experiment with a lower sine frequency parameter (omega_0).
Increase the number of training steps if the network is underfitting.
Blurry reconstruction

If fine details are missing:

Increase hidden-layer width.
Increase the number of hidden layers.
Increase the number of training steps.
Experiment with the SIREN frequency parameters.

These changes should be evaluated using reconstruction metrics rather than visual appearance alone.

Memory consumption

Full-batch training requires evaluating all image coordinates at every step.

For large images, consider implementing mini-batch training.

14. Current Implementation Limitations

The current script is a compact experimental implementation and has several limitations:

It represents one image per trained network.
Training uses full-batch optimization.
The network is trained specifically for the input image.
Higher-resolution rendering does not guarantee recovery of missing details.
The implementation does not include a perceptual loss.
SSIM is mentioned in the documentation comments but is not implemented.
Model size can exceed the original image size.
The model checkpoint saved by the main execution path contains only the state dictionary; the separate load_model() helper expects additional architecture metadata. These formats should be made consistent before using that helper.
The memory-reporting code assumes an original input file exists when calculating the original-file compression ratio, so synthetic-image execution needs a small adjustment to handle a missing original file.
15. Project Workflow Summary

The complete processing pipeline is:

text
             INPUT RGB IMAGE
                    |
                    v
             IMAGE LOADING
                    |
                    v
          RGB NORMALIZATION [-1, 1]
                    |
                    v
          GENERATE COORDINATE GRID
                    |
                    v
             (x, y) -> RGB
                    |
                    v
             SIREN NETWORK
                    |
                    v
            PREDICT RGB VALUES
                    |
                    v
             CALCULATE MSE
                    |
                    v
          BACKPROPAGATION + ADAM
                    |
                    v
              TRAINED SIREN
                    |
          +---------+---------+
          |                   |
          v                   v
   ORIGINAL RESOLUTION     2X RESOLUTION
    RECONSTRUCTION          RENDERING
          |                   |
          v                   v
    QUALITY METRICS       HIGHER-RES IMAGE
          |
          v
      SAVE MODEL
          |
          v
    MEMORY COMPARISON
          |
          v
       OUTPUT FILES

16. Conclusion

This project demonstrates how an image can be represented by a neural network rather than a conventional pixel matrix.

Using SIREN, the network learns a continuous mapping from spatial coordinates to RGB values.

The resulting representation offers several interesting properties:

Continuous image representation.
Resolution-independent rendering.
Ability to represent high-frequency image details.
Flexible sampling at arbitrary spatial coordinates.
Neural representation of image content through learned parameters.

The main advantage demonstrated by this implementation is the ability to reconstruct an image at different resolutions using the same trained neural network.

Although a basic INR is not necessarily more storage-efficient than conventional image formats, it provides an interesting foundation for research into neural image representations, implicit image models, and neural upsampling.

References
Sitzmann, V. et al. (2020). Implicit Neural Representations with Periodic Activation Functions.
PyTorch Documentation.
NumPy Documentation.
Pillow Documentation.

Source code: image_inr.py
