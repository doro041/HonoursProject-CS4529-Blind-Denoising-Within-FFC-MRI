# HonoursProject-CS4529-Blind-Denoising-Within-FFC-MRI

# User Manual

## Introduction

This user manual is crafted to guide you through the utilization of our program, specifically developed for denoising medical imaging data using both traditional and Self-Supervised Machine Learning methods tailored for Fast Field-Cycling Magnetic Resonance Imaging (FFC-MRI). The manual aims to serve healthcare professionals, researchers, and data analysts by providing detailed instructions on how to effectively test these and other algorithms to enhance the clarity and utility of FFC-MRI data.

## Downloading the Software

To begin using the software, follow these steps to download it from our public GitHub repository:

- Navigate to the GitHub repository page.
- Click on the `Code` button located at the top right of the page.
- From the dropdown menu, select `Download ZIP` to download the repository as a ZIP file.
- After the download is complete, extract the contents of the ZIP file to your desired location on your computer.
- If you wish to contribute to the development of the project or customize the software for your needs, consider forking the repository:
    - Click on the `Fork` button at the top right of the repository page. This will create a personal copy of the repository in your own GitHub account, enabling you to make changes without affecting the original project.

## Installation

### Installing Required Software

#### Install Matlab

1. Visit the Matlab official website at: https://www.mathworks.com/products/matlab.html.
2. Choose the appropriate version for your operating system and follow the on-screen instructions to download.
3. Install Matlab by following the setup wizard after the download completes. You may need to log in with your MathWorks account or create one if you don't already have it.
4. Activate Matlab using a valid license. If you are part of an academic institution, you might be eligible for a discounted or free version.

#### Install Python

1. If you do not already have Python installed, download and install it by following the instructions available at: https://www.python.org/.

#### Install Anaconda

1. If you do not already have Anaconda installed, download and install it by following the instructions available at: https://www.anaconda.com/download/.

#### Install Pytorch

1. Visit the PyTorch official site at: https://pytorch.org/.
2. Use the installation selector to choose the right version for your system and setup preferences (including support for CUDA if using NVIDIA GPUs).
3. Follow the generated command to install PyTorch. For example, for a typical Windows installation without GPU support, you might use:
    ```
    pip install torch torchvision torchaudio
    ```
4. Verify the installation by running Python and importing PyTorch:
    ```
    python -c "import torch; print(torch.__version__)"
    ```
5. If the command prints the PyTorch version without errors, the installation was successful.

## Data Overview

This section provides an overview of the data types used in this project, including medical phantom images and simulated data, and instructions on how to effectively utilize these datasets with the denoising algorithms.

### Medical Phantom Images

- **Data Format:** The medical phantom images are stored in TIFF (Tagged Image File Format), a format preferred for its ability to handle high-quality graphic files without loss of information. The TIFF format's compatibility with a wide range of image processing tools allows for versatile manipulation and analysis of the images.

- **Data Location:** The images are organized within the directory at `ImagesForExperimentation/PhantomData`. This directory structure is designed to facilitate easy access and organization of the data.

- **Using the Data:** To apply denoising algorithms to these images, follow these steps:
    1. Navigate to the `ImagesForExperimentation/PhantomData` directory (See Figure 1).
    2. Within this directory, images are organized into subfolders categorized by magnetic field strength, which allows for targeted processing based on specific imaging conditions.
    3. Select the desired subfolder for batch processing of all images within, or choose individual TIFF files for single-image processing. This flexibility supports both comprehensive and focused denoising tasks.

![The folder for the Medical Phantom](StackImages/RealData/phantomdata.png)
Figure 1: The folder for the Medical Phantom

### Simulated Data

- **Using the Data:**
    1. Locate the simulated data within the directory `ImagesForExperimentation/SimulatedMedical`.
    2. Use the file `GroundTruth.mat` to generate additional noisy examples. This allows for extensive testing of denoising algorithms under various conditions.
    3. Apply different types of noise to these examples to simulate real-world imaging scenarios. The types of noise you can introduce include:
        - **Poisson Noise:** Commonly associated with statistical fluctuations in photon detection.
        - **Speckle Noise:** Often observed in ultrasound imaging and can be simulated to evaluate robustness.
    4. Experiment with various Signal-to-Noise Ratios (SNRs) to assess the effectiveness of denoising algorithms at different levels of noise intensity. Available SNR levels include 10, 20, 30, 40, and 50, as well as more extreme noise conditions such as SNR levels of 0, 1, 3, 5, and 10. Our project shows how our algorithm performs under extremely low noise.

**Note: Real medical patient data is not included in this dataset to comply with ethical standards. Distribution of such data is restricted to protect patient privacy and confidentiality. This policy ensures that all experiments and demonstrations are conducted without compromising ethical obligations.**



## Downloading the Software

To begin using the software, follow these steps to download it from our public GitHub repository:

- Navigate to the GitHub repository page.
- Click on the `Code` button located at the top right of the page.
- From the dropdown menu, select `Download ZIP` to download the repository as a ZIP file.
- After the download is complete, extract the contents of the ZIP file to your desired location on your computer.
- If you wish to contribute to the development of the project or customize the software for your needs, consider forking the repository:
    - Click on the `Fork` button at the top right of the repository page. This will create a personal copy of the repository in your own GitHub account, enabling you to make changes without affecting the original project.

## Applying Denoising Algorithms

### Non-Local Means (NLM)

- `skimage.restoration.denoise_nl_means`
- Create background mask using Otsu's thresholding
- Tune `h`, `patch_size`, `patch_distance` parameters
- Apply denoising with adjusted parameters

### BM3D/BM4D

- `import bm3d`
- Create background mask using Otsu's method
- `denoised = bm3d.bm3d(noisy_image, stage_arg=bm3d.BM3DStages.ALL_STAGES)`
- Tune `sigma_psd`, `stage_arg` for optimal performance

### Total Variation (TV)

- Use our TV denoising implementation
- Apply Prime-Dual algorithm
- Iterate through images
- Adjust parameters: `L2_norm=8.1`, `tau`, `sigma/(L2_norm*tau)`, `theta`

### Patch2Self

- Navigate to `patch2self/model` directory
- Adjust parameters in configuration files
- Select regressor model
- Train on each image in dataset
- Plot results for analysis and evaluation

### Noise2Self and Noise2Fast

- Navigate to `noise2self:patch2self` directory
- `conda activate N2F`
- Adjust N2S architecture (if needed)
- Run: `python /path/to/tiff/files N2S.py` or `N2F.py`

## Metrics for Evaluation

### Full-Reference Metrics

Full-reference metrics are essential for assessing the quality of denoising by comparing the denoised image with the original (reference) image. These metrics help quantify the restoration accuracy. Here's how you can implement PSNR, MSE, and SSIM in Python using the `skimage.metrics` module:

```python
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim, mean_squared_error as mse

# Calculating full-reference metrics
psnr_value = psnr(denoised_img, original_img)
ssim_value = ssim(original_img, denoised_img, data
_range=denoised_img.max() - denoised_img.min())
mse_value = mse(original_img, denoised_img)

# Outputting the metric values
print(f"PSNR: {psnr_value}")
print(f"SSIM: {ssim_value}") 
print(f"MSE: {mse_value}")
```

### No-Refewrence Metrics
No-reference metrics evaluate the quality of a denoised image without comparing it to the original image. These metrics are useful when the original image is not available.
```python
import numpy as np
from skimage.filters import laplace
from scipy.stats import entropy as scipy_entropy

def calculate_no_reference_metrics(image):
    """
    Calculate no-reference image quality metrics.

    Parameters:
    - image: Input image as a NumPy array.

    Returns:
    - std_dev: Standard deviation of the image.
    - laplacian_var: Variance of the Laplacian of the image.
    - entropy_val: Entropy of the image.
    """
    # Calculate standard deviation
    std_dev = np.std(image)

    # Apply Laplacian filter
    laplacian_img = laplace(image, ksize=3)
    # Compute variance of the Laplacian
    laplacian_var = np.var(laplacian_img)

    # Calculate global entropy
    hist, _ = np.histogram(image, bins=256, range=(0, 256))
    prob_dist = hist / np.sum(hist)
    entropy_val = scipy_entropy(prob_dist, base=2)

    return std_dev, laplacian_var, entropy_val

# Example usage:
image = np.random.rand(100, 100)  # Example random image
std_dev, laplacian_var, entropy_val = calculate_no_reference_metrics(image)

# Printing the no-reference metric values
print(f"Standard Deviation: {std_dev}")
print(f"Laplacian Variance: {laplacian_var}")
print(f"Entropy: {entropy_val}")
```

### SNR/CNR
To identify Signal-to-Noise ratio and Contrast to noise for medical data, we usually apply Region of interest, and we do not really need a ground-truth image. Please make sure to apply the noisy ROI on the background, the signal ROI, if a brain within white tissue and black tissue.
Use our code for implementing CNR and SNR, as follows and apply to your medical image:
```python
snr = mean_signal / std_noise if std_noise != 0 else np.inf

cnr = (mean_signal - mean_signal2) / std_noise if std_noise != 0 else np.inf

# Define image path and ROI coordinates
image_path = "path_to_your_image.jpg"
roi_coords = (x1, y1, width1, height1)  # Coordinates of signal ROI
noise_coords = (x2, y2, width2, height2)  # Coordinates of noise ROI
roi2_coords = (x3, y3, width3, height3)  # Coordinates of second signal ROI (for CNR)
```
### Edge Preservation
If you want to see whether your images are preserved, you can apply our Edge Preservation implementation using skimage.feature.
Visualizing Edges in Medical Images
The visualize_edges function allows users to visualize the edges detected in medical images using Canny edge detection. This visualization aids in understanding the structural details and boundaries of various tissues or structures within the images.
Prerequisites

Python environment with necessary libraries installed: numpy, matplotlib.pyplot, skimage.io.imread, skimage.feature, skimage.color.rgba2rgb, and skimage.color.rgb2gray.


# Maintenance Manual

## Introduction
This manual serves as a guide for contributors and maintainers of the Blind Denoising Within FFC-MRI. It includes installation instructions, a comprehensive list of dependencies, the file structure of the project, and guidelines for improving the implementation.

## Installation
Please install our code from GitHub as indicated in the user manual. The repository can be accessed at https://rb.gy/3sn5vv.

## Hardware Dependencies
Our methods are designed to operate within modest hardware configurations, ensuring broad accessibility and ease of use. Below are the basic minimum hardware requirements necessary to efficiently run our methods:

- **Processor:** Intel Core i5 or equivalent
- **RAM:** 8 GB
- **Hard Drive:** 500 GB of free space
- **Graphics Card:** Integrated graphics (DirectX 12 support recommended)
- **Operating System:** Windows 10, Linux (Ubuntu 18.04 or later), or macOS Mojave

## Dependencies
All dependencies required for the project are listed in the `requirements.txt` file or within the `requirements.yml` for the Self-Supervised models. It is crucial to install these packages to ensure the functionality of the project. Below is the list of dependencies:

| Package | Version |
| --- | --- |
| comm | 0.1.4 |
| dipy | 1.8.0 |
| image-quality | 1.2.7 |
| ipykernel | 6.25.2 |
| ipython | 8.16.1 |
| jupyter_client | 8.4.0 |
| jupyter_core | 5.4.0 |
| kiwisolver | 1.4.5 |
| matplotlib | 3.8.3 |
| matplotlib-inline | 0.1.6 |
| numpy | 1.26.4 |
| opencv-python | 4.9.0.80 |
| opencv-python-headless | 4.9.0.80 |
| packaging | 23.2 |
| pandas | 1.5.3 |
| pillow | 10.2.0 |
| pytv | 0.3.0 |
| rdflib | 7.0.0 |
| scikit-image | 0.22.0 |
| scikit-learn | 1.4.1.post1 |
| scipy | 1.12.0 |
| six | 1.16.0 |
| smart-open | 6.4.0 |
| spacy | 3.7.4 |
| spacy-legacy | 3.0.12 |
| spacy-loggers | 1.0.5 |
| tifffile | 2024.2.12 |
| tiffile | 2018.10.18 |
| toml | 0.10.2 |
| torch | 2.2.1 |
| torchvision | 0.17.1 |
| transforms3d | 0.4.1 |
| trx-python | 0.2.9 |
| typer | 0.9.0 |
| typing_extensions | 4.10.0 |
| urllib3 | 2.2.1 |
| *skimage* | *Unspecified* |
| *bm3d* | *Unspecified* |
| *bm4d* | *Unspecified* |

### How to contribute

#### Creating a Method Directory
If you are testing or developing a new machine learning method, please initiate by creating a unique subdirectory within the `MLMethods` folder. Use the `ImagesForExperimentation` directory to test your new method rigorously.

#### Storing Processed Outputs
Upon successful testing and validation, proceed to create a corresponding directory in the `DenoisedImages` section. This directory should contain all outputs generated by your new method.

#### Ensuring Reproducibility
To aid others in replicating your results and testing your method, include a detailed environment setup file within your method's subdirectory. We accept files in the following formats: `environment.yml`, `environment.txt`, or a Conda activation script. This documentation is crucial for maintaining the integrity and usability of your contribution.

### Tasks to be Completed
Below is a checklist of tasks that need to be completed for our project:

- [ ] Test the Zero-shot Denoiser from 2023 or 2024, and store the results in the `MLMethods` folder.
- [ ] Test the new method using images from the `ImagesForExperimentation` directory.
- [ ] Employ a Hybrid approach combining Total Variation (TV) with any machine learning method to explore its efficacy.
- [ ] Create a subdirectory within the `TraditionalMethods` directory named `HybridApproaches`.
- [ ] Develop a method specifically designed for volumetric data, such as Fast Field-Cycling MRI, focusing on its unique attributes.
- [ ] Create a corresponding directory in the `DenoisedImages` to store processed images.
- [ ] Include an `environment.yml` or `environment.txt` file to facilitate environment setup for reproducibility.
