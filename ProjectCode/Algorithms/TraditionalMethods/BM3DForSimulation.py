import numpy as np
from skimage.restoration import estimate_sigma
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import scipy.io
import bm4d

def create_background_mask(image_gray):
    thresh = threshold_otsu(image_gray)
    binary_image = image_gray <= thresh
    cleared_image = clear_border(binary_image)
    label_image = label(cleared_image)
    regions = regionprops(label_image)
    largest_area = 0
    background_label = 0
    for region in regions:
        if region.area > largest_area:
            largest_area = region.area
            background_label = region.label
    background_mask = label_image == background_label
    return background_mask

def compute_nr_iqa(image):
    """
    Calculates no-reference image quality assessment (NR-IQA) features for a given image.

    Args:
        image (np.ndarray): The grayscale image for which to calculate NR-IQA features.

    Returns:
        dict: A dictionary containing the calculated NR-IQA features:
            - Standard Deviation: Standard deviation of pixel intensity values.
            - Laplacian Variance: Variance of the Laplacian filter applied to the image.
            - Contrast (Michelson): Michelson contrast measure based on minimum and maximum intensities.
            - Entropy: Image entropy calculated using skimage.metrics.entropy.
    """

    std_dev = np.std(image)
    laplacian_img = laplace(image, ksize=3)  # Apply the Laplacian filter
    laplacian_var = np.var(laplacian_img)  # Compute the variance of the Laplacian
    contrast = (np.max(image) - np.min(image)) / (np.max(image) + np.min(image))
    # Calculate global entropy
    hist, _ = np.histogram(image, bins=256, range=(0, 256))
    prob_dist = hist / np.sum(hist)
    entropy_val = scipy_entropy(prob_dist, base=2)

    return {
        'Standard Deviation': std_dev,
        'Laplacian Variance': laplacian_var,
        'Contrast (Michelson)': contrast,
        'Entropy': entropy_val,
    }


snrs = [10, 5, 3, 1, 0]

for snr in snrs:
    mat = scipy.io.loadmat(f'/Users/dolorious/Downloads/simulation-1/GroundTruth_rician_SNR{snr}.mat')
    mat_ground_truth = scipy.io.loadmat(f'/Users/dolorious/Downloads/simulation-1/GroundTruth.mat')
   
    noisy_image_data = mat['data_rician']
    ground_truth_data = mat_ground_truth['data']

    evol_times = noisy_image_data.shape[3]  # Assuming the last dimension is the number of evolution times
    for j in range(evol_times):
        noisy_image = noisy_image_data[:, :, j,:]  # Assuming the images are 3D (height, width, channels)
        ground_truth = ground_truth_data[:, :,j,:]  # Assuming same indexing as noisy images

        # Convert to grayscale if it's a multichannel image
        if noisy_image.ndim == 3 and noisy_image.shape[2] == 3:
            noisy_image_gray = rgb2gray(noisy_image)
        else:
            noisy_image_gray = noisy_image.mean(axis=2)

        sigma_est = np.mean(estimate_sigma(noisy_image_gray, channel_axis=-1))
        background_mask = create_background_mask(noisy_image_gray)

        # Apply BM4D Denoising
        denoised_image = bm4d.bm4d(noisy_image_gray, sigma_psd=sigma_est)
        denoised_image = np.squeeze(denoised_image)

        # Calculate standard deviation of the background in the noisy image
        std_deviation_noisy = np.std(noisy_image_gray[background_mask])
        # For denoised image, let's also calculate it for comparison
        std_deviation_denoised = np.std(denoised_image[background_mask])

        # Convert ground truth to grayscale
        if ground_truth.ndim == 3 and ground_truth.shape[2] > 1:
            ground_truth_gray = ground_truth.mean(axis=2)
        else:
            ground_truth_gray = ground_truth

        ground_truth_gray = np.squeeze(ground_truth_gray)

        # Ensure both ground_truth_gray and denoised_image have the same number of dimensions
        if ground_truth_gray.ndim != denoised_image.ndim:
            if ground_truth_gray.ndim < denoised_image.ndim:
                ground_truth_gray = np.expand_dims(ground_truth_gray, axis=-1)
            else:
                ground_truth_gray = np.squeeze(ground_truth_gray)

        # Compute the metrics
        image_ssim = ssim(ground_truth_gray, denoised_image, data_range=ground_truth_gray.max() - ground_truth_gray.min())
        image_psnr = psnr(ground_truth_gray, denoised_image)
        image_mse = mse(ground_truth_gray, denoised_image)
        nr_iqa_metrics = compute_nr_iqa(denoised_image)
        
        plt.imsave(f"../TraditionalMethods/Experiments/DenoisedImages/denoised_snr{snr}_evoltime{j+1}.tiff", denoised_image, cmap='gray')
