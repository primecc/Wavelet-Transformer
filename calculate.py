import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm


def calculate_psnr(img1, img2):
    """Calculate PSNR"""
    # Ensure correct image data type
    if img1.dtype != np.uint8:
        img1 = np.clip(img1, 0, 255).astype(np.uint8)
    if img2.dtype != np.uint8:
        img2 = np.clip(img2, 0, 255).astype(np.uint8)
    return cv2.PSNR(img1, img2)


def calculate_ssim(img1, img2):
    """Calculate SSIM (automatically handles color/grayscale images)"""
    if len(img1.shape) == 3:  # Color image
        return ssim(img1, img2, channel_axis=2, data_range=255)
    else:  # Grayscale image
        return ssim(img1, img2, data_range=255)


def calculate_rmse(img1, img2):
    """Calculate RMSE"""
    return np.sqrt(np.mean((img1.astype(float) - img2.astype(float)) ** 2))


def process_images(clean_dir, denoised_dir, output_file="metrics.txt"):
    """
    Process images from two folders
    :param clean_dir: Path to clean image folder
    :param denoised_dir: Path to denoised image folder
    :param output_file: Path to save results
    """
    # Get matching file lists
    clean_files = sorted(
        [f for f in os.listdir(clean_dir) if f.lower().endswith(('png', 'jpg', 'jpeg', 'tif', 'tiff'))])
    denoised_files = sorted(
        [f for f in os.listdir(denoised_dir) if f.lower().endswith(('png', 'jpg', 'jpeg', 'tif', 'tiff'))])

    # Validate file matching
    if len(clean_files) != len(denoised_files):
        raise ValueError("Number of files in the two folders do not match")
    if clean_files != denoised_files:
        raise ValueError("Filenames in the two folders do not match")

    # Initialize statistics
    total_psnr = 0.0
    total_ssim = 0.0
    total_rmse = 0.0
    valid_count = 0

    # Create results file
    with open(output_file, "w") as f:
        f.write("Filename\tPSNR\tSSIM\tRMSE\n")

        # Use tqdm to show progress bar
        for filename in tqdm(clean_files, desc="Processing images"):
            try:
                # Read images
                clean_path = os.path.join(clean_dir, filename)
                denoised_path = os.path.join(denoised_dir, filename)

                # Read as RGB format uniformly
                clean_img = cv2.imread(clean_path, cv2.IMREAD_UNCHANGED)
                denoised_img = cv2.imread(denoised_path, cv2.IMREAD_UNCHANGED)

                # Handle 16-bit images
                if clean_img.dtype == np.uint16:
                    clean_img = (clean_img / 256).astype(np.uint8)
                if denoised_img.dtype == np.uint16:
                    denoised_img = (denoised_img / 256).astype(np.uint8)

                # Validate image validity
                if clean_img is None:
                    raise ValueError(f"Failed to read clean image: {filename}")
                if denoised_img is None:
                    raise ValueError(f"Failed to read denoised image: {filename}")

                # Uniform size
                if clean_img.shape != denoised_img.shape:
                    denoised_img = cv2.resize(denoised_img, (clean_img.shape[1], clean_img.shape[0]))

                # Calculate metrics
                psnr = calculate_psnr(clean_img, denoised_img)
                ssim_value = calculate_ssim(clean_img, denoised_img)
                rmse = calculate_rmse(clean_img, denoised_img)

                # Record results
                f.write(f"{filename}\t{psnr:.4f}\t{ssim_value:.4f}\t{rmse:.4f}\n")

                # Accumulate statistics
                total_psnr += psnr
                total_ssim += ssim_value
                total_rmse += rmse
                valid_count += 1

            except Exception as e:
                print(f"\nError processing file {filename}: {str(e)}")
                continue

        # Calculate averages
        if valid_count > 0:
            avg_psnr = total_psnr / valid_count
            avg_ssim = total_ssim / valid_count
            avg_rmse = total_rmse / valid_count

            f.write("\nAverage Values:\n")
            f.write(f"PSNR: {avg_psnr:.4f} dB\n")
            f.write(f"SSIM: {avg_ssim:.4f}\n")
            f.write(f"RMSE: {avg_rmse:.4f}\n")

            print("\nFinal Average Results:")
            print(f"PSNR: {avg_psnr:.4f} dB")
            print(f"SSIM: {avg_ssim:.4f}")
            print(f"RMSE: {avg_rmse:.4f}")


if __name__ == "__main__":
    # Usage example
    clean_directory = ""  # Replace with clean image folder path
    denoised_directory = ""  # Replace with denoised result folder path

    process_images(
        clean_dir=clean_directory,
        denoised_dir=denoised_directory,
        output_file="image_metrics.txt"
    )