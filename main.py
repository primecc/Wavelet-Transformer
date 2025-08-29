# Environment Configuration
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
import pywt
from basicsr.archs.swinir_arch import SwinIR
from torch.cuda.amp import GradScaler, autocast
from datetime import datetime
import random
from pytorch_msssim import MS_SSIM
from tqdm import tqdm
import sys
from pathlib import Path
import torch.nn.functional as F
from torchvision.models import vgg16
from torchvision.models.feature_extraction import create_feature_extractor

# Data Preprocessing
class WaveletPreprocessor:
    def __init__(self, wavelet='db2', level=2):
        self.wavelet = wavelet
        self.level = level
        self.min_size = max(2 ** self.level, 8) * 4  # Increase adjustment base

    def _adjust_size(self, img):
        h, w = img.shape
        h_new = ((h + self.min_size - 1) // self.min_size) * self.min_size
        w_new = ((w + self.min_size - 1) // self.min_size) * self.min_size
        if h != h_new or w != w_new:
            img = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_LANCZOS4)  # Higher quality interpolation
        return img

    def process(self, img):
        img = self._adjust_size(img)
        coeffs = pywt.swt2(img, self.wavelet, level=self.level, start_level=0)
        cA, (cH, cV, cD) = coeffs[0]
        wavelet_coeffs = np.stack([cA, cH, cV, cD], axis=0)
        # Robust normalization
        max_vals = np.percentile(np.abs(wavelet_coeffs), 99.9, axis=(1, 2), keepdims=True)
        return wavelet_coeffs / (max_vals + 1e-7)

# Custom Dataset
class FMDDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, preprocessor):
        self.noisy_dir = Path(noisy_dir)
        self.clean_dir = Path(clean_dir)
        self.preprocessor = preprocessor
        self.file_list = sorted([f.name for f in self.noisy_dir.glob('*') if f.suffix.lower() in ('.png', '.jpg', '.tif')])
        # Pre-validate file existence
        self.valid_files = []
        for fname in self.file_list:
            noisy_path = self.noisy_dir / fname
            clean_path = self.clean_dir / fname
            if clean_path.exists():
                self.valid_files.append(fname)
            else:
                print(f"Warning: Missing clean image {fname}")

    def __len__(self):
        return len(self.valid_files)

    def _augment_pair(self, noisy, clean):
        # Apply unified geometric transformations
        if np.random.rand() > 0.5:
            noisy = cv2.flip(noisy, 1)
            clean = cv2.flip(clean, 1)
        if np.random.rand() > 0.5:
            k = np.random.choice([1, 3])  # Avoid counterclockwise rotation
            noisy = np.rot90(noisy, k)
            clean = np.rot90(clean, k)
        if np.random.rand() > 0.9:
            alpha = np.random.uniform(500, 1000)
            sigma = np.random.uniform(8, 12)
            noise_x = cv2.GaussianBlur((np.random.rand(*noisy.shape) * 2 - 1), (0, 0), sigma) * alpha
            noise_y = cv2.GaussianBlur((np.random.rand(*noisy.shape) * 2 - 1), (0, 0), sigma) * alpha
            coords_x, coords_y = np.meshgrid(np.arange(noisy.shape[1]), np.arange(noisy.shape[0]))
            noisy = cv2.remap(noisy, (coords_x + noise_x).astype(np.float32), (coords_y + noise_y).astype(np.float32), interpolation=cv2.INTER_LINEAR)
            clean = cv2.remap(clean, (coords_x + noise_x).astype(np.float32), (coords_y + noise_y).astype(np.float32), interpolation=cv2.INTER_LINEAR)
        # Optimize random cropping
        h, w = noisy.shape
        crop_size = 256
        if h >= crop_size and w >= crop_size:
            i = np.random.randint(0, h - crop_size)
            j = np.random.randint(0, w - crop_size)
            noisy = noisy[i:i + crop_size, j:j + crop_size]
            clean = clean[i:i + crop_size, j:j + crop_size]
        else:
            # Smart padding
            noisy = cv2.resize(noisy, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
            clean = cv2.resize(clean, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
        return noisy, clean

    def __getitem__(self, idx):
        fname = self.valid_files[idx]
        noisy = cv2.imread(str(self.noisy_dir / fname), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        clean = cv2.imread(str(self.clean_dir / fname), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        # Joint augmentation
        noisy, clean = self._augment_pair(noisy, clean)
        def _augment(img, operation):
            """Ensure memory continuity of augmented data"""
            result = operation(img)
            return np.ascontiguousarray(result)
        # Optimize noise enhancement
        if np.random.rand() > 0.9:
            sigma = np.random.uniform(0.005, 0.02)  # Reduce maximum noise intensity
            noise_type = np.random.choice(['gaussian', 'poisson', 'mixed'])
            if noise_type == 'gaussian':
                noisy = noisy + np.random.normal(0, sigma, noisy.shape)
            elif noise_type == 'poisson':
                noisy = noisy + np.random.poisson(noisy * 255 * sigma) / 255
            else:
                noisy = noisy + np.random.normal(0, sigma, noisy.shape) + np.random.poisson(noisy * 255 * sigma) / 255
            noisy = np.clip(noisy, 0, 1)
            noisy = np.ascontiguousarray(noisy)
        wavelet_input = self.preprocessor.process(noisy)
        wavelet_input = np.ascontiguousarray(wavelet_input)
        clean = np.ascontiguousarray(clean)
        return (torch.from_numpy(wavelet_input).float(), torch.from_numpy(clean).unsqueeze(0).float())

# Depthwise Separable Convolution Module
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, dilation=dilation, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise(x)
        return self.pointwise(x)

# Dense Block
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate=32, num_layers=10, dilation_rates=[1, 2, 3, 4]):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Sequential(DepthwiseSeparableConv(in_channels + i * growth_rate, growth_rate, dilation=dilation_rates[i % 4]), nn.GroupNorm(4, growth_rate), nn.GELU()))

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        return torch.cat(features, dim=1)

# Transition Layer (for compressing feature map size)
class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DepthwiseSeparableConv(in_channels, out_channels)

    def forward(self, x):
        x = self.conv(x)
        return x  # No longer downsampling

# Improved Feature Fusion Module
class DenseFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=10, growth_rate=32):
        super().__init__()
        self.dense_block = DenseBlock(in_channels, growth_rate=growth_rate, num_layers=num_layers)
        transition_in = in_channels + num_layers * growth_rate
        self.transition = TransitionLayer(transition_in, out_channels)
        self.ca = ChannelAttention(out_channels)

    def forward(self, x):
        x = self.dense_block(x)
        x = self.transition(x)
        x = self.ca(x)
        return x

# WaveletSwinIRWithDnCNN
class WaveletSwinIRWithDnCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.swinir = SwinIR(upscale=1, in_chans=4, img_size=256, window_size=8, depths=[8,8,8,8,8], embed_dim=128, num_heads=[8,8,8,8,8], mlp_ratio=4, upsampler='', resi_connection='3conv')
        # DnCNN branch
        self.cnn_branch = self._build_dncnn()
        # Feature fusion module (corrected input channels)
        self.fusion = DenseFusionModule(in_channels=128 + 4, out_channels=128, num_layers=10, growth_rate=32)
        # Final output layer remains unchanged
        self.final_conv = nn.Sequential(DepthwiseSeparableConv(128, 64), nn.LeakyReLU(0.2), nn.Conv2d(64, 1, 3, padding=1))

    def _build_dncnn(self):
        layers = [nn.Conv2d(4, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
        # Add 15 intermediate layers (Conv+BN+ReLU)
        for _ in range(15):
            layers += [nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True)]
        # Final output layer (no activation function)
        layers.append(nn.Conv2d(128, 128, kernel_size=3, padding=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # SwinIR branch
        swin_feat = self.swinir(x)
        # DnCNN branch
        cnn_feat = self.cnn_branch(x)
        # Feature fusion
        fused = torch.cat([swin_feat, cnn_feat], dim=1)
        fused = self.fusion(fused)
        return self.final_conv(fused)

class ChannelAttention(nn.Module):
    """Channel Attention Mechanism"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channels, channels // reduction), nn.ReLU(), nn.Linear(channels // reduction, channels), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y + x

class GradientLoss(nn.Module):
    """Multi-scale Gradient Loss, Enhanced Edge Preservation"""
    def __init__(self, scales=5):
        super().__init__()
        self.scales = scales
        # Multi-scale Sobel kernels
        self.kernels = []
        for s in range(scales):
            kernel_size = 3 + 2 * s
            sobel_x = self._get_gaussian_kernel(kernel_size, direction='x')
            sobel_y = self._get_gaussian_kernel(kernel_size, direction='y')
            self.kernels.append((sobel_x, sobel_y))
            self.register_buffer(f'sobel_x_{s}', sobel_x)
            self.register_buffer(f'sobel_y_{s}', sobel_y)

    def _get_gaussian_kernel(self, size=3, direction='x'):
        """Differentiable Gaussian Gradient Kernel"""
        coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
        if direction == 'x':
            g = torch.exp(-coords.pow(2) / (2 * (size / 3) ** 2))
            kernel = g.view(1, 1, size, 1) * torch.tensor([[-1, 0, 1]]).float()
        else:
            g = torch.exp(-coords.pow(2) / (2 * (size / 3) ** 2))
            kernel = g.view(1, 1, 1, size) * torch.tensor([[-1], [0], [1]]).float()
        return kernel / kernel.abs().sum()

    def forward(self, pred, target):
        total_loss = 0.0
        for s in range(self.scales):
            # Get operators for current scale
            sobel_x = getattr(self, f'sobel_x_{s}')
            sobel_y = getattr(self, f'sobel_y_{s}')
            # Multi-scale downsampling
            if s > 0:
                pred_s = nn.functional.avg_pool2d(pred, kernel_size=2 ** s)
                target_s = nn.functional.avg_pool2d(target, kernel_size=2 ** s)
            else:
                pred_s, target_s = pred, target
            # Calculate gradient difference
            pred_grad = torch.cat([nn.functional.conv2d(pred_s, sobel_x, padding='same'), nn.functional.conv2d(pred_s, sobel_y, padding='same')], dim=1)
            target_grad = torch.cat([nn.functional.conv2d(target_s, sobel_x, padding='same'), nn.functional.conv2d(target_s, sobel_y, padding='same')], dim=1)
            # Use Charbonnier loss for enhanced robustness
            scale_weight = 1.0 / (2 ** s)
            total_loss += scale_weight * torch.mean(torch.sqrt((pred_grad - target_grad).pow(2) + 1e-6))
        return total_loss / self.scales

class FluorescenceLoss(nn.Module):
    """Hybrid Loss with Adaptive Temperature Weighting"""
    def __init__(self, l1_weight=1.1, ssim_weight=1.2, grad_weight=0.4, freq_weight=0.1, vgg_weight=0, psnr_weight=1.0):
        super().__init__()
        # Initialize loss functions
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.ms_ssim = MS_SSIM(data_range=1.0, channel=1, size_average=True)
        self.grad_loss = GradientLoss(scales=5)
        # Initialize learnable parameters
        self.temperature = nn.Parameter(torch.tensor(0.5))
        self.log_weights = nn.Parameter(torch.log(torch.tensor([l1_weight, ssim_weight, grad_weight, freq_weight, vgg_weight, psnr_weight])))
        # VGG feature extraction (unchanged)
        vgg = vgg16(pretrained=True)
        self.vgg = create_feature_extractor(vgg.features, return_nodes={'3': 'layer1_relu', '8': 'layer2_relu', '15': 'layer3_relu', '22': 'layer4_relu'})
        for param in self.vgg.parameters():
            param.requires_grad = False
        # Wavelet parameters (unchanged)
        self.wavelet = 'db2'
        self.dwt_mode = 'symmetric'

    def _get_vgg_loss(self, pred, target):
        """Improved Multi-layer VGG Perceptual Loss"""
        # Expand single channel to three channels
        pred_rgb = pred.repeat(1, 3, 1, 1)
        target_rgb = target.repeat(1, 3, 1, 1)
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=pred.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=pred.device).view(1, 3, 1, 1)
        pred_norm = (pred_rgb - mean) / std
        target_norm = (target_rgb - mean) / std
        # Extract features
        features_pred = self.vgg(pred_norm)
        features_target = self.vgg(target_norm.detach())
        # Weighted multi-scale loss
        loss = 0.0
        layer_weights = {'layer1_relu': 1.0, 'layer2_relu': 0.8, 'layer3_relu': 0.6, 'layer4_relu': 0.4}
        for layer in features_pred:
            loss += layer_weights[layer] * F.l1_loss(features_pred[layer], features_target[layer])
        return loss

    def _get_freq_loss(self, pred, target):
        """Improved Wavelet Domain Frequency Loss Calculation"""
        def _dwt(x):
            # Separate computation graph and convert to numpy
            x_np = x.squeeze(1).detach().cpu().numpy()  # [B, H, W]
            coeffs = []
            for img in x_np:
                # Use reflection padding to prevent boundary effects
                _, (cH, cV, cD) = pywt.dwt2(img, self.wavelet, mode='symmetric')  # More robust boundary handling
                # Merge high-frequency components and convert back to Tensor
                hf = np.stack([cH, cV, cD], axis=0)  # [3, H, W]
                coeffs.append(torch.from_numpy(hf))
            return torch.stack(coeffs, dim=0).to(x.device)  # [B, 3, H, W]
        # Ensure consistent data types
        pred_hf = _dwt(pred.float())  # [B, 3, H, W]
        target_hf = _dwt(target.float())
        # Use Huber loss for enhanced robustness
        return nn.functional.smooth_l1_loss(pred_hf, target_hf)

    def forward(self, pred, target):
        # Calculate each loss term
        l1_loss = self.l1(pred, target)
        ssim_loss = 1 - self.ms_ssim(pred, target)
        grad_loss = self.grad_loss(pred, target)
        freq_loss = self._get_freq_loss(pred, target)
        vgg_loss = self._get_vgg_loss(pred, target)
        mse_loss = self.mse(pred, target)  # New MSE calculation
        # Dynamic weight calculation
        weights = torch.softmax(self.log_weights / self.temperature.clamp(min=0.1), dim=0)
        # Weighted total loss
        total_loss = (weights[0] * l1_loss + weights[1] * ssim_loss + weights[2] * grad_loss + weights[3] * freq_loss + weights[4] * vgg_loss + weights[5] * mse_loss)
        return total_loss

# Validation Function
def validate(model, val_loader, device):
    model.eval()
    total_mse = 0.0
    count = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            mse = torch.mean((outputs - targets) ** 2)
            total_mse += mse.item() * inputs.size(0)
            count += inputs.size(0)
    avg_mse = total_mse / count
    psnr = 10 * np.log10(1 / avg_mse)
    return psnr

# Test Inference Class
class Denoiser:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device)
        self.preprocessor = WaveletPreprocessor()
        self.model = WaveletSwinIRWithDnCNN().to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def process_folder(self, input_dir, output_dir):
        """Batch process all images in a folder"""
        # Create output folder
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        # Get all image files
        img_exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
        img_files = [f for f in Path(input_dir).glob('*') if f.suffix.lower() in img_exts]
        print(f"Found {len(img_files)} images to process...")
        # Progress bar display
        with tqdm(img_files, desc="Processing Images", unit="img") as pbar:
            for img_path in pbar:
                try:
                    # Process image
                    denoised = self.denoise(str(img_path))
                    # Save result
                    output_path = Path(output_dir) / img_path.name
                    cv2.imwrite(str(output_path), denoised)
                    # Update progress bar description
                    pbar.set_postfix({"status": "done", "current": img_path.name})
                except Exception as e:
                    print(f"\nError processing {img_path.name}: {str(e)}")
                    continue

    def denoise(self, image_path):
        # Read and preprocess
        noisy = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0  # Ensure float32
        wavelet_input = self.preprocessor.process(noisy)
        # Convert to tensor and specify type
        input_tensor = torch.from_numpy(wavelet_input).float()  # Explicitly convert to float32
        input_tensor = input_tensor.unsqueeze(0).to(self.device)
        # Ensure model weight type
        self.model = self.model.float()  # Force model to use float32
        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)
        # Post-processing
        denoised = output.squeeze().cpu().numpy()
        denoised = np.clip(denoised * 255, 0, 255).astype(np.uint8)
        return denoised

# Training Function
def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = GradScaler()
    # Data preparation
    preprocessor = WaveletPreprocessor(wavelet='db2', level=2)
    train_dataset = FMDDataset(noisy_dir='D:/pythonProject/WTNet/data/train/noisy', clean_dir='D:/pythonProject/WTNet/data/train/clean', preprocessor=preprocessor)
    val_dataset = FMDDataset(noisy_dir='D:/pythonProject/WTNet/data/val/noisy', clean_dir='D:/pythonProject/WTNet/data/val/clean', preprocessor=preprocessor)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    # Model initialization
    model = WaveletSwinIRWithDnCNN().to(device)
    criterion = FluorescenceLoss().to(device)
    # Modified optimizer and learning rate configuration
    optimizer = optim.AdamW([{'params': model.swinir.parameters(), 'lr': 1e-4}, {'params': model.cnn_branch.parameters(), 'lr': 2e-4}, {'params': model.fusion.parameters(), 'lr': 2e-4}, {'params': model.final_conv.parameters(), 'lr': 1e-4}], betas=(0.9, 0.999), weight_decay=3e-5, eps=1e-8)
    # Use cosine annealing with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6, last_epoch=-1)
    # Training loop
    accumulation_steps = 8
    best_psnr = 0
    # Add color configuration before training loop
    try:
        from colorama import Fore, Style
        blue = Fore.BLUE
        reset = Style.RESET_ALL
    except:
        blue = ''
        reset = ''
    # Training loop part
    for epoch in range(200):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        # Create blue progress bar
        with tqdm(train_loader, desc=f"{blue}Epoch {epoch + 1:03d}{reset}", bar_format=f"{blue}{{l_bar}}{{bar:30}}{{r_bar}}{reset}", file=sys.stdout, dynamic_ncols=True) as pbar:
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(device), targets.to(device)
                with autocast(dtype=torch.float16):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)  # Correctly call loss function object
                    loss = loss / accumulation_steps
                scaler.scale(loss).backward()
                # Gradient accumulation update
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    grad_clip = 0.1 * (0.95 ** epoch)  # Gradually relax clipping as training progresses
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3, norm_type=2.0)  # Use L2 norm clipping
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                # Update progress bar information
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Get current time
                total_loss += loss.item() * accumulation_steps
                avg_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({f'[{current_time}] loss': f'{avg_loss:.4f}', 'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'})
        scheduler.step()
        # Validation logic (execute every 1 epoch)
        if epoch % 1 == 0:
            val_psnr = validate(model, val_loader, device)
            print(f"Validation PSNR: {val_psnr:.2f} dB")
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save(model.state_dict(), r'D:\pythonProject\WTNet\run\test\best_swinir_wavelet.pth')
                print(f"New best model saved with PSNR: {best_psnr:.2f} dB")
        # Regularly save checkpoints
        if (epoch + 1) % 10 == 0:
            ckpt_path = f'D:/pythonProject/WTNet/run/test/swinir_wavelet_epoch{epoch + 1}.pth'
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint saved at epoch {epoch + 1}")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Main program execution
if __name__ == "__main__":
    # Train model
    train_model()
    '''
    # Test example
    denoiser = Denoiser(model_path=r'D:\pythonProject\WTNet\run\test\best_swinir_wavelet_confocal_FISH.pth')
    # Batch process test set
    test_input_dir = r'D:\pythonProject\WTNet\data\test\noisy'
    test_output_dir = r'D:\pythonProject\WTNet\data\test_denoised\test'
    denoiser.process_folder(input_dir=test_input_dir, output_dir=test_output_dir)
    print(f"All processed images saved to {test_output_dir}")
    '''
    # model = WaveletSwinIRWithDnCNN()
    # total_params = count_parameters(model)
    # print(f"Total model parameters: {total_params:,}")