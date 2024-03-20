import sys
sys.path.append("..")
import cv2
import numpy as np
import os
import torch
from tifffile import imread, imwrite
import torch.nn as nn
from torch.nn import MSELoss
from torch.optim import Adam
import time
from pathlib import Path

if __name__ == "__main__":
    folder = sys.argv[1]
    outfolder = folder+'_noise2self'
    file_list = [f for f in os.listdir(folder)]
    Path(outfolder).mkdir(exist_ok=True)
    
    

    
    def pixel_grid_mask(shape, patch_size, phase_x, phase_y):
        A = torch.zeros(shape[-2:])
        for i in range(shape[-2]):
            for j in range(shape[-1]):
                if (i % patch_size == phase_x and j % patch_size == phase_y):
                    A[i, j] = 1
        return torch.Tensor(A)
    
    def interpolate_mask(tensor, mask, mask_inv):
     device = tensor.device
     n_channels = tensor.size(1)  # Get the number of channels from the input tensor

     kernel = np.array([[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], [0.5, 1.0, 0.5]])
     kernel = np.repeat(kernel[np.newaxis, np.newaxis, :, :], n_channels, axis=0)  # Repeat the kernel for each channel
     kernel = torch.Tensor(kernel).to(device)
     kernel = kernel / kernel.sum()

    # Cast the input tensor and kernel to Double
     tensor = tensor.float()
     kernel = kernel.float()

    # Apply convolution using grouped convolution, where groups equals the number of channels
     filtered_tensor = torch.nn.functional.conv2d(tensor, kernel, stride=1, padding=1, groups=n_channels)

     return filtered_tensor * mask + tensor * mask_inv

    
    class Masker():
        """Object for masking and demasking"""
    
        def __init__(self, width=3, mode='zero', infer_single_pass=False, include_mask_as_input=False):
            self.grid_size = width
            self.n_masks = width ** 2
    
            self.mode = mode
            self.infer_single_pass = infer_single_pass
            self.include_mask_as_input = include_mask_as_input
    
        def mask(self, X, i):
    
            phasex = i % self.grid_size
            phasey = (i // self.grid_size) % self.grid_size
            mask = pixel_grid_mask(X[0, 0].shape, self.grid_size, phasex, phasey)
            mask = mask.to(X.device)
    
            mask_inv = torch.ones(mask.shape).to(X.device) - mask
    
            if self.mode == 'interpolate':
                masked = interpolate_mask(X, mask, mask_inv)
            elif self.mode == 'zero':
                masked = X * mask_inv
            else:
                raise NotImplementedError
                
            if self.include_mask_as_input:
                net_input = torch.cat((masked, mask.repeat(X.shape[0], 1, 1, 1)), dim=1)
            else:
                net_input = masked
    
            return net_input, mask
    
        def __len__(self):
            return self.n_masks
    
        def infer_full_image(self, X, model):
    
            if self.infer_single_pass:
                if self.include_mask_as_input:
                    net_input = torch.cat((X, torch.zeros(X[:, 0:1].shape).to(X.device)), dim=1)
                else:
                    net_input = X
                net_output = model(net_input)
                return net_output
    
            else:
                net_input, mask = self.mask(X, 0)
                net_output = model(net_input)
    
                acc_tensor = torch.zeros(net_output.shape).cpu()
    
                for i in range(self.n_masks):
                    net_input, mask = self.mask(X, i)
                    net_output = model(net_input)
                    acc_tensor = acc_tensor + (net_output * mask).cpu()
    
                return acc_tensor
    
    class DnCNN(nn.Module):
        def __init__(self, channels, num_of_layers=17):
            super(DnCNN, self).__init__()
            kernel_size = 3
            padding = 1
            features = 64
            layers = []
            layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.ReLU(inplace=True))
            for _ in range(num_of_layers - 2):
                layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
                layers.append(nn.BatchNorm2d(features))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
            self.dncnn = nn.Sequential(*layers)
    
        def forward(self, x):
            out = self.dncnn(x)
            return out
for v in range(len(file_list)):
    file_name = file_list[v]
    
    # Skip files that do not end with .tif or .tiff
    if not (file_name.endswith('.tif') or file_name.endswith('.tiff')):
        print(f"Skipping {file_name} as it is not a TIFF file.")
        continue  # Skip the rest of the loop and proceed to the next file
    
    start_time = time.time()
    print(file_name)
    
    try:
        # Correctly joining the directory and file name
        full_path = os.path.join(folder, file_name)
        noisy_image = imread(full_path)
        
        # Ensure image processing only occurs for valid TIFF files
        mean = np.mean(noisy_image)
        std = np.std(noisy_image)

        noisy = torch.Tensor(noisy_image[np.newaxis, np.newaxis])
        noisy = (noisy - mean) / std
        
        # Your processing logic here...
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        masker = Masker(width = 4, mode='interpolate')
        model = DnCNN(1, num_of_layers = 8)
        loss_function = MSELoss()
        optimizer = Adam(model.parameters(), lr=0.01)
        model = model.to(device)
        noisy = noisy.to(device)
        losses = []
        val_losses = []
        best_images = []
        best_val_loss = np.inf
        
        for i in range(500):
            model.train()
            
            net_input, mask = masker.mask(noisy, i % (masker.n_masks - 1))
            net_output = model(net_input)
            
            loss = loss_function(net_output*mask, noisy*mask)
            optimizer.zero_grad()
         
            loss.backward()
            
            optimizer.step()
            
            if i % 100 == 0:
                
                losses.append(loss.item())
                model.eval()
                
                net_input, mask = masker.mask(noisy, masker.n_masks - 1)
                net_output = model(net_input)
            
                val_loss = loss_function(net_output*mask, noisy*mask)
                val_losses.append(val_loss.item())
        
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    denoised = np.clip(model(noisy).detach().cpu().numpy()[0, 0], -1, 1)
                    denoised = (denoised * std) + mean
                    denoised = (denoised - np.min(denoised)) / (np.max(denoised) - np.min(denoised))
                    denoised = cv2.pow(denoised, 1.0)  # Gamma correction with gamma = 0.5
 
                    try:
                        # Save the denoised image
                        imwrite(outfolder + '/' + file_name, denoised)
                    except Exception as e:
                        print(f"Error occurred while saving {file_name}: {e}")
    
        print("--- %s seconds ---" % (time.time() - start_time))

    except Exception as e:
        print(f"Error occurred while processing {file_name}: {e}")
