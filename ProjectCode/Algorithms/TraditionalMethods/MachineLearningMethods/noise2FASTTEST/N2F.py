
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from tifffile import imread, imwrite
import sys
import torch.utils.data as utils_data
import numpy as np
from pathlib import Path
import time


if __name__ == "__main__":
    tsince = 100
    folder = sys.argv[1]
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
    outfolder = folder+'_N2F'
    Path(outfolder).mkdir(exist_ok=True)
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    class TwoCon(nn.Module):
    
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
    
        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            return x
    
    
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = TwoCon(1, 64)
            self.conv2 = TwoCon(64, 64)
            self.conv3 = TwoCon(64, 64)
            self.conv4 = TwoCon(64, 64)  
            self.conv6 = nn.Conv2d(64,1,1)
            
    
        def forward(self, x):
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            x3 = self.conv3(x2)
            x = self.conv4(x3)
            x = torch.sigmoid(self.conv6(x))
            return x
        
    file_list = [f for f in os.listdir(folder)]
    start_time = time.time()
    for v in range(len(file_list)):
        
        file_name =  file_list[v]
        print(file_name)
        if file_name[0] == '.':
            continue
        
        notdone = True
        learning_rate = 0.001
        while notdone:     
            img = imread(folder + '/' + file_name)
            typer = type(img[0,0])
            
            minner = np.amin(img)
            img = img - minner
            maxer = np.amax(img)
            img = img/maxer
            img = img.astype(np.float32)
            shape = img.shape
            
            
        
            listimgH = []
            Zshape = [shape[0],shape[1]]
            if shape[0] % 2 == 1:
                Zshape[0] -= 1
            if shape[1] % 2 == 1:
                Zshape[1] -=1  
            imgZ = img[:Zshape[0],:Zshape[1]]
            
            imgin = np.zeros((Zshape[0]//2,Zshape[1]),dtype=np.float32)
            imgin2 = np.zeros((Zshape[0]//2,Zshape[1]),dtype=np.float32)
            for i in range(imgin.shape[0]):
                for j in range(imgin.shape[1]):
                    if j % 2 == 0:
                        imgin[i,j] = imgZ[2*i+1,j]
                        imgin2[i,j] = imgZ[2*i,j]
                    if j % 2 == 1:
                        imgin[i,j] = imgZ[2*i,j]
                        imgin2[i,j] = imgZ[2*i+1,j]
            imgin = torch.from_numpy(imgin)
            imgin = torch.unsqueeze(imgin,0)
            imgin = torch.unsqueeze(imgin,0)
            imgin = imgin.to(device)
            imgin2 = torch.from_numpy(imgin2)
            imgin2 = torch.unsqueeze(imgin2,0)
            imgin2 = torch.unsqueeze(imgin2,0)
            imgin2 = imgin2.to(device)
            listimgH.append(imgin)
            listimgH.append(imgin2)
            
            listimgV = []
            Zshape = [shape[0],shape[1]]
            if shape[0] % 2 == 1:
                Zshape[0] -= 1
            if shape[1] % 2 == 1:
                Zshape[1] -=1  
            imgZ = img[:Zshape[0],:Zshape[1]]
            
            imgin3 = np.zeros((Zshape[0],Zshape[1]//2),dtype=np.float32)
            imgin4 = np.zeros((Zshape[0],Zshape[1]//2),dtype=np.float32)
            for i in range(imgin3.shape[0]):
                for j in range(imgin3.shape[1]):
                    if i % 2 == 0:
                        imgin3[i,j] = imgZ[i,2*j+1]
                        imgin4[i,j] = imgZ[i, 2*j]
                    if i % 2 == 1:
                        imgin3[i,j] = imgZ[i,2*j]
                        imgin4[i,j] = imgZ[i,2*j+1]
            imgin3 = torch.from_numpy(imgin3)
            imgin3 = torch.unsqueeze(imgin3,0)
            imgin3 = torch.unsqueeze(imgin3,0)
            imgin3 = imgin3.to(device)
            imgin4 = torch.from_numpy(imgin4)
            imgin4 = torch.unsqueeze(imgin4,0)
            imgin4 = torch.unsqueeze(imgin4,0)
            imgin4 = imgin4.to(device)
            listimgV.append(imgin3)
            listimgV.append(imgin4)
            
        
            img = torch.from_numpy(img)
            img = torch.unsqueeze(img,0)
            img = torch.unsqueeze(img,0)
            img = img.to(device)
            
            listimgV1 = [[listimgV[0],listimgV[1]]]
            listimgV2 = [[listimgV[1],listimgV[0]]]
            listimgH1 = [[listimgH[1],listimgH[0]]]
            listimgH2 = [[listimgH[0],listimgH[1]]]
            listimg = listimgH1+listimgH2+listimgV1+listimgV2
            
            net = Net()
            net.to(device)
            criterion = nn.BCELoss()
            optimizer = optim.Adam(net.parameters(), lr=learning_rate)
            
            
            
            running_loss1=0.0
            running_loss2=0.0
            maxpsnr = -np.inf
            timesince = 0
            last10 = [0]*105
            last10psnr = [0]*105
            cleaned = 0
            while timesince <= tsince:
                indx = np.random.randint(0,len(listimg))
                data = listimg[indx]
                inputs = data[0]
                labello = data[1]
                
                optimizer.zero_grad()
                outputs = net(inputs)
                loss1 = criterion(outputs, labello)
                loss = loss1
                running_loss1+=loss1.item()
                loss.backward()
                optimizer.step()
                
                
                running_loss1=0.0
                with torch.no_grad():
                    last10.pop(0)
                    last10.append(cleaned*maxer+minner)
                    outputstest = net(img)
                    cleaned = outputstest[0,0,:,:].cpu().detach().numpy()
                    
                    noisy = img.cpu().detach().numpy()
                    ps = -np.mean((noisy-cleaned)**2)
                    last10psnr.pop(0)
                    last10psnr.append(ps)
                    if ps > maxpsnr:
                        maxpsnr = ps
                        outclean = cleaned*maxer+minner
                        timesince = 0
                    else:
                        timesince+=1.0
            H = np.mean(last10, axis=0)
            if np.sum(np.round(H[1:-1,1:-1]-np.mean(H[1:-1,1:-1]))>0) <= 25 and learning_rate != 0.000005:
                learning_rate = 0.000005
                print("Reducing learning rate")
            else:
                notdone = False
                print("--- %s seconds ---" % (time.time() - start_time))
                start_time = time.time()
                
        
        
        
        imwrite(outfolder + '/' + file_name, np.round(H).astype(typer))
        
        
        
        torch.cuda.empty_cache()
