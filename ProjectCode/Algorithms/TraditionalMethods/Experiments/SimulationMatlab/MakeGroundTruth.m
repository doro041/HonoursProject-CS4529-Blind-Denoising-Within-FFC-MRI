% this script makes a ground truth image for FFC image filtering test
% purposes
% LB
% GNU GPLv3 license

clear

% set the image parameters
Nrow = 90;
Ncol = 90;
Ntevo = 5;
Nbevo = 3;
data = zeros(Nrow,Ncol,Ntevo,Nbevo);

% set the experiment parameters
% TODO: this section should be replaced to use tissue data
% t_evo = logspace(log10(0.8),log10(0.03),Ntevo);
t_evo = [   455   242   129    68    36;
        282   150    80    42    23;
        136    73    39    21    11]/1000;      % list of evolution times in ms
B = logspace(log10(0.2),log10(0.0022),Nbevo);   % list of evolution fields in T
Bpol = 0.2;                                     % polarisation field

% here we use a power law model for the T1 NMRD profiles (R1 =
% alpha*B^beta), for all 5 regions in the image
% TODO: this can also be extracted from tissue data
alpha = [2.8 2.2 1.3 0.6 1.9]*2;
beta = -[.1 .15 .3 .005 .08];
proton = [3 1 2 .1 2.02]/3;

% create the segmented image
% TODO: we could draw pseudo-random shapes to simulate random parts of the
% body instead of just the head.
figure(1) % draw a vector image in a figure
Nregions = numel(alpha);
imshow(squeeze(data(:,:,1,1)),[0 Nregions])
draw_ellipse(43,38,45,45, 1,0)      % fat
draw_ellipse(40,35,45,45, 2,0)      % head
draw_ellipse(25,15,50,34, 3,10)     % lobe 1
draw_ellipse(25,15,50,56, 3,-10)    % lobe 2
draw_ellipse(8 ,5 ,60,40, 4,10)     % CSF
draw_ellipse(8 ,5 ,60,50, 4,-10)    % CSF
draw_ellipse(12,4 ,40,60, 5,-10)    % lesion
caxis([0 Nregions])
drawnow
F = getframe(gca);      % convert from vector image to frame 
[dataHighres, Map] = frame2im(F); % convert to uint8 image (not nice but I could not find better, this wastes some signal because of the 8-bit digitisation)
dataHighres = double(squeeze(dataHighres(2:end-1,2:end-1,1))); % recovers the digitisation errors (ugly fix but it works)
dataHighres = floor(dataHighres/255*Nregions);
sze = size(dataHighres);
[X,Y] = meshgrid(1:sze(2),1:sze(1));
[x,y] = meshgrid(linspace(1,sze(2),Nrow),linspace(1,sze(1),Ncol));
dataref = interp2(X,Y,dataHighres,x,y,'nearest'); % interpolated the imnage to the correct image size to mimick sampling

% find isolated voxels and replace them (these are due to the getframe rounding procedure)
se = strel('disk',1);
for n = 1:Nregions
    d = dataref == n;
    d = d - imopen(d,se);
    ind = find(d);
    dataref(ind) = dataref(ind+1);
end

% now generate the magnetisation images ROI by ROI
for b = 1:Nbevo
    for t = 1:size(data,3)
        for n = 1:Nregions
            indlist = find(dataref==n);
            R1(n,b) = alpha(n)*B(b).^beta(n);
            d = zeros(Nrow,Ncol);
            d(indlist) = proton(n)*((-Bpol - B(b))*exp(-t_evo(b,t)*R1(n,b))  +B(b))/Bpol;
            data(:,:,t,b) = data(:,:,t,b) + d;
        end
    end
end

data = rescale(data);

save GroundTruth.mat data



% SNR levels in dB
snr_levels = [10, 5, 3, 1, 0]  

% Loop over SNR levels and apply noise
for i = 1:length(snr_levels)
    snr_db = snr_levels(i);

    % Add Gaussian noise to data
    data_gaussian = data;  % Initialize with the original data
    for b = 1:Nbevo
        for t = 1:Ntevo
            data_gaussian(:,:,t,b) = add_gaussian_noise(data(:,:,t,b), snr_db);
        end
    end
    % Save Gaussian noisy data
    save(sprintf('GroundTruth_gaussian_SNR%d.mat', snr_db), 'data_gaussian');

    % Add Rician noise to data
    data_rician = data;  % Initialize with the original data
    for b = 1:Nbevo
        for t = 1:Ntevo
            data_rician(:,:,t,b) = add_rician_noise(data(:,:,t,b), snr_db);
        end
    end
    % Save Rician noisy data
    save(sprintf('GroundTruth_rician_SNR%d.mat', snr_db), 'data_rician');
end

% Display one example of noisy data
imagesc(data_gaussian(:,:,1,1));
title(sprintf('Example of Gaussian Noisy Data with SNR %d dB', snr_levels(end)));
 
% Function to add Gaussian noise based on SNR
function noisy_img = add_gaussian_noise(img, snr_db)
    P_signal = var(img(:));
    P_noise = P_signal / 10^(snr_db / 10);
    noise = sqrt(P_noise) * randn(size(img));
    noisy_img = img + noise;
end

% Function to add Rician noise based on SNR
function noisy_img = add_rician_noise(img, snr_db)
    P_signal = var(img(:));
    P_noise = P_signal / 10^(snr_db / 10);
    noise_real = sqrt(P_noise / 2) * randn(size(img));
    noise_imag = sqrt(P_noise / 2) * randn(size(img));
    noisy_img = sqrt((img + noise_real).^2 + noise_imag.^2);
end
