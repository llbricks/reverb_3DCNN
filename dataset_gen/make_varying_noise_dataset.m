% sim_path = '/data/llbricks/datasets/field2/20181206_ImageNet_anechoic/';
% sim_path = '/data/llbricks/datasets/field2/20190226_1/';
sim_path = '/data/llbricks/datasets/field2/20190330_1/';

addpath /data/llbricks/US_denoising/preprocessing/field2/matlab_helper_functions
save_path = fullfile(sim_path,'test_files');
mkdir(save_path)

j = 1; 
i = 1025;
      
%     try
simfile = [sim_path 'pre_f2p/imagenet_' sprintf('%08d',i) '_01.h5'];
disp(['Processing ' simfile])

% add noise and focus the file
% thermal_dB = -100;
for n = -15:10
    disp(['n = ' num2str(n)])
    reverb_dB = n;
    thermal_dB = n;
    [focsig_noise, focsig_noNoise] = add_noise_and_focus_file_setNoise(simfile, '', thermal_dB,reverb_dB);

    % Clip off bright top and bottom
    focsig_noise = focsig_noise(10:214,:,:);
    focsig_noNoise = focsig_noNoise(10:214,:,:);

    % display the loaded images
    if 0
        figure(1)
        tmp = abs(sum(focsig_noise,3));
        subplot(1,2,1), imagesc(db(tmp/max(tmp(:))), [-50 0]); axis image
        tmp = abs(sum(focsig_noNoise,3));
        subplot(1,2,2), imagesc(db(tmp/max(tmp(:))), [-50 0]); axis image
        colormap gray
        drawnow
%             waitforbuttonpress
    end

    % Put in the format matlab will expect
    try
        y(j,:,:,:) = permute(focsig_noNoise,[4,1,2,3]);
    catch
        y = permute(focsig_noNoise,[4,1,2,3]); 
    end
    try
        x(j,:,:,:) = permute(focsig_noise,[4,1,2,3]);
    catch
        x = permute(focsig_noise,[4,1,2,3]); 
    end

    j = j + 1;
end

filename = ['noiseTest_sweepReverb_plusThermal.mat'];
save(fullfile(save_path,filename) ,'x','y','-v7.3')
clear x y;
disp('saved')
j = 1;

