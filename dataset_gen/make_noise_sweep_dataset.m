% define where the datset is, this should be a path to a folder containing a folder named 'pre_f2p'
sim_path = '/home/llbricks/lesion0p5mm_val';

% make the destination path
save_path = fullfile(sim_path,'mat_files');
mkdir(save_path)      

% must have the interp2_gpumex repo from Dongwoon Hyun, add it's directory path here
addpath /home/llbricks/interp2_gpumex

% loop through the simulations
for sim_idx = 1:3
    
    disp(['-----Simulation ' num2str(sim_idx) '-----'])

    % define the source file for the field II pro simulation
    simfile = fullfile(sim_path,['pre_f2p/anechoic_0p5mm_lesion_' num2str(sim_idx) '.h5']);
    
    % loop through noise added
    for n = -15:10
        
        disp(['Noise = ' num2str(n) ' dB'])
        
        % Set thermal and reverberation noise
        dbT = n;
        dbR = n;
        
        % Focus the rf data, with and without noise added
        [focsig_noise, focsig_noNoise] = add_noise_and_focus_file_setNoise(simfile, dbT, dbR);

        % Clip off edges, make centered at 200x200
        focsig_noise = focsig_noise(12:211,12:211,:);
        focsig_noNoise = focsig_noNoise(12:211,12:211,:);

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

        % Put in the format that python will expect
        y = permute(focsig_noNoise,[4,1,2,3]); 
        x = permute(focsig_noise,[4,1,2,3]); 

        filename = ['sim' num2str(sim_idx) '_' num2str(n) 'dB.mat'];
        save(fullfile(save_path,filename) ,'x','y','dbT','dbR','sim_idx','-v7.3')
    end
end



