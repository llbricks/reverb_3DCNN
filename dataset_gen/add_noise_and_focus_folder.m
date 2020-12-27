sim_path = '/home/llbricks/lesion_val_dataset/20190711_val_0p5mmLesion';
save_path = fullfile(sim_dest_path,'mat_files');
mkdir(save_path)
%%
j = 0; 
batch = 1;
for i = 1:2

    simfile = [sim_source_path 'pre_f2p/imagenet_' sprintf('%08d',i) '_01.h5'];
    disp(['Processing ' simfile])

    % add noise and focus the file
    [focsig_noise, focsig_noNoise, dbT, dbR] = add_noise_and_focus_file_rand_noise(simfile, 1,dbT,dbR,1);

    % Clip off bright top and bottom
    focsig_noise = focsig_noise(10:214,:,:);
    focsig_noNoise = focsig_noNoise(10:214,:,:);

    % Put in the format matlab will expect
    y = permute(focsig_noNoise,[4,1,2,3]); 
    x = permute(focsig_noise,[4,1,2,3]); 
    filename = ['batch_' num2str(i) '.mat'];
    save(fullfile(save_path,filename) ,'x','y','dbT','dbR','-v7.3')
    clearvars x y;
    disp(['saved file ' fullfile(save_path,filename)])

end