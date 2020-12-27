% This code aligns the raw simulation data from single element
% transmits and uses the synthetic aperture technique to
% reconstruct image pixels at desired locations in the field.
%

function [focsig_noise, focsig_noNoise] = add_noise_and_focus_file_setNoise(simfile, thermal_dB,reverb_dB)

    % Create an anonymous function to easily draw from a uniform distribution
    rrange = @(xmin,xmax) rand() .* (xmax-xmin) + xmin;  

    % Define noise added
%     thermal_dB = rrange(-20, 0); % Thermal noise dB level
%     reverb_dB = rrange(-20, 10); % Reverb noise dB level

    %% Load simulation parameters
    par = load_simdata_h5(simfile);
    fs = par.fs;
    fc = par.fc;
    bw = par.bandwidth;
    par.nsub_y = 19;

    %% Use pre-defined image grid
    img_x = par.phantom.img_x;
    img_y = par.phantom.img_y;
    img_z = par.phantom.img_z;
    nix = length(img_x);
    niy = length(img_y);
    niz = length(img_z);

    %% Actually, use a smaller grid than the pre-defined image grid
    % Need to do this to avoid aliasing later, when resampling the channel data

    %% Compute delays from each pixel point to each element
    % (Delays are the same for both transmit and receive)
    
    % Find positions of transducer elements
    ele_x = (1:par.nelem_x) * par.pitch_x;
    ele_y = (1:par.nelem_y) * par.pitch_y;
    ele_x = ele_x-mean(ele_x);
    ele_y = ele_y-mean(ele_y);
    [ele_y,ele_x] = meshgrid(ele_y,ele_x);
    ele_z = zeros(numel(ele_x),1);
    nelems = par.nelem_x * par.nelem_y;
   
    % Compute the delay in seconds-5
    delays = zeros(nix*niy*niz,nelems);
    [z, x, y] = ndgrid(img_z, img_x, img_y);
    z = z + 0.3527e-3;
    for E = 1:nelems
        delays(:,E) = 1 / par.c * ...
            sqrt( (x(:)-ele_x(E)).^2 + (y(:)-ele_y(E)).^2 + (z(:)-ele_z(E)).^2 );
    end
    delays = reshape(delays,niz,nix*niy,nelems);

    %% Apply synthetic aperture focusing
    % Iterate through phantoms
    
    % Pre-allocate memory for image output
    focsig_noise = zeros(niz,nix*niy,nelems);
    focsig_noNoise = zeros(niz,nix*niy,nelems);

    % Also pre-compute channel matrix to be used as input to interp2
    chanmat = single(repmat(1:nelems, nix*niy*niz, 1));

    max_nsamps = 0;
    % Iterate across transmit elements%
%     disp(['Focusing...'])

    for T = 1:nelems

%         disp(T)
        % Make sure data is available
        if par.status(T) ~= 1
            disp(['Waiting for data for transmit element ' num2str(T-1) '...'])
            while par.status(T) ~= 1
                pause(300);
                par = load_simdata_h5(simfile,T);
            end
        end

        % Load rf Data
        [par, rfdata_noNoise, start_sample] = load_simdata_h5(simfile, T-1);
        error_if_nan(rfdata_noNoise)	

        % Add noise
        rfdata_noise = addNoise_rf(rfdata_noNoise,reverb_dB,thermal_dB,fc,fs,bw);
        
        % Find the time vector for the data
        t = ((1:size(rfdata_noNoise,1))'-1 + start_sample) / par.fs - par.tshift;

        % Compute the roundtrip delay from every pixel to each element for this transmit
        delayRT = bsxfun(@plus, delays, delays(:,:,T));
        delayRT = reshape(delayRT, nix*niy*niz, nelems);
                
        % Get the focused signal for all pixel locations for this transmit

        delayRT_t = single((delayRT + par.tshift) * par.fs - start_sample+1);
        focsig_noise = focsig_noise + beamform_fullSynth_f2p(rfdata_noise,delayRT_t,chanmat,nix,niy,niz,nelems,T);


        % Beamform both the noisy and noise free data
        focsig_noNoise = focsig_noNoise + beamform_fullSynth_f2p(rfdata_noNoise,delayRT_t,chanmat,nix,niy,niz,nelems,T);

%             % Display
%             if show_display && mod(T,1) == 0
%                 tmp = abs(sum(focsig_noise,3));
%                 figure(1)
%                 subplot(121), imagesc(img_x, img_z, db(tmp/max(tmp(:))), [-50 0]); axis image
%                 subplot(122), imagesc(img_x, img_z, db(gt/max(gt(:))), [-50 0]); axis image
%                 colormap gray
%                 drawnow
%                 figure(2)
%                 tmp = abs(sum(focsig_noNoise,3));
%                 subplot(121), imagesc(img_x, img_z, db(tmp/max(tmp(:))), [-50 0]); axis image
%                 subplot(122), imagesc(img_x, img_z, db(gt/max(gt(:))), [-50 0]); axis image
%                 colormap gray
%                 drawnow
%             end
    end

    % Demodulate
    focsig_noise = bsxfun(@times, focsig_noise, exp(-2j*pi*par.fc*img_z/(par.c/2)));
    focsig_noNoise = bsxfun(@times, focsig_noNoise, exp(-2j*pi*par.fc*img_z/(par.c/2)));

    % Decimate ( upsamp actually = downsamp) 
    upsamp = 1;
    focsig_noise = focsig_noise(1:upsamp:end,:,:);
    focsig_noNoise = focsig_noNoise(1:upsamp:end,:,:);

    
end
