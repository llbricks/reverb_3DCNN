function noisyrf = addNoise_rf(rfdata,reverb_dB,thermal_dB,fc,fs,bw)

    % Make reverberation clutter
    % Prepare the bandpass filter for reverberation noise
    filtorder = 100;
    a = 1;
    
    % Compute the axial sampling frequency for the patch
    f1 = fc-fc*bw/2;
    f2 = fc+fc*bw/2;
    f = [f1 f2]*2/fs;
    bz = fir1(filtorder,f,'bandpass');
    
    % Compute reverberation noise
    rnoise = randn(size(rfdata,1)+filtorder, size(rfdata,2), 'single');
    rnoise = filter(bz,a,rnoise,[],1);
    rnoise = rnoise(filtorder+1:end, :);
    rnoise = rnoise / rms(rnoise(:)) * rms(rfdata(:)) * 10^(reverb_dB/20);

    % Compute thermal noise 
    tnoise = randn(size(rfdata,1), size(rfdata,2), 'single') ;
    tnoise = tnoise / rms(tnoise(:)) * rms(rfdata(:)) * 10^(thermal_dB/20);

    % add the noise to create the noisy data
    noisyrf = rfdata + tnoise + rnoise;
    
end