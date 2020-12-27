function focsig = beamform_fullSynth_f2p(rfdata,delayRT_t,chanmat,nix,niy,niz,nelems,T)

    rfdata = single(rfdata);
    rfdata = rfdata / max(rfdata(:));
    rfdata = rfdata / rms(rfdata(:));

    % Get complex representation of RF signal
    iqdata = hilbert(rfdata);
    clear rfdata

    max_nsamps = size(iqdata,1);
%     if T == 1
        % Specify three input arguments to initialize interp2_gpumex
    focR = interp2_gpumex(real(iqdata), chanmat, delayRT_t);
%     else
%         % If the data is smalsler than previous, re-use existing CUDA arrays
%         if size(iqdata,1) <= max_nsamps
%             % Specify four input arguments to re-use allocated arrays
%             focR = interp2_gpumex(real(iqdata), chanmat, delayRT_t, 1);
%         else
%             % If the data is larger than before, allocate new CUDA arrays
%             focR = interp2_gpumex(real(iqdata), chanmat, delayRT_t);
%             max_nsamps = size(iqdata,1);
%         end
%     end
    focI = interp2_gpumex(imag(iqdata));
    foctmp = focR + 1i*focI;

    % Apply proper phase rotation to foctmp and add to focsig running sum
    focsig = reshape(foctmp, niz, nix*niy, nelems);
end
