function [par, rfdata, start_sample] = load_simdata_h5(h5file, tx_elem)

par.status        = h5readatt(h5file,'/data','status');
par.transducer	  = h5readatt(h5file,'/parameters','transducer');
par.fs			  = h5readatt(h5file,'/parameters','fs');
par.c			  = h5readatt(h5file,'/parameters','c');
par.fc			  = h5readatt(h5file,'/parameters','fc');
par.nelem_x		  = h5readatt(h5file,'/parameters','nelem_x');
par.nelem_y		  = h5readatt(h5file,'/parameters','nelem_y');
par.kerf_x		  = h5readatt(h5file,'/parameters','kerf_x');
par.kerf_y		  = h5readatt(h5file,'/parameters','kerf_y');
par.pitch_x		  = h5readatt(h5file,'/parameters','pitch_x');
par.pitch_y		  = h5readatt(h5file,'/parameters','pitch_y');
par.width		  = h5readatt(h5file,'/parameters','width');
par.height		  = h5readatt(h5file,'/parameters','height');
par.elev_foc	  = h5readatt(h5file,'/parameters','elev_foc');
par.bandwidth	  = h5readatt(h5file,'/parameters','bandwidth');
par.tshift		  = h5readatt(h5file,'/parameters','tshift');
par.impulse		  = h5readatt(h5file,'/parameters','impulseResponse');
par.excitation	  = h5readatt(h5file,'/parameters','excitationPulse');
par.phantom.img_x = h5readatt(h5file,'/phantom', 'img_x');
par.phantom.img_y = h5readatt(h5file,'/phantom', 'img_y');
par.phantom.img_z = h5readatt(h5file,'/phantom', 'img_z');

par.phantom.ground_truth = h5read(h5file,'/phantom/ground_truth');

if nargin > 1
	str = ['/data/tx_elem_' num2str(tx_elem)];
	rfdata = h5read(h5file, str);
	start_sample = double(h5readatt(h5file,str,'start_sample'));
end
