# DL_RTZS
A deep learning method for obtaining ultraclean pure shift NMR spectra. 

# Profiles
Real time ZS experimental data of azithromycin, quinine and strychnine:
1. exp_azithromycin.mat
2. exp_quinine.mat
3. exp_strychnine.mat

Model weight file (model_rtzs.h5) can be download from the weblink: https://www.dropbox.com/s/1yfoj91r7n2d505/model_rtzs.h5

Model can be tested or used by the file:
1. modelTest.py

Model training code, using AC-ResNet and SM-CDMANE, is shown in the file：
1. modelTrain.py

# Dependencies
1. keras == 2.2.4
2. numpy == 1.16.0
3. tensorfolw == 1.14.0
4. h5py == 3.1.0
5. matplotlib == 3.3.1

Model has been written and tested with the above dependencies. Performance with other module versions has not been tested.

# Preparing Data
The input to the DNN must be in '.mat' format, which can be edited in MATLAB or Python.

Prior to input into the network model, the FID need to be zero-filled to 4096 complex points and Fourier transformed to the spectrum. Then, the spectrum is phased and normalized to 1.
