# DL_RTZS
A deep learning method for obtaining ultraclean pure shift NMR spectra. 

# Profiles
Real time ZS experimental data of quinine (divided into 2 spectra, according to the preparing method of dealing with spectral width of over 4096 Hz in the paper), azithromycin and strychnine:
1. exp_quinine_1.mat
2. exp_quinine_2.mat
2. exp_azithromycin.mat
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
The input to the DNN must be in '.mat' format with variable name of 'data', which can be edited in MATLAB or Python. 

Prior to input into the network model, the FID data containing less than 4096 complex points needs to be zero-filled to 4096 complex points and Fourier transformed to the spectrum. Then, the spectrum is phased, taken as the real part, and normalized to 1. 

If FID contains more than 4096 complex points or the spectral width is larger than 4096 Hz, FID can be zero-filled to 8192 or more (integer multiple of 4096) complex points, then the corresponding spectrum can be divided into two or more spectra with 4096 real points as inputs, and finally the processed spectra can be concatenated into a complete spectrum.
