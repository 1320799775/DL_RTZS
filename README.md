# DL_RTZS
A deep learning method for obtaining ultraclean pure shift NMR spectra. 

# Profiles
Real time ZS experimental data of quinine (divided into 2 spectra, according to the preprocessing method of dealing with spectral width of over 4096 Hz in the paper), azithromycin and strychnine:
1. exp/exp_quinine_1.txt
2. exp/exp_quinine_2.txt
3. exp/exp_azithromycin.txt
4. exp/exp_strychnine.txt

Real time ZS experimental data of quinine from the University of Manchester (weblink: https://data.mendeley.com/datasets/rgj4jwcsnz):
1. exp/Pure_shift_archive_Varian

Model weight file (model_rtzs.h5) can be download from the weblink: https://www.dropbox.com/s/1yfoj91r7n2d505/model_rtzs.h5

Model can be tested or used by the file:
1. modelTest.py

Model training code, using AC-ResNet and SM-CDMANE, is shown in the file：
1. modelTrain.py

The code for generating dataset is shown in the file:
1. dataset_rtZS.m

The code for loading the real time ZS FID data acquired on the Agilent NMR System is shown in the file:
1. load_Varian_fid_to_spec.m

The configuration files of 'load_Varian_fid_to_spec.m' is shown in the files:
1. auto_phase.m
2. WeiF_LoadFid.m

# Dependencies
1. keras == 2.2.4
2. numpy == 1.16.0
3. tensorfolw == 2.4.0
4. h5py == 3.1.0
5. matplotlib == 3.3.1

Model has been written and tested with the above dependencies. Performance with other module versions has not been tested.

# Generating Dataset and Training Model


# Preparing Data for Testing
When testing, the inputs to 'modelTest.py' must be in '.txt' format, where different data points are separated by Spaces, eg. 'point1 point2 ... point4096'. 'load_Varian_fid_to_spec.m' provides the methods to prepare data for testing and save data as '.txt' format. Running 'load_Varian_fid_to_spec.m' can convert the real time ZS FID data acquired on the Agilent NMR System into '.txt' format required by 'modelTest.py'.

Prior to input into the network model, the FID data containing less than 4096 complex points needs to be zero-filled to 4096 complex points and Fourier transformed. Then, the spectrum is phased, and only the real part is taken and normalized to 1.

If FID contains more than 4096 complex points or the spectral width is larger than 4096 Hz, FID can be zero-filled to 8192 or more (integer multiple of 4096) complex points, then the corresponding spectrum can be divided into two or more spectra with 4096 real points as inputs, and finally the processed spectra can be concatenated into a complete spectrum.
