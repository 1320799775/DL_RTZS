from keras import backend as K
K.set_image_data_format('channels_first')
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

# Read experimental data
# fdata = open('./exp/exp.txt', 'r')
# fdata = open('./exp/exp_quinine_1.txt', 'r')
# fdata = open('./exp/exp_quinine_2.txt', 'r')
# fdata = open('./exp/exp_azithromycin.txt', 'r')
fdata = open('./exp/exp_strychnine.txt', 'r')
data = fdata.readlines()
X = data[0].strip('\n').split('\t')
for i in range(len(X)):
    X[i] = float(X[i])  # 4096, list[float]

# Pre-process
# --------------------------------------------------------------
# If the spectrum is not divided into 2 or more sub-spectra, the spectrum can be normalized to 1 directly.
# If the spectrum is divided into 2 or more sub-spectra, the spectrum need to be normalized to 1 before being divided.
# --------------------------------------------------------------
# X = X / np.nanmax(X)  #  Normalizing data to 1
num_raws = 4096
X = np.array(X).reshape(1, num_raws).astype('float16')
X = np.expand_dims(X, 2)
print('Data shape: ', X.shape)

# Do not need loss function when testing
def HADAMARD_YZX(y_true, y_pred):
    return 0

# Load model
model = load_model('model/model_rtzs.h5', custom_objects={'HADAMARD_YZX': HADAMARD_YZX})

# Post-process
predict = model.predict(X, verbose=1)

# Show
xt = X[0]
plt.subplot(211)
plt.plot(xt)
plt.subplot(212)
plt.plot(predict.reshape(num_raws, 1))
plt.show()