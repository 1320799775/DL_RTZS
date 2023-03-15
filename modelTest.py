from keras import backend as K
K.set_image_data_format('channels_first')
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import h5py

# Read experimental data
a = h5py.File('exp_quinine_1.mat')
# a = h5py.File('exp_quinine_2.mat')
# a = h5py.File('exp_azithromycin.mat')
# a = h5py.File('exp_strychnine.mat')

# Pre-process
X = a['data']
num_raws = 4096
X = np.array(X).reshape(1, num_raws).astype('float16')
X = np.expand_dims(X, 2)
print('Data shape: ', X.shape)

# Do not need loss function when testing
def HADAMARD_YZX(y_true, y_pred):
    return 0

# Load model
model = load_model('model_rtzs.h5', custom_objects={'HADAMARD_YZX': HADAMARD_YZX})

# Post-process
predict = model.predict(X, verbose=1)

# Show
xt = X[0]
plt.subplot(211)
plt.plot(xt)
plt.subplot(212)
plt.plot(predict.reshape(num_raws, 1))
plt.show()