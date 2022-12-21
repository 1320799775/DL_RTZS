from keras import backend as K
K.set_image_data_format('channels_first')
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from keras.models import load_model
import h5py

# read experimental data
a = h5py.File('exp/exp_azithromycin.mat')
# a = h5py.File('exp/exp_quinine.mat')
# a = h5py.File('exp/exp_strychnine.mat')
data_all = a['data']
X = data_all['data_x']
num_raws = 4096
X = np.array(X).reshape(1, num_raws).astype('float16')
X = np.expand_dims(X, 2)
print('Data shape: ', X.shape)

# loss function
def HADAMARD_YZX(y_true, y_pred):
    return 0

# load model
model = load_model('model_rtzs.h5', custom_objects={'HADAMARD_YZX': HADAMARD_YZX})

# process
xt = X[0]
plt.subplot(211)
plt.plot(xt)
# x = xt.reshape(num_raws, 1).tolist()

predict = model.predict(X, verbose=1)
plt.subplot(212)
plt.plot(predict.reshape(num_raws, 1))
plt.show()
# predict = predict.reshape(num_raws, 1).tolist()

