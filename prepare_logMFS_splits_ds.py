from __future__ import division
import numpy as np
import glob
from sklearn.preprocessing import StandardScaler
from joblib import dump
from fg_constants import *



duration = np.array([902,882,876,976,924,878,1084,676])




# With MFS Features
mfs_list = glob.glob(AUDIO_STIMULI + '*.mfs')


feature_list = [np.genfromtxt(mfs_fn,delimiter=',')
                for mfs_fn in sorted(mfs_list)]

# cut values smaller 1 and take log
for i in xrange(len(feature_list)):
    feature_list[i][feature_list[i]<1] = 1
    feature_list[i] = np.log(feature_list[i])

ft_freq = feature_list[0].shape[1]

def reduce_MFS(a, b):
    '''deletes offset'''
    return np.concatenate([a[:-8*10], b[8*10:]], axis=0)

feature_list = [feat[:10*dur] for feat, dur in zip(feature_list, duration)]
features = reduce(reduce_MFS, feature_list)
features = features[:-(features.shape[0] % 10)]


features = np.reshape(features, (-1, ft_freq*20))

strides = (features.strides[0],) + features.strides

# rolling window of length 4 samples
shape = (features.shape[0] - 4 + 1, 4, features.shape[1])

features = np.lib.stride_tricks.as_strided(features[::-1,:].copy(),
                                          shape=shape,
                                          strides=strides)[::-1, :, :]

features = np.reshape(features, (features.shape[0], -1))

# we kick out the most recent sample
features = features[:, :-20*ft_freq]


# and downsample to 1 Hz
features = np.mean(np.reshape(features, (features.shape[0], 6, 10, ft_freq)), axis=-2)
features = np.reshape(features, (-1, 6*ft_freq))

features = StandardScaler().fit_transform(features)

dump(features,
     PREPRO_DIR+'logMFS_ds_stimuli.pkl')
