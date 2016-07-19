from __future__ import division
from joblib import dump, Memory
import numpy as np
import sys
from sklearn.cross_validation import KFold
from nilearn.input_data import MultiNiftiMasker
from fg_constants import *
import os


mask_file = './masks/template_mask_thick.nii.gz'
memory = Memory(cachedir=JOBLIB_DIR, verbose=1)

spenc_dir = '/storage/workspace/mboos/'

@memory.cache
def split_fmri_data_ko(subj, n_splits=10):
    subj_preprocessed_path = spenc_dir+'clean_data/sub{0:03d}/'.format(subj)

    run_fn = [subj_preprocessed_path+'run{0:03d}.nii.gz'.format(i) for i in xrange(8)]

    masker = MultiNiftiMasker(mask_img=mask_file)
    data = np.concatenate(masker.fit_transform(run_fn), axis=0).astype('float32')
    duration = np.array([902,882,876,976,924,878,1084,676])

    # i did not kick out the first/last 4 samples per run yet
    slice_nr_per_run = [dur/2 for dur in duration]

    # use broadcasting to get indices to delete around the borders
    idx_borders = np.cumsum(slice_nr_per_run[:-1])[:,np.newaxis] + \
                  np.arange(-4,4)[np.newaxis,:]

    data = np.delete(data, idx_borders, axis=0)

    # and we're going to remove the last fmri slice
    # since it does not correspond to a movie part anymore
    data = data[:-1, :]

    # shape of TR samples
    data = data[3:]

    voxel_kfold = KFold(data.shape[1], n_folds=n_splits)
    return [data[:, split] for _, split in voxel_kfold]

if __name__=='__main__':
    subj = sys.argv[1]
    subj = int(subj)
    for i, split in enumerate(split_fmri_data(subj)):
        dump(split,
             spenc_dir+'prepro/fmri_subj_{}_split_{}.pkl'.format(subj, i),
             compress=3)
