#!/usr/bin/python

import subprocess
import os

import numpy as np
import scipy
from nibabel import load, save, Nifti1Image
from nilearn import image
from nilearn.input_data import NiftiMasker
from joblib import Parallel, delayed
from fg_constants import *

def confounds(num, subj):
    conf = os.path.join(DATA_DIR, subj, 'BOLD', 'task001_run00%d'%(num+1), 'bold_dico_moco.txt')
    pca = np.load(os.path.join(CONFOUNDS_DIR, subj, '%i.npy' % (num,)))
    motion = []
    with open(conf, 'r') as f:
        for line in f:
            motion.append(map(float, line.split()))
    motion = np.array(motion)
    return np.hstack((motion, pca))


def warp_mask(subj, mask_warp_dir):
    mask_path = os.path.join(DATA_DIR, subj, 'templates', 'bold7Tp1', 'brain_mask.nii.gz')
    template_path = os.path.join(DATA_DIR, 'templates', 'grpbold7Tp1', 'brain.nii.gz')
    warp_path = os.path.join(DATA_DIR, subj, 'templates', 'bold7Tp1', 'in_grpbold7Tp1', 'subj2tmpl_warp.nii.gz')

    output_path = os.path.join(mask_warp_dir, '%s.nii.gz' % subj)

    print 'Warping image for %s...' % subj
    subprocess.call(['fsl5.0-applywarp', '-i', mask_path, '-o', output_path, '-r', template_path, '-w', warp_path, '-d', 'float'])


def preprocess(num, subj, subj_dir, subj_warp_dir, force_warp=False):
    bold_path = 'BOLD/task001_run00%i/bold_dico_bold7Tp1_to_subjbold7Tp1.nii.gz' % (num+1)
    bold_path = os.path.join(DATA_DIR, subj, bold_path)
    template_path = os.path.join(DATA_DIR, 'templates', 'grpbold7Tp1', 'brain.nii.gz')
    warp_path = os.path.join(DATA_DIR, subj, 'templates', 'bold7Tp1', 'in_grpbold7Tp1', 'subj2tmpl_warp.nii.gz')

    output_path = os.path.join(subj_warp_dir, 'run00%i.nii.gz' % num)

    if force_warp or not os.path.exists(output_path):
        print 'Warping image #%i...' % num
        subprocess.call(['fsl5.0-applywarp', '-i', bold_path, '-o', output_path, '-r', template_path, '-w', warp_path, '-d', 'float'])
    else:
        print 'Reusing cached warp image #%i' % num

    print 'Loading image #%i...' % num
    bold = load(output_path)

    # TODO extract masker somewhere
    mask_file = './masks/template_mask_thick.nii.gz'
    masker = NiftiMasker(load(mask_file))
    # masker = niftimasker(load(mask_file), detrend=true, smoothing_fwhm=4.0,
    #                     high_pass=0.01, t_r=2.0, standardize=true)
    masker.fit()
    print 'Removing confounds from image #%i...' % num
    data = masker.transform(bold, confounds(num, subj))
    print 'Detrending image #%i...' % num
    filtered = np.float32(scipy.signal.savgol_filter(data, 61, 5, axis=0))
    img = masker.inverse_transform(data-filtered)
    print 'Smoothing image #%i...' % num
    img = image.smooth_img(img, 4.0)
    print 'Saving image #%i...' % num
    save(img, os.path.join(subj_dir, 'run00%i.nii.gz' % num))
    print 'Finished with image #%i' % num


def convert_subject(subj):
    subj_dir = os.path.join(PREP_DATA_DIR, subj)
    if not os.path.exists(subj_dir):
        os.makedirs(subj_dir)

    subj_warp_dir = os.path.join(WARP_DIR, subj)
    if not os.path.exists(subj_warp_dir):
        os.makedirs(subj_warp_dir)

    Parallel(n_jobs=8)(delayed(preprocess)(i, subj, subj_dir, subj_warp_dir) for i in range(TOTAL_SEGMENTS))


if __name__ == '__main__':
    subj_list = SUBJECTS
    for sub_num in subj_list:
        subj = 'sub%03d' % sub_num
        convert_subject(subj)
        # warp_mask(subj, './mask_warp')
