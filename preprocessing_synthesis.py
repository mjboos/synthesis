# coding: utf-8
import subprocess
import os
import numpy as np
from nibabel import load, save
from nilearn.input_data import NiftiMasker
from joblib import Parallel, delayed
from fg_constants import *
from sklearn.preprocessing import StandardScaler
import sys

def preprocess_varpar(num, subj, subj_dir, **kwargs):
    from nistats.design_matrix import make_design_matrix
    from nistats.first_level_model import run_glm
    bold_path = 'BOLD/task001_run00%i/bold_dico_bold7Tp1_to_subjbold7Tp1.nii.gz' % (num+1)
    bold_path = os.path.join(DATA_DIR, subj, bold_path)
    mask = os.path.join(DATA_DIR, subj, 'templates', 'bold7Tp1', 'brain_mask.nii.gz')
    bold = load(bold_path)
    masker = NiftiMasker(mask)
    data = masker.fit_transform(bold)
    dmat = make_design_matrix(np.arange(data.shape[0])*TR, hrf_model='fir', drift_order=5,
                              **kwargs)
    labels, results = run_glm(data, dmat, noise_model='ols', verbose=1)
    img = masker.inverse_transform(StandardScaler().fit_transform(results[0.0].resid))
#    return StandardScaler().fit_transform(results[0.0].resid)
    save(img, os.path.join(subj_dir, 'run00%i.nii.gz' % num))

def preprocess(num, subj, subj_dir, subj_warp_dir, force_warp=False, group_mode=False):
    bold_path = 'BOLD/task001_run00%i/bold_dico_bold7Tp1_to_subjbold7Tp1.nii.gz' % (num+1)
    bold_path = os.path.join(DATA_DIR, subj, bold_path)
    template_path = os.path.join(DATA_DIR, 'templates', 'grpbold7Tp1', 'brain.nii.gz')
    warp_path = os.path.join(DATA_DIR, subj, 'templates', 'bold7Tp1', 'in_grpbold7Tp1', 'subj2tmpl_warp.nii.gz')

    output_path = os.path.join(subj_warp_dir, 'run00%i.nii.gz' % num)
    if group_mode:
        if force_warp or not os.path.exists(output_path):
            print 'Warping image #%i...' % num
            subprocess.call(['fsl5.0-applywarp', '-i', bold_path, '-o', output_path, '-r', template_path, '-w', warp_path, '-d', 'float'])
        else:
            print 'Reusing cached warp image #%i' % num
        mask = None
        bold = output_path
    else:
        mask = os.path.join(DATA_DIR, subj, 'templates', 'bold7Tp1', 'brain_mask.nii.gz')
        bold = bold_path
    masker = NiftiMasker(mask, standardize=True, detrend=True)
    img = masker.inverse_transform(masker.fit_transform(bold))
    print 'Saving image #%i...' % num
    save(img, os.path.join(subj_dir, 'run00%i.nii.gz' % num))
    print 'Finished with image #%i' % num

def convert_subject_partest(subj, app, **kwargs):
    subj_dir = os.path.join(PREP_DATA_DIR, app, subj)
    if not os.path.exists(subj_dir):
        os.makedirs(subj_dir)
    Parallel(n_jobs=8)(delayed(preprocess_varpar)(i, subj, subj_dir, **kwargs) for i in range(TOTAL_SEGMENTS))


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
    for sub_num in subj_list[1:]:
        subj = 'sub%03d' % sub_num
        convert_subject(subj)

