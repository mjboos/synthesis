# coding: utf-8
from __future__ import division
from sklearn.linear_model import RidgeCV
from sklearn.cross_validation import KFold
from sklearn.metrics import r2_score
from sklearn.externals.joblib import load, dump
import sys
import argparse
from joblib import Parallel, delayed

spenc_dir = '/storage/workspace/mboos/'
syn_spenc_dir = '/storage/workspace/mboos/synthesis/'

def ridge_fit(stim_train, fmri_train, alpha_grid):
    return RidgeCV(alphas=alpha_grid
           ).fit(stim_train, fmri_train)

def create_models_from_splits(fmri_data, model_name, subj, split, alpha_grid, work_dir=syn_spenc_dir, n_jobs=8):
    stimuli = load(work_dir+'prepro/'+model_name+\
            '_stimuli.pkl'.format(subj))

    cv = KFold(stimuli.shape[0], n_folds=8, random_state=500)

    models_list = Parallel(n_jobs=n_jobs)(
                       delayed(ridge_fit)(stimuli[train], fmri_data[train],
                                          alpha_grid)
                       for train, test in cv)

    dump(models_list,
         work_dir+'models/{}_subj_{}_split_{}.pkl'.format(model_name,
                                                          subj, split),
         compress=3)

def create_models(model_name, subj, split, alpha_grid, work_dir=syn_spenc_dir, n_jobs=8):
    fmri_data = load(work_dir+'splits/'+\
            'fmri_subj_{}_split_{}.pkl'.format(subj, split))
    create_models_from_splits(fmri_data, model_name, subj, split, alpha_grid,
                                   work_dir=work_dir, n_jobs=n_jobs)

if __name__=='__main__':
    alpha_grid = [1e2, 1e3, 1e4]

    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str)
    parser.add_argument('subj', type=int)
    parser.add_argument('split', type=int)
    parser.add_argument('--alpha', default=alpha_grid, type=float, nargs='*')

    arg_namespace = vars(parser.parse_args())

    model_name, subj, split, alpha_grid = [arg_namespace[key]
                                           for key in ['model_name',
                                                       'subj',
                                                       'split',
                                                       'alpha']]
    create_models(model_name, subj, split, alpha_grid)
