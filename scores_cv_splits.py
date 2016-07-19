# coding: utf-8
from __future__ import division
from sklearn.linear_model import RidgeCV
from sklearn.cross_validation import KFold
from sklearn.metrics import r2_score
from sklearn.externals.joblib import load, dump
import sys
import argparse
from joblib import Parallel, delayed
import numpy as np

spenc_dir = '/storage/workspace/mboos/'
syn_spenc_dir = '/storage/workspace/mboos/synthesis/'

def ridge_fit(stim_train, fmri_train, stim_test, fmri_test, alpha_grid):
    predictions = RidgeCV(alphas=alpha_grid
                  ).fit(stim_train, fmri_train).predict(stim_test)
    return np.array([np.corrcoef(predictions[:, i], fmri_test[:, i])[0,1] for i in xrange(predictions.shape[1])])

def create_scores_from_splits(fmri_data, model_name, subj, split, alpha_grid, work_dir=syn_spenc_dir, n_jobs=8, savefolder='logMFS_ds'):
    stimuli = load(work_dir+'prepro/'+model_name+\
            '_stimuli.pkl')

    cv = KFold(stimuli.shape[0], n_folds=8, random_state=500)

    scores_list = Parallel(n_jobs=n_jobs)(
                       delayed(ridge_fit)(stimuli[train], fmri_data[train],
                                          stimuli[test], fmri_data[test],
                                          alpha_grid)
                       for train, test in cv)
    scores = np.hstack([score[:, None] for score in scores_list])
    dump(scores,
         work_dir+'scores/'+savefolder+'{}_subj_{}_split_{}.pkl'.format(model_name,
                                                          subj, split),
         compress=3)

def create_scores(model_name, subj, split, alpha_grid, work_dir=syn_spenc_dir, n_jobs=8):
    fmri_data = load(work_dir+'splits/'+\
            'fmri_subj_{}_split_{}.pkl'.format(subj, split))
    create_scores_from_splits(fmri_data, model_name, subj, split, alpha_grid,
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
    create_scores(model_name, subj, split, alpha_grid)
