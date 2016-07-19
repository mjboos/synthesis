from joblib import Parallel, delayed
from scores_cv_splits import create_scores_from_splits
from fg_constants import *

def prepare_for_fitting_fmri(n_splits=10):
    from prepare_fmri_splits_ko import split_fmri_data_ko
    for subj in [1]:
        yield split_fmri_data_ko(subj, n_splits)

def prepare_for_fitting_audio():
    pass

def preprocess_fmri():
    #TODO
    pass

if __name__=='__main__':
    model = 'logMFS_ds'
    alpha_grid = [ 1e-1, 1e-2, 1e-3]
    preprocess_fmri()
    for subj, split_list in zip(SUBJECTS, prepare_for_fitting_fmri()):
        for split_nr, split in enumerate(split_list):
            create_scores_from_splits(split, model, subj, split_nr, alpha_grid, work_dir='/storage/workspace/mboos/')
