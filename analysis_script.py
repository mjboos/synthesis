from joblib import Parallel, delayed
from scores_cv_splits import create_scores_from_splits

def prepare_for_fitting_fmri(n_splits=10):
    from prepare_fmri_splits import split_fmri_data
    for subj in xrange(11, 20):
        yield split_fmri_data(subj, n_splits)

def prepare_for_fitting_audio():
    pass

def preprocess_fmri():
    #TODO
    pass

if __name__=='__main__':
    model = 'logMFS_ds'
    alpha_grid = [1e6, 1e5, 1e4, 1e3, 1e2]
    preprocess_fmri()
    for subj, split_list in enumerate(prepare_for_fitting_fmri()):
        for split_nr, split in enumerate(split_list):
            create_scores_from_splits(split, model, subj, split_nr, alpha_grid)
