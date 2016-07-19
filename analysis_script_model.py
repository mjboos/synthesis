from joblib import Parallel, delayed
from scores_cv_splits import create_scores_from_splits
from prepare_fmri_splits import split_fmri_data_model

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
    alpha_grid = [1e6, 1e5, 1e4, 1e3, 1e2, 1e1, 1e-1]
    for model_nr in xrange(8):
        for split_nr, split in enumerate(split_fmri_data_model(1, 'model_{}'.format(model_nr))):
            create_scores_from_splits(split, model, 1, split_nr, alpha_grid, savefolder='model_{}'.format(model_nr))

