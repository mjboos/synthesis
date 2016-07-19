from __future__ import division
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import r2_score
from sklearn.externals import joblib
from json import load
from nilearn.plotting import plot_stat_map
from nilearn.masking import unmask
from nilearn.image import threshold_img
import sys
from sklearn.mixture import log_multivariate_normal_density
from statsmodels.sandbox.stats.multicomp import fdrcorrection0
from fg_constants import *

spenc_dir = '/storage/workspace/mboos/'

class Encoding(object):
    def __init__(self, scores, metric='p_adj'):
        self.scores = scores
        self.metric = metric
        self.best_voxels = np.argsort(scores)
        if metric in ('p', 'p_adj'):
            p_vals = metric_functions[metric](self.scores)
            if metric == 'p':
                self.threshold = np.min(self.scores[np.logical_and(p_vals<0.05,self.scores>0.0)])
            else:
                self.threshold = np.min(self.scores[np.logical_and(p_vals[0],self.scores>0.0)])
        else:
            self.threshold = 0.0

    def get_best_voxels(self, threshold):
        if threshold is None:
            return self.best_voxels
        if self.metric == 'p':
            return self.best_voxels[-np.sum(self.scores<threshold):]
        else:
            return self.best_voxels[-np.sum(self.scores>threshold):]

def p_adj_map_from_scores(r, n=3539):
    '''Creates a p map with adjusted p values from scores (correlations)'''
    from scipy.stats import betai
    df = n-2
    t_squared = r*r * (df / ((1.0 - r) * (1.0 + r)))
    prob = betai(0.5*df, 0.5, df / (df+t_squared))
    return fdrcorrection0(prob)

def p_map_from_scores(r, n=3539):
    '''Creates a p map from scores (correlations)'''
    from scipy.stats import betai
    df = n-2
    t_squared = r*r * (df / ((1.0 - r) * (1.0 + r)))
    prob = betai(0.5*df, 0.5, df / (df+t_squared))
    return prob

def r2_map_from_predictions(preds_pc, data_to_map):
    '''Creates an r2 score map from predictions'''
    from sklearn.cross_validation import KFold
    cv = KFold(data_to_map.shape[1], n_folds=5)
    scores = np.concatenate([
        r2_score(data_to_map[:, split], preds_pc[:, split],
                 multioutput='raw_values') for _, split in cv])
    return scores.astype('float32')

def r_map_from_predictions(preds_pc, data_to_map):
    '''Creates an r map from predictions'''
    from sklearn.preprocessing import StandardScaler
    mx = StandardScaler().fit_transform(preds_pc)
    my = StandardScaler().fit_transform(data_to_map)
    n = mx.shape[0]
    r = (1/(n-1))*((mx*my).sum(axis=0))
    return r.astype('float32')

def p_adj_map_from_predictions(preds_pc, data_to_map):
    '''Creates a p map with adjusted p values from predictions'''
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import betai
    mx = StandardScaler().fit_transform(preds_pc)
    my = StandardScaler().fit_transform(data_to_map)
    n = mx.shape[0]
    r = (1/(n-1))*((mx*my).sum(axis=0))
    df = n-2
    t_squared = r*r * (df / ((1.0 - r) * (1.0 + r)))
    prob = betai(0.5*df, 0.5, df / (df+t_squared))
    return fdrcorrection0(prob)

def p_map_from_predictions(preds_pc, data_to_map):
    '''Creates a p map from predictions'''
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import betai
    mx = StandardScaler().fit_transform(preds_pc)
    my = StandardScaler().fit_transform(data_to_map)
    n = mx.shape[0]
    r = (1/(n-1))*((mx*my).sum(axis=0))
    df = n-2
    t_squared = r*r * (df / ((1.0 - r) * (1.0 + r)))
    prob = betai(0.5*df, 0.5, df / (df+t_squared))
    return prob

metric_functions = {'p' : p_map_from_scores,
                    'p_adj' : p_adj_map_from_scores}


def map_from_part(preds_pc, data, threshold=0.0, metric='r2'):
    '''Creates a statistical map from part of the data'''
    map_function = metric_functions[metric]
    data_to_map = data.get_fmri(threshold)
    if metric == 'p':
        scores_pc = np.ones_like(data.scores)
    else:
        scores_pc = np.zeros_like(data.scores)
    scores_pc[data.get_best_voxels(threshold)] = map_function(
        preds_pc, data_to_map)
    return scores_pc

def load_data_ko(subj, model='logMFS_ds', metric='p_adj'):
    '''Loads data and returns them as a namedtuple'''
    scores = np.vstack([joblib.load('/storage/workspace/mboos/scores/'+\
                                    '{}_subj_{}_split_{}.pkl'.format(model, subj, split))
                        for split in xrange(10)])
    scores = np.mean(scores, axis=1)
    return Encoding(scores, metric=metric)

def load_data(subj, model='logBSC_H200', metric='r2'):
    '''Loads data and returns them as an Encoding instance'''
    fmri_data = np.hstack(
            [joblib.load(('/home/data/scratch/mboos/prepro/'
                          'fmri_subj_{}_split_{}.pkl').format(subj, i))
                for i in xrange(10)]).astype('float32')

    model_preds = joblib.load(spenc_dir + ('MaThe/predictions/'
                             '{}_subj_{}_all.pkl').format(model, subj))
    if metric in ('p', 'p_adj'):
        scores_model = metric_functions['r'](model_preds, fmri_data)
    else:
        scores_model = metric_functions[metric](model_preds, fmri_data)

    return Encoding(fmri_data, model_preds,
                    scores_model, metric=metric)


def reconstruct_all_component(filtered_data, pca, segments, component=0):
    '''Reconstructs predictions from one component common to all participants 
    and returns the predictions per segment in a list'''
    preds_pc = np.zeros_like(filtered_data)
    preds_pc[:, component] = filtered_data[:, component]
    preds_pc = pca.inverse_transform(preds_pc)
    preds_pc = [preds_pc[:, segments[i]:segments[i+1]]
                for i in xrange(len(segments)-1)]
    return preds_pc.astype('float32')

def reconstruct_component(filtered_data, pca, component=0):
    '''Reconstructs predictions from only one component'''
    filt_pc = np.zeros_like(filtered_data)
    filt_pc[:, component] = filtered_data[:, component]
    try:
        preds_pc = pca.inverse_transform(filt_pc)
    except AttributeError:
        preds_pc = filt_pc.dot(pca.components_)
    return preds_pc.astype('float32')

def plot_subj_ko(subj, scores, threshold=0.01, coords=None):
    '''plots subject scoremap using nilearn and returns display object'''
    subj_mask = './masks/template_mask_thick.nii.gz'
    background_img = os.path.join(DATA_DIR, 'templates', 'grpbold7Tp1', 'brain.nii.gz')
    scores = scores.copy()
    scores[scores<threshold] = 0.0
    unmasked = unmask(scores, subj_mask)
    unmasked = threshold_img(unmasked, 0.001)
    display = plot_stat_map(
                    unmasked, cut_coords=coords, bg_img=background_img,
                    symmetric_cbar=False,
                    title='metric per voxel', dim=-1, aspect=1.25,
                    threshold=0.001, draw_cross=False)
    fig = plt.gcf()
    fig.set_size_inches(12, 4)
    return display

def plot_subj(subj, scores, threshold=0.01, coords=None):
    '''plots subject scoremap using nilearn and returns display object'''
    subj_mask = spenc_dir+'temporal_lobe_mask_brain_subj{0:02}bold.nii.gz'.format(subj)
    background_img = '/home/data/psyinf/forrest_gump/anondata/sub{0:03}/'.format(subj)+\
            'templates/bold7Tp1/brain.nii.gz'
    scores = scores.copy()
    scores[scores<threshold] = 0.0
    unmasked = unmask(scores, subj_mask)
    unmasked = threshold_img(unmasked, 0.001)
    display = plot_stat_map(
                    unmasked, cut_coords=coords, bg_img=background_img,
                    symmetric_cbar=False,
                    title='metric per voxel', dim=-1, aspect=1.25,
                    threshold=0.001, draw_cross=False)
    fig = plt.gcf()
    fig.set_size_inches(12, 4)
    return display

def save_map(subj, scores, threshold=0.0, model='logBSC_H200'):
    '''Saves brainmap as nifti file'''
    subj_mask = spenc_dir+'temporal_lobe_mask_brain'+\
            '_subj{0:02}bold.nii.gz'.format(subj)
    if threshold is not None:
        scores[scores<threshold] = 0
    unmasked = unmask(scores, subj_mask)
    unmasked.to_filename(
            spenc_dir+'MaThe/maps/model_{}_subj_'.format(model)+\
                '{}_map.nii.gz'.format(subj))

def who_speaks(idx, joint_speech):
    '''returns the speaker identity'''
    if idx.size == 0:
        return 'none'
    elif 'person' not in joint_speech[idx[0]]['parsed']:
        return 'NARRATOR'
    return joint_speech[idx[0]]['parsed']['person']

def index_to_dialog(index):
    index *= 2
    timeframe = 8 + np.array((-8, -2)) + index
    return np.where(np.logical_not(np.logical_or(timeframe[0]>speech_arr[:, 1], timeframe[1]<speech_arr[:, 0])))

def compute_speech_overlap(index):
    '''Computes how much speech the sample contains in s'''
    index *= 2
    timeframe = 8 + np.array((-8, -2)) + index
    overlap_lb = np.max(np.hstack([np.repeat(timeframe[0],
        speech_arr.shape[0])[:, np.newaxis], speech_arr[:, 0][:, np.newaxis]]), axis=1)
    overlap_ub = np.min(np.hstack([np.repeat(timeframe[1],
        speech_arr.shape[0])[:, np.newaxis], speech_arr[:, 1][:, np.newaxis]]), axis=1)
    overlap = (overlap_ub - overlap_lb) > 0
    if not np.any(overlap):
        return 0
    else:
        return (np.max(overlap_ub[overlap]) - np.min(overlap_lb[overlap]))

def transform_by_pca(data, threshold, pca):
    '''Reduces Encoding data using pca
    returns filtered data and fit pca object'''
    filtered_data = pca.fit_transform(data.get_predictions(threshold)).astype('float32')
    return (filtered_data, pca)

def pca_cv(predictions, ref_pcs, fmri_data, pca, n_folds=8):
    '''Reduces predictions to dim of ref_pcs using pca
    separately for each fold'''
    from sklearn.cross_validation import KFold
    cv = KFold(ref_pcs.shape[0], n_folds=n_folds)
    pc_list = []
    for train, test in cv:
        pca.fit(predictions[train])
        tmp_pcs = pca.transform(predictions[test])
        fmri_pcs = pca.transform(fmri_data[test])
        rev_flags = np.eye(N=fmri_pcs.shape[1])
        np.fill_diagonal(rev_flags, np.array([1 if np.corrcoef(tmp_pc, ref_pcs[test, i])[0, 1] >= 0
            else -1
            for i, tmp_pc in enumerate(tmp_pcs.T)]))

        pc_list.append(fmri_pcs.dot(rev_flags))
    return np.vstack(pc_list)

def group_pca_sign_flip(mat, n_pb):
    '''aligns the signs in mat [ obs*n_pb x components] for n_pb'''
    flip_hist = []
    seg_len = mat.shape[0] / n_pb
    for comp in xrange(mat.shape[1]):
        which_flags = np.array([np.corrcoef(mat[:seg_len, comp], part)[0,1] < 0
                                for part in np.reshape(mat[:, comp], (n_pb, -1))])
        flip_hist.append(which_flags)
        rev_flags = np.eye(N=n_pb)
        np.fill_diagonal(rev_flags, [-1 if wh else 1 for wh in which_flags])
        mat[:, comp] = np.reshape(mat[:, comp], (n_pb, -1)).T.dot(rev_flags).T.flatten()
    return (mat, flip_hist)


def reg_df(measures, fmri_pcs):
    '''Returns a dataframe of correlations for PCs'''
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.cross_validation import cross_val_predict
    meslen = len(measures.keys())
    corrs = np.zeros((16, meslen))
    for pb in xrange(16):
        for i, (key, meas) in enumerate(measures.items()):
            corrs[pb, i] = np.corrcoef(cross_val_predict(LinearRegression(),
                                      fmri_pcs[pb], meas), meas)[0, 1]

    corrs_df = pd.DataFrame(data={'Correlation': corrs.flatten(),
                            'Participant':np.repeat(np.arange(1,17), meslen) ,
                            'Component':np.tile(measures.keys(),16)})
    return corrs_df

def subj_flip(pred_pc, data_pc):
    '''Flips signs in pred_pc so it aligns with data_pc'''
    flip_pcs = np.where([np.corrcoef(pred_pc[:, i], data_pc[:,i])[0,1]<0 for i in xrange(data_pc.shape[1])])[0]
    for flip in flip_pcs:
        pred_pc[:, flip] *= -1
    return pred_pc


def correlation_1d_df(measure, fmri_pcs):
    '''Returns a dataframe of correlations for PCs'''
    import pandas as pd
    comps = fmri_pcs.shape[-1]
    corrs = np.zeros((16, comps))
    for pb in xrange(16):
        for comp in xrange(comps):
            corrs[pb, comp] = np.corrcoef(measure, fmri_pcs[pb, :, comp])[0, 1]

    corrs_df = pd.DataFrame(data={'Correlation': corrs.flatten(), 'Participant':np.repeat(np.arange(1,17), comps) ,'Component':np.tile(np.arange(1,comps+1),16)})
    return corrs_df

def correlation_df(preds, fmri_pcs):
    '''Returns a dataframe of correlations for PCs'''
    import pandas as pd
    comps = fmri_pcs.shape[-1]
    corrs = np.zeros((16, comps))
    for pb in xrange(16):
        for comp in xrange(comps):
            corrs[pb, comp] = np.abs(np.corrcoef(preds[pb,:,comp], fmri_pcs[pb, :, comp])[0, 1])

    corrs_df = pd.DataFrame(data={'Correlation': corrs.flatten(), 'Participant':np.repeat(np.arange(1,17), comps) ,'Component':np.tile(np.arange(1,comps+1),16)})
    return corrs_df

def pca_cv_tmp(pred_pcs, predictions, pca, n_folds=8):
    '''Reduces predictions to dim of ref_pcs using pca
    separately for each fold'''
    from sklearn.cross_validation import KFold
    cv = KFold(pred_pcs.shape[0], n_folds=n_folds)
    inv_list = []
    for train, test in cv:
        pca.fit(predictions[train])
        tmp_pcs = pca.transform(predictions[test])
        rev_flags = np.eye(N=tmp_pcs.shape[1])
        np.fill_diagonal(rev_flags, np.array([1 if np.corrcoef(tmp_pc, pred_pcs[test, i])[0, 1] >= 0
            else -1
            for i, tmp_pc in enumerate(tmp_pcs.T)]))

        inv_list.append(pred_pcs[test].dot(rev_flags))
    return np.vstack(inv_list)

def encoding_decoding(score_func, predictions, y):
    '''For scoring function returns the decoding accuracy for X_val,y_val'''
    probabilities = [score_func(np.roll(predictions, -i, axis=0), y)
                     for i in xrange(predictions.shape[0])]
    probabilities = np.vstack([np.roll(row, i)
                               for i, row in enumerate(probabilities)])
    return probabilities

def pca_enc_score(pca_predy, pca_y, pca_cov):
    samples = pca_predy.shape[0]
    if samples != pca_y.shape[0]:
        raise RuntimeError('X and y need to have the same number of samples')
    log_likelihood = np.empty((samples,))
    for i in xrange(samples):
        log_likelihood[i] = log_multivariate_normal_density(
                pca_y[i][None,:], pca_predy[i][None,:],
                pca_cov[None,:,:], covariance_type='full')

    return log_likelihood


if __name__=='__main__':

    if len(sys.argv) > 1:
        subj = int(sys.argv[1])
    else:
        subj = 18

    subj_preprocessed_path = os.path.join(spenc_dir, 'PreProcessed',
                                          'FG_subj%dpp.gzipped.hdf5' % subj)
    with open('DialogData/german_dialog_20150211.json') as fh:
        dialog = load(fh)
    with open('DialogData/german_audio_description_20150211.json') as fh:
        description = load(fh)

    dialog_SE = [(anno['begin'],anno['end']) for anno in dialog['annotations']]
    description_SE = [(anno['begin'],anno['end']) for anno in description['annotations']]
    speech_SE = dialog_SE + description_SE

    joint_speech = np.concatenate([np.array(dialog['annotations']), np.array(description['annotations'])])
    joint_speech = joint_speech[np.argsort(np.array(speech_SE)[:,0])]


    speech_arr = np.array(speech_SE)
    speech_arr = speech_arr[np.argsort(speech_arr[:,0]),:]

    #MFS stepsize is 10ms, speech begin/end is in ms, so we divide by 10
    speech_arr = speech_arr / 1000

    duration = np.array([902,882,876,976,924,878,1084,676])

    mfs_ft = joblib.load('MaThe/prepro/logMFS_stimuli.pkl')

