from fg_constants import *
from preprocessing_synthesis import convert_subject_partest
import os

subj = 'sub001'
subj_dir = os.path.join(PREP_DATA_DIR, 'test', subj)

max_period = 441.0 * TR

period_cut = [max_period, max_period * 0.5, max_period * 0.25, max_period * 0.1]
drift_model =  ['polynomial', 'cosine']

exh_params = []
for peri in period_cut:
    for drift in drift_model:
        kwargs = {'period_cut' : peri,
                  'drift_model': drift}
        exh_params.append(kwargs)

for i, pars in enumerate(exh_params):
    convert_subject_partest(subj, 'model_'+str(i), **pars)
