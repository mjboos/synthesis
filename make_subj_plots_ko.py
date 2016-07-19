import featurespace as fsf
import matplotlib.pyplot as plt
from fg_constants import *

for subj in xrange(18):
    try:
        data = fsf.load_data_ko(subj)
    except ValueError:
        continue
    fsf.plot_subj_ko(subj, data.scores, data.threshold)
    plt.savefig('/storage/workspace/mboos/plots/common_subj_{}.png'.format(subj))
    plt.clf()

