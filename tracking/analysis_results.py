import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist

trackers = []
dataset_name = 'lasot'
# dataset_name = 'lasot_extension_subset'
# dataset_name = 'tnl2k'
# dataset_name = 'uav'

"""compresstracker"""
trackers.extend(trackerlist(name='compresstracker', parameter_name='compresstracker_vitb_256_4', dataset_name=dataset_name,
                            run_ids=None, display_name='CompressTracker'))

dataset = get_dataset(dataset_name)
print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec', 'prec'))
