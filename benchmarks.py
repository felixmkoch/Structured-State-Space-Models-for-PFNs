#------------------------------------------------------------------------------------------------
#                                        IMPORTS
#------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt

from tabpfn.scripts import tabular_baselines

import seaborn as sns
import numpy as np

from tabpfn.datasets import load_openml_list, valid_dids_classification, test_dids_classification, open_cc_dids
from tabpfn.scripts.tabular_baselines import *
from tabpfn.scripts.tabular_evaluation import evaluate
from tabpfn.scripts.tabular_metrics import calculate_score, make_ranks_and_wins_table, make_metric_matrix
from tabpfn.scripts import tabular_metrics

from tabpfn.notebook_utils import *

import os

os.chdir("tabpfn")

#------------------------------------------------------------------------------------------------
#                                       END IMPORTS
#------------------------------------------------------------------------------------------------

cc_test_datasets_multiclass, cc_test_datasets_multiclass_df = \
    load_openml_list(open_cc_dids, multiclass=True, shuffled=True, filter_for_nan=False, max_samples = 10000, num_feats=100, return_capped=True)

#------------------------------------------------------------------------------------------------
#                                        FUNCTIONS
#------------------------------------------------------------------------------------------------



def get_datasets(selector, task_type, suite='openml'):
    
    # Note - logic of this function was wrong in the notebook provided by the TabPFN authors.

    return cc_test_datasets_multiclass


def eval_method(task_type, 
                method, 
                dids, 
                selector, 
                eval_positions, 
                max_time, 
                metric_used, 
                split_number, 
                append_metric=True, 
                fetch_only=False, 
                verbose=False):
    
    dids = dids if type(dids) is list else [dids]
    
    for did in dids:

        ds = get_datasets(selector, task_type, suite=suite)

        ds = ds if did is None else ds[did:did+1]

        clf = clf_dict[method]

        time_string = '_time_'+str(max_time) if max_time else ''
        metric_used_string = '_'+tabular_baselines.get_scoring_string(metric_used, usage='') if append_metric else ''

        result = evaluate(datasets=ds, 
                          model=clf, 
                          method=method+time_string+metric_used_string, 
                          bptt=bptt, base_path=base_path, 
                          eval_positions=eval_positions, 
                          device=device, 
                          max_splits=1, 
                          overwrite=overwrite, 
                          save=True, 
                          metric_used=metric_used, 
                          path_interfix=task_type, 
                          fetch_only=fetch_only, 
                          split_number=split_number, 
                          verbose=verbose, 
                          max_time=max_time)
    
    return result


#------------------------------------------------------------------------------------------------
#                                      END FUNCTIONS
#------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------
#                                       PARAMETERS
#------------------------------------------------------------------------------------------------

eval_positions = [1000]
max_features = 100
bptt = 2000
selector = 'test'
base_path = os.path.join('.')
overwrite=False
max_times = [15, 20]
metric_used = tabular_metrics.auc_metric
methods = ['transformer', 'logistic']
task_type = 'multiclass'

device = 'cuda'

clf_dict= {'gp': gp_metric, 
           'knn': knn_metric, 
           'catboost': catboost_metric, 
           'xgb': xgb_metric, 
           'transformer': transformer_metric, 
           'logistic': logistic_metric, 
           'autosklearn': autosklearn_metric, 
           'autosklearn2': autosklearn2_metric, 
           'autogluon': autogluon_metric}


# Redo all the calucation stuff
REDO = True

#------------------------------------------------------------------------------------------------
#                                     END PARAMETERS
#------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------
#                                        DO STUFF
#------------------------------------------------------------------------------------------------



suite = 'cc'

if REDO: 
    test_datasets = get_datasets('test',task_type, suite=suite)
    overwrite=True
    jobs = [
        eval_method(task_type, m, did, selector, eval_positions, max_time, metric_used, split_number)
        for did in range(0, len(test_datasets))
        for selector in ['test']
        for m in methods
        for max_time in max_times
        for split_number in [2, 4]         #for split_number in [1, 2, 3, 4, 5]
    ]

#------------------------------------------------------------------------------------------------
#                                      END DO STUFF
#------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------
#                                       COMPARISON
#------------------------------------------------------------------------------------------------





#------------------------------------------------------------------------------------------------
#                                     END COMPARISON
#------------------------------------------------------------------------------------------------



print("worked")
