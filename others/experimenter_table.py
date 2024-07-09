from py_experimenter.experimenter import PyExperimenter
from tabpfn.scripts.tabular_evaluation import evaluate
from tabpfn.datasets import load_openml_list, valid_dids_classification, test_dids_classification, open_cc_dids
from tabpfn.scripts.tabular_metrics import calculate_score, make_ranks_and_wins_table, make_metric_matrix
from tabpfn.scripts.tabular_baselines import *
from tabpfn.scripts import tabular_metrics
from tabpfn.scripts import tabular_baselines

import os
#os.chdir("tabpfn")

#------------------------------------------------------------------------------------------------
#                                       PARAMETERS
#------------------------------------------------------------------------------------------------

cc_test_datasets_multiclass, cc_test_datasets_multiclass_df = \
    load_openml_list(
        open_cc_dids, 
        multiclass=True, 
        shuffled=True, 
        filter_for_nan=False, 
        max_samples = 10000, 
        num_feats=100, 
        return_capped=True
        )

eval_positions = [1000]
max_features = 100
bptt = 2000
base_path = os.path.join('.')
overwrite=True
metric_used = tabular_metrics.auc_metric
task_type = 'multiclass'
suite="cc"

device = 'cuda'

clf_dict= {'gp': gp_metric, 
           'knn': knn_metric, 
           'catboost': catboost_metric, 
           'xgb': xgb_metric, 
           'transformer': transformer_metric, 
           'logistic': logistic_metric, 
           'autosklearn': autosklearn_metric, 
           'autosklearn2': autosklearn2_metric, 
           'autogluon': autogluon_metric,
           'mamba': mamba_metric
           }

metric_renamer = {'roc': 'ROC AUC', 
                  'cross_entropy': 'Cross entropy', 
                  'rank_roc': 'Mean ROC AUC Rank', 
                  'rank_cross_entropy': 'Mean Cross entropy Rank', 
                  'wins_roc': 'Mean ROC AUC Wins', 
                  'wins_cross_entropy': 'Mean Cross entropy Wins', 
                  'time': 'actual time taken'
                  }

max_times_renamer = {0.5: "0.5s", 1: "1s", 5: "5s", 15: "15s", 30: "30s", 60: "1min", 300: "5min", 900: "15min", 3600: "1h", 14400: "4h"}



#------------------------------------------------------------------------------------------------
#                                     END PARAMETERS
#------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------
#                                      OTHER METHODS
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
                          max_time=max_time,
                          method_name=method)

        # RESULT IS ONLY USED ONCE IN THIS FOR LOOP!

        # Result is a dict with keys: ['metric_used', 
        # 'bptt', 
        # 'eval_positions', 
        # 'balance-scale_best_configs_at_1000', 
        # 'balance-scale_outputs_at_1000', 
        # 'balance-scale_ys_at_1000', 
        # 'balance-scale_time_at_1000', ...]

        # The mean of all is the key caled "mean_metric"
    
    return result

#------------------------------------------------------------------------------------------------
#                                    END OTHER METHODS
#------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------
#                                  RUN EXPERIMENT METHOD
#------------------------------------------------------------------------------------------------

def run_experiment(config, result_processor, custom_fields):

    #did = config["did"]                    # Dataset ID from the OpenML dataset to test on.
    selector = config["selector"]           # Dataset selector, default only "test".
    method = config["method"]               # The method used.
    max_time = config["max_time"]           # Max time of the experiment.
    split_number = config["split_number"]   # Split number of the dataset.

    eval_dict = eval_method(
        task_type=task_type,
        method=method,
        dids=None,
        selector=selector,
        eval_positions=eval_positions,
        max_time=max_time,
        metric_used=metric_used,
        split_number=split_number,
        append_metric=True,
        fetch_only=True,
        verbose=False
    )

    res = eval_dict["mean_metric"] # Tensor with just 1 item in it, the result.

    result_processor.process_results({
        "y": res.item()
    })


#------------------------------------------------------------------------------------------------
#                                END RUN EXPERIMENT METHOD
#------------------------------------------------------------------------------------------------



if __name__ == "__main__":

    # PyExperimenter object
    pyexp = PyExperimenter(experiment_configuration_file_path="expsetup_first.cnf")

    max_times = [0.5, 1, 15, 30, 60, 300, 900, 3600]
    methods = ['transformer', 'mamba']
    selectors = ["test"]
    split_numbers = [1, 2, 3, 4, 5]

    combinations = [
       {"max_time": max_time, "method": method, "selector": selector, "split_number": split_number}
       for max_time in max_times
       for method in methods
       for selector in selectors
       for split_number in split_numbers
    ]

    pyexp.fill_table_from_combination(combinations)

    print(pyexp.get_table())

    pyexp.execute(run_experiment, max_experiments=1)

    print("worked")