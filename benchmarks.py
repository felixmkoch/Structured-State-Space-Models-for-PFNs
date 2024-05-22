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
import json

os.chdir("tabpfn")

import tikzplotlib

#------------------------------------------------------------------------------------------------
#                                       END IMPORTS
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

# cc_test_dataset_muticlass_df: [did, Num_Features, Num_Classes...] in a dataframe
# cc_test_datasets_multiclass: list[list[object]]; List1: List of Tasks; List2: name, tensor, tensor, ...;


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
split_numbers = [2, 4]
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
           'autogluon': autogluon_metric
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

# Redo all the calucation stuff
REDO = False

#------------------------------------------------------------------------------------------------
#                                     END PARAMETERS
#------------------------------------------------------------------------------------------------

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

        # RESULT IS ONLY USED ONCE IN THIS FOR LOOP!

        # Result is a dict with keys: ['metric_used', 
        # 'bptt', 
        # 'eval_positions', 
        # 'balance-scale_best_configs_at_1000', 
        # 'balance-scale_outputs_at_1000', 
        # 'balance-scale_ys_at_1000', 
        # 'balance-scale_time_at_1000', ...]
    
    return result


def generate_ranks_and_wins_table(global_results_filtered, metric_key, max_time, split_number, time_matrix):

    global_results_filtered_split = {**global_results_filtered}
    
    for k in global_results_filtered_split.keys():
        print(f"Key: {k}")
        print('_time_'+str(max_time)+tabular_baselines.get_scoring_string(metric_used, usage='')+'_split_'+str(split_number))

    global_results_filtered_split = {k: global_results_filtered_split[k] 
                                     for k in global_results_filtered_split.keys() 
                                     if '_time_'+str(max_time)+tabular_baselines.get_scoring_string(metric_used, usage='')+'_split_'+str(split_number) in k 
                                     or 'transformer_split_'+str(split_number) in k}


    matrix, matrix_stds, matrix_splits = make_metric_matrix(global_results_filtered_split, methods, pos, metric_key, test_datasets)
    for method in methods:
        if time_matrix[method] > max_time * 2:
            matrix[method] = np.nan
        # = np.nan

    if metric_key == 'cross_entropy':
        matrix = -(matrix.fillna(-100))
    else:
        matrix = matrix.fillna(-1)
    return make_ranks_and_wins_table(matrix.copy())



def make_tabular_results_plot(metric_key, exclude, max_times, df_, grouping=True):
    f, ax = plt.subplots(figsize=(7, 7))
    #ax.set(xscale="log")
    
    df_.loc[:, 'time_log'] = np.log10(df_.time)
    df_.loc[:, 'real_time_log'] = np.log10(df_.real_time)
    time_column = 'time_log' if grouping else 'real_time_log'

    sns.set_palette("tab10")
    for method in methods:
        if method in exclude or method=='transformer':
            continue
        df_method = df_[df_.method==method].copy()
        ax = sns.lineplot(time_column, 'metric'+metric_key, data=df_method, marker='o', label=method, ax=ax)
    #sns.scatterplot(data=df_, x='time', y='metric', hue='method', ax=ax, style='method') #
    df_trans = df_[df_.method=='transformer']
    if time_column == 'real_time_log':
        # Removing dots for line for transformers
        df_trans = df_trans[np.logical_or(df_trans.real_time == df_trans.real_time.min(), df_trans.real_time == df_trans.real_time.max())]
        df_trans.loc[:, 'metric'+metric_key] = df_trans['metric'+metric_key].mean()
        df_trans.loc[:, time_column] = np.log(1) # Hacky code to get the right time from our measurements
    ax = sns.lineplot(time_column, 'metric'+metric_key, data=df_trans, linestyle='--', marker='o', ci="sd", ax=ax)
    
    #ax = sns.scatterplot(data = df_trans, x=time_column, y='metric'+metric_key, s=800, marker='*', color='grey') #
    #ax = plt.scatter(df_trans[time_column], df_trans['metric'+metric_key], s=600, marker=['*']) #
    
    if grouping:
        ax.set_xlabel("Time (s, requested, not actual)")
    else:
        ax.set_xlabel("Time taken")
    ax.set_ylabel(metric_renamer[metric_key])

    #ax.legend()
    
    times = np.log10(max_times)
    ax.set_xticks(times)
    ax.set_xticklabels([max_times_renamer[t] for t in max_times])
    
    #ax.legend([],[], frameon=False)
    
    return ax


#------------------------------------------------------------------------------------------------
#                                      END FUNCTIONS
#------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------
#                                        DO STUFF
#------------------------------------------------------------------------------------------------

suite = 'cc'

# Is: list[list[object]]; List1: List of Tasks; List2: name, tensor, tensor, ...;
test_datasets = get_datasets('test', task_type, suite=suite)


if REDO: 
    overwrite=True
    # Each entry in this list contains a result, meaning a dict containing information about the run and different settings on datasets as keys.
    jobs = [
        eval_method(task_type, m, did, selector, eval_positions, max_time, metric_used, split_number)
        for did in range(0, len(test_datasets))
        for selector in ['test']
        for m in methods
        for max_time in max_times
        for split_number in split_numbers         #for split_number in [1, 2, 3, 4, 5]
    ]

    print(type(jobs))

#------------------------------------------------------------------------------------------------
#                                      END DO STUFF
#------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------
#                                       COMPARISON
#------------------------------------------------------------------------------------------------

pos = str(eval_positions[0])

global_results = {}
overwrite=False

# Add by hand, TabPFN did not include that.
baseline_methods = methods

for method in baseline_methods:
    for max_time in max_times:
        #for split_number in range(1,5+1):
        for split_number in split_numbers:
            global_results[
                method+'_time_'+str(max_time)+tabular_baselines.get_scoring_string(metric_used, usage='')+'_split_'+str(split_number)
                ] = eval_method(task_type, 
                                method,  
                                None, 
                                selector, 
                                eval_positions, 
                                fetch_only=True, 
                                verbose=False, 
                                max_time=max_time,
                                metric_used=metric_used, 
                                split_number=split_number
                                )
            

# Global results: ["transformer_time_15roc_auc_split_2", "transformer_time_15roc_auc_split_4", ...]
# Where each dict looks like ['metric_used', 
        # 'bptt', 
        # 'eval_positions', 
        # 'balance-scale_best_configs_at_1000', 
        # 'balance-scale_outputs_at_1000', 
        # 'balance-scale_ys_at_1000', 
        # 'balance-scale_time_at_1000', ...]


''''
path_ = 'prior_tuning_result.pkl'


try:
    output = open(path_, 'rb')
    _, metrics, _, _, _, _ = CustomUnpickler(output).load()
except:
    output = open(path_, 'rb')
    _, metrics, _, _, _ = CustomUnpickler(output).load()
if isinstance(metrics, list):
    for i in range(1, len(metrics[1])+1):
        global_results['transformer_split_'+str(i)] = metrics[2][i-1]

'''

#
# Print result JSON
#


# Verify integrity of results
for bl in set(global_results.keys()):
    if 'split_1' in bl:
        for ds in test_datasets:
            if f'{ds[0]}_ys_at_1000' not in global_results[bl]:
                continue
            match = (global_results[bl][f'{ds[0]}_ys_at_1000'] == global_results['transformer_split_1'][f'{ds[0]}_ys_at_1000']).float().mean()
            if not match:
                raise Exception("Not the same labels used")
            

limit_to = ''
calculate_score(tabular_metrics.auc_metric, 'roc', global_results, test_datasets, eval_positions + [-1], limit_to=limit_to)
calculate_score(tabular_metrics.cross_entropy, 'cross_entropy', global_results, test_datasets, eval_positions + [-1], limit_to=limit_to)
calculate_score(tabular_metrics.accuracy_metric, 'acc', global_results, test_datasets, eval_positions + [-1])
calculate_score(tabular_metrics.time_metric, 'time', global_results, test_datasets, eval_positions + [-1], aggregator='sum', limit_to=limit_to)
calculate_score(tabular_metrics.time_metric, 'time', global_results, test_datasets, eval_positions + [-1], aggregator='mean', limit_to=limit_to)
calculate_score(tabular_metrics.count_metric, 'count', global_results, test_datasets, eval_positions + [-1], aggregator='sum', limit_to=limit_to)



# Here global results remain the same for most parts
# Except all other scored are added in a string-like manner to the dict.

df_ = []
metric_keys = ['roc', 'cross_entropy', 'time']

for max_time in max_times:
    global_results_filtered = {**global_results}
    global_results_filtered = {k: global_results_filtered[k] for k in global_results_filtered.keys() if '_time_'+str(max_time)+tabular_baselines.get_scoring_string(metric_used, usage='')+'_' in k or 'transformer' in k}
    
    time_matrix, _, _ = make_metric_matrix(global_results_filtered, methods, pos, 'time', test_datasets)   # Error here
    time_matrix = time_matrix.mean()
    
    if len(global_results_filtered) == 0:
        continue
    '''
    # Calculate ranks and wins per split
    for metric_key in metric_keys:
        for split_number in split_numbers:
            ranks, wins = generate_ranks_and_wins_table(global_results_filtered, metric_key, max_time, split_number, time_matrix)

            for method in methods:
                method_ = method+'_time_'+str(max_time)+tabular_baselines.get_scoring_string(metric_used, usage='') if method != 'transformer' else method
                global_results[method_+'_split_'+str(split_number)]['mean_rank_'+metric_key+f'_at_{pos}'] = ranks[method]
                global_results[method_+'_split_'+str(split_number)]['mean_wins_'+metric_key+f'_at_{pos}'] = wins[method]
    '''
    #for method in global_results.keys():
    #    global_results[method]['mean_rank_'+metric_key+f'_at_{pos}'] = ranks[]
    '''
    avg_times = {}
    for method_ in methods:
        avg_times[method_] = []
        for split_number in split_numbers:
            if method_ != 'transformer':
                method = method_+'_time_'+str(max_time)+tabular_baselines.get_scoring_string(metric_used, usage='')+'_split_'+str(split_number)
            else:
                method = method_+'_split_'+str(split_number)
            avg_times[method_] += [global_results[method][f'mean_time_at_{pos}']]
    avg_times = pd.DataFrame(avg_times).mean()
    '''
    for metric_key in metric_keys:
        #for ranking in ['', 'rank_', 'wins_']:
        for ranking in ['']:
            for method_ in methods:
                for split_number in split_numbers:
                    method = method_
                    #if method_ != 'transformer':
                    method = method_+'_time_'+str(max_time)+tabular_baselines.get_scoring_string(metric_used, usage='')+'_split_'+str(split_number)
                    #else:
                    #    method = method_+'_split_'+str(split_number)

                    if global_results[method][f'sum_count_at_{pos}'] <= 29:
                        print('Warning not all datasets generated for '+method+' '+ str(global_results[method][f'sum_count_at_{pos}']))
                        
                    time = global_results[method]['mean_time'] if ranking == '' else max_time
                    time = max_time # Todo: This is not the real time
                    df_ += [{'metric'+ranking+metric_key: global_results[method]['mean_'+ranking+metric_key+f'_at_{pos}'], 'time': time, 'method': method_, 'split_number': split_number}]
                    #df_ += [{'metric'+ranking+metric_key: global_results[method]['mean_'+ranking+metric_key+f'_at_{pos}'], 'real_time': avg_times[method_], 'time': time, 'method': method_, 'split_number': split_number}]
                    # For Roc AUC Plots
                    #if 'transformer' in method:
                    #    df_ += [{'metric'+ranking+metric_key: global_results[method]['mean_'+ranking+metric_key+f'_at_{pos}'], 'real_time': avg_times[method_], 'time': time, 'method': method_, 'split_number': split_number}]
                    #    df_ += [{'metric'+ranking+metric_key: global_results[method]['mean_'+ranking+metric_key+f'_at_{pos}'], 'real_time': max(avg_times), 'time': max(max_times), 'method': method_, 'split_number': split_number}]


df_ = pd.DataFrame(df_)

df_absolute = df_.copy()
#df_absolute = df_absolute[np.logical_or(df_.method != 'autogluon', df_.time >= 30)] # Autogluon did not yield any useful results before 30s

#knn_extend = df_absolute[np.logical_and(df_absolute.method=='knn', df_absolute.time == 3600)].copy()
knn_extend = df_absolute.copy() #Added by me
knn_extend['real_time'] = 14400
knn_extend['time'] = 14400
df_absolute = pd.concat([df_absolute, knn_extend], ignore_index=True).reindex()
# df_absolute = df_absolute.append(knn_extend, ignore_index=True).reindex()  --> Deprecated/Removed from pandas

knn_extend = df_absolute[np.logical_and(df_absolute.method=='logistic', df_absolute.time == 3600)].copy()
knn_extend['real_time'] = 14400
knn_extend['time'] = 14400

#df_absolute = df_absolute.append(knn_extend, ignore_index=True).reindex()  --> Deprecated / Removed from pandas
df_absolute = pd.concat([df_absolute, knn_extend], ignore_index=True).reindex()

        
#------------------------------------------------------------------------------------------------
#                                     END COMPARISON
#------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------
#                                          PLOT
#------------------------------------------------------------------------------------------------

exclude=['']
#ax = make_tabular_results_plot('time', exclude=exclude)
ax = make_tabular_results_plot('roc', df_=df_absolute, exclude=exclude, grouping=False, max_times=[1, 5, 30, 60*5, 60*60])
ax.set_ylim([0.84, 0.9])
ax.set_xlim([np.log10(0.7), np.log10(3600)])
ax.legend([],[], frameon=False)

plt.savefig('figure.png')

#tikzplotlib.save(f'roc_over_time.tex', axis_height='5cm', axis_width='6cm', strict=True)


#------------------------------------------------------------------------------------------------
#                                        END PLOT
#------------------------------------------------------------------------------------------------


print("worked")
