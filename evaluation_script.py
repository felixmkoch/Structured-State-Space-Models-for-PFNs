from tabpfn.scripts.tabular_evaluation import evaluate
from tabpfn.scripts import tabular_metrics
import os
from tabpfn.datasets import load_openml_list
from evaluation_helper import EvalHelper
from tabpfn.scripts.mamba_prediction_interface import load_model_workflow as mamba_load_model_workflow
from tabpfn.scripts.transformer_prediction_interface import load_model_workflow as transformer_load_model_workflow
from tabpfn.scripts.tabular_baselines import *

import pandas as pd

#EVALUATION_TYPE = "openmlcc18_large"
EVALUATION_TYPE = "openmlcc18"

EVALUATION_METHODS = ["transformer", "mamba"]

METRIC_USED = tabular_metrics.auc_metric

RESULT_CSV_SAVE_DIR = os.path.join("result_csvs", "cc18_small_result.csv")

#MAMBA_MODEL_NAME = "tabpfn/models_diff/mamba_test_model.cpkt"
MAMBA_MODEL_NAME = "tabpfn/models_diff/mamba_150e.cpkt"
TRANSFORMER_MODEL_NAME = "tabpfn/models_diff/tabpfn_transformer_model.cpkt"

def do_evaluation(eval_list):

    result_dict = {}

    # Set up the evaluation Helper class
    eval_helper = EvalHelper()

    #
    # MAMBA EVALUATION
    #
    if "mamba" in eval_list:
        # Load Mamba Model (Yes this is a bit scuffed).
        m_model, mamba_config, results_file = mamba_load_model_workflow(2, -1, add_name="", base_path="", device="cuda",eval_addition='', 
                                                    only_inference=True, model_path_custom=MAMBA_MODEL_NAME)

        # That's the real mamba model here.
        mamba_model = m_model[2]

        # Key is the dataset id (did) and value the mean error on it.
        result_dict["mamba"] = eval_helper.do_evaluation_custom(mamba_model, bptt=mamba_config["bptt"], eval_positions=mamba_config["eval_positions"], metric=METRIC_USED, device="cuda", method_name="mamba",
                                        evaluation_type=EVALUATION_TYPE)

    #
    # TRANSFORMER EVALUATION
    #
    if "transformer" in eval_list:
        # Load Transformer Model (Yes this is a bit scuffed).
        t_model, transformer_config, results_file = transformer_load_model_workflow(2, -1, add_name="", base_path="", device="cuda",eval_addition='', 
                                                    only_inference=True, model_path_custom=TRANSFORMER_MODEL_NAME)

        # That's the real transformer model here.
        transformer_model = t_model[2]

        # Key is the dataset id (did) and value the mean error on it.
        result_dict["transformer"] = eval_helper.do_evaluation_custom(transformer_model, bptt=transformer_config["bptt"], eval_positions=transformer_config["eval_positions"], metric=METRIC_USED, device="cuda", method_name="transformer",
                                        evaluation_type=EVALUATION_TYPE)


    #
    # XGBoost Evaluation
    #
    if "xgboost" in eval_list:
        # That's the xgboost metric function serving as a model
        xgboost_model = xgb_metric

        # Key is the dataset id (did) and value the mean error on it. We use mamba model params as bptt and eval_positions
        # NOTE: Current max time is 300, aka 5 minutes. Need to change this maybe.
        result_dict["xgboost"] = eval_helper.do_evaluation_custom(xgboost_model, bptt=mamba_config["bptt"], eval_positions=mamba_config["eval_positions"], metric=METRIC_USED, device="cuda", method_name="xgb",
                                        evaluation_type=EVALUATION_TYPE)


    #
    # KNN Evaluation
    #
    if "knn" in eval_list:
        # That's the k-nearest-neighbor metric function serving as a model
        knn_model = knn_metric

        # Key is the dataset id (did) and value the mean error on it. We use mamba model params as bptt and eval_positions
        result_dict["knn"] = eval_helper.do_evaluation_custom(knn_model, bptt=mamba_config["bptt"], eval_positions=mamba_config["eval_positions"], metric=METRIC_USED, device="cuda", method_name="knn",
                                        evaluation_type=EVALUATION_TYPE)
    

    return result_dict




'''
# results into csv
result_arr = []
header = ["did", "transformer", "mamba", "knn"] # , "xgb"]

for key in eval_helper_mamba_results.keys(): result_arr.append(
                                    [
                                        key, 
                                        eval_helper_transformer_results[key], 
                                        eval_helper_mamba_results[key],
                                        #eval_helper_xgb_results[key]
                                        eval_helper_knn_results[key]
                                     ]
)

df_out = pd.DataFrame(result_arr, columns=header)

df_out.to_csv(RESULT_CSV_SAVE_DIR)
'''

if __name__ == "__main__":

    result_dict = do_evaluation(EVALUATION_METHODS)

    header = ["did"] + EVALUATION_METHODS

    result_arr = []

    keys = list(result_dict[list(result_dict.keys())[0]].keys())

    for key in keys:
        to_add = [key]

        for method in EVALUATION_METHODS: 
            to_add.append(result_dict[method][key])

        result_arr.append(to_add)

    df_out = pd.DataFrame(result_arr, columns=header)

    df_out.to_csv(RESULT_CSV_SAVE_DIR)

    print("worked")