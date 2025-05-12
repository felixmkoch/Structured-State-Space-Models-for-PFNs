from tabpfn.scripts.tabular_evaluation import evaluate
from tabpfn.scripts import tabular_metrics
import os
from tabpfn.datasets import load_openml_list
from evaluation_helper import EvalHelper
from tabpfn.scripts.mamba_prediction_interface import load_model_workflow as mamba_load_model_workflow
from tabpfn.scripts.transformer_prediction_interface import load_model_workflow as transformer_load_model_workflow
from tabpfn.scripts.hydra_prediction_interface import load_model_workflow as hydra_load_model_workflow
from tabpfn.scripts.tabular_baselines import *
from scipy import stats

import pandas as pd

#EVALUATION_TYPE = "openmlcc18_large"
EVALUATION_TYPE = "openmlcc18"

openml_cc18_small_dids = [11, 14, 15, 16, 18, 22, 23, 29, 31, 37, 50, 54, 188, 458, 469, 1049, 1050, 1063, 1068, 1462, 1464, 1480, 1494, 1510, 6332, 23381, 40966, 40975, 40982, 40994]

#
# Here: True means to keep them, false to omit
#
EVALUATION_TYPE_FILTERS = {
    "categorical": True,
    "nans": True,
    "multiclass": True
}

# Only one method here
EVALUATION_METHODS = ["transformer", "mamba", "hydra"]

METRICS_USED = [tabular_metrics.accuracy_metric, tabular_metrics.auc_metric]
METRIC_TO_STR = {
    tabular_metrics.accuracy_metric: "Acc",
    tabular_metrics.auc_metric: "AUC"
}


RESULT_CSV_SAVE_DIR = os.path.join("result_csvs", "table")
RESULT_CSV_PREFIX = ""

MAMBA_MODEL_NAME = "tabpfn/models_diff/mamba_small.cpkt"
TRANSFORMER_MODEL_NAME = "tabpfn/models_diff/tabpfn_transformer_model.cpkt"
HYDRA_MODEL_NAME = "tabpfn/models_diff/hydra_small.cpkt"

NUM_SPLITS = 16

SPLIT_NUMBERS = [i + 1 for i in range(NUM_SPLITS)]

bptt_here = 1024
# Relevant for the AutoML approaches
max_time = 3600

JRT_PROMPT = False
SINGLE_EVAL_PROMPT = False
# Default 1, number of permutations that will be averaged.
PERMUTATION_BAGGING = 1
# Default 0. Number of bootstrap samples to be bagged.
SAMPLE_BAGGING = 0

device = "cuda:0"

eval_helper = EvalHelper()


def do_evaluation(eval_list, metric):

    result_dict = {}

    #
    # MAMBA EVALUATION
    #
    if "mamba" in eval_list:
        # Load Mamba Model (Yes this is a bit scuffed).
        m_model, mamba_config, results_file = mamba_load_model_workflow(2, -1, add_name="", base_path="", device=device,eval_addition='', 
                                                    only_inference=True, model_path_custom=MAMBA_MODEL_NAME)

        # That's the real mamba model here.
        mamba_model = m_model[2]

        # Key is the dataset id (did) and value the mean error on it.
        result_dict["mamba"] = eval_helper.do_evaluation_custom(mamba_model, bptt=bptt_here, eval_positions=mamba_config["eval_positions"], metric=metric, device=device, method_name="mamba",
                                        evaluation_type=EVALUATION_TYPE, split_numbers=SPLIT_NUMBERS, jrt_prompt=JRT_PROMPT, permutation_bagging=PERMUTATION_BAGGING, sample_bagging=SAMPLE_BAGGING, eval_filters=EVALUATION_TYPE_FILTERS)

    #
    # TRANSFORMER EVALUATION
    #
    if "transformer" in eval_list:
        # Load Transformer Model (Yes this is a bit scuffed).
        t_model, transformer_config, results_file = transformer_load_model_workflow(2, -1, add_name="", base_path="", device=device,eval_addition='', 
                                                    only_inference=True, model_path_custom=TRANSFORMER_MODEL_NAME)

        # That's the real transformer model here.
        transformer_model = t_model[2]

        # Key is the dataset id (did) and value the mean error on it.
        result_dict["transformer"] = eval_helper.do_evaluation_custom(transformer_model, bptt=bptt_here, eval_positions=transformer_config["eval_positions"], metric=metric, device=device, method_name="transformer",
                                        evaluation_type=EVALUATION_TYPE, split_numbers=SPLIT_NUMBERS, jrt_prompt=JRT_PROMPT, eval_filters=EVALUATION_TYPE_FILTERS)


    #
    # HYDRA EVALUATION
    #
    if "hydra" in eval_list:
        # Load Transformer Model (Yes this is a bit scuffed).
        h_model, hydra_config, results_file = hydra_load_model_workflow(2, -1, add_name="", base_path="", device=device,eval_addition='', 
                                                    only_inference=True, model_path_custom=HYDRA_MODEL_NAME)

        # That's the real transformer model here.
        hydra_model = h_model[2]

        # Key is the dataset id (did) and value the mean error on it.
        result_dict["hydra"] = eval_helper.do_evaluation_custom(hydra_model, bptt=bptt_here, eval_positions=hydra_config["eval_positions"], metric=metric, device=device, method_name="hydra",
                                        evaluation_type=EVALUATION_TYPE, split_numbers=SPLIT_NUMBERS, jrt_prompt=JRT_PROMPT, single_evaluation_prompt=SINGLE_EVAL_PROMPT, permutation_bagging=PERMUTATION_BAGGING, sample_bagging=SAMPLE_BAGGING, eval_filters=EVALUATION_TYPE_FILTERS)


    #
    # XGBoost Evaluation
    # Note: This doesn't work anymore in Python 3.10 ...
    #
    if "xgboost" in eval_list:
        # That's the xgboost metric function serving as a model
        xgboost_model = xgb_metric

        # Key is the dataset id (did) and value the mean error on it. We use mamba model params as bptt and eval_positions
        # NOTE: Current max time is 300, aka 5 minutes. Need to change this maybe.
        result_dict["xgboost"] = eval_helper.do_evaluation_custom(xgboost_model, bptt=bptt_here, eval_positions=[999999], metric=metric, device=device, method_name="xgb",
                                        evaluation_type=EVALUATION_TYPE, max_time=max_time, eval_filters=EVALUATION_TYPE_FILTERS)


    #
    # KNN Evaluation
    #
    if "knn" in eval_list:
        # That's the k-nearest-neighbor metric function serving as a model
        knn_model = knn_metric

        # Key is the dataset id (did) and value the mean error on it. We use mamba model params as bptt and eval_positions
        result_dict["knn"] = eval_helper.do_evaluation_custom(knn_model, bptt=bptt_here, eval_positions=mamba_config["eval_positions"], metric=metric, device=device, method_name="knn",
                                        evaluation_type=EVALUATION_TYPE, max_time=max_time, eval_filters=EVALUATION_TYPE_FILTERS)
    

    return result_dict



if __name__ == "__main__":

    header = ["did"] + [str(i) for i in SPLIT_NUMBERS]

    for metric_used in METRICS_USED:

        result_dict = do_evaluation(EVALUATION_METHODS, metric_used)

        metric_str = METRIC_TO_STR[metric_used]

        # Calc Mean and Confidence Intervals

        for evaluation_method in EVALUATION_METHODS:

            res_dict_method = result_dict[evaluation_method]

            csv_name = f"{RESULT_CSV_PREFIX}{evaluation_method}_{metric_used}.csv"

            output_path = os.path.join(RESULT_CSV_SAVE_DIR, csv_name)

            to_print = []

            for did in openml_cc18_small_dids:

                splitted_vals = res_dict_method[did]

                to_print.append([did] + splitted_vals)

            df_out = pd.DataFrame(to_print, columns=header)

            df_out.to_csv(output_path, index=False)
