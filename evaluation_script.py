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

EVALUATION_METHODS = ["mamba"]

METRIC_USED = tabular_metrics.auc_metric

RESULT_CSV_SAVE_DIR = os.path.join("result_csvs", "test.csv")

MAMBA_MODEL_NAME = "tabpfn/models_diff/mamba_small.cpkt"
TRANSFORMER_MODEL_NAME = "tabpfn/models_diff/tabpfn_transformer_model.cpkt"
HYDRA_MODEL_NAME = "tabpfn/models_diff/test.cpkt"

SPLIT_NUMBERS = [1]

bptt_here = 1000
max_time = 3600
CONFIDENCE_LEVEL = 0.95

JRT_PROMPT = False
SINGLE_EVAL_PROMPT = False
PERMUTATION_BAGGING = 1
SAMPLE_BAGGING = 10

device = "cuda:0"

eval_helper = EvalHelper()

def calc_moe(data):

    sem = stats.sem(data)

    confidence_level = CONFIDENCE_LEVEL
    degrees_of_freedom = len(data) - 1
    t_score = stats.t.ppf((1 + confidence_level) / 2, degrees_of_freedom)

    margin_of_error = t_score * sem

    return margin_of_error


def do_evaluation(eval_list):

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
        result_dict["mamba"] = eval_helper.do_evaluation_custom(mamba_model, bptt=bptt_here, eval_positions=mamba_config["eval_positions"], metric=METRIC_USED, device=device, method_name="mamba",
                                        evaluation_type=EVALUATION_TYPE, split_numbers=SPLIT_NUMBERS, jrt_prompt=JRT_PROMPT, permutation_bagging=PERMUTATION_BAGGING, sample_bagging=SAMPLE_BAGGING)

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
        result_dict["transformer"] = eval_helper.do_evaluation_custom(transformer_model, bptt=bptt_here, eval_positions=transformer_config["eval_positions"], metric=METRIC_USED, device=device, method_name="transformer",
                                        evaluation_type=EVALUATION_TYPE, split_numbers=SPLIT_NUMBERS, jrt_prompt=JRT_PROMPT)


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
        result_dict["hydra"] = eval_helper.do_evaluation_custom(hydra_model, bptt=bptt_here, eval_positions=hydra_config["eval_positions"], metric=METRIC_USED, device=device, method_name="hydra",
                                        evaluation_type=EVALUATION_TYPE, split_numbers=SPLIT_NUMBERS, jrt_prompt=JRT_PROMPT, single_evaluation_prompt=SINGLE_EVAL_PROMPT, permutation_bagging=PERMUTATION_BAGGING, sample_bagging=SAMPLE_BAGGING)


    #
    # XGBoost Evaluation
    #
    if "xgboost" in eval_list:
        # That's the xgboost metric function serving as a model
        xgboost_model = xgb_metric

        # Key is the dataset id (did) and value the mean error on it. We use mamba model params as bptt and eval_positions
        # NOTE: Current max time is 300, aka 5 minutes. Need to change this maybe.
        result_dict["xgboost"] = eval_helper.do_evaluation_custom(xgboost_model, bptt=bptt_here, eval_positions=mamba_config["eval_positions"], metric=METRIC_USED, device=device, method_name="xgb",
                                        evaluation_type=EVALUATION_TYPE, max_time=max_time)


    #
    # KNN Evaluation
    #
    if "knn" in eval_list:
        # That's the k-nearest-neighbor metric function serving as a model
        knn_model = knn_metric

        # Key is the dataset id (did) and value the mean error on it. We use mamba model params as bptt and eval_positions
        result_dict["knn"] = eval_helper.do_evaluation_custom(knn_model, bptt=bptt_here, eval_positions=mamba_config["eval_positions"], metric=METRIC_USED, device=device, method_name="knn",
                                        evaluation_type=EVALUATION_TYPE, max_time=max_time)
    

    return result_dict


# This needs rework at some time. Super not efficient.
if __name__ == "__main__":

    result_dict = do_evaluation(EVALUATION_METHODS)

    header = ["did"] + EVALUATION_METHODS

    result_arr = []

    # Calc Mean and Confidence Intervals
    for method in EVALUATION_METHODS:
        split_means = []

        for split in range(len(SPLIT_NUMBERS)):
            vals = result_dict[method].values()
            split_errs = [x[split] for x in vals]
            split_means.append(sum(split_errs) / len(split_errs))

        print(f"{method} Stats: ")
        print(f"Split Means: {split_means}")
        print(f"Mean Overall: {sum(split_means) / len(split_means)}")
        print(f"MOE: {calc_moe(split_means)}")


    keys = list(result_dict[list(result_dict.keys())[0]].keys())

    for key in keys:
        to_add = [key]

        for method in EVALUATION_METHODS: 
            res = result_dict[method][key]
            to_add.append(sum(res) / len(res))

        result_arr.append(to_add)

    df_out = pd.DataFrame(result_arr, columns=header)

    df_out.to_csv(RESULT_CSV_SAVE_DIR)

    print("worked")
