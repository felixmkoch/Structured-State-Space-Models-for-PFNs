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
import torch.nn.functional as F

import pandas as pd

EVALUATION_TYPE = "dummy"

NUM_DUMMY_ROWS = [20, 100, 500, 1000, 2000, 5000, 10000]
NUM_DUMMY_FEATS = 99            # Stay below the maximum of the model, because this doesn't include the class column yet.

EVALUATION_METHODS = ["transformer", "hydra"]

METRIC_USED = tabular_metrics.auc_metric

RESULT_CSV_SAVE_DIR = os.path.join("result_csvs", "dummy_script.csv")

MAMBA_MODEL_NAME = "tabpfn/models_diff/mamba_small.cpkt"
TRANSFORMER_MODEL_NAME = "tabpfn/models_diff/tabpfn_transformer_model.cpkt"
HYDRA_MODEL_NAME = "tabpfn/models_diff/hydra_small.cpkt"

# Only use one element in the list please
SPLIT_NUMBERS = [1]
max_time = 3600

JRT_PROMPT = False

bptt_here = 999999999
eval_positions = [99999999]

device = "cuda:0"

PERMUTATION_BAGGING = 1
SAMPLE_BAGGING = 0

eval_helper = EvalHelper()

def do_time_evaluation(eval_method, random_premutation=False, dummy_tensor_dim=(1000, 99)):

    result_dict = {}

    #
    # MAMBA EVALUATION
    #
    if "mamba" == eval_method:
        # Load Mamba Model (Yes this is a bit scuffed).
        m_model, mamba_config, results_file = mamba_load_model_workflow(2, -1, add_name="", base_path="", device=device,eval_addition='', 
                                                    only_inference=True, model_path_custom=MAMBA_MODEL_NAME)

        # That's the real mamba model here.
        mamba_model = m_model[2]

        # Key is the dataset id (did) and value the mean error on it.
        result_dict["mamba"] = eval_helper.do_evaluation_custom(mamba_model, bptt=bptt_here, eval_positions=eval_positions, metric=METRIC_USED, device=device, method_name="mamba",
                                        evaluation_type=EVALUATION_TYPE, split_numbers=SPLIT_NUMBERS, jrt_prompt=JRT_PROMPT, permutation_random=random_premutation, 
                                        permutation_bagging=PERMUTATION_BAGGING, sample_bagging=SAMPLE_BAGGING, return_whole_output=True, dummy_size=dummy_tensor_dim)

    #
    # TRANSFORMER EVALUATION
    #
    if "transformer" == eval_method:
        # Load Transformer Model (Yes this is a bit scuffed).
        t_model, transformer_config, results_file = transformer_load_model_workflow(2, -1, add_name="", base_path="", device=device,eval_addition='', 
                                                    only_inference=True, model_path_custom=TRANSFORMER_MODEL_NAME)

        # That's the real transformer model here.
        transformer_model = t_model[2]

        # Key is the dataset id (did) and value the mean error on it.
        result_dict["transformer"] = eval_helper.do_evaluation_custom(transformer_model, bptt=bptt_here, eval_positions=eval_positions, metric=METRIC_USED, device=device, method_name="transformer",
                                        evaluation_type=EVALUATION_TYPE, split_numbers=SPLIT_NUMBERS, jrt_prompt=JRT_PROMPT, 
                                        permutation_random=random_premutation, return_whole_output=True, dummy_size=dummy_tensor_dim)


    #
    # HYDRA EVALUATION
    #
    if "hydra" == eval_method:
        # Load Transformer Model (Yes this is a bit scuffed).
        h_model, hydra_config, results_file = hydra_load_model_workflow(2, -1, add_name="", base_path="", device=device,eval_addition='', 
                                                    only_inference=True, model_path_custom=HYDRA_MODEL_NAME)

        # That's the real transformer model here.
        hydra_model = h_model[2]

        # Key is the dataset id (did) and value the mean error on it.
        result_dict["hydra"] = eval_helper.do_evaluation_custom(hydra_model, bptt=bptt_here, eval_positions=eval_positions, metric=METRIC_USED, device=device, method_name="hydra",
                                        evaluation_type=EVALUATION_TYPE, split_numbers=SPLIT_NUMBERS, jrt_prompt=JRT_PROMPT, permutation_random=random_premutation, 
                                        permutation_bagging=PERMUTATION_BAGGING, sample_bagging=SAMPLE_BAGGING, return_whole_output=True, dummy_size=dummy_tensor_dim)


    #
    # XGBoost Evaluation
    #
    if "xgboost" == eval_method:
        # That's the xgboost metric function serving as a model
        xgboost_model = xgb_metric

        # Key is the dataset id (did) and value the mean error on it. We use mamba model params as bptt and eval_positions
        # NOTE: Current max time is 300, aka 5 minutes. Need to change this maybe.
        result_dict["xgboost"] = eval_helper.do_evaluation_custom(xgboost_model, bptt=bptt_here, eval_positions=mamba_config["eval_positions"], metric=METRIC_USED, device=device, method_name="xgb",
                                        evaluation_type=EVALUATION_TYPE, max_time=max_time, permutation_random=random_premutation, return_whole_output=True)


    #
    # KNN Evaluation
    #
    if "knn" == eval_method:
        # That's the k-nearest-neighbor metric function serving as a model
        knn_model = knn_metric

        # Key is the dataset id (did) and value the mean error on it. We use mamba model params as bptt and eval_positions
        result_dict["knn"] = eval_helper.do_evaluation_custom(knn_model, bptt=bptt_here, eval_positions=mamba_config["eval_positions"], metric=METRIC_USED, device=device, method_name="knn",
                                        evaluation_type=EVALUATION_TYPE, max_time=max_time, permutation_random=random_premutation, return_whole_output=True)
    

    return result_dict



if __name__ == "__main__":

    results = {}

    for model_type in EVALUATION_METHODS:

        results[model_type] = []

        for num_rows in NUM_DUMMY_ROWS:

            dummy_tensor_dim = (num_rows, NUM_DUMMY_FEATS)

            # output is method_dict -> did -> split_number -> output_dict
            outputs = do_time_evaluation(model_type, random_premutation=True, dummy_tensor_dim=dummy_tensor_dim)[model_type]["dummy"][0]

            results[model_type].append(outputs[f"dummy_set_time_at_{eval_positions[0]}"])

        
    df = pd.DataFrame(results)

    dummy_size = [v * (NUM_DUMMY_FEATS+1) for v in NUM_DUMMY_ROWS]

    df.insert(0, "table_entries", dummy_size)

    df.to_csv(RESULT_CSV_SAVE_DIR)

    print("worked")
