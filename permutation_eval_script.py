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


#EVALUATION_TYPE = "openmlcc18_large"
EVALUATION_TYPE = "openmlcc18"

NUM_RANDOM_PERMUTATIONS = 2

EVALUATION_METHOD = "transformer"

METRIC_USED = tabular_metrics.auc_metric

RESULT_CSV_SAVE_DIR = os.path.join("result_csvs", "permutation_eval_res.csv")

MAMBA_MODEL_NAME = "tabpfn/models_diff/mamba_small.cpkt"
TRANSFORMER_MODEL_NAME = "tabpfn/models_diff/tabpfn_transformer_model.cpkt"
HYDRA_MODEL_NAME = "tabpfn/models_diff/mamba_test.cpkt"

# Only use one element in the list please
SPLIT_NUMBERS = [1]
max_time = 3600

JRT_PROMPT = True

bptt_here = 1000

device = "cuda:0"

eval_helper = EvalHelper()

def do_permutation_evaluation(eval_method, random_premutation=True):

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
        result_dict["mamba"] = eval_helper.do_evaluation_custom(mamba_model, bptt=bptt_here, eval_positions=mamba_config["eval_positions"], metric=METRIC_USED, device=device, method_name="mamba",
                                        evaluation_type=EVALUATION_TYPE, split_numbers=SPLIT_NUMBERS, jrt_prompt=JRT_PROMPT, permutation_random=random_premutation, return_whole_output=True)

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
        result_dict["transformer"] = eval_helper.do_evaluation_custom(transformer_model, bptt=bptt_here, eval_positions=transformer_config["eval_positions"], metric=METRIC_USED, device=device, method_name="transformer",
                                        evaluation_type=EVALUATION_TYPE, split_numbers=SPLIT_NUMBERS, jrt_prompt=JRT_PROMPT, permutation_random=random_premutation, return_whole_output=True)


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
        result_dict["hydra"] = eval_helper.do_evaluation_custom(hydra_model, bptt=bptt_here, eval_positions=hydra_config["eval_positions"], metric=METRIC_USED, device=device, method_name="hydra",
                                        evaluation_type=EVALUATION_TYPE, split_numbers=SPLIT_NUMBERS, jrt_prompt=JRT_PROMPT, permutation_random=random_premutation, return_whole_output=True)


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


def calc_kl_div(p, q):

    # Convert p to log probabilities
    log_p = torch.log(p)

    # Compute KL divergence: F.kl_div(log_p, q) computes D_KL(P || Q)
    kl_divergence = F.kl_div(log_p, q, reduction='batchmean')

    return kl_divergence.item()


if __name__ == "__main__":

    results = []
    mean_metrics = []

    for i in range(NUM_RANDOM_PERMUTATIONS):
        # output is method_dict -> did -> split_number -> output_dict
        outputs = do_permutation_evaluation(EVALUATION_METHOD, random_premutation=True)

        last_outputs = {}
        mean_metric = {}

        for did in eval_helper.get_dids_by_string("openmlcc18"):
            last_outputs[did] = outputs[EVALUATION_METHOD][did][0]["last_outputs"]
            mean_metric[did] = outputs[EVALUATION_METHOD][did][0]["mean_metric"]

        mean_metrics.append(mean_metric)
        results.append(last_outputs)


    kl_per_did = []

    for did in eval_helper.get_dids_by_string("openmlcc18"):
            
        kl_per_did.append(calc_kl_div(results[0][did], results[1][did]))

    
    print(f"KL-Divergence Overall: {sum(kl_per_did) / len(kl_per_did)}")


    print("worked")