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

#
# Here: True means to keep them, false to omit
#
EVALUATION_TYPE_FILTERS = {
    "categorical": True,
    "nans": True,
    "multiclass": True
}

openml_cc18_small_dids = [11, 14, 15, 16, 18, 22, 23, 29, 31, 37, 50, 54, 188, 458, 469, 1049, 1050, 1063, 1068, 1462, 1464, 1480, 1494, 1510, 6332, 23381, 40966, 40975, 40982, 40994]

EVALUATION_METHODS = ["transformer"]

metrics_dict = {
    "acc": tabular_metrics.accuracy_metric,
    "auc": tabular_metrics.auc_metric
}

METRICS_USED = ["acc"]

RESULT_CSV_SAVE_DIR = os.path.join("result_csvs", "perm_bars")

RESULT_CSV_PREFIX = ""

MAMBA_MODEL_NAME = "tabpfn/models_diff/mamba_small.cpkt"
TRANSFORMER_MODEL_NAME = "tabpfn/models_diff/tabpfn_transformer_model.cpkt"
HYDRA_MODEL_NAME = "tabpfn/models_diff/hydra_small.cpkt"

# Only use one element in the list please
NUM_SPLITS = 2
SPLIT_NUMBERS = [i+1 for i in range(NUM_SPLITS)]

JRT_PROMPT = False
SINGLE_EVAL_PROMPT = False

bptt_here = 1024

device = "cuda:0"

PERMUTATION_BAGGINGS = [2]
SAMPLE_BAGGING = 0

eval_helper = EvalHelper()

def do_evaluation(eval_list, metric, permutation_bagging):

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
        result_dict["mamba"] = eval_helper.do_evaluation_custom(mamba_model, bptt=bptt_here, 
                                        eval_positions=mamba_config["eval_positions"], metric=metric, device=device, method_name="mamba", permutation_random=True,
                                        evaluation_type=EVALUATION_TYPE, split_numbers=SPLIT_NUMBERS, jrt_prompt=JRT_PROMPT, permutation_bagging=permutation_bagging, sample_bagging=SAMPLE_BAGGING, eval_filters=EVALUATION_TYPE_FILTERS, return_whole_output=True)

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
        result_dict["hydra"] = eval_helper.do_evaluation_custom(hydra_model, bptt=bptt_here, 
                                        eval_positions=hydra_config["eval_positions"], metric=metric, device=device, method_name="hydra", permutation_random=True,
                                        evaluation_type=EVALUATION_TYPE, split_numbers=SPLIT_NUMBERS, jrt_prompt=JRT_PROMPT, single_evaluation_prompt=SINGLE_EVAL_PROMPT, permutation_bagging=permutation_bagging , sample_bagging=SAMPLE_BAGGING, eval_filters=EVALUATION_TYPE_FILTERS, return_whole_output=True)


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
        result_dict["transformer"] = eval_helper.do_evaluation_custom(transformer_model, bptt=bptt_here, 
                                        eval_positions=transformer_config["eval_positions"], metric=metric, device=device, method_name="transformer", permutation_random=True,
                                        evaluation_type=EVALUATION_TYPE, split_numbers=SPLIT_NUMBERS, jrt_prompt=JRT_PROMPT, single_evaluation_prompt=SINGLE_EVAL_PROMPT, permutation_bagging=permutation_bagging , sample_bagging=SAMPLE_BAGGING, eval_filters=EVALUATION_TYPE_FILTERS, return_whole_output=True)

    return result_dict


def calc_kl_div(p, q):

    # Convert p to log probabilities
    log_p = torch.log(p)

    # Compute KL divergence: F.kl_div(log_p, q) computes D_KL(P || Q)
    kl_divergence = F.kl_div(log_p, q, reduction='batchmean')

    return kl_divergence.item()


if __name__ == "__main__":

    header = ["did"] + [str(i) for i in SPLIT_NUMBERS]

    for permutation_bagging in PERMUTATION_BAGGINGS:
        
        for metric_used in METRICS_USED:

            metric = metrics_dict[metric_used]

            result_dict = do_evaluation(EVALUATION_METHODS, metric, permutation_bagging)
            result_dict2 = do_evaluation(EVALUATION_METHODS, metric, permutation_bagging)

            for evaluation_method in EVALUATION_METHODS:

                res_dict_method = result_dict[evaluation_method]
                res_dict_method2 = result_dict2[evaluation_method]

                csv_name = f"{RESULT_CSV_PREFIX}{evaluation_method}_{metric_used}_pb_{permutation_bagging}.csv"

                output_path = os.path.join(RESULT_CSV_SAVE_DIR, csv_name)
                
                to_print = []

                for did in eval_helper.get_dids_by_string("openmlcc18"):

                    to_print.append([did] + [calc_kl_div(res_dict_method[did][k]["last_outputs"], res_dict_method2[did][k]["last_outputs"]) for k in range(NUM_SPLITS)])
                    
                df_out = pd.DataFrame(to_print, columns=header)

                df_out.to_csv(output_path, index=False)


    print("worked")
