from tabpfn.scripts.tabular_evaluation import evaluate
from tabpfn.scripts import tabular_metrics
import os
from tabpfn.datasets import load_openml_list
from evaluation_helper import EvalHelper
from tabpfn.scripts.mamba_prediction_interface import load_model_workflow as mamba_load_model_workflow
from tabpfn.scripts.transformer_prediction_interface import load_model_workflow as transformer_load_model_workflow
from tabpfn.scripts.tabular_baselines import *

import pandas as pd

EVALUATION_TYPE = "openmlcc18_large"
#EVALUATION_TYPE = "openmlcc18"

EVALUATION_METHODS = ["mamba", "transformer"]

METRIC_USED = tabular_metrics.auc_metric

RESULT_CSV_SAVE_DIR = os.path.join("result_csvs", "bptt_cc18_large_cropped.csv")

#MAMBA_MODEL_NAME = "tabpfn/models_diff/mamba_test_model.cpkt"
MAMBA_MODEL_NAME = "../tabpfn/models_diff/mamba_current.cpkt"
TRANSFORMER_MODEL_NAME = "../tabpfn/models_diff/transformer_120e_tabpfn.cpkt"

#BPTTS = [i for i in range(50, 2500, 50)]
BPTTS = [10, 20]

device = "cuda:0"

def do_evaluation(eval_list, bptt):

    result_dict = {}

    # Set up the evaluation Helper class
    eval_helper = EvalHelper()

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
        result_dict["mamba"] = eval_helper.do_evaluation_custom(mamba_model, bptt=bptt, eval_positions=mamba_config["eval_positions"], metric=METRIC_USED, device=device, method_name="mamba",
                                        evaluation_type=EVALUATION_TYPE)

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
        result_dict["transformer"] = eval_helper.do_evaluation_custom(transformer_model, bptt=bptt, eval_positions=transformer_config["eval_positions"], metric=METRIC_USED, device=device, method_name="transformer",
                                        evaluation_type=EVALUATION_TYPE)
    

    return result_dict



if __name__ == "__main__":

    bptt_dict = {}

    count = len(EVALUATION_METHODS) * len(BPTTS)
    counter = 1

    for method in EVALUATION_METHODS:
        bptt_dict[method] = {}
        for bptt in BPTTS:
            print(f"Currently at {counter} / {count}")
            counter += 1
            result_dict = do_evaluation(method, bptt)
            vals = result_dict.values()
            bptt_dict[bptt] = sum(vals) / len(vals)

    header = ["bptt"] + EVALUATION_METHODS

    result_arr = []

    for bptt in BPTTS:
        to_add = [bptt]

        for method in EVALUATION_METHODS: 
            to_add.append(bptt_dict[method][bptt])

        result_arr.append(to_add)

    df_out = pd.DataFrame(result_arr, columns=header)

    df_out.to_csv(RESULT_CSV_SAVE_DIR)

    print("worked")
