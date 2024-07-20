from tabpfn.scripts.tabular_evaluation import evaluate
from tabpfn.scripts import tabular_metrics
import os
from tabpfn.datasets import load_openml_list
from evaluation_helper import EvalHelper
from tabpfn.scripts.mamba_prediction_interface import load_model_workflow as mamba_load_model_workflow
from tabpfn.scripts.transformer_prediction_interface import load_model_workflow as transformer_load_model_workflow
from tabpfn.scripts.tabular_baselines import *
from scipy import stats

import pandas as pd


#EVALUATION_TYPE = "openmlcc18_large"
EVALUATION_TYPE = "openmlcc18"

EVALUATION_METHODS = ["mamba"]

METRIC_USED = tabular_metrics.auc_metric

RESULT_CSV_SAVE_DIR = os.path.join("result_csvs", "bptt_cc18_large_cropped.csv")

#MAMBA_MODEL_NAME = "tabpfn/models_diff/mamba_test_model.cpkt"
MAMBA_MODEL_NAME = "tabpfn/models_diff/mamba_150e.cpkt"
TRANSFORMER_MODEL_NAME = "tabpfn/models_diff/transformer_120e_tabpfn.cpkt"

#BPTTS = [i for i in range(50, 2500, 50)]
BPTTS = [100, 150]
SPLIT_NUMBERS = [1, 2, 3, 4, 5]
CONFIDENCE_LEVEL = 0.95

device = "cuda:0"

# Set up the evaluation Helper class
eval_helper = EvalHelper()

def calc_confidence_interval(data):

    mean = np.mean(data)

    sem = stats.sem(data)

    confidence_level = CONFIDENCE_LEVEL
    degrees_of_freedom = len(data) - 1
    t_score = stats.t.ppf((1 + confidence_level) / 2, degrees_of_freedom)

    margin_of_error = t_score * sem

    confidence_interval = (mean - margin_of_error, mean + margin_of_error)

    return confidence_interval


def do_evaluation(eval_list, bptt):

    result_dict = {}

    eval_position = [(bptt / 2) + 1]

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
        result_dict["mamba"] = eval_helper.do_evaluation_custom(mamba_model, bptt=bptt, eval_positions=eval_position, metric=METRIC_USED, device=device, method_name="mamba",
                                        evaluation_type=EVALUATION_TYPE, split_numbers=SPLIT_NUMBERS)

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
        result_dict["transformer"] = eval_helper.do_evaluation_custom(transformer_model, bptt=bptt, eval_positions=eval_position, metric=METRIC_USED, device=device, method_name="transformer",
                                        evaluation_type=EVALUATION_TYPE, split_numbers=SPLIT_NUMBERS)
    

    return result_dict



if __name__ == "__main__":

    bptt_dict = {}

    count = len(EVALUATION_METHODS) * len(BPTTS)
    counter = 1

    EVALUATION_METHOD_HEADER = []
    for m in EVALUATION_METHODS:
        EVALUATION_METHOD_HEADER.append(m)
        EVALUATION_METHOD_HEADER.append(m+"_clow")
        EVALUATION_METHOD_HEADER.append(m+"_chigh")

    for method in EVALUATION_METHODS:
        bptt_dict[method] = {}
        for bptt in BPTTS:
            bptt_dict[method][bptt] = []
            print(f"Currently at {counter} / {count}")
            counter += 1
            result_dict = do_evaluation(method, bptt)[method]
            vals = result_dict.values()
            for i in range(len(SPLIT_NUMBERS)):
                s_res = [x[i] for x in vals]
                bptt_dict[method][bptt].append(sum(s_res) / len(s_res)) # Mean of the split

    header = ["bptt"] + EVALUATION_METHOD_HEADER

    result_arr = []

    for bptt in BPTTS:
        to_add = [bptt]

        for method in EVALUATION_METHODS: 
            split_vals = bptt_dict[method][bptt] # arr split values
            to_add.append(sum(split_vals) / len(split_vals)) # Normal mean
            conf_interval = calc_confidence_interval(split_vals)
            to_add.append(conf_interval[0]) # Lower Confidence
            to_add.append(conf_interval[1]) # Upper Confidence

        result_arr.append(to_add)

    df_out = pd.DataFrame(result_arr, columns=header)

    df_out.to_csv(RESULT_CSV_SAVE_DIR)

    print("worked")
