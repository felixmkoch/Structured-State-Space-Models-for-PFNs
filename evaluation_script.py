from tabpfn.scripts.tabular_evaluation import evaluate
from tabpfn.scripts import tabular_metrics
import os
from tabpfn.datasets import load_openml_list
from evaluation_helper import EvalHelper
from tabpfn.scripts.mamba_prediction_interface import load_model_workflow as mamba_load_model_workflow
from tabpfn.scripts.transformer_prediction_interface import load_model_workflow as transformer_load_model_workflow

import pandas as pd


METRIC_USED = tabular_metrics.auc_metric

RESULT_CSV_SAVE_DIR = os.path.join("result_csvs", "cc18_results.csv")

MAMBA_MODEL_NAME = "tabpfn/models_diff/mamba_test_model.cpkt"
TRANSFORMER_MODEL_NAME = "tabpfn/models_diff/transformer_test_model.cpkt"


# Set up the evaluation Helper class
eval_helper = EvalHelper()

#
# MAMBA EVALUATION
#

# Load Mamba Model (Yes this is a bit scuffed).
m_model, mamba_config, results_file = mamba_load_model_workflow(2, -1, add_name="", base_path="", device="cuda",eval_addition='', 
                                             only_inference=True, model_path_custom=MAMBA_MODEL_NAME)

# That's the real mamba model here.
mamba_model = m_model[2]

# Key is the dataset id (did) and value the mean error on it.
eval_helper_mamba_results = eval_helper.do_evaluation_custom(mamba_model, bptt=mamba_config["bptt"], eval_positions=mamba_config["eval_positions"], metric=METRIC_USED, device="cuda", method_name="mamba",
                                 evaluation_type="openmlcc18")

#
# TRANSFORMER EVALUATION
#

# Load Transformer Model (Yes this is a bit scuffed).
t_model, transformer_config, results_file = transformer_load_model_workflow(2, -1, add_name="", base_path="", device="cuda",eval_addition='', 
                                             only_inference=True, model_path_custom=TRANSFORMER_MODEL_NAME)

# That's the real transformer model here.
transformer_model = t_model[2]

# Key is the dataset id (did) and value the mean error on it.
eval_helper_transformer_results = eval_helper.do_evaluation_custom(mamba_model, bptt=transformer_config["bptt"], eval_positions=transformer_config["eval_positions"], metric=METRIC_USED, device="cuda", method_name="transformer",
                                 evaluation_type="openmlcc18")


# results into csv
result_arr = []
header = ["did", "transformer", "mamba"]

for key in eval_helper_mamba_results.keys(): result_arr.append([key, eval_helper_transformer_results[key], eval_helper_mamba_results[key]])

df_out = pd.DataFrame(result_arr, columns=header)

df_out.to_csv(RESULT_CSV_SAVE_DIR)


print("worked")