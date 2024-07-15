from openml import config, study, tasks, runs, extensions, datasets
from tabpfn.scripts.tabular_evaluation import evaluate
from tabpfn.scripts import tabular_metrics
from openml.tasks import TaskType
import numpy as np
import wandb
from tabpfn.datasets import load_openml_list
import openml

class EvalHelper:

    def __init__(self, dids="default"):

        # Got these datasets from the scripts of tabpfn
        if dids == "test": valid_dids_classification = [973, 1596, 40981, 1468, 40984, 40975, 41163, 41147, 1111, 41164, 1169, 1486, 41143, 1461, 41167, 40668, 41146, 41169, 41027, 23517, 41165, 41161, 41159, 41138, 1590, 41166, 1464, 41168, 41150, 1489, 41142, 3, 12, 31, 54, 1067]
        else: valid_dids_classification = [13, 59, 4, 15, 40710, 43, 1498]

        # OpenCC Dids filtered by N_samples < 2000, N feats < 100, N classes < 10
        self.openml_cc18_dids = [3, 6, 11, 12, 14, 15, 16, 18, 22, 23, 28, 29, 31, 32, 37, 44, 46, 50, 54, 151, 182, 188, 38, 307, 300, 458, 469, 554, 1049, 1050, 1053, 1063, 1067, 1068, 1590, 4134, 1510, 1489, 1494, 1497, 1501, 1480, 1485, 1486, 1487, 1468, 1475, 1462, 1464, 4534, 6332, 1461, 4538, 1478, 23381, 40499, 40668, 40966, 40982, 40994, 40983, 40975, 40984, 40979, 40996, 41027, 23517, 40923, 40927, 40978, 40670, 40701]
        self.openml_cc18_dids_small = [11, 14, 15, 16, 18, 22, 23, 29, 31, 37, 50, 54, 188, 458, 469, 1049, 1050, 1063, 1068, 1462, 1464, 1480, 1494, 1510, 6332, 23381, 40966, 40975, 40982, 40994]
        self.openml_cc18_dids_large = [3, 6, 12, 28, 32, 38, 44, 46, 151, 182, 300, 307, 554, 1053, 1067, 1461, 1468, 1475, 1478, 1485, 1486, 1487, 1489, 1497, 1501, 1590, 4134, 4534, 4538, 23517, 40499, 40668, 40670, 40701, 40923, 40927, 40978, 40979, 40983, 40984, 40996, 41027]

        print("Loading the OpenML cc18 Dicts ...")
        self.openml_cc18_dataset_data = {}
        for did in self.openml_cc18_dids:
            self.openml_cc18_dataset_data[did] = load_openml_list([did], num_feats=99999, max_samples=999999, max_num_classes=999)[0]

        # Validation and so on - not OpenML cc18
        print("Loading validation Datasets ...")    
        self.dids = dids
        self.datasets_data, _ = load_openml_list(valid_dids_classification)


    
    def do_evaluation(self, model, bptt, eval_positions, metric, device, method_name):
        '''
        Evaluation on the validation set for everything.
        '''
        
        result = evaluate(self.datasets_data, bptt, eval_positions, metric, model, device,method_name=method_name)

        return result['mean_metric']
    

    def do_evaluation_custom(self, model, bptt, eval_positions, metric, device, method_name, evaluation_type):

        '''
        Evaluation on customly settable datasets.
        '''

        if evaluation_type == "openmlcc18":
            print("Evaluating on the OpenML cc18 Dataset ...")

            result = {}

            for did_idx, did in enumerate(self.openml_cc18_dids_small):
                result[did] = evaluate(self.openml_cc18_dataset_data[did], bptt, eval_positions, metric, model, device, method_name=method_name)["mean_metric"].item()

            return result
        
        if evaluation_type == "openmlcc18_large":
            print("Evaluating on the large part of the OpenML cc18 Dataset ...")

            result = {}

            for did_idx, did in enumerate(self.openml_cc18_dids_large):
                result[did] = evaluate(self.openml_cc18_dataset_data[did], bptt, eval_positions, metric, model, device, method_name=method_name)["mean_metric"].item()

            return result


        # Standard case: Normal eval dataset
        return evaluate(self.datasets_data, bptt, eval_positions, metric, model, device,method_name=method_name)['mean_metric']
    


    def do_naive_evaluation(self):

        performances = []

        for ds_name, X, y, categorical_feats, _, _ in self.datasets_data:
            
            unique, counts = np.unique(y, return_counts=True)
            majority_class = unique[np.argmax(counts)]

            predictions = np.full_like(y, majority_class)

            metric = tabular_metrics.auc_metric
            performance = metric(y, predictions)

            performances.append(performance)

        print(f"Naive Evluation has mean of {sum(performances) / len(performances)}")

        return sum(performances) / len(performances)


    def log_wandb_naive_evaluation(self, num_steps=100, log_name="mamba_mean_acc"):

        wandb_project = "mamba_project"
        wandb_job_type = "naive_model_exec"
        wandb_run_name = "Majority Class Prediction"

        wandb_config= {}

        wandb_run = wandb.init(project=wandb_project,job_type=wandb_job_type,config=wandb_config, name=wandb_run_name)

        naive_result = self.do_naive_evaluation()

        for _ in range(num_steps): wandb.log({f"test/{log_name}": naive_result}) 

        wandb_run.finish()


if __name__ == "__main__":
    h = EvalHelper()
    #h.do_naive_evaluation()
    #h.log_wandb_naive_evaluation(num_steps=200, log_name="mamba_mean_acc")





