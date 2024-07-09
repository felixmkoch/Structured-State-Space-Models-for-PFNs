from openml import config, study, tasks, runs, extensions, datasets
from tabpfn.scripts.tabular_evaluation import evaluate
from tabpfn.scripts import tabular_metrics
from openml.tasks import TaskType
import numpy as np
import wandb
from tabpfn.datasets import load_openml_list

class EvalHelper:

    def __init__(self, dids="default"):

        # Got these datasets from the scripts of tabpfn
        if dids == "test": valid_dids_classification = [973, 1596, 40981, 1468, 40984, 40975, 41163, 41147, 1111, 41164, 1169, 1486, 41143, 1461, 41167, 40668, 41146, 41169, 41027, 23517, 41165, 41161, 41159, 41138, 1590, 41166, 1464, 41168, 41150, 1489, 41142, 3, 12, 31, 54, 1067]
        else: valid_dids_classification = [13, 59, 4, 15, 40710, 43, 1498]

        # OpenCC Dids filtered by N_samples < 2000, N feats < 100, N classes < 10
        self.openml_cc18_dids = [11,14,15,16,18,22,23,29,31,37,50,54,188,458,469,1049,1050,1063,1068,1510,1494,1480,1462,1464,6332,23381,40966,40982,40994,40975]

        print("Loading the OpenML cc18 Dicts ...")
        self.openml_cc18_dataset_data = list(zip(*[load_openml_list([did]) for did in self.openml_cc18_dids]))[0]

        # OpenML cc18 purely numerical without missing values (18 items) 

        self.dids = dids

        print("Loading validation Datasets ...")    
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

            for did_idx, did in enumerate(self.openml_cc18_dids):
                result[did] = evaluate(self.openml_cc18_dataset_data[did_idx], bptt, eval_positions, metric, model, device, method_name=method_name)["mean_metric"].item()

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





