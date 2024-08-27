from openml import config, study, tasks, runs, extensions, datasets
from tabpfn.scripts.tabular_evaluation import evaluate
from tabpfn.scripts import tabular_metrics
from openml.tasks import TaskType
import numpy as np
import wandb
from collections import Counter
from tabpfn.datasets import load_openml_list
import openml
import torch

class EvalHelper:

    def __init__(self):

        # Got these datasets from the scripts of tabpfn
        self.test_dids_classification = [973, 1111, 1169, 1596, 40981, 41138, 41142, 41143, 41146, 41147, 41150, 41159, 41161, 41163, 41164, 41165, 41166, 41167, 41168, 41169]
        self.valid_dids_classification = [13, 59, 40710, 43, 1498]

        # OpenCC Dids filtered by N_samples < 2000, N feats < 100, N classes < 10
        self.openml_cc18_dids = [3, 6, 11, 12, 14, 15, 16, 18, 22, 23, 28, 29, 31, 32, 37, 44, 46, 50, 54, 151, 182, 188, 38, 307, 300, 458, 469, 554, 1049, 1050, 1053, 1063, 1067, 1068, 1590, 4134, 1510, 1489, 1494, 1497, 1501, 1480, 1485, 1486, 1487, 1468, 1475, 1462, 1464, 4534, 6332, 1461, 4538, 1478, 23381, 40499, 40668, 40966, 40982, 40994, 40983, 40975, 40984, 40979, 40996, 41027, 23517, 40923, 40927, 40978, 40670, 40701]
        self.openml_cc18_dids_small = [11, 14, 15, 16, 18, 22, 23, 29, 31, 37, 50, 54, 188, 458, 469, 1049, 1050, 1063, 1068, 1462, 1464, 1480, 1494, 1510, 6332, 23381, 40966, 40975, 40982, 40994]
        self.openml_cc18_dids_large = [3, 6, 12, 28, 32, 38, 44, 46, 151, 182, 300, 307, 554, 1053, 1067, 1461, 1468, 1475, 1478, 1485, 1486, 1487, 1489, 1497, 1501, 1590, 4134, 4534, 4538, 23517, 40499, 40668, 40670, 40701, 40923, 40927, 40978, 40979, 40983, 40984, 40996, 41027]

        all_dids = self.openml_cc18_dids + self.test_dids_classification + self.valid_dids_classification

        self.datasets_data = {}
        self.limit_dict = {}


    def check_datasets_data(self, dids):

        data_keys = list(self.datasets_data.keys())

        for did in dids:
            if not (did in data_keys):
                self.datasets_data[did] = load_openml_list([did], num_feats=99999, max_samples=999999, max_num_classes=999)[0]

    
    def do_evaluation(self, model, bptt, eval_positions, metric, device, method_name, max_classes=10, max_features=100, max_time=300):
        '''
        Evaluation on the validation set for everything.
        '''
        results = {}

        self.check_datasets_data(self.valid_dids_classification)

        self.make_limit_datasets(max_classes, max_features, self.valid_dids_classification)

        for did in self.valid_dids_classification:
            results[did] = evaluate(self.limit_dict[did], bptt, eval_positions, metric, model, device,method_name=method_name)["mean_metric"]

        vals = results.values()

        return sum(vals) / len(vals)
    

    def do_test(self, model, bptt, eval_positions, metric, device, method_name, max_classes=10, max_features=100, max_time=300):
        '''
        Test on the validation set for everything.
        '''
        results = {}

        self.check_datasets_data(self.test_dids_classification)

        self.make_limit_datasets(max_classes, max_features, self.test_dids_classification)

        for did in self.test_dids_classification:
            results[did] = evaluate(self.limit_dict[did], bptt, eval_positions, metric, model, device,method_name=method_name)["mean_metric"]

        vals = results.values()

        return sum(vals) / len(vals)
    
    
    def limit_dataset(self, ds_name, X, y, categorical_feats, max_classes, max_features):
        # Ensure X and y are tensors
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y)
        
        # Limit features
        X_limited = X[:, :max_features]
        
        # Get top max_classes majority classes
        y_np = y.numpy()
        class_counts = Counter(y_np)
        top_classes = [cls for cls, _ in class_counts.most_common(max_classes)]
        
        # Create a mask for top classes
        top_classes_set = set(top_classes)
        mask = torch.tensor([item in top_classes_set for item in y_np])
        
        # Filter rows to include only the top max_classes
        X_limited = X_limited[mask]
        y_limited = y[mask]
    
        # Return the limited dataset
        return (ds_name, X_limited, y_limited, categorical_feats, None, None)
    
    

    def make_limit_datasets(self, max_classes, max_features, limit_dids):

        for did in limit_dids:
            ds_name, X, y, categorical_feats, _, _ = self.datasets_data[did][0]
            new_data = self.limit_dataset(ds_name, X, y, categorical_feats, max_classes, max_features)
            self.limit_dict[did] = [new_data]

    

    def do_evaluation_custom(self, 
                             model, 
                             bptt, 
                             eval_positions, 
                             metric, 
                             device, 
                             method_name, 
                             evaluation_type, 
                             max_classes=10, 
                             max_features=100, 
                             max_time=300, 
                             split_numbers=[1],
                             jrt_prompt=False,
                             permutation_random=False,
                             return_whole_output=False):

        '''
        Evaluation on customly settable datasets.
        '''

        predefined_eval_types = ["openmlcc18", "openmlcc18_large", "test"]

        # Standard case: Normal eval dataset
        if evaluation_type not in predefined_eval_types: print("Using single DID in Evaluation")

        # The dataset to iterate over
        ds = None
        if evaluation_type == "openmlcc18": ds = self.openml_cc18_dids_small

        if evaluation_type == "openmlcc18_large": ds = self.openml_cc18_dids_large

        if evaluation_type == "test": ds = self.test_dids_classification

        if evaluation_type not in predefined_eval_types: ds = [evaluation_type]
            
        self.check_datasets_data(ds)

        self.make_limit_datasets(max_classes, max_features, ds)

        print("Evaluating custom dataset ... ")

        result = {}

        for did in ds:
            result[did] = []
            for split_number in split_numbers:
                if return_whole_output:
                    result[did].append(evaluate(self.limit_dict[did], bptt, eval_positions, metric, model, device, method_name=method_name, max_time=max_time, split_number=split_number, jrt_prompt=jrt_prompt, random_premutation=permutation_random))
                else:
                    result[did].append(evaluate(self.limit_dict[did], bptt, eval_positions, metric, model, device, method_name=method_name, max_time=max_time, split_number=split_number, jrt_prompt=jrt_prompt, random_premutation=permutation_random)["mean_metric"].item())
                
        return result
    


    ''' Need to fix this later
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
    '''


if __name__ == "__main__":
    h = EvalHelper(dids="test")
    #h.do_naive_evaluation()
    #h.log_wandb_naive_evaluation(num_steps=200, log_name="mamba_mean_acc")





