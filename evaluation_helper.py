from openml import config, study, tasks, runs, extensions, datasets
from tabpfn.scripts.tabular_evaluation import evaluate
from tabpfn.scripts import tabular_metrics
from openml.tasks import TaskType
import numpy as np
from tabpfn.datasets import load_openml_list

class EvalHelper:

    def __init__(self, dids="default"):

        # Got these datasets from the scripts of tabpfn
        if dids == "test": valid_dids_classification = [973, 1596, 40981, 1468, 40984, 40975, 41163, 41147, 1111, 41164, 1169, 1486, 41143, 1461, 41167, 40668, 41146, 41169, 41027, 23517, 41165, 41161, 41159, 41138, 1590, 41166, 1464, 41168, 41150, 1489, 41142, 3, 12, 31, 54, 1067]
        else: valid_dids_classification = [13, 59, 4, 15, 40710, 43, 1498]

        self.dids = dids
            
        self.datasets_data, _ = load_openml_list(valid_dids_classification)


    def do_evaluation(self, model, bptt, eval_positions, metric, device, method_name):

        result = evaluate(self.datasets_data, bptt, eval_positions, metric, model, device,method_name=method_name)

        return result['mean_metric']
    


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


if __name__ == "__main__":
    h = EvalHelper()
    h.do_naive_evaluation()



