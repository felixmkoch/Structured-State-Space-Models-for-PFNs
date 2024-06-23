from openml import config, study, tasks, runs, extensions, datasets
from tabpfn.scripts.tabular_evaluation import evaluate
from openml.tasks import TaskType
from tabpfn.datasets import load_openml_list

class EvalHelper:

    def __init__(self):

        # Got these datasets from the scripts of tabpfn
        valid_dids_classification = [13, 59, 4, 15, 40710, 43, 1498]
            
        self.datasets_data, _ = load_openml_list(valid_dids_classification)



    def do_evaluation(self, model, bptt, eval_positions, metric, device, method_name):

        result = evaluate(self.datasets_data, bptt, eval_positions, metric, model, device,method_name=method_name)

        return result['mean_metric']
    

    def do_naive_evaluation(self):
        pass

        



if __name__ == "__main__":
    h = EvalHelper()



