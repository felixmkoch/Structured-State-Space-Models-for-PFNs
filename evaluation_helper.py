from openml import config, study, tasks, runs, extensions, datasets
from tabpfn.scripts.tabular_evaluation import evaluate
from openml.tasks import TaskType

class EvalHelper:

    def __init__(self):
        
        # Task dids from all bench suite (271) that are not contained in
        #all_bench_dids = [2073, 3945, 7593, 10090, 146818, 168350, 168757, 168784, 168868, 168909, 168910, 168911, 189354, 189355, 189356, 189922, 190137, 190146, 190392, 190410, 190411, 190412, 211979, 211986, 359953, 359954, 359955, 359956, 359957, 359958, 359959, 359960, 359961, 359962, 359963, 359964, 359965, 359966, 359967, 359968, 359969, 359970, 359971, 359972, 359973, 359974, 359975, 359976, 359977, 359979, 359980, 359981, 359982, 359983, 359984, 359985, 359986, 359987, 359988, 359989, 359990, 359991, 359992, 359993, 359994, 360112, 360113, 360114, 360975]
        # TaskIDs from OpenML with constraints (num_features, num_instances, num_classes) and excluded from cc18:
        #       [3492, 3493, 3494, 3512, 3543, 3561, 9980, 14968, 125921]    
        print("Start of evaluation helper init")

        #filtered_tasks = tasks.list_tasks(tag="OpenML100", output_format="dataframe")

        #cc18_tasks = study.get_suite(99).tasks

        #filtered_tasks = filtered_tasks[filtered_tasks['NumberOfNumericFeatures'] <= 100]
        #filtered_tasks = filtered_tasks[filtered_tasks['NumberOfInstances'] <= 1000]
        #filtered_tasks = filtered_tasks[filtered_tasks['NumberOfClasses'] <= 10]
        #filtered_tasks = filtered_tasks[~filtered_tasks['tid'].isin(cc18_tasks)]

        task_ids = [3492, 3493, 3494, 3512, 3543, 3561, 9980, 14968, 125921]

        task_list = tasks.get_tasks(task_ids)

        datasets = [task.get_dataset() for task in task_list]

        ds_data = []

        for d in datasets:
            X, y, categorical_indicator, attribute_names = d.get_data(target=d.default_target_attribute)
            name = d.name
            categorical_features = [attribute_names[i] for i in range(len(attribute_names)) if categorical_indicator[i]]

            ds_data.append((name, X, y, categorical_features, None, None))

        self.datasets_data = ds_data



    def do_evaluation(self, model, bptt, eval_positions, metric, device, method_name):

        result = evaluate(self.datasets_data, bptt, eval_positions, metric, model, device,method_name=method_name)

        return result['mean_metric']
        






if __name__ == "__main__":
    h = EvalHelper()



