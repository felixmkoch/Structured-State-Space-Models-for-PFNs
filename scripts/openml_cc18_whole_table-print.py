import openml

# Fetch the list of datasets in the OpenML CC-18 suite
cc18_datasets = openml.study.get_suite('OpenML-CC18').data

# Helper function to get dataset details
def get_dataset_details(dataset_id):
    dataset = openml.datasets.get_dataset(dataset_id)
    qualities = dataset.qualities
    name = dataset.name
    num_instances = int(qualities['NumberOfInstances'])
    num_features = int(qualities['NumberOfFeatures'])
    num_classes = int(qualities['NumberOfClasses'])
    num_nans = int(qualities.get('NumberOfMissingValues', 0))
    min_class_size = int(qualities.get('MinorityClassSize', 0))
    is_num = (not dataset.get_features_by_type("nominal")) and (not dataset.get_features_by_type("date")) and (not dataset.get_features_by_type("string"))
    
    return {
        'id': dataset_id,
        'name': name,
        'num_instances': num_instances,
        'num_features': num_features,
        'num_classes': num_classes,
        'num_nans': num_nans,
        'min_class_size': min_class_size,
        'all_numeric': is_num
    }

# Fetch details for all datasets in CC-18
datasets = [get_dataset_details(ds_id) for ds_id in cc18_datasets]

# Split datasets into groups based on the given criteria
group_1 = []
group_2 = []
group_3 = []

for ds in datasets:
    if ds['num_instances'] <= 2000 and ds['num_features'] <= 100 and ds['num_classes'] <= 10:
        if ds['all_numeric']:
            group_1.append(ds)
        else:
            group_2.append(ds)
    else:
        group_3.append(ds)

# Sort each group by Dataset ID
group_1.sort(key=lambda x: x['id'])
group_2.sort(key=lambda x: x['id'])
group_3.sort(key=lambda x: x['id'])

# Print the datasets in the desired format
def print_datasets(datasets):
    for ds in datasets:
        print(f"{ds['id']} & {ds['name']} & {ds['num_instances']} & {ds['num_features']} & {ds['num_classes']} & {ds['num_nans']} & {ds['min_class_size']} \\\\")

print("Group 1: (<= 2000 instances, <= 100 features, <= 10 classes, all numerical values)")
print_datasets(group_1)
print("Group 2: (<= 2000 instances, <= 100 features, <= 10 classes, not all numerical values)")
print_datasets(group_2)
print("Group 3: (rest of the datasets)")
print_datasets(group_3)