import json

def print_dict_recursive(dictionary, indent=0):
    for key, value in dictionary.items():
        if isinstance(value, dict):
            print('  ' * indent + f"{key}:")
            print_dict_recursive(value, indent + 1)
        else:
            print('  ' * indent + f"{key}: {value}")

# Read the contents of the config.json file
with open('config.json', 'r') as json_file:
    config_data = json.load(json_file)

                        
# Print the contents of the config.json file
print("Contents of config.json:")
print_dict_recursive(config_data)
