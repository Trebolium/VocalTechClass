import pickle, os
import sys

results_path = './results'
model_name = sys.argv[1]

log_file_name = 'config_params.pkl'
search_path = os.path.join(results_path, model_name, log_file_name)

config_variables = pickle.load(open(search_path, 'rb'))
print()
for param in vars(config_variables):
    if len(param) < 7:
        print(param, '\t\t', getattr(config_variables, param))
    else:
        print(param, '\t', getattr(config_variables, param))
print()
