import itertools
import yaml


hyper_params = {
    'max_learning_rate': [0.4, 0.6],
    'data_aug_cutout_size': [5, 12],
    'batch_size': [128, 512],
    'momentum': [0.9, 0.99],
    'batch_norm': ['on']
}

keys, values = zip(*hyper_params.items())
hyperparam_allsets = [dict(hyperparam_set=dict(zip(keys, v))) for v in itertools.product(*values)]

print("Total number of hyperparameter sets: "+str(len(hyperparam_allsets)))

with open('hyperparams.yml', 'w') as outfile:
    yaml.dump(hyperparam_allsets, outfile, default_flow_style=False)

print("Hyperparameter sets saved to: "+'hyperparams.yml')


