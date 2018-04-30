from sklearn.model_selection import ParameterGrid
import json
import os

DIR = 'configs'

parameter_dict = {
  "version": [4],
  "num_epochs": [48],
  "num_iter_per_epoch": [84],
  "learning_rate": [2e-4],
  "batch_size": [6],
  "state_size": [[192, 192, 1]],
  "max_to_keep": [1],
  "is_training": [1],
  "use_aug": [1],
  "activation": ['prelu'],
  "pooling": ['avg', 'max'],
  "target": ['Media'],
  "run": range(1, 6)
}

grid = sorted(list(ParameterGrid(parameter_dict)), key=lambda x: x['target'])

try:  # create the DIR if not existeds
    os.stat(DIR)
except:
    os.mkdir(DIR)

for i in grid:
    i['exp_name'] = 'V{}-Paper-ImageSize{}-Epoch{}-Iter{}-LR{}-BS{}-{}-{}-{}-{}'.format(
        i['version'],
        i['state_size'][0],
        i['num_epochs'], 
        i['num_iter_per_epoch'], 
        i['learning_rate'],
        i['batch_size'],
        i['activation'],
        i['pooling'],
        i['target'],
        i['run']
        )

    with open('./configs/{}.json'.format(i['exp_name']), 'w') as f:
        print('./configs/{}.json'.format(i['exp_name']))
        json.dump(i, f)

    with open('./mains/run.sh', 'a') as f:
        f.write('python3.5 main.py -c ../configs/{}.json\n'.format(i['exp_name']))

