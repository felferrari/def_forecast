from pydoc import locate
from copy import deepcopy
from datasets.vector import DataModule
from utils.ops import deepupdate

def build(experiments, run_name):
    #run_description = experiments['default'] | experiments[run_name]
    run_description = deepupdate(experiments['default'], experiments[run_name])

    if 'parent_experiment' in run_description.keys():
        #run_description =  experiments[run_description['parent_experiment']] | run_description
        run_description =  deepupdate(experiments[run_description['parent_experiment']], run_description)
        
    run_description['data_module']['params']['features_list'] =  run_description['features_list']
    
    data_module = locate(run_description['data_module']['class_path'])(**run_description['data_module']['params'])
    
    if not 'params' in run_description['model'].keys():
        model = locate(run_description['model']['class_path'])()
    else:
        run_description['model']['params']['input_sample'] = data_module.train_dataloader().dataset[0]
        model = locate(run_description['model']['class_path'])(**run_description['model']['params'])
    
    criterion = locate(run_description['criterion']['class_path'])(**run_description['criterion']['params'])
    
    optimizer = locate(run_description['optimizer']['class_path'])
    
    optimizer_params = run_description['optimizer']['params']
    
    train_params = run_description['train_params']
    
    predict_params = run_description['predict_params']
        
    return run_description, model, data_module, criterion, optimizer, optimizer_params, train_params, predict_params