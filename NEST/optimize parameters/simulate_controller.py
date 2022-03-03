# -*- coding: utf-8 -*-
"""
Created on Feb 2022
@author: benedetta gambosi - benedetta.gambosi@polimi.it
"""
import os
import subprocess
import json
import numpy as np

path =  "/home/benedetta/controller_versions/controller_tuning/complete_control/"
params_file = "params.json"
error_file = "error_xy.txt"
tuning_file = "tuning_params.json"
new_params_file = "new_params.json"
def update_weights(controller_parameters, path_saving_files):
  
    path_params_file = path + params_file
    f = open(path_params_file,"rb")
    params = json.load(f)
    f.close()
    
    path_tuning_file = path + tuning_file
    f = open(path_tuning_file,"rb")
    params_tuning = json.load(f)
    f.close()
    
    i = 0
    for k in params_tuning.keys():
        items = params_tuning[k].items()
        for j in items:
            if isinstance(j[1], list):
                for l in j[1]:
                    params[k][j[0]][l] = controller_parameters[i]
                    i+=1
            else:
                params[k][j[0]][j[1]] = controller_parameters[i]
                i+=1
    params["path"] = path_saving_files + '/'
    path_new_params_file = path + new_params_file
    f = open(path_new_params_file,"w")
    json.dump(params, f, indent =6)
    f.close()
    
def get_errors():
  
    path_file = path + error_file
    f = open(path_file,"rb")
    error = np.loadtxt(f)
    f.close()
    
    return np.abs(error[0]),np.abs(error[1]),np.abs(error[1]) 
    
def simulate_controller(controller_parameters, path_saving_files):
    # TODO #2 minimaze maen and std on multiple trials as fitness 
    import os
    import subprocess
    
    update_weights(controller_parameters, path_saving_files)
    _ = subprocess.run(['sh', path+'run_controller.sh'], stdout=subprocess.PIPE)
    
    error_x, error_y,error_all = get_errors()

    return  error_x, error_y, error_all



if __name__ == '__main__':
    import sys

    path_file = path + params_file
    f = open(path_file,"rb")
    params = json.load(f)
    f.close()



    # weights_parameters = [params[], params[], params[], params[], params[], params[], params[], params[]]

    print("Initial weights: ",weights_parameters)

    simulate_controller(weights_parameters, "./")
