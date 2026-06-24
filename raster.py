import numpy as np
import json
import tqdm
import pickle as pkl
import multiprocessing as mp

from model import Model

scenario = 'no_juv_trans_50'

with open('rasters.json', 'r') as data:
	param_set = json.load(data)[scenario]

#Raster parameters
mode = 'path'
output_path = 'data/' + scenario + '_' + mode + '.p'
size = 10								#Raster dimension
N_iter = 200                            #Evolutionary iterations
cores = 8								#Number of CPU cores

#Simulation parameters
var_1 = param_set['var_1']				#First parameter rastered
var_2 = param_set['var_2']				#Second parameter rastered

var_1_vals = np.linspace(param_set['var_1_vals'][0], param_set['var_1_vals'][1], size)
var_2_vals = np.linspace(param_set['var_2_vals'][0], param_set['var_2_vals'][1], size)

def pass_to_sim(model):
	return model.run_sim()

if __name__ == '__main__':
	coords = []     #x, y coordinates of each simulation in raster
	models = []     #Empty tuple for model classes

	print('Running Raster: ' + scenario)

	#Create raster of model classes for each parameter combination
	print('Initializing Models...')
	for i in range(size):
		for j in range(size):
			coords.append((i,j))
			var_params = {param_set['var_1']: var_1_vals[i], param_set['var_2']: var_2_vals[j]}
			params = var_params | param_set['params'] | {'N_iter': N_iter, 'mode': mode}
			new_model = Model(**params)

			models.append(new_model)
		
	#Run simluations for 4 core processor
	pool = mp.Pool(processes=cores)	
	
	print('Running Simulations...')
	results = []
	for result in tqdm.tqdm(pool.imap(pass_to_sim, models), total=len(models)):
		results.append(result)

	raster = []
	for i in range(size):
		inds = [j for j in range(len(coords)) if coords[j][1] == i]
		raster.append([results[j] for j in inds])

	with open(output_path, 'wb') as f:
		pkl.dump([models, results], f)