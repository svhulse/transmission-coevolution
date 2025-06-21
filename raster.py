import numpy as np
import tqdm
import pickle as pkl
import multiprocessing as mp

from model import Model

#Raster parameters
output_path = 'host.p'			#Name of raster scenario
size = 10						#Raster dimension
cores = 8						#Number of CPU cores

#Simulation parameters
var_1 = 'm'						#First parameter rastered
var_2 = 'z'						#Second parameter rastered
mode = 'host'
N_iter = 200

def pass_to_sim(model):
	return model.run_sim()

if __name__ == '__main__':
	coords = []     #x, y coordinates of each simulation in raster
	models = []     #Empty tuple for model classes

	m_vals = np.linspace(0.25, 1, size)
	z_vals = np.linspace(1, 10, size)

	#Create raster of model classes for each parameter combination
	print('Initializing Models...')
	for i in range(size):
		for j in range(size):
			coords.append((i,j))
			params = {'m': m_vals[i], 'z': z_vals[j], 'N_iter': N_iter, 'mode': mode}
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