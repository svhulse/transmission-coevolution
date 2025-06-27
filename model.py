import numpy as np
from scipy.integrate import solve_ivp

class Model:
	def __init__(self, **kwargs):
		self.H_alleles = 100 	#Number of host alleles
		self.P_alleles = 100 	#Number of pathogen alleles
		self.N_iter = 100 		#Number of evolutionary time steps
		self.evol_range = 2		#Base for tradeoff function

		self.mode = 'coevol' 	#Can be set to coevol, path, or host

		self.b = 1					#Birth rate
		self.mu = 0.2				#Death rate
		self.k = 0.001				#Coefficient of density-dependent growth
		self.beta = 0.001			#Baseline transmission rate
		self.m = 0.5				#Maturation rate
		self.z = 1					#Adult resistance bias
		self.assort = 1				#Age-based assortativity

		for key, value in kwargs.items():
			setattr(self, key, value)

		self.resJ = np.logspace(-1, 1, self.H_alleles, base=self.evol_range)
		self.infJ = np.logspace(-1, 1, self.P_alleles, base=self.evol_range)
		self.resA = 1/(self.resJ*self.z)
		self.infA = 1/self.infJ
		self.infJ *= self.beta
		self.infA *= self.beta

	def mutation(self, genotypes, mut=0.05):
		N_alleles = len(genotypes)

		M = np.diag(np.full(N_alleles, 1 - mut))
		M = M + np.diag(np.ones(N_alleles - 1)*mut/2, 1)
		M = M + np.diag(np.ones(N_alleles - 1)*mut/2, -1)
		M[0,1] = mut
		M[N_alleles - 1, N_alleles - 2] = mut

		return np.dot(M, genotypes)

	#Define the dynamical system	
	def df(self, t, X):
		Sj = X[:self.H_alleles]
		Sa = X[self.H_alleles:2*self.H_alleles] 
		Ij = X[2*self.H_alleles:2*self.H_alleles+self.P_alleles]
		Ia = X[2*self.H_alleles+self.P_alleles:]

		N = np.sum(Sa) + np.sum(Ia)
		dSj = Sa*self.b - Sj*(self.mu + self.m + self.k*N + self.resJ*np.dot(self.infJ, Ia/self.assort + Ij*self.assort))
		dSa = Sj*self.m - Sa*(self.mu + self.resA*np.dot(self.infA, Ia*self.assort + Ij/self.assort))
		dIj = (Ia/self.assort + Ij*self.assort)*(np.dot(self.resJ, Sj)*self.infJ) - Ij*(self.mu + self.m + self.k*N)
		dIa = (Ia*self.assort + Ij/self.assort)*(np.dot(self.resA, Sa)*self.infA - self.mu) + Ij*self.m
		
		X_out = np.concatenate((dSj, dSa, dIj, dIa))

		return X_out

	#Run simulation
	def run_sim(self):
		#Define initial conditions
		Sj_0 = np.zeros(self.H_alleles)
		Sa_0 = np.zeros(self.H_alleles)
		Ij_0 = np.zeros(self.P_alleles)
		Ia_0 = np.zeros(self.P_alleles)

		Sj_0[50] = 100
		Sa_0[50] = 200
		Ij_0[50] = 10
		Ia_0[50] = 10

		X_0 = np.concatenate((Sj_0, Sa_0, Ij_0, Ia_0))

		Sj_eq = np.zeros((self.H_alleles, self.N_iter))
		Sa_eq = np.zeros((self.H_alleles, self.N_iter))
		Ij_eq = np.zeros((self.P_alleles, self.N_iter))
		Ia_eq = np.zeros((self.P_alleles, self.N_iter))

		t = (0, 1500)
		zero_threshold = 0.1 #Threshold to set abundance values to zero

		for i in range(self.N_iter):
			sol = solve_ivp(self.df, t, X_0)
			
			Sj_eq[:, i] = sol.y[:self.H_alleles, -1]
			Sa_eq[:, i] = sol.y[self.H_alleles:2*self.H_alleles, -1]
			Ij_eq[:, i] = sol.y[2*self.H_alleles:2*self.H_alleles+self.P_alleles, -1]
			Ia_eq[:, i] = sol.y[2*self.H_alleles+self.P_alleles:, -1]

			#Set any population below threshold to 0
			Sj_eq[:, i][Sj_eq[:,i] < zero_threshold] = 0
			Sa_eq[:, i][Sa_eq[:,i] < zero_threshold] = 0
			Ij_eq[:, i][Ij_eq[:,i] < zero_threshold] = 0
			Ia_eq[:, i][Ia_eq[:,i] < zero_threshold] = 0

			#Assign the values at the end of the ecological simulation to the 
			#first value so the simulation can be re-run
			
			if self.mode == 'coevol' or self.mode == 'host':
				Sj_0 = self.mutation(Sj_eq[:, i])
				Sa_0 = self.mutation(Sa_eq[:, i])
			else:
				Sj_0 = Sj_eq[:, i]
				Sa_0 = Sa_eq[:, i]

			if self.mode == 'coevol' or self.mode == 'path':
				Ij_0 = self.mutation(Ij_eq[:, i])
				Ia_0 = self.mutation(Ia_eq[:, i])
			else:
				#Ij_0 = Ij_eq[:, i] > 1
				#Ia_0 = Ia_eq[:, i] > 1
				Ij_0 = Ij_eq[:, i]
				Ia_0 = Ia_eq[:, i]

			X_0 = np.concatenate((Sj_0, Sa_0, Ij_0, Ia_0))

		return (Sj_eq, Sa_eq, Ij_eq, Ia_eq)