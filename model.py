import numpy as np
from scipy.integrate import solve_ivp

class Model:
	def __init__(self, **kwargs):
		self.H_alleles = 100 	#Number of host alleles
		self.P_alleles = 100 	#Number of pathogen alleles
		self.N_iter = 100 		#Number of evolutionary time steps
		self.evol_range = 2		#Base for tradeoff function

		self.fast_path = False	#Doubles the pathogen evolution rate
		self.mode = 'coevol' 	#Can be set to coevol, path, or host
		self.tradeoff = 'log'	#Can be log or linear
		self.juv_trans = False	#Set whether juveniles transmit the pathogen

		self.b = 1					#Birth rate
		self.mu = 0.2				#Death rate
		self.k = 0.001				#Coefficient of density-dependent growth
		self.beta = 0.001			#Baseline transmission rate
		self.m = 0.5				#Maturation rate
		self.z = 1					#Adult resistance bias
		self.alpha = 0				#Induced mortality

		#Set inital condition
		self.Sj_gtp0 = 50
		self.Sa_gtp0 = 50
		self.Ij_gtp0 = 50
		self.Ia_gtp0 = 50

		for key, value in kwargs.items():
			setattr(self, key, value)

		self.Sj_0 = np.zeros(self.H_alleles)
		self.Sa_0 = np.zeros(self.H_alleles)
		self.Ij_0 = np.zeros(self.P_alleles)
		self.Ia_0 = np.zeros(self.P_alleles)
		
		self.Sj_0[self.Sj_gtp0] = 100
		self.Sa_0[self.Sa_gtp0] = 200
		self.Ij_0[self.Ij_gtp0] = 10
		self.Ia_0[self.Ia_gtp0] = 20

		if self.tradeoff == 'log':
			self.resJ = np.logspace(-1, 1, self.H_alleles, base=self.evol_range)
			self.infJ = np.logspace(-1, 1, self.P_alleles, base=self.evol_range)
			self.resA = 1/(self.resJ*self.z)
			self.infA = 1/self.infJ

		if self.tradeoff == 'linear':
			self.resJ = np.linspace(0.5, 1.5, self.H_alleles)
			self.infJ = np.linspace(0.5, 1.5, self.P_alleles)
			self.resA = np.linspace(1.5, 0.5, self.H_alleles) / self.z
			self.infA = np.linspace(1.5, 0.5, self.P_alleles)
		
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
		if self.juv_trans:
			Sj = X[:self.H_alleles]
			Sa = X[self.H_alleles:2*self.H_alleles] 
			Ij = X[2*self.H_alleles:2*self.H_alleles+self.P_alleles]
			Ia = X[2*self.H_alleles+self.P_alleles:]

			N = np.sum(Sa) + np.sum(Ia)
			dSj = Sa*self.b - Sj*(self.mu + self.m + self.k*N + self.resJ*np.dot(self.infJ, Ia + Ij))
			dSa = Sj*self.m - Sa*(self.mu + self.resA*np.dot(self.infA, Ia + Ij))
			dIj = (Ia + Ij)*(np.dot(self.resJ, Sj)*self.infJ) - Ij*(self.mu + self.alpha + self.m + self.k*N)
			dIa = (Ia + Ij)*(np.dot(self.resA, Sa)*self.infA) - Ia*(self.mu + self.alpha) + Ij*self.m
			
			X_out = np.concatenate((dSj, dSa, dIj, dIa))

			return X_out			
		
		if not self.juv_trans:
			Sj = X[:self.H_alleles]
			Sa = X[self.H_alleles:2*self.H_alleles] 
			Ij = X[2*self.H_alleles:2*self.H_alleles+self.P_alleles]
			Ia = X[2*self.H_alleles+self.P_alleles:]

			N = np.sum(Sa) + np.sum(Ia)
			dSj = Sa*self.b - \
				Sj*(self.mu + self.m + self.k*N + self.resJ*np.dot(self.infJ, Ia))
			dSa = Sj*self.m - \
				Sa*(self.mu + self.resA*np.dot(self.infA, Ia))
			dIj = Ia*(np.dot(self.resJ, Sj)*self.infJ) - \
				Ij*(self.mu + self.m + self.k*N)
			dIa = Ia*(np.dot(self.resA, Sa)*self.infA - self.mu) + \
				Ij*self.m

			X_out = np.concatenate((dSj, dSa, dIj, dIa))

			return X_out

	#Run simulation
	def run_sim(self):
		#Define initial conditions
		X_0 = np.concatenate((self.Sj_0, self.Sa_0, self.Ij_0, self.Ia_0))
		zero_threshold = 0.1 #Threshold to set abundance values to zero

		if self.fast_path:
			t = (0, 750)

			Sj_eq = np.zeros((self.H_alleles, self.N_iter*2))
			Sa_eq = np.zeros((self.H_alleles, self.N_iter*2))
			Ij_eq = np.zeros((self.P_alleles, self.N_iter*2))
			Ia_eq = np.zeros((self.P_alleles, self.N_iter*2))
			
			for i in range(self.N_iter*2):
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
				
				if i%2 == 1:
					Ij_0 = self.mutation(Ij_eq[:, i])
					Ia_0 = self.mutation(Ia_eq[:, i])
				else:
					Sj_0 = self.mutation(Sj_eq[:, i])
					Sa_0 = self.mutation(Sa_eq[:, i])
					Ij_0 = self.mutation(Ij_eq[:, i])
					Ia_0 = self.mutation(Ia_eq[:, i])

				X_0 = np.concatenate((Sj_0, Sa_0, Ij_0, Ia_0))	

			Sj_eq = Sj_eq[:, ::2]
			Sa_eq = Sa_eq[:, ::2]
			Ij_eq = Ij_eq[:, ::2]
			Ia_eq = Ia_eq[:, ::2]

		else:
			t = (0, 1500)

			Sj_eq = np.zeros((self.H_alleles, self.N_iter))
			Sa_eq = np.zeros((self.H_alleles, self.N_iter))
			Ij_eq = np.zeros((self.P_alleles, self.N_iter))
			Ia_eq = np.zeros((self.P_alleles, self.N_iter))

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
					Ij_0 = Ij_eq[:, i]
					Ia_0 = Ia_eq[:, i]

				X_0 = np.concatenate((Sj_0, Sa_0, Ij_0, Ia_0))

		return (Sj_eq, Sa_eq, Ij_eq, Ia_eq)