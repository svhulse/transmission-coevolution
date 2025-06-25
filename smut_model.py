import numpy as np

from scipy.integrate import solve_ivp

class SmutModel:
	def __init__(self, **kwargs):
		self.H_alleles = 100 #number of alleles
		self.P_alleles = 100 #number of alleles
		self.N_iter = 100 #number of evolutionary time steps

		self.mode = 'coevol' 	#Can be set to coevol, path, or host

		#Set default parameters and resistance-cost curve
		self.b = 1				#Birth rate
		self.mu = 0.2			#Death rate
		self.k = 0.001			#Coefficient of density-dependent growth
		self.beta_j = 0.001		#Baseline density-dependent transmission rate for junveniles
		self.beta_a = 1			#Baseline frequency-dependent transmission rate for adults
		self.m = 0.5			#Maturation rate

		for key, value in kwargs.items():
			setattr(self, key, value)

		self.resJ = np.logspace(-1, 1, self.H_alleles, base=3)
		self.infJ = np.logspace(-1, 1, self.P_alleles, base=3)
		self.resA = 1/self.resJ
		self.infA = 1/self.infJ
		self.infJ *= self.beta_j
		self.infA *= self.beta_a

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

		dSj = Sa*self.b - Sj*(self.mu + self.m + self.k*N + self.resJ*np.dot(self.infJ, Ia))
		dSa = Sj*self.m - Sa*(self.mu + self.resA*np.dot(self.infA, Ia/N))
		dIj = Ia*(np.dot(self.resJ, Sj)*self.infJ) - Ij*(self.mu + self.m + self.k*N)
		dIa = (Ia/N)*np.dot(self.resA, Sa)*self.infA - Ia*self.mu + Ij*self.m
		
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
				Ij_0 = Ij_eq[:, i]
				Ia_0 = Ia_eq[:, i]

			X_0 = np.concatenate((Sj_0, Sa_0, Ij_0, Ia_0))

		return (Sj_eq, Sa_eq, Ij_eq, Ia_eq)
	
class ModelAnalysis:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.model = SmutModel(**kwargs)
        self.Sj, self.Sa, self.Ij, self.Ia = self.model.run_sim()

    def inf_prev(self):
        prev = (np.sum(self.Ij, axis=0) + np.sum(self.Ia, axis=0)) / (np.sum(self.Ij, axis=0) + np.sum(self.Ia, axis=0) + np.sum(self.Sj, axis=0) + np.sum(self.Sa, axis=0))

        return prev

    def juv_prop(self):
        Sj_sum = np.sum(self.Sj, axis=0)
        Sa_sum = np.sum(self.Sa, axis=0)
        Ij_sum = np.sum(self.Ij, axis=0)
        Ia_sum = np.sum(self.Ia, axis=0)

        return (Sj_sum + Ij_sum) / (Sj_sum + Sa_sum + Ij_sum + Ia_sum)        
    
    def force_of_infection(self):
        beta_j = np.zeros(self.Sj.shape[1])
        beta_a = np.zeros(self.Sa.shape[1])
		
        N = np.sum(self.Sa, axis=0) + np.sum(self.Ia, axis=0) 

        for i in range(self.Sj.shape[1]):
            beta_j[i] = np.dot(self.Sj[:, i], self.model.resJ*np.dot(self.model.infJ, self.Ia[:, i])) / np.sum(self.Sj[:, i])
            beta_a[i] = np.dot(self.Sa[:, i], self.model.resA*np.dot(self.model.infA, self.Ia[:, i])/N[i]) / np.sum(self.Sa[:, i])
        
        return beta_j, beta_a  
    
    def foi_bias(self):
        bias = np.log10(self.force_of_infection()[0] / self.force_of_infection()[1])

        return bias