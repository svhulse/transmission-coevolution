import numpy as np
import pickle as pkl

from PIL import Image

from model import Model

class RasterAnalysis:
    def __init__(self, fname, **kwargs):

        with open(fname, 'rb') as f:
            self.models, self.results = pkl.load(f)

        self.arg_x = 'm'
        self.arg_y = 'z'
        
        for key, value in kwargs.items():
            setattr(self, key, value)

        x_vals = np.zeros(len(self.models))
        y_vals = np.zeros(len(self.models))
        
        for i, model in enumerate(self.models):
            x_vals[i] = getattr(model, self.arg_x)
            y_vals[i] = getattr(model, self.arg_y)
			
        self.x_vals = np.sort(list(set(x_vals)))
        self.y_vals = np.sort(list(set(y_vals)))

        self.dim_x = len(self.x_vals)
        self.dim_y = len(self.y_vals)
        
    def evol_plot(self):
        host_evol, path_evol = np.zeros((2, self.dim_x, self.dim_y))

        for i, model in enumerate(self.models):
            x_ind = np.where(self.x_vals == getattr(model, self.arg_x))
            y_ind = np.where(self.y_vals == getattr(model, self.arg_y))

            _, Sa, _, Ia = self.results[i]

            host_evol[y_ind, x_ind] = np.dot(Sa[:,-1], range(Sa.shape[0])) / np.sum(Sa[:, -1])
            path_evol[y_ind, x_ind] = np.dot(Ia[:,-1], range(Ia.shape[0])) / np.sum(Ia[:, -1])
		
        return host_evol, path_evol
    
    def abundance(self):
        Sj_raster, Sa_raster, Ij_raster, Ia_raster = np.zeros((4, self.dim_x, self.dim_y))
        
        for i, model in enumerate(self.models):
            x_ind = np.where(self.x_vals == getattr(model, self.arg_x))
            y_ind = np.where(self.y_vals == getattr(model, self.arg_y))
            
            Sj, Sa, Ij, Ia = self.results[i]
            
            Sj_raster[y_ind, x_ind] = np.sum(Sj[:, -1])
            Sa_raster[y_ind, x_ind] = np.sum(Sa[:, -1])
            Ij_raster[y_ind, x_ind] = np.sum(Ij[:, -1])
            Ia_raster[y_ind, x_ind] = np.sum(Ia[:, -1])
        
            
        return Sj_raster, Sa_raster, Ij_raster, Ia_raster
    
    def mean_transmission(self):
        beta_j, beta_a = np.zeros((2, self.dim_x, self.dim_y))

        for i, model in enumerate(self.models):
            x_ind = np.where(self.x_vals == getattr(model, self.arg_x))
            y_ind = np.where(self.y_vals == getattr(model, self.arg_y))

            Sj, Sa, Ij, Ia = self.results[i]

            beta_j[y_ind, x_ind] = np.dot(Sj[:, -1], model.resJ*np.dot(model.infJ, Ij[:, -1] + Ia[:, -1])) / np.sum(Sj[:, -1])
            beta_a[y_ind, x_ind] = np.dot(Sa[:, -1], model.resA*np.dot(model.infA, Ij[:, -1] + Ia[:, -1])) / np.sum(Sa[:, -1])
            
        return beta_j, beta_a
  
    def prevalence(self):
        prevalence = np.zeros((self.dim_x, self.dim_y))

        for i, model in enumerate(self.models):
            x_ind = np.where(self.x_vals == getattr(model, self.arg_x))
            y_ind = np.where(self.y_vals == getattr(model, self.arg_y))

            Sj, Sa, Ij, Ia = self.results[i]

            prevalence[y_ind, x_ind] = np.sum(Ij[:, -1] + Ia[:, -1]) / np.sum(Sj[:, -1] + Sa[:, -1] + Ij[:, -1] + Ia[:, -1])
            
        return prevalence

    def label_axes(self, ax, n_ticks, label_x=True, label_y=True, x_string=r'maturation rate ($m$)', y_string=r'adult resistance advantage ($z$)'):        
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        
        x_ticks = np.linspace(x_lim[0], x_lim[1], n_ticks[0])
        y_ticks = np.linspace(y_lim[0], y_lim[1], n_ticks[1])
        
        x_tick_labels = np.round(np.linspace(np.min(self.x_vals), np.max(self.x_vals), n_ticks[0]), 3)
        y_tick_labels = np.round(np.linspace(np.min(self.y_vals), np.max(self.y_vals), n_ticks[1]), 3)
        
        ax.set_xticks(x_ticks, x_tick_labels)
        ax.set_yticks(y_ticks, y_tick_labels)
        
        if label_x: ax.set_xlabel(x_string)
        if label_y: ax.set_ylabel(y_string)
        
    def plot_transition(self):
        _, I = self.evol_plot()
        vals = (np.min(I), np.max(I))

        boundary_vals = np.zeros(I.shape[0])

        for i in range(I.shape[0]):
            base_val = np.argmin(np.abs(vals - I[i, 0]))

            for j in range(I.shape[1]):
                if np.argmin(np.abs(vals - I[i, j])) != base_val:
                    boundary_vals[i] = j
                    break;

        return boundary_vals
        
class ModelAnalysis:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.model = Model(**kwargs)
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

        for i in range(self.Sj.shape[1]):
            beta_j[i] = np.dot(self.Sj[:, i], self.model.resJ*np.dot(self.model.infJ, self.Ij[:, i] + self.Ia[:, i])) / np.sum(self.Sj[:, i])
            beta_a[i] = np.dot(self.Sa[:, i], self.model.resA*np.dot(self.model.infA, self.Ij[:, i] + self.Ia[:, i])) / np.sum(self.Sa[:, i])
        
        return beta_j, beta_a  
    
    def foi_bias(self):
        bias = np.log10(self.force_of_infection()[0] / self.force_of_infection()[1])

        return bias
    
def make_image(Sa, Ia):
    Sa_img = Sa / np.max(Sa)
    Ia_img = Ia / np.max(Ia)
    Sa_img = np.array(Image.fromarray(Sa).resize(size=(100,100)))
    Ia_img = np.array(Image.fromarray(Ia).resize(size=(100,100)))
    Sa_img[Sa_img >= 0.2] = 1
    Sa_img[Sa_img < 0.2] = 0
    Ia_img[Ia_img >= 0.2] = 1
    Ia_img[Ia_img < 0.2] = 0

    Sa_alpha = np.ones(Sa_img.shape)*0.9
    Ia_alpha = np.ones(Ia_img.shape)*0.9
    Sa_alpha[Sa_img == 0] = 0
    Ia_alpha[Ia_img == 0] = 0

    return (Sa_img, Ia_img, Sa_alpha, Ia_alpha)
