import numpy as np

class LR_dataset_generator:
    def __init__(self, feature_dim, n_sample = 100, noise = 0):
        self._feature_dim = feature_dim
        self._n_sample = n_sample
        self._noise = noise
        
        self._coefficient_list = None
        self._distribution_params = None
        
        self._dataset = None
        
        self._init_coefficient()
        self._init_distribution_params()
        
    def _init_coefficient(self):
        self._coefficient_list = [0] + [1 for _ in range(self._feature_dim)]
        
    def _init_distribution_params(self):
        self._distribution_params = {f:{'mean':0, 'std':1}
                                     for f in range(1, self._feature_dim+1)}
    
    def make_dataset(self):
        x_data = np.zeros(shape = (self._n_sample, 1))
        y_data = np.zeros(shape = (self._n_sample, 1))
        
        for f_idx in range(1, self._feature_dim + 1):
            feature_data = np.random.normal(loc = self._distribution_params[f_idx]['mean'], scale = self._distribution_params[f_idx]['std'], size = (self._n_sample, 1))
            x_data = np.hstack((x_data, feature_data))
            y_data += self._coefficient_list[f_idx]*feature_data
        y_data += self._coefficient_list[0]
        
        self._dataset = np.hstack((x_data, y_data))
        return self._dataset
    
    def set_n_sample(self, n_sample):
        self._n_sample = n_sample
    
    def set_noise(self, noise):
        self._noise = noise
    
    def set_distribution_params(self, distribution_params):
        for param_key, param_value in distribution_params.items():
            self._distribution_params[param_key] = param_value
    
    def set_coefficient(self, coefficient_list):
        self._coefficient_list = coefficient_list




# Before Customizing
data_gen = LR_dataset_generator(feature_dim = 3)

dataset = data_gen.make_dataset()
x_data, y_data = dataset[:,:-1], dataset[:,-1].reshape(-1,1)

for feature_idx in range(1, 4):
    print(np.mean(x_data[:,feature_idx]))
    print(np.std(x_data[:,feature_idx]), '\n')

# After Customizing
n_sample = 1000
distribution_params = {
    1:{'mean':1, 'std':2},
    2:{'mean':2, 'std':3}}
noise = 0.5

data_gen.set_distribution_params(distribution_params)
data_gen.set_n_sample(n_sample)
data_gen.set_noise(noise)
dataset = data_gen.make_dataset()

x_data, y_data = dataset[:,:-1], dataset[:,-1].reshape(-1,1)
print("====== After Customizing =====")
for feature_idx in range(1, 4):
    print(np.mean(x_data[:,feature_idx]))
    print(np.std(x_data[:,feature_idx]), '\n')