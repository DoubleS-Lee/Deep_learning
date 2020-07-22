import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

plt.style.use('seaborn')
np.random.seed(0)

##### Start Your Code(Targer Function) #####
t_x = np.linspace(-3, 3, 2)
t_y = 3*t_x
##### End Your Code(Targer Function) #####

fig, ax = plt.subplots(figsize = (7,7))
ax.plot(t_x, t_y, linestyle = ':')

##### Start Your Code(Dataset Generation) #####
n_sample = 100
x_data = np.random.normal(0, 1, (n_sample,))
y_data = 3*x_data + 0.5*np.random.normal(0, 1, (n_sample,))
##### End Your Code(Dataset Generation) #####

# Target Function Visualization
fig, ax = plt.subplots(figsize = (7,7))
ax.plot(t_x, t_y, linestyle = ':')

# Dataset Visualization
ax.scatter(x_data, y_data, color = 'r')
ax.tick_params(axis = 'both', labelsize = 20)
ax.set_title("Dataset", fontsize = 30)
ax.set_xlabel("x data", fontsize = 20)
ax.set_ylabel("y data", fontsize = 20)
