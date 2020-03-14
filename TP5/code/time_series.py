from model import RosslerModel, train_model
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Rossler:
    def __init__(self, delta_t):
        self.delta_t = delta_t #if discrete model your delta_t
                              #if continuous model chose one <=1e-2
        self.nb_steps = 100000

        self.rosler_nn = train_model(print_debug=True)

    def full_traj(self,initial_condition=np.array([-5.75, -1.6,  0.02])): 
        # run your model to generate the time series with nb_steps
        # just the y cordinate is necessary.
        y = [initial_condition]
        X = torch.FloatTensor(initial_condition)
        with torch.no_grad():
            for t in range(int(self.nb_steps)):
                X = self.rosler_nn(X)
                y.append(X.detach().numpy())
        return np.array(y)

    def plot_traj(self, y):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(y[:,0], y[:,1], y[:,2])

    def save_traj(self,y):
        pass
        #save the trajectory in y.dat file 
    
if __name__ == '__main__':

    delta_t = 1e-2

    ROSSLER = Rossler(delta_t)

    y = ROSSLER.full_traj()

    ROSSLER.save_traj(y)

