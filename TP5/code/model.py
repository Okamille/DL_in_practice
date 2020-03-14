import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from copy import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

from rossler_map import RosslerMap

class RosslerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(3, 100)
        self.h1 = nn.Linear(100, 200)
        self.h2 = nn.Linear(200, 300)
        self.output_layer = nn.Linear(300, 3)

    def forward(self, input):
        out = F.relu(self.input_layer(input))
        out = F.relu(self.h1(out))
        out = F.relu(self.h2(out))
        out = self.output_layer(out)
        return out

def generate_trajectory(Niter = 100000, delta_t=1e-2, x_init=np.array([-5.75, -1.6,  0.02])):
    ROSSLER_MAP = RosslerMap(delta_t=delta_t)
    traj,t = ROSSLER_MAP.full_traj(Niter, x_init)
    return traj, t


def plot_trajectory(model, traj_size, epoch, init_val, eval_traj):
    predicted_traj = np.zeros((traj_size, 3), dtype=float)
    predicted_traj[0] = traj[0]

    with torch.no_grad():
        print('Generating val trajectory')
        for i in tqdm(range(1, traj_size)):
            predicted_traj[i] = model(torch.FloatTensor(predicted_traj[i-1]).view(-1, 3)).detach().numpy().flatten()
        fig = plt.figure()
        plt.title('Epoch : %d' % epoch)
        ax = fig.gca(projection='3d')
        ax.plot(predicted_traj[:,0], predicted_traj[:,1], predicted_traj[:,2], alpha=0.5)
        ax.plot(eval_traj[:,0], eval_traj[:,1], eval_traj[:,2], alpha=0.5)
        plt.savefig('../imgs/%d_eval.png' % epoch)


def corr(t1, t2):
    pass


def autocorr(ts):
    mean = np.mean(ts, axis=0)
    std = np.std(ts, axis=0)


def generate_data(traj):
    x_t = traj[:,0][:-1]
    y_t = traj[:,1][:-1]
    z_t = traj[:,2][:-1]
    X = np.zeros((len(x_t),3))
    X[:,0] = x_t
    X[:,1] = y_t
    X[:,2] = z_t
    ## Outputs
    x_tp1 = np.roll(traj[:,0],-1)[:-1]
    y_tp1 = np.roll(traj[:,1],-1)[:-1]
    z_tp1 = np.roll(traj[:,2],-1)[:-1]
    y = np.zeros((len(x_t),3))
    y[:,0] = x_tp1
    y[:,1] = y_tp1
    y[:,2] = z_tp1
    return X,y


def select(array,i,step,N):
  selected=[]
  n=0
  while len(selected)<N:
    selected.append(array[i])
    n+=1
    i=i+step
  return np.array(selected).reshape((n,array.shape[1]))


def train_model(n_epochs=100, plot_fig=False):

    Niter = 10000
    step = 100

    traj, t = generate_trajectory(Niter=Niter*step, delta_t=1e-4)
    eval_traj, eval_t = generate_trajectory(Niter=Niter, delta_t=1e-2)

    traj_size = Niter - 1

    # Data generation procedure proposed by another group
    training = []
    init=0
    starts=[l for l in range(step)]
    for start in starts:
      init+=1
      training.append(np.array(select(traj,start,step,Niter).tolist()))

    X_train,y_train= np.zeros((traj_size*len(starts),3)),np.zeros((traj_size*len(starts),3))
    for i in range(len(starts)):
        X,y=generate_data(training[i])
        X_train[i*traj_size:(i+1)*traj_size,:]=X
        y_train[i*traj_size:(i+1)*traj_size,:]=y
    X_val,y_val = generate_data(eval_traj)

    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)

    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val)

    model = RosslerModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=traj_size, shuffle=False)

    eval_dataset = TensorDataset(X_val, y_val)
    eval_dataloader = DataLoader(eval_dataset, batch_size=traj_size)

    models = []

    for epoch in range(n_epochs):
        running_loss = 0
        for i, (X, y) in tqdm(enumerate(dataloader)):
            optimizer.zero_grad()

            y_pred = model(X)
            loss = criterion(y, y_pred)
            loss.backward()
            optimizer.step()
        
        models.append(copy(model))
        
        if epoch % 1 == 0:
            print("Epoch %d : Loss %.4f" % (epoch, loss.item()))

        if epoch % 10 == 0 and plot_fig:
            predicted_traj = np.zeros((traj_size, 3), dtype=float)
            predicted_traj[0] = traj[0]

            with torch.no_grad():
                print('Generating val trajectory')
                for i in tqdm(range(1, traj_size)):
                    predicted_traj[i] = model(torch.FloatTensor(predicted_traj[i-1]).view(-1, 3)).detach().numpy().flatten()
                fig = plt.figure()
                plt.title('Epoch : %d' % epoch)
                ax = fig.gca(projection='3d')
                ax.plot(predicted_traj[:,0], predicted_traj[:,1], predicted_traj[:,2], alpha=0.5)
                ax.plot(eval_traj[:,0], eval_traj[:,1], eval_traj[:,2], alpha=0.5)
                plt.savefig('../imgs/%d_eval.png' % epoch)

    return models