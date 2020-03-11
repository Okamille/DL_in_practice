import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from rossler_map import RosslerMap

class RosslerModel(nn.Module):
    def __init__(self, input_size=3, hidden_layer_size=30, output_size=3):
        super().__init__()
        self.input_linear = nn.Linear(input_size, hidden_layer_size)
        self.hidden_linear = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.out_linear = nn.Linear(hidden_layer_size, output_size)
        self.activation = F.relu

    def forward(self, input):
        out = self.input_linear(input)
        out = self.activation(out)
        out = self.hidden_linear(out)
        out = self.activation(out)
        out = self.out_linear(out)
        return out

def generate_trajectory(Niter = 100000, delta_t=1e-2, x_init=np.array([-5.75, -1.6,  0.02])):
    ROSSLER_MAP = RosslerMap(delta_t=delta_t)
    traj,t = ROSSLER_MAP.full_traj(Niter, x_init)
    return traj, t


def train_model(batch_size=16, n_epochs=10, print_debug=False):
    traj, t = generate_trajectory()

    X_train = torch.FloatTensor(traj[:-1])
    y_train = torch.FloatTensor(traj[1:])

    model = RosslerModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    for epoch in range(n_epochs):
        running_loss = 0
        for i, (X, y) in enumerate(dataloader):
            optimizer.zero_grad()

            y_pred = model(X)
            loss = criterion(y, y_pred)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 1000 == 1 and print_debug:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    
    return model