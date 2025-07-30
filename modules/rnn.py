import torch
import torch.nn as nn
import numpy as np

# Define the vanilla RNN model from scratch
class RNN(nn.Module):
    def __init__(self, info):
        super(RNN, self).__init__()
        self.batch_size  = info["batch_size"]
        self.input_size  = info["input_size"]
        self.num_layers  = info["num_layers"]
        self.hidden_size = info["hidden_size"]
        self.device      = info["device"]
        self.seq_len     = info["seq_len"]
        self.noise_level = info["noise_level"]
        self.wheel_len   = info["wheel_len"]
        
        # RNN parameters
        self.W_ih = nn.ParameterList([nn.Parameter(torch.Tensor(self.hidden_size, self.input_size if i == 0 else self.hidden_size)) for i in range(self.num_layers)])
        self.b_ih = nn.ParameterList([nn.Parameter(torch.Tensor(self.hidden_size)) for _ in range(self.num_layers)])
        self.W_hh = nn.ParameterList([nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size)) for _ in range(self.num_layers)])
        self.b_hh = nn.ParameterList([nn.Parameter(torch.Tensor(self.hidden_size)) for _ in range(self.num_layers)])

        self.fun = torch.tanh
        #self.fun = nn.ReLU()
        

        self.fc = nn.Linear(self.hidden_size, self.wheel_len)
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / np.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x, noise=None):
        h_t = [self.init_hidden() for _ in range(self.num_layers)]
        if noise is None:
            noise = torch.zeros(self.batch_size, self.seq_len, self.num_layers, self.hidden_size).to(self.device)        
        for t in range(self.seq_len):
            x_t = x[:, t, :]
            for layer in range(self.num_layers):
                h_t[layer] = self.fun(x_t @ self.W_ih[layer].T + self.b_ih[layer] + 
                                        h_t[layer] @ self.W_hh[layer].T + self.b_hh[layer] 
                                        + noise[:, t, layer, :])
                x_t = h_t[layer]
        
        out = self.fc(h_t[-1])
        return out

    def init_hidden(self):
        return torch.zeros(self.batch_size, self.hidden_size).to(self.device)



# Criterion function for training (borrowed from Piwek for training)
def criterion(y_pred, y, use_mean = True):
    yhat = torch.exp(y_pred)
    yhatsum = yhat.sum(axis=1)
    yhatsum = yhatsum.repeat(yhat.shape[1], 1)
    yhat = yhat / yhatsum.T

    err1 = ((y - yhat)*find_phase_diff(y))**2
    err1 = err1.sum(axis = 1)


    if use_mean:
        return err1.mean()
    else:
        return err1

# Criterion function for studying (How often is the output correct)
def criterion_pi(y_pred, y, use_mean = True):
    yhat = torch.exp(y_pred)
    yhatsum = yhat.sum(axis=1)
    yhatsum = yhatsum.repeat(yhat.shape[1], 1)
    yhat = yhat / yhatsum.T

    idx = torch.argmax(yhat, axis = 1)
    idx2 = torch.argmax(y, axis = 1)
    phase_diffs = find_phase_diff(y)
        
    err1 = torch.zeros(idx.shape[0])
    acc  = torch.zeros(idx.shape[0])
    for i in range(idx.shape[0]):
        err1[i] = phase_diffs[i, idx[i]]
        acc[i] = idx2[i] == idx[i]
        
    
    if use_mean:
        return err1.mean(), acc.mean()
    else:
        return err1, acc

def find_phase_diff(y):
    phs = torch.arange(y.shape[1])
    phs = phs.repeat(y.shape[0], 1).to(y.device)


    shifts = (y*phs).sum(axis=1)
    shifts = shifts.repeat(y.shape[1],1).T

    out = 2*np.pi*(phs - shifts)/y.shape[1]
    out = (out+ np.pi) % (2 * np.pi) - np.pi

    return out


if __name__ == '__main__':
    pass