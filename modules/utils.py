# Set up
import os
from pathlib import Path
import copy

# Computation modules
import torch
import numpy as np

# Define the Recurrent Neural Network Type
from .rnn import RNN, criterion, criterion_pi

# Training data
from .dataset import get_dataset_single_batch


# Train the model
def train_model(config, model=None, file=None, verbose=False, loss_old = 1e10):
    
    if model == None:
        model = RNN(config).to(config["device"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"]) 
    # optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"]) 
    # optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"])
    
    loss_mean = 0
    count = 0
    model_aux = copy.deepcopy(model)
    optim_aux = copy.deepcopy(optimizer)
    for i in range(config["num_batches"]):
        x_it, L_it, n_it, _, _ = get_dataset_single_batch(config)
        x_it = torch.from_numpy(x_it.astype(np.float32)).to(config['device'])
        L_it = torch.from_numpy(L_it.astype(np.int64)).to(config['device'])
        n_it = torch.from_numpy(n_it.astype(np.float32)).to(config['device'])
        outputs = model(x_it, n_it) 

        loss = criterion(outputs.view(L_it.shape), L_it, config)  
        with torch.no_grad():
            loss_mean = loss_mean + loss/config["check_num"]
            
         # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 

        # Print current results and use average losses to decide if we keep the model
        if i == 0:
            with torch.no_grad():
                if verbose == True:
                    _, acc = criterion_pi(outputs.view(L_it.shape), L_it, use_mean = True) 
                    print (f'Epoch [{(i+1)}/{config["num_batches"]}],  Accuracy: {acc:.4f}, Loss: {loss_mean*config["check_num"]:.4f}')    
        if ((i+1) % config['check_num'] == 0):
            with torch.no_grad():
                if verbose == True:
                    _, acc = criterion_pi(outputs.view(L_it.shape), L_it, use_mean = True)  
                    print (f'Epoch [{(i+1)}/{config["num_batches"]}],  Accuracy: {acc:.4f}, Loss: {loss_mean:.4f}')

                # Reset if the average loss has increased over the last config['check_num'] batches
                if loss_mean > loss_old:
                    model = copy.deepcopy(model_aux)
                    optimizer = copy.deepcopy(optim_aux)
                else:
                    model_aux = copy.deepcopy(model)
                    optim_aux = copy.deepcopy(optimizer)
                    loss_old = loss_mean
                loss_mean = 0

    if count != 0:
        model = copy.deepcopy(model_aux)

    
    if file != None:
        torch.save(config, file + "_config.pth")
        torch.save(model.state_dict(), file + "_sd.pth")
    
    return model, loss_old

# Test the model
def test_model(config, model, file=None, verbose=False):
    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        loss_pi = torch.zeros(config["num_test_batches"],config["batch_size"]).to(config["device"])
        set_size = np.zeros((config["num_test_batches"],config["batch_size"]))
        
        n_correct = 0
        n_samples = 0
        for i in range(config["num_test_batches"]):
            x_it, L_it, n_it, s_it, _ = get_dataset_single_batch(config)
            x_it = torch.from_numpy(x_it.astype(np.float32)).to(config['device'])
            L_it = torch.from_numpy(L_it.astype(np.int64)).to(config['device'])
            n_it = torch.from_numpy(n_it.astype(np.float32)).to(config['device'])
            outputs = model(x_it, n_it)
            
            loss_pi[i, :], acc = criterion_pi(outputs.view(L_it.shape), L_it, use_mean = False)
            set_size[i, :] = s_it

            
            n_samples += L_it.size(0)
            n_correct += (acc).sum().item()
        acc = 100.0 * n_correct / n_samples
        if verbose == True:
            print(f'Accuracy of the network on  test images: {acc:.2f} %')


    test = {
                "loss_pi": loss_pi,
                "acc": acc,
                "set_size": set_size
    }
    if file != None:
        torch.save(test, file + "_test.pth")
        
    
    return test

def save_model(config=None, model=None, file=None, test=None):
    if file != None:
        if config!= None:
            torch.save(config, file + "_config.pth")
        if model != None:
            torch.save(model.state_dict(), file + "_sd.pth")
        if test!=None:
            torch.save(test, file + "_test.pth")
    else:
        pass

# Load after training and/or testing
def load_model(file, load_test=True):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config = torch.load(file+ "_config.pth", map_location = device)
    config["device"] = device
    

    model = RNN(config).to(config["device"])    
    model.load_state_dict(torch.load(file+ "_sd.pth", map_location = device))
    model.eval()

    if load_test:
        test = torch.load(file+"_test.pth", map_location = device)
    else:
        test = None

    return model, config, test

if __name__ == '__main__':
    pass