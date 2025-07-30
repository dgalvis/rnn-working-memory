# Import modules
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

# This function generates one batch of synthetic training data
def get_dataset_single_batch(params: dict, seed: Optional[int] = None):
    # use seed for reproducibility
    rng = np.random.default_rng(seed)
    
    # Training Synthetic Dataset
    x_train = np.zeros((params["batch_size"], params["seq_len"], params["input_size"])) # batch training data
    labels_train = np.zeros((params["batch_size"], params["wheel_len"])) # one hot encoded

    # some randomness to the training data cues
    shifts = params["shift_level"] * np.random.randn(params["batch_size"], params["max_set_size"], params["wheel_len"])


    # set_size in [min_colours, max_colours] for each sample
    set_size = np.zeros(params["batch_size"])

    # colour choice for each entry in the set, for each sample in batch (-1 means no colour)
    colour_scheme = -np.ones((params["batch_size"], params["max_set_size"]))
    
    # Fill in the training set, iterate over samples in the batch
    for j in range(params["num_batches"]):
        
        # how many colours are presented for this sample
        num_colours = rng.integers(params["min_colours"], params["max_colours"]+1)
        set_size[j] = num_colours

        # Choose num_colours locations to place the stimuli
        col_locs = rng.permutation(params["max_set_size"])
        col_locs = col_locs[0:num_colours]

        # Choose the colour for each stimulus
        cues = np.zeros((num_colours), dtype=np.int64)
        for c in range(num_colours):
            cues[c] = rng.integers(0, params["wheel_len"])

        # Which colour is retrocued, i.e., which needs to be maintained in memory
        retro_cues = rng.permutation(num_colours)
        retro_cues = retro_cues[0]
            
        # Place cue data (a noisy von Mises distribution around the correct colour)
        for k in range(num_colours):
            colour_scheme[j, col_locs[k]] = cues[k]
            for s in range(params["wheel_len"]):
                x_train[j, params["cue_time"]-1, s+params["max_set_size"]+params["wheel_len"] * col_locs[k]] = von_Mises(
                    cues[k],
                    s,
                    params["wheel_len"],
                    kappa=params["kappa"],
                    phase_shift=shifts[j,k,s]) 

        # Place Retrocues
        x_train[j, params["retro_cue_time"]-1, col_locs[retro_cues]] = 1
        # Place labels (one hot encoded)
        labels_train[j, cues[retro_cues]] = 1
            
    
    return x_train, labels_train, generate_noise(params, seed=rng.integers(1e10)), set_size, colour_scheme

# Von Mises Distribution
def von_Mises(idx1, idx2, wheel_len, kappa = 5, phase_shift = 0):
    numer = np.exp(kappa*np.cos(2*np.pi*(idx1 - idx2+ phase_shift)/wheel_len))
    denom = np.exp(kappa*np.cos(0))
    
    return numer/denom

# Noise for the RNN    
def generate_noise(params: dict, seed: Optional[int] = None):
    if params["noise_level"] != 0:
        noise = params["noise_level"] * np.random.randn(params["batch_size"], params["seq_len"], params["num_layers"], params["hidden_size"])
    else:
        noise = np.zeros((params["batch_size"], params["seq_len"], params["num_layers"], params["hidden_size"]))
    return noise


def plot_dataset(params):
    x_train, labels_train, _, _, _ = get_dataset_single_batch(params)
    batch_size = x_train.shape[0]

    
    for bs in range(batch_size):
        # Clear any existing figure (helps in interactive/back-to-back use)
        plt.close('all')  # closes all previous figures
        fig, ax = plt.subplots(figsize=(params['seq_len']/3, 1 + 3*params['max_set_size']))

        # Plot input data
        ax.imshow(x_train[bs].T, aspect='auto')

        # Draw separating lines
        ax.plot([-0.5, params['seq_len'] - 0.5], [params['max_set_size'] - 0.5]*2, 'w')
        for k in range(params['max_set_size']):
            y = params['max_set_size'] + params['wheel_len'] * k - 0.5
            ax.plot([-0.5, params['seq_len'] - 0.5], [y, y], 'w')

        # Ticks
        ax.set_xticks(range(params['seq_len']))
        ax.set_xticklabels(range(1, params['seq_len']+1))

        y_labels = np.concatenate([
            np.arange(1, params['max_set_size'] + 1),
            np.tile(np.arange(1, params['wheel_len'] + 1), params['max_set_size'])
        ])
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels)

        label_idx = np.argmax(labels_train[bs])
        ax.set_title(f'The label is: L={label_idx + 1}')

        plt.tight_layout()
        plt.show()

        # Ask user if they want to see another
        ask = input("Show another [y/n]: ").strip().lower()
        if ask != 'y':
            break
 

if __name__ == '__main__':
    pass
