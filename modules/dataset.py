# Import modules
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

# This function generates one batch of synthetic training data
def get_dataset_single_batch(config: dict, seed: Optional[int] = None):
    # use seed for reproducibility
    rng = np.random.default_rng(seed)
    
    # Training Synthetic Dataset
    x_train = np.zeros((config["batch_size"], config["seq_len"], config["input_size"])) # batch training data
    labels_train = np.zeros((config["batch_size"], config["wheel_len"])) # one hot encoded

    # some randomness to the training data cues
    shifts = config["shift_level"] * np.random.randn(config["batch_size"], config["max_set_size"], config["wheel_len"])


    # set_size in [min_colours, max_colours] for each sample
    set_size = np.zeros(config["batch_size"])

    # colour choice for each entry in the set, for each sample in batch (-1 means no colour)
    colour_scheme = -np.ones((config["batch_size"], config["max_set_size"]))
    
    # Fill in the training set, iterate over samples in the batch
    for j in range(config["num_batches"]):
        
        # how many colours are presented for this sample
        num_colours = rng.integers(config["min_colours"], config["max_colours"]+1)
        set_size[j] = num_colours

        # Choose num_colours locations to place the stimuli
        col_locs = rng.permutation(config["max_set_size"])
        col_locs = col_locs[0:num_colours]

        # Choose the colour for each stimulus
        cues = np.zeros((num_colours), dtype=np.int64)
        for c in range(num_colours):
            cues[c] = rng.integers(0, config["wheel_len"])

        # Which colour is retrocued, i.e., which needs to be maintained in memory
        retro_cues = rng.permutation(num_colours)
        retro_cues = retro_cues[0]
            
        # Place cue data (a noisy von Mises distribution around the correct colour)
        for k in range(num_colours):
            colour_scheme[j, col_locs[k]] = cues[k]
            for s in range(config["wheel_len"]):
                x_train[j, config["cue_time"]-1, s+config["max_set_size"]+config["wheel_len"] * col_locs[k]] = von_Mises(
                    cues[k],
                    s,
                    config["wheel_len"],
                    kappa=config["kappa"],
                    phase_shift=shifts[j,k,s]) 

        # Place Retrocues
        x_train[j, config["retro_cue_time"]-1, col_locs[retro_cues]] = 1
        # Place labels (one hot encoded)
        labels_train[j, cues[retro_cues]] = 1
            
    
    return x_train, labels_train, generate_noise(config, seed=rng.integers(1e10)), set_size, colour_scheme

# Von Mises Distribution
def von_Mises(idx1, idx2, wheel_len, kappa = 5, phase_shift = 0):
    numer = np.exp(kappa*np.cos(2*np.pi*(idx1 - idx2+ phase_shift)/wheel_len))
    denom = np.exp(kappa*np.cos(0))
    
    return numer/denom

# Noise for the RNN    
def generate_noise(config: dict, seed: Optional[int] = None):
    if config["noise_level"] != 0:
        noise = config["noise_level"] * np.random.randn(config["batch_size"], config["seq_len"], config["num_layers"], config["hidden_size"])
    else:
        noise = np.zeros((config["batch_size"], config["seq_len"], config["num_layers"], config["hidden_size"]))
    return noise


def plot_dataset(config):
    x_train, labels_train, _, _, _ = get_dataset_single_batch(config)
    batch_size = x_train.shape[0]

    
    for bs in range(batch_size):
        # Clear any existing figure (helps in interactive/back-to-back use)
        plt.close('all')  # closes all previous figures
        fig, ax = plt.subplots(figsize=(config['seq_len']/3, 1 + 3*config['max_set_size']))

        # Plot input data
        ax.imshow(x_train[bs].T, aspect='auto')

        # Draw separating lines
        ax.plot([-0.5, config['seq_len'] - 0.5], [config['max_set_size'] - 0.5]*2, 'w')
        for k in range(config['max_set_size']):
            y = config['max_set_size'] + config['wheel_len'] * k - 0.5
            ax.plot([-0.5, config['seq_len'] - 0.5], [y, y], 'w')

        # Ticks
        ax.set_xticks(range(config['seq_len']))
        ax.set_xticklabels(range(1, config['seq_len']+1))

        y_labels = np.concatenate([
            np.arange(1, config['max_set_size'] + 1),
            np.tile(np.arange(1, config['wheel_len'] + 1), config['max_set_size'])
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
