# Recurrent Neural Network for Working Memory Task

This project implements a recurrent neural network (RNN) trained on synthetic data to perform a working memory task involving **N stimuli**. It builds on and extends the work presented in:

> **A recurrent neural network model of prefrontal brain activity during a working memory task**  
> *Emilia P. Piwek, Mark G. Stokes, Christopher Summerfield (2023)*  
> [DOI:10.1371/journal.pcbi.1011555](https://doi.org/10.1371/journal.pcbi.1011555)

Whereas the original paper focused on tasks with **2 stimuli**, this repository generalizes the model and task structure to support **N-way memory tasks**, enabling broader exploration of working memory behavior.

---

## Environment Setup

To ensure reproducibility, we provide OS-specific Conda environment files. <os_name> is linux or mac.

```bash 
conda env create -f requirements_<os_name>.yml
conda activate rnn_env
```

## Synthetic Dataset

To see the synthetic data, use the Jupyter Notebook plot_dataset.ipynb.


## Parameters and Hyperparameters

Parameters for the synthetic dataset and for the recurrent neural network implementation 

```python
from modules import params
config = params.get_parameters(**kwargsP)
config = params.get_hyperparameters(config, **kwargsH)
```

You can customise the behaviour by passing keyword arguments with the following options:


#### `kwargsP` Parameter options
|Key|Type|Default|Description|
|----------------|------|---------|------------------------------------| 
| `config`| Optional[dict] | `None` | Dictionary to append |
| `wheel_len`| int |  `31` | Discretisation of the colour wheel (i.e., number of colours) |
| `seq_len`| int |  `15` | The number of total time units for each sample |
| `max_set_size` | int |  `2` | The number of spaces to hold stimuli (N) |
| `max_colours` | Optional[int] | `max_set_size` | The maximum number of stimuli for each sample (must be `max_colours<=max_set_size` ) |
| `min_colours` | Optional[int] | `2` | The minimum number of stimuli for each sample (must be `2<=min_colours<=max_colours`) |
| `cue_time` | int | `1` | Time to present the stimuli (1 corresponds to first time point) |
| `retro_cue_time` | int | `8` | Time to present the retrocue. The retrocue tells the subject which stimulus to keep in memory |
| `kappa` | float | `5.` | Parameter of the von Mises distribution. Determines stimulus spread around the true colour |
| `shift_level` | float | `0.` | Standard deviation of noise added as a shift in the von Mises distribution to each entry in the cue for each sample |


#### `kwargsH` Hyperparameter options
|Key|Type|Default|Description|
|----------------|------|---------|------------------------------------| 
| `config` | Optional[dict] | `None`  | Dictionary to append |
| `hidden_size` | int | `256` | Size of RNN hidden layers |
| `learning_rate` | float | `0.001` | Learning rate |
| `num_layers` | int | `1` | Number of RNN hidden layers |
| `noise_level` | float | `0.` | Amount of noise added when training |
| `num_batches` | int | `2000` | Total batches for training |
| `num_test_batches` | int | `500` | Total batches for testing |
| `batch_size` | int64 |  `2000` | The number of training examples in on batch |
