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
|`max_set_size`| int64 |  `2` | The number of spaces to hold stimuli (N)|

#### `kwargsH` Hyperparameter options
|Key|Type|Default|Description|
|----------------|------|---------|------------------------------------| 
|`batch_size`| int64 |  `2000` | The number of training examples in on batch |
