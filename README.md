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

## Plot synthetic dataset

To see the synthetic data, use the Jupyter Notebook plot_dataset.ipynb.
