# Import Packages
import torch
from typing import Optional

# Parameters for the dataset that is created
def get_parameters(
    config: Optional[dict] = None,
    wheel_len: int = 31,
    seq_len: int = 15,
    max_set_size: int = 2,
    max_colours: Optional[int] = None,
    min_colours: Optional[int] = None,
    cue_time: int = 1,
    retro_cue_time: int = 8,
    kappa: float = 5,
    shift_level: float = 0,
) -> dict:
    
    if config is None:
        config = {}

    min_colours = min_colours if min_colours is not None else 2
    max_colours = max_colours if max_colours is not None else max_set_size

    if not (2 <= min_colours <= max_set_size):
        raise ValueError(f"min_colours must be between 2 and max_set_size ({max_set_size}), got {min_colours}.")

    # Validate max_colours
    if not (min_colours <= max_colours <= max_set_size):
        raise ValueError(f"max_colours must be between min_colours ({min_colours}) and max_set_size ({max_set_size}), got {max_colours}.")

    
    config.update({
        "wheel_len": wheel_len,
        "seq_len": seq_len,
        "max_set_size": max_set_size,
        "max_colours": max_colours,
        "min_colours": min_colours,
        "cue_time": cue_time,
        "retro_cue_time": retro_cue_time,
        "kappa": kappa,
        "shift_level": shift_level,
        "input_size": max_set_size * wheel_len + max_set_size,
    })

    return config


# Parameters for the neural network
def get_hyperparameters(
    config: Optional[dict] = None,
    hidden_size: int = 256,
    learning_rate: float = 0.001,
    num_layers: int = 1,
    noise_level: float = 0.0,
    num_batches: int = 2000,
    num_test_batches: int = 500,
    batch_size: int = 2000,
) -> dict:
    
    if config is None:
        config = {}

    config.update({
        "hidden_size": hidden_size,
        "learning_rate": learning_rate,
        "num_layers": num_layers,
        "noise_level": noise_level,
        "num_batches": num_batches,
        "num_test_batches": num_test_batches,
        "batch_size": batch_size,
        "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    })

    return config


if __name__ == '__main__':
    # Example usage
    params = get_parameters()
    params = get_hyperparameters(params)
    print(params)
