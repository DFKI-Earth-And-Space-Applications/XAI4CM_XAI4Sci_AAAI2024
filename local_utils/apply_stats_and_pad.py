import torch
import numpy as np

def apply_stats_and_pad(arr,dataset,apply_stat=True, pad=True):
    """Apply min-max rescaling
    Args:
        arr: vector of shape (n_samples, n_bands, n_time_steps)
        dataset: xarray dataset containing the min and max values of the bands.
        apply_stat (bool, optional): option to apply the min-max scaling or not. Defaults to True.
        pad (bool, optional): option to pad the nan values with -1 or not. Defaults to True.

    Returns:
        _type_: _description_
    """

    if "stats-min" not in dataset:
        dataset = dataset.assign({ #calculate stats for each band across axis 0 (index) - N
            "stats-min": dataset.s2.min(axis=(0,2), skipna=True),
            "stats-max": dataset.s2.max(axis=(0,2), skipna=True),
        })
    
    # apply min-max rescaling along bands
    if apply_stat:
        arr = np.transpose(arr,(0,2,1))
        min_ = dataset["stats-min"].values 
        max_ = dataset["stats-max"].values
        arr = (arr - min_)/(max_ - min_)
        arr = np.transpose(arr,(0,2,1))

    # padd nan values with -1
    if pad:
        nan_mask = torch.isnan(torch.Tensor(arr))
        arr[nan_mask] = -1
    else:
        nan_mask = None

    arr = torch.Tensor(arr).float()
    return arr, nan_mask