import numpy as np

def get_mask_grouped_features(samples,grp_axis=1):
    """ Group features based on temporal or spectral dimension.
    arg:
    - samples: 3 dimesional array: (batchsize, bands, timesteps)
    - grp_axis: axis index for grouping: 2 for temporal dimansion and 1 for spectral
    returns:
    - a mask where each integer value refer to elements of same group.
    """
    mask = np.ones_like(samples)
    n_ts, n_bd = samples.shape[1:]
    if grp_axis==1:
        for i in range(n_ts):
            mask[:,i,:] = i
    elif grp_axis==2:
        for i in range(n_bd):
            mask[:,:,i] = i
    return mask
