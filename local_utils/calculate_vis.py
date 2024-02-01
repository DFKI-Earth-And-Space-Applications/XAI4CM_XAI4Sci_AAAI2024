import numpy as np
import xarray as xr

def calculate_ndvi(dataset):
    # Normalized Difference Vegetation Index (NDVI)
    # Formula: NDVI = (NIR – RED) / (NIR + RED)
    nir = dataset.s2.sel(bands='B08').values
    red = dataset.s2.sel(bands='B04').values
    #red = red.astype(np.float32)
    ndvi_div = nir + red
    ndvi = np.zeros_like(red, dtype=np.float32)
    np.divide(nir - red, ndvi_div, out=ndvi, where=ndvi_div != 0)
    return ndvi

def calculate_nndvi(dataset):
    # Narrow-Normalized Difference Vegetation Index (NNDVI)
    # Formula: NNDVI = (NNIR – RED) / (NNIR + RED)
    nnir = dataset.s2.sel(bands='B8A').values
    red = dataset.s2.sel(bands='B04').values
    nndvi_div = nnir + red
    nndvi = np.zeros_like(red, dtype=np.float32)
    np.divide(nnir - red, nndvi_div, out=nndvi, where=nndvi_div != 0)
    return nndvi

def calculate_ndre(dataset):
    # Normalized Difference Red Edge Vegetation Index (NDRE)
    # Formula: NDRE = (NIR – RED EDGE 1) / (NIR + RED EDGE 1)
    rededge = dataset.s2.sel(bands='B05').values
    nir = dataset.s2.sel(bands='B08').values
    ndre_div = nir + rededge
    ndre = np.zeros_like(rededge, dtype=np.float32)
    np.divide(nir - rededge, ndre_div, out=ndre, where=ndre_div != 0)
    return ndre

def calculate_ndre2(dataset):
    # Normalized Difference Red Edge Vegetation Index (NDRE2)
    # Formula: NDRE2 = (NIR – RED EDGE 2) / (NIR + RED EDGE 2)
    rededge2 = dataset.s2.sel(bands='B06').values
    nir = dataset.s2.sel(bands='B08').values
    ndre_div = nir + rededge2
    ndre = np.zeros_like(rededge2, dtype=np.float32)
    np.divide(nir - rededge2, ndre_div, out=ndre, where=ndre_div != 0)
    return ndre

def calculate_ndre3(dataset):
    # Normalized Difference Red Edge Vegetation Index (NDRE3)
    # Formula: NDRE3 = (NIR – RED EDGE 3) / (NIR + RED EDGE 3)
    rededge3 = dataset.s2.sel(bands='B07').values
    nir = dataset.s2.sel(bands='B08').values
    ndre_div = nir + rededge3
    ndre = np.zeros_like(rededge3, dtype=np.float32)
    np.divide(nir - rededge3, ndre_div, out=ndre, where=ndre_div != 0)
    return ndre

def calculate_ndmi(dataset):
    # Normalized Difference Moisture Index (NDMI)
    # Formula: ndmi = (NIR – SWIR) / (NIR + SWIR)
    swir = dataset.s2.sel(bands='B11').values
    nir = dataset.s2.sel(bands='B08').values
    ndmi_div = nir + swir
    ndmi = np.zeros_like(swir, dtype=np.float32)
    np.divide(nir - swir, ndmi_div, out=ndmi, where=ndmi_div != 0)
    return ndmi

def calculate_ndmi2(dataset):
    # Normalized Difference Moisture Index (NDMI) 2
    # Formula: ndmi2 = (NIR – SWIR2) / (NIR + SWIR2)
    swir2 = dataset.s2.sel(bands='B12').values
    nir = dataset.s2.sel(bands='B08').values
    ndmi2_div = nir + swir2
    ndmi2 = np.zeros_like(swir2, dtype=np.float32)
    np.divide(nir - swir2, ndmi2_div, out=ndmi2, where=ndmi2_div != 0)
    return ndmi2

def calculate_ireci(dataset):
    # IRECI   Inverted Red-Edge Chlorophyll Index        (RE3 - R) / (RE1 / RE2)
    red = dataset.s2.sel(bands='B04').values
    rededge1 = dataset.s2.sel(bands='B05').values
    rededge2 = dataset.s2.sel(bands='B06').values
    rededge3 = dataset.s2.sel(bands='B07').values
    ireci_div = np.zeros_like(red, dtype=np.float32)
    np.divide(rededge1, rededge2, out=ireci_div, where=rededge2 != 0)
    ireci = np.zeros_like(red, dtype=np.float32)
    np.divide(rededge3-red, ireci_div, out=ireci, where=ireci_div != 0)
    return ireci



vi_name_func = {
    'ndvi':calculate_ndvi,
    'nndvi':calculate_nndvi,
    'ndmi':calculate_ndmi,
    'ndre':calculate_ndre,
    'ndre2':calculate_ndre2,
    'ndre3':calculate_ndre3,
    'ndmi2':calculate_ndmi2,
    'ireci':calculate_ireci,
}    



def add_vi_to_dataset(dataset,VIs_name):
    """Add additional bands to the dataset for the selected VIs

    Args:
        dataset: xarray dataset with s2 bands.
        VIs_name: list of selected VIs to add.

    Returns:
        dataset: the input bands with selected VIs as additional bands.
    """

    if isinstance(VIs_name,str): 
        VIs_name = [VIs_name]

    for i,vi in enumerate(VIs_name):
        vi_arr = vi_name_func[vi](dataset)[:,None,:]
        if i==0:
            VIs_arr = vi_arr
        else:
            VIs_arr = np.concatenate([VIs_arr,vi_arr],axis=1)


    #assert len(VIs_arr.shape)==3
    da_ = xr.DataArray(
        data = np.concatenate([dataset.s2.values,VIs_arr],axis=1),
        dims = ('index','bands','dates'),
        coords={
            'index':dataset.index,
            'bands':list(dataset.bands.values)+VIs_name,
            'dates':dataset.dates,
        },
    )
    del dataset['s2']
    if "stats-min" in dataset:
        del dataset["stats-min"]
        del dataset["stats-max"]
    dataset = dataset.assign_coords(bands=list(dataset.bands.values)+VIs_name)
    dataset['s2'] = da_
    return dataset


def swap_to_vis(dataset,use_vis='none',logger=None):
    """exchange s2 bands with selected VIs

    Args:
        dataset: xarray dataset with s2 bands.
        use_vis (str, optional): 'none' to not add VIs, or a list of VIs seperated by `_`. Default to 'none'.
        logger (optional): main logger in which selected VIs will be printed. Default to None.

    Returns:
        dataset: xarray dataset with s2 bands removed and exchanged with VIs if passed, and min-max values of the bands updated.
    """

    # exchange s2 bands with VIs
    if use_vis.lower()!='none': 
        VIs = use_vis.split('_')
        for vi in VIs:
            assert vi in vi_name_func, f"{vi} is not a valid vegetation index, please correct the list passed ({use_vis})."
            
        if logger: 
            logger.info(f'Training with VIs: {VIs}')
        dataset = add_vi_to_dataset(dataset,VIs)
        dataset = dataset.sel(bands=VIs+['cloud_mask'])
        if "stats-min" in dataset:
            del dataset["stats-min"]
            del dataset["stats-max"]

    # update the min-max values of the new bands (=VIs)
    if "stats-min" not in dataset:
        dataset = dataset.assign({ 
            "stats-min": dataset.s2.min(axis=(0,2), skipna=True),
            "stats-max": dataset.s2.max(axis=(0,2), skipna=True),
        })
    return dataset