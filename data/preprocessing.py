
import os
import json
import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime
from local_utils.variables import S2_BANDS


def read_json(pth):
    with open(pth) as f:
        data = json.load(f)
    return data


def create_crop_dataset(data_dir,split_scheme,max_seq_len=228):
    """Create the xarray crop dataset containing s2 modality alone.

    Args:
        data_dir (str): directory of dowloaded Ghana or Sudan datasets form https://beta.source.coop/ website
        split_scheme (str): specify the dataset, either 'ghana' or 'southsudan'.
        max_seq_len (int, optional): fixed length of time series. Default to 228.
    """
    dates = np.arange(max_seq_len)


    convert_crop_id_to_base_id = {
        "southsudan":{1:0,2:1,3:2,4:3},
        "ghana":{1:3,2:1,3:2,4:4,5:5,7:0},
    }
    crops = ["sorghum", "maize", "rice", "groundnut", "soyabean", "yam"]
    num_classes = len(crops)
    

    print(f"\n\nCreating ncfile for {split_scheme} dataset\n".upper())
    dataset = None
    total_crop_pixels = 0
    IDs = [ file.split('_')[-1] for file in os.listdir(os.path.join(data_dir,"truth")) ]
    for i,id in enumerate(IDs):
        if i==0:
            print("Xarray dataset creating progress:")
        if i>0 and i%100==0: 
            print(f"\t{i} out of {len(IDs)}")

        gt = np.load(Path(data_dir)/"original"/"truth_npy"/f"{split_scheme}_64x64_{id}_label.npy")
        crop_mask = np.isin(gt, list(convert_crop_id_to_base_id[split_scheme].keys()))
        crop_pixels = crop_mask.sum()
        total_crop_pixels += crop_pixels
        if crop_pixels==0:
            continue

        s2 = np.load(Path(data_dir)/"original"/"s2_npy"/f"s2_{split_scheme}_{id}.npy") # shape (10-bands, 64, 64, dates)
        s2_cld = np.load(Path(data_dir)/"original"/"s2_npy"/f"s2_{split_scheme}_{id}_cloudmask.npy") # shape (64, 64, dates)
        s2_dates = read_json(Path(data_dir)/"original"/"s2_npy"/f"s2_{split_scheme}_{id}.json")["dates"]
        s2 = np.concatenate([s2,s2_cld[None,]])
        s2_crops = s2[:,crop_mask,:].transpose((1,0,2))  # shape (samples, bands, dates)

        indices = [ f"{split_scheme}_{id}_{e}" for e in range(crop_pixels) ]

        # Create data variables
        samples_data = s2_crops  
        patch_id_data = np.array([id,]*crop_pixels)
        crop_data = [ convert_crop_id_to_base_id[split_scheme][int(crp)] for crp in gt[crop_mask].reshape(-1)]
        dates_data = np.repeat(np.array(s2_dates)[None,:],crop_pixels,axis=0)
        doy_data = [ float(datetime.strptime(dt, '%Y-%m-%d').timetuple().tm_yday) for dt in s2_dates ]
        doy_data = np.repeat(np.array(doy_data)[None,:],crop_pixels,axis=0)

        # pad missing time steps to match max_seq_len
        padding_steps = max_seq_len-len(s2_dates)
        samples_data = np.pad(samples_data,((0,0), (0,0), (padding_steps,0), ), mode='constant', constant_values=np.nan)
        dates_data = np.pad(dates_data,((0,0), (padding_steps,0), ), mode='constant', constant_values=np.nan)
        doy_data = np.pad(doy_data,((0,0), (padding_steps,0), ), mode='constant', constant_values=np.nan)

        # Create xarray data variables
        sample = xr.DataArray(samples_data, dims=['index', 'bands', 'dates'], coords={'index': indices, 'bands': S2_BANDS, 'dates': dates})
        patch_id = xr.DataArray(patch_id_data, dims=['index'], coords={'index': indices})
        crop = xr.DataArray(crop_data, dims=['index'], coords={'index': indices},)
        date = xr.DataArray(dates_data, dims=['index','dates'], coords={'index': indices, 'dates': dates}, )
        doy = xr.DataArray(doy_data, dims=['index','dates'], coords={'index': indices, 'dates': dates}, )

        # Create xarray dataset with data variables
        if dataset is None:
            dataset = xr.Dataset({
                's2': sample,
                'patch_id': patch_id,
                'crop': crop,
                'date':date,
                'doy':doy,
            })
        else:
            id_dataset = xr.Dataset({
                's2': sample,
                'patch_id': patch_id,
                'crop': crop,
                'date':date,
                'doy':doy,
            })
            dataset = xr.concat([dataset,id_dataset ], dim='index')

    dataset['crop'].attrs = { str(i):crop for i,crop in enumerate(crops) }
    dataset = dataset.assign({ #calculate stats for each band across axis 0 (index) - N
        "stats-min": dataset.s2.min(axis=(0,2), skipna=True),
        "stats-max": dataset.s2.max(axis=(0,2), skipna=True),
    })

    output_pth = Path(data_dir)/f'netcdfs/s2_data_crop_pixels_{max_seq_len}ts_{num_classes}_classes.nc'
    os.makedirs(output_pth.parent, exist_ok=True)
    dataset.to_netcdf(output_pth)
    print(f"NetCDF file creation for {split_scheme} data completed, and saved at {output_pth}.")
    print(f"Total crop pixels: {total_crop_pixels}.")
    return dataset


