import numpy as np
from torch.utils.data import Dataset
from local_utils.apply_stats_and_pad import apply_stats_and_pad

class CropDataset(Dataset):
    def __init__(self, ds, indices=None, num_classes=6):
        """Dataset subclass for the crops data

        Args:
            ds (xarray.Dataset): netcdf dataset loadedcontaining the full data.
            indices (list or array, optional): ids of the samples to select along the 'index' dimension of the dataset `ds` . Defaults to None.
            num_classes (int, optional): number of target classes. Defaults to 6.
        """
        if indices is None:
            indices = ds.index.values
        self.ds = ds.sel(index=indices)
        self.indices = indices
        self.num_classes = num_classes

        if False:
            self.ds = self.ds.assign({ #calculate stats for each band across axis 0 (index) - N
                "stats-min": self.ds.s2.min(axis=(0,2), skipna=True),
                "stats-max": self.ds.s2.max(axis=(0,2), skipna=True),
            })

            arr, _ = apply_stats_and_pad(self.ds['s2'].sel(index=self.indices).values,self.ds)
            self.ds['s2'].loc[{'index':self.indices}] = arr

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        index = self.indices[index]
        s2_data = self.ds.sel(index=index).s2.values
        label = self.ds.sel(index=index).crop.values
        onehot_label = np.zeros((self.num_classes))
        onehot_label[int(label)] = 1
        return {'s2':s2_data,'label': onehot_label,'original_index':index}