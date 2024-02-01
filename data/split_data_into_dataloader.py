import torch
import numpy as np
from collections import Counter
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedGroupKFold
from data.crop_dataset import CropDataset

def get_dataloaders(dataset,n_splits=20,batch_size=64,random_state=1,use_sampler=True):
    """Create training and validation dataloader from the xarray crop dataset.

    Args:
        dataset: An xarray dataset to be split into training and validation sets.
        n_splits (int, optional): Number of splits for StratifiedGroupKFold cross-validation, such that validation set gets 1/n_splits of the whole data. Defaults to 20.
        batch_size (int, optional): Batch size for DataLoader. Defaults to 64.
        random_state (int, optional): Random seed for reproducibility. Defaults to 1.
        use_sampler (bool, optional): If True, uses WeightedRandomSampler for imbalanced class handling. Defaults to True.

    Returns:
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
    """

    num_classes = 6
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_id, val_id = next(sgkf.split(dataset.index.values, dataset.crop.values, dataset.patch_id.values))  # get first split only
    train_dataset = CropDataset(dataset, indices=dataset.index.values[train_id], num_classes=num_classes)
    val_dataset = CropDataset(  dataset, indices=dataset.index.values[val_id], num_classes=num_classes)
    if use_sampler:
        # https://discuss.pytorch.org/t/how-to-handle-imbalanced-classes/11264/2
        crop_counter = Counter(train_dataset.ds.crop.values)
        class_counts = [crop_counter[i] for i in range(num_classes)]
        total_samples = sum(class_counts)
        class_weights = [total_samples / (1+count) for count in class_counts]
        class_weights = (np.array(class_weights)/sum(class_weights)) #.clip(0.05)
        samples_weight = torch.from_numpy(np.array([class_weights[target] for target in train_dataset.ds.crop.values])).double()
        sampler = WeightedRandomSampler(weights=samples_weight, num_samples=len(samples_weight), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # validation loader (doesn't need specific sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader, val_loader