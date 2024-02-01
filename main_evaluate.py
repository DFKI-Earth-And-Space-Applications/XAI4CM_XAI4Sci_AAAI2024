import os
import click
import logging
import torch
import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from local_utils.calculate_vis import swap_to_vis
from local_utils.variables import CROPS
from local_utils.apply_stats_and_pad import apply_stats_and_pad
from data.split_data_into_dataloader import get_dataloaders


def evaluate_fct(model_pth,ghana_data_dir,ssudan_data_dir,batch_size,logger=None,use_vis=''):
    # prepare logging and results directory
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_pth = Path(model_pth)
    save_dir = model_pth.parent / f"evaluation_{model_pth.name.split('.')[0]}"
    os.makedirs(save_dir, exist_ok=True)
    log_filepath = save_dir / f'logs_eval_{now}.log'
    if logger is None:
        logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=log_filepath,
                        filemode='w')
        logger = logging.getLogger(__name__)
    logger.info('  --- Model evaluation started --- ')
    logger.info(f'Evaluated model: {model_pth}')

    # prepare data
    max_seq_len = 228
    num_classes = 6
    ghana_ds = xr.open_dataset(Path(ghana_data_dir)/f's2_data_crop_pixels_{max_seq_len}ts_{num_classes}_classes.nc')
    ssudan_ds = xr.open_dataset(Path(ssudan_data_dir)/f's2_data_crop_pixels_{max_seq_len}ts_{num_classes}_classes.nc')
    dataset = xr.concat([ghana_ds, ssudan_ds], dim='index')
    if "stats-min" in dataset: # after the datasets are concatenate, these statustics need to be updated
        del dataset["stats-min"]
        del dataset["stats-max"]
    dataset = swap_to_vis(dataset,use_vis,logger)
    train_loader, val_loader = get_dataloaders(dataset,batch_size=batch_size,use_sampler=False)
    logger.info(f'Dataset: \n{dataset}')

    # prepare predictions ds
    pred_ds = dataset[['s2','crop']]
    pred_arr = np.full(pred_ds.index.shape, fill_value=np.nan)  #, dtype='int64')
    pred_ds['prediction'] = xr.DataArray(
        data=pred_arr, 
        dims=pred_ds.crop.dims, 
        coords=pred_ds.crop.coords
    )
    split_arr = np.full(pred_ds.index.shape, fill_value='')
    pred_ds['split'] = xr.DataArray(
        data=split_arr, 
        dims=pred_ds.crop.dims, 
        coords=pred_ds.crop.coords
    )

    # prepare model and attributor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_pth,map_location=device)
    model = model.to(device,dtype=torch.float32)
    logger.info(f'Device: {device}')
    logger.info(f'Model: \n{model}')
    model.eval()
    if not hasattr(model,'return_probs'):
        model.return_probs = isinstance(model.final[-1],torch.nn.Softmax)


    train_indices = []
    val_indices = []
    with torch.no_grad():
        logger.info(f'Infering predicted class for all samples..')
        for i,data in enumerate(train_loader):
            logger.info(f'\tTraining set, step {1+i}/{len(train_loader)}')
            inputs, labels, original_indices = data['s2'], data['label'].to(device), data['original_index']
            inputs, _ = apply_stats_and_pad(inputs,dataset)
            outputs = model(inputs.to(device))
            if not model.return_probs:
                probabilities = torch.nn.Softmax(dim=1)(outputs).data
            else:
                probabilities = outputs.data
            _, predicted = torch.max(probabilities, 1)
            pred_ds['prediction'].loc[{'index':original_indices}] = predicted.detach().cpu().numpy()
            pred_ds['split'].loc[{'index':original_indices}] = ['train']*len(labels)
            train_indices.extend(original_indices)
        for i,data in enumerate(val_loader):
            logger.info(f'\tValidation set, step {1+i}/{len(val_loader)}')
            inputs, labels, original_indices = data['s2'], data['label'].to(device), data['original_index']
            inputs, _ = apply_stats_and_pad(inputs,dataset)
            outputs = model(inputs.to(device))
            if not model.return_probs:
                probabilities = torch.nn.Softmax(dim=1)(outputs).data
            else:
                probabilities = outputs.data
            _, predicted = torch.max(probabilities, 1)
            pred_ds['prediction'].loc[{'index':original_indices}] = predicted.detach().cpu().numpy()
            pred_ds['split'].loc[{'index':original_indices}] = ['val']*len(labels)
            val_indices.extend(original_indices)

    # save predictions dataset after each crop is processed
    logger.info("Completed infering predictions")
    logger.info(f"Predictions dataset: \n{pred_ds}")
    pred_ds.to_netcdf(save_dir / "predictions_ds.nc")
    logger.info(f"Dataset saved at {save_dir / 'predictions_ds.nc'}.")

    logger.info('  --- Model evaluation completed --- ')
    return pred_ds
    
def visualize_accuracies(pred_ds,model_pth,val=False):

    num_classes = 6
    model_pth = Path(model_pth)
    save_dir = model_pth.parent / f"evaluation_{model_pth.name.split('.')[0]}"
    if val: 
        suffix = '_validation'
    else: 
        suffix = '_full_dataset'
    os.makedirs(save_dir, exist_ok=True)

    # compute class accuracies and plot prediction heatmap
    cm = confusion_matrix(pred_ds.crop.values, pred_ds.prediction.values, labels=range(num_classes))
    class_accuracies = np.diag(cm) / (cm.sum(axis=1)+0.001)
    df = pd.DataFrame(pd.Series(class_accuracies),columns=['accuracy']).reset_index(drop=False)
    df["crop"] = df['index'].apply(lambda id : CROPS[num_classes][id])
    df = df[['index','crop','accuracy']]

    # compute F1, Precision and Recall scores
    df["precision"] = precision_score(pred_ds.crop.values, pred_ds.prediction.values, labels=range(num_classes), average=None) #average='weighted')
    df["recall"] = recall_score(pred_ds.crop.values, pred_ds.prediction.values, labels=range(num_classes), average=None) #average='weighted')
    df["f1"] = f1_score(pred_ds.crop.values, pred_ds.prediction.values, labels=range(num_classes), average=None) #average='weighted')

    # compute TP, FN, and FN for each class
    TP, FN, FP = np.zeros(num_classes), np.zeros(num_classes), np.zeros(num_classes)
    for i in range(num_classes):
        TP[i] = cm[i, i]
        FN[i] = np.sum(cm[i, :]) - TP[i]
        FP[i] = np.sum(cm[:, i]) - TP[i]
    df["TP"], df["FN"], df["FP"] = TP, FN, FP

    # save the dataframe
    df.to_csv(save_dir/f'class_accuracies_and_scores{suffix}.csv')
    df.to_excel(save_dir/f'class_accuracies_and_scores{suffix}.xlsx')

    #plot prediction heatmap
    sns.set_theme(style="ticks", palette="pastel", font_scale=1.)
    sns.set_style("darkgrid", {"axes.facecolor": ".95","grid.color": ".9"})

    ticklabels = [CROPS[num_classes][i] for i in range(len(class_accuracies))]
    fig, ax = plt.subplots(figsize=(7,5))
    sns.heatmap(cm, annot=True, cmap="crest", fmt=".0f", cbar=False, xticklabels=ticklabels, yticklabels=ticklabels)
    plt.title('Classes count')
    plt.xlabel('Predictions')
    plt.ylabel('Ground Truth')
    plt.tight_layout()
    fig.savefig(save_dir / f'class_count_heatmap{suffix}.png',dpi=200)

    fig, ax = plt.subplots(figsize=(7,5))
    div = np.repeat(cm.sum(axis=1)[:,None],num_classes,axis=1) + 0.001
    sns.heatmap(100*cm/div, annot=True, cmap="crest", fmt=".0f", linewidth=.5, xticklabels=ticklabels, yticklabels=ticklabels)
    plt.title('Classes accuracies (in %)')
    plt.xlabel('Predictions')
    plt.ylabel('Ground Truth')
    plt.tight_layout()
    fig.savefig(save_dir / f'class_accuracy_heatmap{suffix}.png',dpi=200)




@click.group()
def main():
    """Entry method"""
    pass

@main.command()
@click.option('--model_pth', required=True, help='Path to the model checkpoint to be explained')
@click.option('--ghana_data_dir', required=True, help='Path to Ghana crop data directory')
@click.option('--ssudan_data_dir', required=True, help='Path to SouthSudan crop data directory')
@click.option('-b', '--batch_size', type=int, default=1024, help='evaluation batch size')
@click.option('-v', '--use_vis', type=str, default='none', help='if training with VIs, pass a list (comma-separated) of the indices to use')
def evaluate(model_pth,ghana_data_dir,ssudan_data_dir,batch_size,use_vis):
    
    pth_ds = Path(model_pth).parent / "predictions_ds.nc"
    if pth_ds.exists():
        pred_ds = xr.open_dataset(str(pth_ds))
    else:
        pred_ds = evaluate_fct(model_pth,ghana_data_dir,ssudan_data_dir,batch_size,logger=None,use_vis=use_vis)
    visualize_accuracies(pred_ds,model_pth)
    visualize_accuracies(pred_ds.sel(index=pred_ds.split=='v'),model_pth,val=True)

if __name__ == '__main__':
    main()
 