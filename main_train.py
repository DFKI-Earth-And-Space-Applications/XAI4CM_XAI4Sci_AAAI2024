
import os
import click
import logging
import torch
import xarray as xr
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from datetime import datetime

from model.GRU import GRU
from data.split_data_into_dataloader import get_dataloaders
from local_utils.calculate_vis import swap_to_vis
from local_utils.plot_training_curves import plot_training_curves
from local_utils.apply_stats_and_pad import apply_stats_and_pad
from main_evaluate import evaluate_fct, visualize_accuracies
from main_explain import explain_fct



@click.group()
def main():
    """Entry method"""
    pass

@main.command()
@click.option('--results_dir_pth', required=True, help='Path to results directory where the experiment results will be saved under a new directory')
@click.option('--ghana_data_dir', required=True, help='Path to Ghana crop data directory')
@click.option('--ssudan_data_dir', required=True, help='Path to SouthSudan crop data directory')
@click.option('-e', '--num_epochs', type=int, default=100, help='number of epochs')
@click.option('-b', '--batch_size', type=int, default=512, help='training batch size')
@click.option('-p', '--dropout', type=float, default=0.30, help='model dropout')
@click.option('-v', '--use_vis', type=str, default='none', help='if training with VIs, pass a list (comma-separated) of the indices to use')
@click.option('-l', '--evaluate', type=bool, default=True, help='evaluate the best and final models')
@click.option('-x', '--explain', type=bool, default=True, help='explain the best model')
def train(results_dir_pth,ghana_data_dir,ssudan_data_dir,num_epochs,batch_size,dropout,use_vis,evaluate,explain):

    # prepare logging and results directory
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir_pth = Path(results_dir_pth)
    suffix_ = f"{now}_{dropout}drp_{batch_size}bs"
    use_vis = use_vis.replace(',','_')
    if use_vis.lower()!='none': 
        suffix_ += f'_usevis_{use_vis}'
    else: 
        suffix_ += '_s2'
    save_dir = results_dir_pth / suffix_
    os.makedirs(save_dir, exist_ok=True)
    log_filepath = save_dir / 'logs.log'
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename=log_filepath,
                    filemode='w')
    logger = logging.getLogger(__name__)
    logger.info(f'Use VIs: {use_vis}')
    logger.info(f'Number epochs: {num_epochs}')
    logger.info(f'Batch size: {batch_size}')
    
    # prepare data
    max_seq_len = 228
    num_classes = 6
    ghana_ds = xr.open_dataset(Path(ghana_data_dir)/f's2_data_crop_pixels_{max_seq_len}ts_{num_classes}_classes.nc')
    ssudan_ds = xr.open_dataset(Path(ssudan_data_dir)/f's2_data_crop_pixels_{max_seq_len}ts_{num_classes}_classes.nc')
    dataset = xr.concat([ghana_ds, ssudan_ds], dim='index')
    if "stats-min" in dataset:  # need to be updated after concatenating two datasets
        del dataset["stats-min"]
        del dataset["stats-max"]
    dataset = swap_to_vis(dataset,use_vis,logger)
    train_loader, val_loader = get_dataloaders(dataset,batch_size=batch_size, use_sampler=True)
    logger.info(f'Dataset: \n{dataset}')
    logger.info(f'Number of training instances: {len(train_loader)*batch_size}')
    logger.info(f'Number of validation instances: {len(val_loader)*batch_size}')

    # initialize the model
    learning_rate = 0.0003
    input_size = len(dataset.bands.values)  
    hidden_size = 512
    num_layers = 2
    hidden_units = 64
    logger.info(f'GRU hidden units:{hidden_size} in GRU layers ({num_layers}), {hidden_units} in the fully connected layers.')
    logger.info(f'dropout:{dropout}')
    model = GRU(input_size, hidden_size, num_layers, num_classes, hidden_units, dropout=dropout)

    # prepare for training
    evaluate_every = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = torch.tensor([1.]*num_classes)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))
    #criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(device))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model = model.to(device,dtype=torch.float32)
    #model = model.to(device)
    logger.info(f'Learning rate:{learning_rate}')
    logger.info(f'Device:{device}')
    logger.info(f'Model:\n{model}\n\n')

    # Set up early stopping parameters
    early_stopping_patience = 10
    best_val_loss = float('inf')
    counter = 0
    df = pd.DataFrame(columns=['Epoch','train_loss','train_accuracy','val_loss','val_accuracy'])

    # Training and evaluation loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_running_loss = 0.0
        total, correct = 0, 0

        for i, data in enumerate(train_loader):
            #logger.info(f'epoch {epoch}, step {1+i}/{len(train_loader)}')
            inputs, labels = data['s2'], data['label'].to(device)
            inputs, _ = apply_stats_and_pad(inputs,dataset)
            optimizer.zero_grad()

            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels)  
            loss.backward()
            optimizer.step()
            train_running_loss += loss.item()

            total += labels.size(0)
            probabilities = outputs.data
            _, predicted = torch.max(probabilities, 1)
            _, labels = torch.max(labels.data, 1)
            correct += (predicted == labels).sum().item()
        train_accuracy = 100 * correct / total
        train_loss = train_running_loss / len(train_loader)

        val_loss, val_accuracy = 0.0, 0.0
        if epoch % evaluate_every == 0:
            model.eval()
            correct = 0
            total = 0
            val_running_loss = 0.0

            with torch.no_grad():
                for data in val_loader:
                    inputs, labels = data['s2'], data['label'].to(device)
                    inputs, _ = apply_stats_and_pad(inputs,dataset)
                    outputs = model(inputs.to(device))
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item()
                    total += labels.size(0)
                    probabilities = outputs.data
                    _, predicted = torch.max(probabilities, 1)
                    _, labels = torch.max(labels.data, 1)
                    correct += (predicted == labels).sum().item()

            val_accuracy = 100 * correct / total
            val_loss = val_running_loss / len(val_loader)
            logger.info(f'Epoch {epoch},\n  Training: Loss: {train_loss}, Accuracy: {train_accuracy}%\n  Validation: Loss: {val_loss}, Accuracy: {val_accuracy}%\n')

            # Early stopping check
            if val_loss < best_val_loss:
                logger.info(f"New lowest validation loss: {best_val_loss} > {val_loss}")
                best_val_loss = val_loss
                counter = 0
                torch.save(model, save_dir/'best_model.pth')
            else:
                counter += 1
                if counter >= early_stopping_patience:
                    logger.info(f'Early stopping after {epoch} epochs')
                    break

        df.loc[len(df)] = [epoch,train_loss,train_accuracy,val_loss,val_accuracy]
        df.to_csv(save_dir/"accuracies_and_losses.csv")
        plot_training_curves(df,save_dir,show=False)
    
    
    torch.save(model, save_dir/'final_model.pth')
    logger.info('Saved final model.')

    if evaluate:
        # evaluate the best model
        pred_ds = evaluate_fct(str(save_dir/'best_model.pth'),ghana_data_dir,ssudan_data_dir,batch_size,logger=logger,use_vis=use_vis)
        visualize_accuracies(pred_ds,str(save_dir/'best_model.pth'),num_classes)
        visualize_accuracies(pred_ds.sel(index=pred_ds.split=='v'),str(save_dir/'best_model.pth'),num_classes)
        # evaluate the final model
        pred_ds = evaluate_fct(str(save_dir/'final_model.pth'),ghana_data_dir,ssudan_data_dir,batch_size,logger=logger,use_vis=use_vis)
        visualize_accuracies(pred_ds.sel(index=pred_ds.split=='v'),str(save_dir/'final_model.pth'),num_classes)

    if explain:
        explain_fct(str(save_dir/'best_model.pth'),ghana_data_dir, ssudan_data_dir, attr_batch_size=batch_size, max_per_class=5000, use_vis=use_vis, logger=logger)


    logger.info('Training and evaluation complete.')

if __name__ == '__main__':
    main()
 