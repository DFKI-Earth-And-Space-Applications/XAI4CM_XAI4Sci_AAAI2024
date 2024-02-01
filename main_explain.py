import os
import click
import logging
import torch
import traceback
import numpy as np
import xarray as xr
from glob import glob
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader

from data.crop_dataset import CropDataset
from local_utils.variables import CROPS
from local_utils.apply_stats_and_pad import apply_stats_and_pad
from local_utils.get_mask_grouped_features import get_mask_grouped_features
from local_utils.plot_spectral_imp import plot_spectral_imp
from main_evaluate import evaluate_fct, visualize_accuracies

from captum.attr import ShapleyValueSampling


def explain_fct(
    model_pth,
    ghana_data_dir,
    ssudan_data_dir,
    attr_batch_size=512,
    grp_ftrs_by_band=True,
    max_per_class=5000,
    num_samples=25,
    use_vis='none',
    show_progress=False,
    logger=None,
):

    # prepare logging and results directory
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_pth = Path(model_pth)
    save_dir = model_pth.parent / f"attributions_{model_pth.name.split('.')[0]}"
    os.makedirs(save_dir, exist_ok=True)
    if logger is None:
        log_filepath = save_dir / f'logs_attr_{now}.log'
        logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=log_filepath,
                        filemode='w')
        logger = logging.getLogger(__name__)
    logger.info(f'Group features by bands: {grp_ftrs_by_band}')
    logger.info(f'Maximum number of samples per class (correctly predicted) to be explained: {max_per_class}')
    logger.info(f'Number of perturbation samples: {num_samples}')

    # get predictions ds
    num_classes = 6
    pred_ds_pth = model_pth.parent / f"evaluation_{model_pth.name.split('.')[0]}" / "predictions_ds.nc"
    if pred_ds_pth.exists():
        pred_ds = xr.open_dataset(pred_ds_pth)
    else:
        pred_ds = evaluate_fct(str(model_pth),ghana_data_dir,ssudan_data_dir,attr_batch_size,logger=logger,use_vis=use_vis)
        visualize_accuracies(pred_ds,model_pth)
    dataset = pred_ds
    logger.info(f'Dataset: \n{dataset}')

    # prepare attribution ds
    if len(glob(str(save_dir/"*attribution_ds.nc")))>0:
        attr_ds = xr.open_dataset(glob(str(save_dir/"*attribution_ds.nc"))[0])
        attr_precomputed = True
    else:
        attr_precomputed = False
        attr_ds = dataset[['s2','crop']]
        attr_arr = np.full(attr_ds.s2.shape, fill_value=np.nan, dtype=np.float32)
        attr_ds['processed_attribution'] = xr.DataArray(
            data=attr_arr, 
            dims=attr_ds.s2.dims, 
            coords=attr_ds.s2.coords
        )
        attr_ds['raw_attribution'] = xr.DataArray(
            data=attr_arr, 
            dims=attr_ds.s2.dims, 
            coords=attr_ds.s2.coords
        )
        xai_arr = np.full(attr_ds.index.shape, fill_value=False)
        attr_ds['explained'] = xr.DataArray(
            data=xai_arr, 
            dims=attr_ds.crop.dims, 
            coords=attr_ds.crop.coords
        )
        attr_ds = attr_ds.assign({ #calculate stats for each band across axis 0 (index) - N
            "stats-min": attr_ds.s2.min(axis=(0,2), skipna=True),
            "stats-max": attr_ds.s2.max(axis=(0,2), skipna=True),
        })

        # prepare model and attributor
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.load(model_pth,map_location=device)
        model = model.to(device,dtype=torch.float32)
        shap_ex = ShapleyValueSampling(model)
        perturbations_per_eval = 1
        explained_ids = []
        logger.info(f'Device: {device}')
        logger.info(f'Model: \n{model}')

    # loop over the crops
    for crop_id in range(num_classes):

        if not attr_precomputed:
            logger.info(f"Starting computing attributions of crop {crop_id} ({CROPS[num_classes][crop_id]})")

            # select samples corresponding to the crop to be explained
            crop_ds = dataset.sel(index=dataset.crop==crop_id)
            if len(crop_ds.index.values)==0:
                logger.info(f"skipping crop {crop_id}, no data")
                continue

            # select data samples with correct prediction
            crop_ds_correct_preds = crop_ds.sel(index=(crop_ds.crop==crop_ds.prediction))
            if len(crop_ds_correct_preds.index.values)==0:
                logger.info(f"skipping crop {crop_id}, no correct predictions")
                continue
            logger.info(f"{CROPS[num_classes][crop_id]} has {len(crop_ds_correct_preds.index.values)} correct predictions, will be explaining {min(len(crop_ds_correct_preds.index),max_per_class)}.")

            # create baseline: mean over correctly predicted samples from same crop
            crop_data = crop_ds_correct_preds.s2.values
            crop_baseline = np.expand_dims( np.nanmean(crop_data,axis=0) , axis=0)
            crop_baseline, _ = apply_stats_and_pad(crop_baseline,dataset,pad=False)

            # select a maximum of `max_per_class` random samples
            rng = np.random.default_rng(seed=1)
            indices = rng.choice(crop_ds_correct_preds.index.values, size=min(len(crop_ds_correct_preds.index),max_per_class), replace=False)
            attr_ds['explained'].loc[{'index':indices}] = [True]*len(indices)
            xai_dataset = CropDataset(crop_ds_correct_preds, indices=indices, num_classes=num_classes)
            xai_loader = DataLoader(xai_dataset,batch_size=attr_batch_size)
            explained_ids.extend(indices)


            for i, data in enumerate(xai_loader):
                samples, _, original_index = data['s2'], data['label'], data['original_index']
                samples, _ = apply_stats_and_pad(samples,dataset)

                # for each sample to be explained, match the missing data in its corresponding baseline vector
                crop_baselines = np.repeat(crop_baseline,len(samples),axis=0)
                crop_baselines_nanmasked = torch.Tensor(np.where(
                    samples==-1,
                    -1,
                    crop_baselines
                    )).to(device)

                # create temporal groupings, to generate attribution for each band
                if grp_ftrs_by_band:
                    grp_mask = get_mask_grouped_features(samples, grp_axis=1)
                    grp_mask = torch.Tensor(grp_mask).long().to(device)
                else:
                    grp_mask = None
                
                attribution = shap_ex.attribute(
                    inputs=samples.to(device), 
                    baselines=crop_baselines_nanmasked, # if baselines is not None else None,
                    target=crop_id,
                    n_samples=num_samples, # number of feature permutations tested. captum default: 25
                    feature_mask=grp_mask,
                    perturbations_per_eval=perturbations_per_eval,  # Allows multiple ablations to be processed simultaneously in one call to the model. default=1
                    show_progress=show_progress,
                )

                pos_attribution = attribution.clamp(min=0)
                totals = pos_attribution.sum(axis=(1,2))[:,None,None]
                scaled_pos_attr = pos_attribution/totals

                attr_ds['raw_attribution'].loc[{'index':original_index}] = attribution.detach().cpu().numpy()
                attr_ds['processed_attribution'].loc[{'index':original_index}] = scaled_pos_attr.detach().cpu().numpy()
                if i%5==0:
                    logger.info(f"... progress: {i+1}/{len(xai_loader)} completed for crop {crop_id} ({CROPS[num_classes][crop_id]})")
                torch.cuda.empty_cache()

        else:
            crop_ds = attr_ds.sel(index=attr_ds.crop==crop_id)
            indices = crop_ds.sel(index=crop_ds.explained).index.values

        # visualization
        crop_ds = attr_ds.sel(index=indices)
        try:
            plot_spectral_imp(crop_ds,crop_id,save_dir,now,logger)
        except Exception as e:
            logger.info(f"Failed getting attributions for crop {crop_id} ({CROPS[num_classes][crop_id]}) with error:\n {e}\n\n")
            traceback.print_exc()
        # save attributions dataset after each crop is processed
        logger.info(f"Completed explaining crop {crop_id} ({CROPS[num_classes][crop_id]})")
        logger.info(f"Attributions Dataset: \n{attr_ds.sel(index=indices)}")
        logger.info(f"Attributions: \n{attr_ds.sel(index=indices).raw_attribution}")
        if not attr_precomputed:
            attr_ds.to_netcdf(save_dir / f"{now}_attribution_ds.nc")
    
    # keep only the explained samples in the attributions dataset
    if not attr_precomputed:
        attr_ds = attr_ds.sel(index=explained_ids)
        attr_ds.to_netcdf(save_dir / f"{now}_attribution_ds_selected_ids.nc")
    logger.info("Completed explaining all crops.")


@click.group()
def main():
    """Entry method"""
    pass

@main.command()
@click.option('--model_pth', required=True, help='Path to the model checkpoint to be explained')
@click.option('--ghana_data_dir', required=True, help='Path to Ghana crop data directory')
@click.option('--ssudan_data_dir', required=True, help='Path to SouthSudan crop data directory')
@click.option('-b', '--attr_batch_size', type=int, default=512, help='loader batch size which specifies how many samples are explained at once')
@click.option('-g', '--grp_ftrs_by_band', type=bool, default=True, help='group the features per band to explain band importance only')
@click.option('-m', '--max_per_class', type=int, default=10000, help='maximum number of samples correctly predicted per class to be explained')
@click.option('-n', '--num_samples', type=int, default=25, help='number of perturbation used in the attribution method')
@click.option('-s', '--show_progress', type=bool, default=False, help='show attribution progress bar')
@click.option('-v', '--use_vis', type=str, default='none', help='if training with VIs, pass a list (comma-separated) of the indices to use')
def explain(model_pth,ghana_data_dir,ssudan_data_dir,attr_batch_size,grp_ftrs_by_band,max_per_class,num_samples,use_vis,show_progress):
    explain_fct(
        model_pth=model_pth,
        ghana_data_dir=ghana_data_dir,
        ssudan_data_dir=ssudan_data_dir,
        attr_batch_size=attr_batch_size,
        grp_ftrs_by_band=grp_ftrs_by_band,
        max_per_class=max_per_class,
        num_samples=num_samples,
        use_vis=use_vis,
        show_progress=show_progress,
    )


if __name__ == '__main__':
    main()
 