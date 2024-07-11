
# Implmentation of "XAI-Guided Enhancement of Vegetation Indices for Crop Mapping"

> by Hiba Najjar, Marlon Nuske and Andreas Dengel.

> Accepted as a short paper at the **AAAI2024** workshop: ["*XAI4Sci: Explainable machine learning for sciences*"](https://xai4sci.github.io/) and at **IGARSS2024**.

## Overview


The paper presents a comprehensive framework guiding practitioners through the vast array of over a hundred spectral indices documented in the literature. To validate our method on the crop mapping task, we trained multiple models using individual indices and combinations of two, and demonstrated their ability to outperform the model trained on all spectral bands. Beyond its application in crop type identification, our approach holds promise for broader applications in remote sensing such as forest health assessment, drought monitoring, soil moisture estimation and flood mapping.


## Dependencies

Install the dependencies required to run the code using the following command:

```bash
pip install -r requirements.txt
```


## Data

The crop data used needs to first be downloaded and preprocessed through the following steps:
1. Download the crop mapping datasets of [Ghana](https://beta.source.coop/repositories/stanford/africa-crops-ghana/description/) and [South Sudan](https://beta.source.coop/repositories/stanford/african-crops-south-sudan/download/).
2. Run the data preprocessing code:

```bash
python data_preprocessing.py prepare_datasets \
  /path/to/crops_ghana/ \
  /path/to/crops_ssudan/ \
  --max_seq_len 300
```

## Experiments

To **train** the GRU model, **evaluate** the checkpoint with the smallest loss and **explain** it, run the following command:

```bash
python main_train.py train \
    --results_dir_pth results \
    --ghana_data_dir "/path/to/crops_ghana/netcdfs/" \
    --ssudan_data_dir "/path/to/crops_ssudan/netcdfs/" \
    --num_epochs 100 \
    --dropout  0.30 \
    --evaluate True \
    --explain True \
```

To **train with vegetation indices (VIs)** instead of the raw satellite bands, use the `--use_vis` option and list the VIs seprated by a comma: 
```bash
python main_train.py train \
    --results_dir_pth results \
    --ghana_data_dir "/path/to/crops_ghana/netcdfs/" \
    --ssudan_data_dir "/path/to/crops_ssudan/netcdfs/" \
    --num_epochs 100 \
    --dropout  0.30 \
    --use_vis ndvi,ndre \
    --evaluate True \
    --explain True 
```

To **evaluate** the model using a specific checkpoint, run the following command (make sure to specify the same VIs used in the training, if any):
```bash
python main_evaluate.py evaluate \
    --model_pth "/path/to/checkpoint.pth" \
    --ghana_data_dir "/path/to/crops_ghana/netcdfs/" \
    --ssudan_data_dir "/path/to/crops_ssudan/netcdfs/" \
    --use_vis ndvi,ndre 
```

To **explain** the model using a specific checkpoint, specify the number of samples -correctly classified- to explain per class, and run the following command (make sure to also specify the same VIs used in the training, if any):
```bash
python main_explain.py explain \
    --model_pth "/path/to/checkpoint.pth" \
    --ghana_data_dir "/path/to/crops_ghana/netcdfs/" \
    --ssudan_data_dir "/path/to/crops_ssudan/netcdfs/" \
    --max_per_class 1000 \
    --use_vis ndvi,ndre \
```

## Acknowledgments

H.Najjar acknowledges support through a scholarship from the University of Kaiserslautern-Landau. All authors are affiliated to the German Research Centre for Artificial Intelligence (DFKI).

