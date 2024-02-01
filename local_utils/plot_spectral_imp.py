import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from math import ceil
from matplotlib.ticker import MultipleLocator

from local_utils.variables import CROPS, S2_BANDS


def plot_spectral_imp(crop_ds,crop_id,save_dir,now,logger):
    """BarPlot of the spectral importance of a specific crop

    Args:
        crop_ds: dataset containing the attributions of the crop of interest
        crop_id: id of the crop of interest
        save_dir: directory path where plot will be saved
        now: current time
    """

    num_classes=6
    # remove samples with nan attributions
    mask = np.isnan(crop_ds.raw_attribution.values[:,0,0])
    crop_ds = crop_ds.sel(index=~mask)
    attr = crop_ds.processed_attribution.values
    logger.info(f'... After removing NAN attributions, {len(crop_ds.index.values)} samples will be visualized for crop {crop_id} ({CROPS[num_classes][crop_id]}).')
    spectral_attr = attr.sum(axis=2).mean(axis=0)
    df = pd.DataFrame([(bd,at) for bd,at in zip(S2_BANDS,spectral_attr)],columns=['band','attr'])

    sns.set_theme(style="ticks", palette="pastel", font_scale=1.)
    sns.set_style("darkgrid", {"axes.facecolor": ".95","grid.color": ".9"})
    fig, ax = plt.subplots(figsize=(7,5))
    sns.barplot(
        data=df, x='band', y='attr',
        ax=ax, linewidth=0.5, width=0.75,
    )
    ymax = ceil((df.attr.max()+0.01)*10)*0.1
    _ = ax.set(title=f'Spectral attribution of crop {CROPS[num_classes][crop_id]}\n\n',xlabel='', ylabel='Average attribution',ylim=(0,ymax)) #ymax))
    plt.xticks(rotation=45,rotation_mode='anchor',ha='right')
    plt.tight_layout()
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    os.makedirs(save_dir/str(now), exist_ok=True)
    fig.savefig(save_dir/(f'{now}/spectral_imp_{CROPS[num_classes][crop_id]}.png'),dpi=200)
    logger.info(f'Visualization completed for crop {crop_id} ({CROPS[num_classes][crop_id]}).')
    plt.close('all')
