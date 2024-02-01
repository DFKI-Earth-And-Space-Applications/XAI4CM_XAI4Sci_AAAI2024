import click
from data.preprocessing import create_crop_dataset

@click.group()
def main():
    """Entry method"""
    pass

@main.command()
@click.argument('ghana_data_dir', type=str)
@click.argument('ssudan_data_dir', type=str)
@click.option('-s', '--max_seq_len', type=int, default=228, help='max fixed sequence length of all time series')
def prepare_datasets(ghana_data_dir,ssudan_data_dir,max_seq_len):
    # Ghana
    split_scheme = "ghana" 
    create_crop_dataset(ghana_data_dir,split_scheme,max_seq_len=max_seq_len)
    # South Sudan
    split_scheme = "southsudan"
    create_crop_dataset(ssudan_data_dir,split_scheme,max_seq_len=max_seq_len)


if __name__ == '__main__':
    main()
 