from data_provider.data_loader import PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader,UCRAnomalyloader,TrainSegLoader
from torch.utils.data import DataLoader
import pandas as pd

data_dict = {
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
}

"""
Implementation from

"""
def read_meta(root_path, dataset):
    meta_path = root_path + "/DETECT_META.csv"
    meta = pd.read_csv(meta_path)
    meta = meta.query(f'file_name.str.contains("{dataset}")', engine="python")
    file_paths = root_path + f"/{meta.file_name.values[0]}"
    train_lens = meta.train_lens.values[0]
    return file_paths, train_lens

def read_UCR_meta(dataset):
    assert dataset.endswith('.txt')
    parts = dataset.split('_')
    if len(parts) < 3:
        return None

    border_str = parts[-3]
    border_1_str = parts[-2]
    border_2_str = parts[-1]
    if '.' in border_2_str:
        border_2_str = border_2_str[:border_2_str.find('.')]

    try:
        border = int(border_str)
        border_1 = int(border_1_str)
        border_2 = int(border_2_str)
        return border, border_1, border_2
    except ValueError:
        return None

def data_provider(args, flag):
    timeenc = 0 if args.embed != 'timeF' else 1
    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq

    if args.task_name == 'anomaly_detection':
        drop_last = False
        if args.data == 'UCR':
            border, border_1, border_2 = read_UCR_meta(args.data_path)
            data_set = UCRAnomalyloader(args,
                args.root_path,
                args.seq_len,
                args.data_path,
                train_length=border,
                border_1=border_1,
                border_2=border_2,
                flag=flag
                )
        elif args.data == 'SWAN' :
            file_paths, train_lens = read_meta(root_path=args.root_path, dataset=args.data_path)
            data_set = TrainSegLoader(file_paths, train_lens, win_size=args.seq_len, flag=flag, discrete_channels = None)

        elif args.data == 'GECCO':
            file_paths, train_lens = read_meta(root_path=args.root_path, dataset=args.data_path)
            data_set = TrainSegLoader(file_paths, train_lens, win_size=args.seq_len, flag=flag, discrete_channels=None)

        elif args.data == 'SMAP':
            data_set = SMAPSegLoader(args=args,
                                    root_path=args.root_path,
                                    win_size=args.seq_len,
                                    flag=flag,
                                    discrete_channels=range(1, 25))

        elif args.data == 'SMD':
            data_set = SMDSegLoader(args = args,
                root_path=args.root_path,
                win_size=args.seq_len,
                flag=flag,
                discrete_channels = [4,7,16,17,26,28,36,37])#

        elif args.data == 'MSL':
            data_set = MSLSegLoader(args = args,
                root_path=args.root_path,
                win_size=args.seq_len,
                flag=flag,
                discrete_channels = None)


        else:
            Data = data_dict[args.data]
            data_set = Data(
                args = args,
                root_path=args.root_path,
                win_size=args.seq_len,
                flag=flag,
            )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            pin_memory=True)
        return data_set, data_loader

    else:
        raise ValueError('Error argument in task_name')


