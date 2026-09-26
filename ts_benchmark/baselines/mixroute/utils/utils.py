
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ts_benchmark.baselines.time_series_library.utils.timefeatures import (
    time_features,
)
from ts_benchmark.utils.data_processing import split_time
from ts_benchmark.baselines.mixroute.layers.Embed import patch_geometry


class EarlyStopping:
    

    def __init__(self, patience=7, delta=0, min_improvement_ratio=0.001):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.min_improvement_ratio = min_improvement_ratio
        self.improvement_history = []

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            print(f"Validation loss initialized: {val_loss:.6f}. Saving model ...")
            return True

        significant = (val_loss < self.best_score - self.delta and
                       (self.best_score - val_loss) / self.best_score > self.min_improvement_ratio)
        improved = False
        if significant:
            self.improvement_history.append((self.best_score - val_loss) / self.best_score)
            self.best_score = val_loss
            self.counter = 0
            improved = True
            print(f"✓ improved -> {val_loss:.6f}. counter reset.")
        else:
            self.counter += 1
            gap = (self.best_score - val_loss) / self.best_score if self.best_score != 0 else 0
            print(f"✗ No significant improvement (current: {val_loss:.6f}, best_score: {self.best_score:.6f}, "
                  f"gap: {abs(gap) * 100:.2f}%). EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                avg_improvement = np.mean(self.improvement_history[-3:]) if self.improvement_history else 0
                print(f"Early stopping triggered! Average recent improvement: {avg_improvement * 100:.2f}%")
        return improved


def train_val_split(train_data, ratio, seq_len):
    if ratio == 1:
        return train_data, None

    elif seq_len is not None:
        border = int((train_data.shape[0]) * ratio)

        train_data_value, valid_data_rest = split_time(train_data, border)
        train_data_rest, valid_data = split_time(train_data, border - seq_len)
        return train_data_value, valid_data
    else:
        border = int((train_data.shape[0]) * ratio)

        train_data_value, valid_data_rest = split_time(train_data, border)
        return train_data_value, valid_data_rest


def decompose_time(
        time: np.ndarray,
        freq: str,
) -> np.ndarray:
    
    df_stamp = pd.DataFrame(pd.to_datetime(time), columns=["date"])
    freq_scores = {
        "m": 0,
        "w": 1,
        "b": 2,
        "d": 2,
        "h": 3,
        "t": 4,
        "s": 5,
    }
    max_score = max(freq_scores.values())
    df_stamp["month"] = df_stamp.date.dt.month
    if freq_scores.get(freq, max_score) >= 1:
        df_stamp["day"] = df_stamp.date.dt.day
    if freq_scores.get(freq, max_score) >= 2:
        df_stamp["weekday"] = df_stamp.date.dt.weekday
    if freq_scores.get(freq, max_score) >= 3:
        df_stamp["hour"] = df_stamp.date.dt.hour
    if freq_scores.get(freq, max_score) >= 4:
        df_stamp["minute"] = df_stamp.date.dt.minute
    if freq_scores.get(freq, max_score) >= 5:
        df_stamp["second"] = df_stamp.date.dt.second
    return df_stamp.drop(["date"], axis=1).values


def get_time_mark(
        time_stamp: np.ndarray,
        timeenc: int,
        freq: str,
) -> np.ndarray:
    
    if timeenc == 0:
        origin_size = time_stamp.shape
        data_stamp = decompose_time(time_stamp.flatten(), freq)
        data_stamp = data_stamp.reshape(origin_size + (-1,))
    elif timeenc == 1:
        origin_size = time_stamp.shape
        data_stamp = time_features(pd.to_datetime(time_stamp.flatten()), freq=freq)
        data_stamp = data_stamp.transpose(1, 0)
        data_stamp = data_stamp.reshape(origin_size + (-1,))
    else:
        raise ValueError("Unknown time encoding {}".format(timeenc))
    return data_stamp.astype(np.float32)


def forecasting_data_provider(data, config, timeenc, batch_size, shuffle, drop_last, generator=None):
    
    
    
    _, _input_len = patch_geometry(config.seq_len, config.patch_len)
    dataset = DatasetForTransformer(
        dataset=data,
        history_len=_input_len,
        prediction_len=config.horizon,
        label_len=0,
        timeenc=timeenc,
        freq=config.freq,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        drop_last=drop_last,
        generator=generator,
    )

    return dataset, data_loader


class DatasetForTransformer:
    def __init__(
            self,
            dataset: pd.DataFrame,
            history_len: int = 10,
            prediction_len: int = 2,
            label_len: int = 5,
            timeenc: int = 1,
            freq: str = "h",
    ):
        self.dataset = dataset
        self.history_length = history_len
        self.prediction_length = prediction_len
        self.label_length = label_len
        self.timeenc = timeenc
        self.freq = freq
        self.__read_data__()

    def __len__(self) -> int:
        return len(self.dataset) - self.history_length - self.prediction_length + 1

    def __read_data__(self):
        df_stamp = self.dataset.reset_index()
        df_stamp = df_stamp[["date"]].values.transpose(1, 0)
        data_stamp = get_time_mark(df_stamp, self.timeenc, self.freq)[0]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.history_length
        r_begin = s_end - self.label_length
        r_end = r_begin + self.label_length + self.prediction_length

        seq_x = self.dataset[s_begin:s_end]
        seq_y = self.dataset[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        seq_x = torch.tensor(seq_x.values, dtype=torch.float32)
        seq_y = torch.tensor(seq_y.values, dtype=torch.float32)
        seq_x_mark = torch.tensor(seq_x_mark, dtype=torch.float32)
        seq_y_mark = torch.tensor(seq_y_mark, dtype=torch.float32)
        return seq_x, seq_y, seq_x_mark, seq_y_mark
