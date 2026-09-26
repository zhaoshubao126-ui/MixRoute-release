

import copy
import logging
import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch import optim
from torch.utils.data import DataLoader

from ts_benchmark.baselines.mixroute.models.mixroute_model import MixRouteModel
from ts_benchmark.baselines.mixroute.utils.utils import (
    forecasting_data_provider,
    train_val_split,
    get_time_mark,
    EarlyStopping,
)
from ts_benchmark.models.model_base import ModelBase, BatchMaker

logger = logging.getLogger(__name__)

DEFAULT_HYPER_PARAMS = {
    
    "loss": "MAE",
    "batch_size": 64,
    "lr": 0.0001,
    "lradj": "cosine",
    "warmup_epochs": 3,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "num_epochs": 100,
    "patience": 30,
    "num_workers": 0,
    "parallel_strategy": "DP",
    "norm": True,

    
    "seq_len": 96,
    "patch_len": 24,
    "d_model": 256,
    "d_ff": 1024,
    "e_layers": 2,
    "n_heads": 8,
    "num_glob_tokens": 4,
    "dropout": 0.1,
    "drop_path_rate": 0.1,
    "hidden_act": "gelu",
    "rms_norm_eps": 1e-6,
    "max_position_embeddings": 1024,
    "rope_theta": 10000,

    
    "use_instance_norm": True,
    "air_orth_weight": 0.0,
}


class Config:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_HYPER_PARAMS.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)

        if hasattr(self, "horizon"):
            logger.warning(
                "The model parameter horizon is deprecated. Please use pred_len."
            )
            setattr(self, "pred_len", self.horizon)


class MixRoute(ModelBase):

    def __init__(self, **kwargs):
        super(MixRoute, self).__init__()
        self.config = Config(**kwargs)
        self.scaler1 = StandardScaler()
        self.scaler2 = StandardScaler()

    def _init_model(self):
        return MixRouteModel(self.config)

    def _adjust_lr(self, optimizer, epoch, config):
        
        if config.lradj == "cosine":
            T = config.num_epochs
            scale = 0.01 + 0.5 * 0.99 * (1 + math.cos(math.pi * epoch / T))
        elif config.lradj == "warmup_cosine":
            warmup = config.warmup_epochs
            T = config.num_epochs
            if epoch < warmup:
                scale = (epoch + 1) / (warmup + 1)
            else:
                progress = (epoch - warmup) / max(T - warmup, 1)
                scale = 0.01 + 0.5 * 0.99 * (1 + math.cos(math.pi * progress))
        elif config.lradj == "constant":
            return
        else:
            raise ValueError(f"Unknown lradj schedule: {config.lradj}")

        for i, pg in enumerate(optimizer.param_groups):
            pg["lr"] = self._initial_lrs[i] * scale

    def save_checkpoint(self, models):
        
        return {key: copy.deepcopy(model.state_dict()) for key, model in models.items()}

    def _init_criterion(self):
        
        if self.config.loss == "MSE":
            criterion = nn.MSELoss(reduction="mean")
        elif self.config.loss == "MAE":
            criterion = nn.L1Loss(reduction="mean")
        else:
            raise ValueError(f"Unknown loss: {self.config.loss}")
        return criterion

    def _init_optimizer(self):
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

    def _process(self, input, target, input_mark, target_mark, exog_future=None):
        
        model_output = self.model(input, exog_future)

        if isinstance(model_output, (tuple, list)):
            output = model_output[0]
            additional_loss = model_output[1] if len(model_output) >= 2 else None
        else:
            output = model_output
            additional_loss = None

        out_loss = {"output": output}
        if self.model.training and additional_loss is not None:
            out_loss["additional_loss"] = additional_loss
        else:
            out_loss["additional_loss"] = None
        return out_loss

    def _post_process(self, output, target):
        return output, target

    def _init_early_stopping(self):
        return EarlyStopping(patience=self.config.patience)

    @property
    def model_name(self):
        return "MixRoute"

    @staticmethod
    def required_hyper_params() -> dict:
        return {
            "seq_len": "input_chunk_length",
            "horizon": "output_chunk_length",
            "norm": "norm",
        }

    def __repr__(self) -> str:
        return self.model_name

    def multi_forecasting_hyper_param_tune(self, train_data: pd.DataFrame):
        freq = pd.infer_freq(train_data.index)
        if freq == None:
            raise ValueError("Irregular time intervals")
        elif freq[0].lower() not in ["m", "w", "b", "d", "h", "t", "s"]:
            self.config.freq = "s"
        else:
            self.config.freq = freq[0].lower()

        column_num = train_data.shape[1]
        self.config.enc_in = column_num
        setattr(self.config, "label_len", self.config.seq_len // 2)

    def single_forecasting_hyper_param_tune(self, train_data: pd.DataFrame):
        freq = pd.infer_freq(train_data.index)
        if freq == None:
            raise ValueError("Irregular time intervals")
        elif freq[0].lower() not in ["m", "w", "b", "d", "h", "t", "s"]:
            self.config.freq = "s"
        else:
            self.config.freq = freq[0].lower()

        column_num = train_data.shape[1]
        self.config.enc_in = column_num
        setattr(self.config, "label_len", self.config.horizon)

    def _padding_time_stamp_mark(
            self, time_stamps_list: np.ndarray, padding_len: int
    ) -> np.ndarray:
        
        padding_time_stamp = []
        for time_stamps in time_stamps_list:
            start = time_stamps[-1]
            expand_time_stamp = pd.date_range(
                start=start,
                periods=padding_len + 1,
                freq=self.config.freq.upper(),
            )
            padding_time_stamp.append(expand_time_stamp.to_numpy()[-padding_len:])
        padding_time_stamp = np.stack(padding_time_stamp)
        whole_time_stamp = np.concatenate(
            (time_stamps_list, padding_time_stamp), axis=1
        )
        padding_mark = get_time_mark(whole_time_stamp, 1, self.config.freq)
        return padding_mark

    def validate(
            self, valid_data_loader: DataLoader, series_dim: int, criterion: torch.nn.Module
    ) -> float:
        
        config = self.config
        total_loss = []
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            for input, target, input_mark, target_mark in valid_data_loader:
                input, target, input_mark, target_mark = (
                    input.to(device),
                    target.to(device),
                    input_mark.to(device),
                    target_mark.to(device),
                )
                exog_future = target[:, -config.horizon:, series_dim:]
                out_loss = self._process(
                    input, target, input_mark, target_mark, exog_future
                )
                additional_loss = 0
                output = out_loss["output"]
                if out_loss.get("additional_loss") is not None:
                    additional_loss = out_loss["additional_loss"]
                    if isinstance(additional_loss, torch.Tensor) and additional_loss.dim() > 0:
                        additional_loss = additional_loss.mean()

                target = target[:, -config.horizon:, :series_dim]
                output = output[:, -config.horizon:, :series_dim]
                output, target = self._post_process(output, target)
                all_loss = criterion(output, target) + additional_loss
                loss = all_loss.detach().cpu().numpy()
                total_loss.append(loss)

        total_loss = np.mean(total_loss)
        self.model.train()
        return total_loss

    def forecast_fit(
            self,
            train_valid_data: pd.DataFrame,
            *,
            covariates: Optional[dict] = None,
            train_ratio_in_tv: float = 1.0,
            **kwargs,
    ) -> "ModelBase":
        
        if covariates is None:
            covariates = {}
        series_dim = train_valid_data.shape[-1]
        exog_data = covariates.get("exog", None)
        if exog_data is not None:
            train_valid_data = pd.concat([train_valid_data, exog_data], axis=1)
            exog_dim = exog_data.shape[-1]
        else:
            exog_dim = 0

        if train_valid_data.shape[1] == 1:
            train_drop_last = False
            self.single_forecasting_hyper_param_tune(train_valid_data)
        else:
            train_drop_last = True
            self.multi_forecasting_hyper_param_tune(train_valid_data)

        self.config.series_dim = series_dim
        self.model_pred_len = self.config.patch_len

        criterion = self._init_criterion()
        self.model = self._init_model()
        device_ids = np.arange(torch.cuda.device_count()).tolist()
        if len(device_ids) > 1 and self.config.parallel_strategy == "DP":
            self.model = nn.DataParallel(self.model, device_ids=device_ids)
        print(
            "----------------------------------------------------------",
            self.model_name,
        )
        config = self.config
        train_data, valid_data = train_val_split(
            train_valid_data, train_ratio_in_tv, config.seq_len
        )

        
        if exog_dim > 0:
            self.scaler1.fit(train_data.values[:, :series_dim])
            self.scaler2.fit(train_data.values[:, series_dim:])

            if config.norm:
                scaled_series = self.scaler1.transform(
                    train_data.values[:, :series_dim]
                )
                scaled_exog = self.scaler2.transform(train_data.values[:, series_dim:])
                final_train_data = np.concatenate((scaled_series, scaled_exog), axis=1)
                train_data = pd.DataFrame(
                    final_train_data,
                    columns=train_data.columns,
                    index=train_data.index,
                )
        else:
            self.scaler1.fit(train_data.values)
            if config.norm:
                train_data = pd.DataFrame(
                    self.scaler1.transform(train_data.values),
                    columns=train_data.columns,
                    index=train_data.index,
                )

        if train_ratio_in_tv != 1:
            if config.norm:
                if exog_dim > 0:
                    scaled_series = self.scaler1.transform(
                        valid_data.values[:, :series_dim]
                    )
                    scaled_exog = self.scaler2.transform(
                        valid_data.values[:, series_dim:]
                    )
                    final_valid_data = np.concatenate(
                        (scaled_series, scaled_exog), axis=1
                    )
                    valid_data = pd.DataFrame(
                        final_valid_data,
                        columns=valid_data.columns,
                        index=valid_data.index,
                    )
                else:
                    valid_data = pd.DataFrame(
                        self.scaler1.transform(valid_data.values),
                        columns=valid_data.columns,
                        index=valid_data.index,
                    )
            valid_dataset, valid_data_loader = forecasting_data_provider(
                valid_data,
                config,
                timeenc=1,
                batch_size=config.batch_size,
                shuffle=True,
                drop_last=False,
            )

        
        
        train_dataset, self.train_data_loader = forecasting_data_provider(
            train_data,
            config,
            timeenc=1,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=train_drop_last,
            generator=torch.Generator().manual_seed(torch.initial_seed()),
        )
        optimizer = self._init_optimizer()
        self._initial_lrs = [pg["lr"] for pg in optimizer.param_groups]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.early_stopping = self._init_early_stopping()
        self.model.to(device)
        total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"Total trainable parameters: {total_params}")

        valid_loss = None
        for epoch in range(config.num_epochs):
            self.model.train()

            epoch_train_losses = []
            for i, (input, target, input_mark, target_mark) in enumerate(
                    self.train_data_loader
            ):
                optimizer.zero_grad()
                input, target, input_mark, target_mark = (
                    input.to(device),
                    target.to(device),
                    input_mark.to(device),
                    target_mark.to(device),
                )
                exog_future = target[:, -config.horizon:, series_dim:].to(device)

                out_loss = self._process(
                    input, target, input_mark, target_mark, exog_future
                )
                additional_loss = 0
                output = out_loss["output"]
                if out_loss.get("additional_loss") is not None:
                    additional_loss = out_loss["additional_loss"]
                    if isinstance(additional_loss, torch.Tensor) and additional_loss.dim() > 0:
                        additional_loss = additional_loss.mean()

                target = target[:, -config.horizon:, :series_dim]
                output = output[:, -config.horizon:, :series_dim]
                output, target = self._post_process(output, target)
                loss = criterion(output, target)

                total_loss = loss + (additional_loss if additional_loss is not None else 0)

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=config.grad_clip
                )
                optimizer.step()

                if self.config.lradj == "TST":
                    self._adjust_lr(optimizer, epoch + 1, config)

                epoch_train_losses.append(loss.detach().cpu().item())

            avg_train_loss = np.mean(epoch_train_losses)
            print(f"Epoch [{epoch + 1}/{config.num_epochs}] - Train Loss: {avg_train_loss:.6f}", end="")

            if train_ratio_in_tv != 1:
                valid_loss = self.validate(valid_data_loader, series_dim, criterion)
                assert not np.isnan(valid_loss), "valid loss is nan"

                print(f" - Val Loss: {valid_loss:.6f}")
                improved = self.early_stopping(valid_loss, self.model)
                if improved:
                    print(f"Validation loss decreased ({self.early_stopping.best_score:.6f} --> {valid_loss:.6f}).  Saving model ...")
                    self.check_point = self.save_checkpoint({"Model": self.model})
                else:
                    print(f"EarlyStopping counter: {self.early_stopping.counter} out of {config.patience}")

                if self.early_stopping.early_stop:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break
            else:
                print()

            if self.config.lradj != "TST":
                self._adjust_lr(optimizer, epoch + 1, config)

    def forecast(
            self,
            horizon: int,
            series: pd.DataFrame,
            *,
            covariates: Optional[dict] = None,
    ) -> np.ndarray:
        raise NotImplementedError("Use batch_forecast for rolling evaluation.")

    def batch_forecast(
            self, horizon: int, batch_maker: BatchMaker, exog_futures, i, **kwargs
    ) -> np.ndarray:
        
        if self.check_point is not None:
            self.model.load_state_dict(self.check_point["Model"])
        if self.model is None:
            raise ValueError("Model not trained. Call the fit() function first.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        input_data = batch_maker.make_batch(self.config.batch_size, self.config.seq_len)
        input_np = input_data["input"]
        series_dim = input_np.shape[-1]
        batch_size = self.config.batch_size
        if input_data["covariates"] is None:
            covariates = {}
        else:
            covariates = input_data["covariates"]
        exog_data = covariates.get("exog")
        if exog_data is not None:
            exog_dim = exog_data.shape[-1]
            input_np = np.concatenate((input_np, exog_data), axis=2)
            if (
                    hasattr(self.config, "output_chunk_length")
                    and horizon != self.config.output_chunk_length
            ):
                raise ValueError(
                    f"Error: 'exog' is enabled during training, but horizon ({horizon}) != output_chunk_length ({self.config.output_chunk_length}) during forecast."
                )
        else:
            exog_dim = 0
        if self.config.norm:
            if exog_dim > 0:
                series_data = input_np[..., :series_dim]
                origin_shape1 = series_data.shape
                flattened_data = series_data.reshape((-1, series_data.shape[-1]))
                series_data = self.scaler1.transform(flattened_data).reshape(
                    origin_shape1
                )
                exog_data = input_np[..., series_dim:]
                origin_shape2 = exog_data.shape
                flattened_data = exog_data.reshape((-1, exog_data.shape[-1]))
                exog_data = self.scaler2.transform(flattened_data).reshape(
                    origin_shape2
                )

                input_np = np.concatenate((series_data, exog_data), axis=2)
            else:
                origin_shape = input_np.shape
                flattened_data = input_np.reshape((-1, input_np.shape[-1]))
                input_np = self.scaler1.transform(flattened_data).reshape(origin_shape)

        
        if exog_futures is not None:
            exog_future = torch.tensor(
                exog_futures[i * batch_size: (i + 1) * batch_size, -horizon:, :]
            ).to(device)
        else:
            exog_future = None

        if self.config.norm and exog_dim > 0:
            flattened_data = exog_future.reshape((-1, exog_future.shape[-1]))
            flattened_data_np = flattened_data.cpu().numpy()
            exog_future = self.scaler2.transform(flattened_data_np).reshape(
                exog_future.shape
            )
            exog_future = torch.tensor(exog_future).to(device)
        input_index = input_data["time_stamps"]
        padding_len = (
                              math.ceil(horizon / self.config.horizon) + 1
                      ) * self.config.horizon
        all_mark = self._padding_time_stamp_mark(input_index, padding_len)

        answers = self._perform_rolling_predictions(
            horizon, input_np, exog_future, series_dim, all_mark, device
        )

        if self.config.norm:
            flattened_data = answers.reshape((-1, answers.shape[-1]))
            answers = self.scaler1.inverse_transform(flattened_data).reshape(
                answers.shape
            )

        return answers[..., :series_dim]

    def _perform_rolling_predictions(
            self,
            horizon: int,
            input_np: np.ndarray,
            exog_future: torch.Tensor,
            series_dim: int,
            all_mark: np.ndarray,
            device: torch.device,
    ) -> list:
        
        rolling_time = 0
        input_np, target_np, input_mark_np, target_mark_np = self._get_rolling_data(
            input_np, None, all_mark, rolling_time
        )
        if exog_future is not None:
            rolling_time_sum = horizon // self.model_pred_len + 1
            padding = rolling_time_sum * self.model_pred_len - horizon
            padding_tensor = torch.zeros(
                exog_future.shape[0], padding, exog_future.shape[-1]
            ).to(device)
            exog_future = torch.cat(
                (exog_future, padding_tensor),
                dim=1
            )
            exog_future = exog_future.float()
        with torch.no_grad():
            answers = []
            while not answers or sum(a.shape[1] for a in answers) < horizon:
                input, dec_input, input_mark, target_mark = (
                    torch.tensor(input_np, dtype=torch.float32).to(device),
                    torch.tensor(target_np, dtype=torch.float32).to(device),
                    torch.tensor(input_mark_np, dtype=torch.float32).to(device),
                    torch.tensor(target_mark_np, dtype=torch.float32).to(device),
                )

                exog_future_sample = exog_future[
                                     :, rolling_time * self.model_pred_len:, :,
                                     ] if exog_future is not None else None

                out_loss = self._process(
                    input, dec_input, input_mark, target_mark, exog_future_sample
                )
                output = out_loss["output"]
                output1 = output[:, -self.config.horizon:, :series_dim]

                column_num = output.shape[-1]
                real_batch_size = output.shape[0]
                output = torch.cat(
                    [output1, output[:, -self.config.horizon:, series_dim:]], dim=-1
                )
                answer = (
                    output.cpu()
                    .numpy()
                    .reshape(real_batch_size, -1, column_num)[
                    :, -self.config.horizon:, :
                    ]
                )
                answers.append(answer)
                if sum(a.shape[1] for a in answers) >= horizon:
                    break
                rolling_time += 1
                output = output.cpu().numpy()[:, -self.config.horizon:, :]
                new_exog_future_sample = exog_future[
                                         :,
                                         rolling_time * self.model_pred_len:rolling_time * self.model_pred_len + self.model_pred_len,
                                         :,
                                         ].cpu().numpy() if exog_future is not None else None
                output = np.concatenate((output, new_exog_future_sample), axis=-1)
                input_np, target_np, input_mark_np, target_mark_np = self._get_rolling_data(
                    input_np,
                    output,
                    all_mark,
                    rolling_time
                )

        answers = np.concatenate(answers, axis=1)
        return answers[:, -horizon:, :]

    def _get_rolling_data(
            self,
            input_np: np.ndarray,
            output: Optional[np.ndarray],
            all_mark: np.ndarray,
            rolling_time: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        if rolling_time > 0:
            input_np = np.concatenate((input_np, output), axis=1)
            input_np = input_np[:, -self.config.seq_len:, :]
        target_np = np.zeros(
            (
                input_np.shape[0],
                self.config.label_len + self.config.horizon,
                input_np.shape[2],
            )
        )
        target_np[:, : self.config.label_len, :] = input_np[
                                                   :, -self.config.label_len:, :
                                                   ]
        advance_len = rolling_time * self.config.horizon
        input_mark_np = all_mark[:, advance_len: self.config.seq_len + advance_len, :]
        start = self.config.seq_len - self.config.label_len + advance_len
        end = self.config.seq_len + self.config.horizon + advance_len
        target_mark_np = all_mark[
                         :,
                         start:end,
                         :,
                         ]
        return input_np, target_np, input_mark_np, target_mark_np
