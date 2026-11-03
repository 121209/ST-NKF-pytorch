"""PyTorch KernelPointProcess used by the ST-NKF / ETAS training script.

The conditional intensity and the compensated log-likelihood follow
``models.py`` in zhancx-edu/NeuralKernelFunction:

    l_ts = mu + sum_i kappa(m_i) * h(t - t_i) * w(x - x_i, y - y_i)
    l_t  = mu * area + sum_i kappa(m_i) * h(t - t_i)
    Int  = mu * area * dt + sum_i kappa(m_i) * (F(t - t_i) - F(t_prev - t_i))
    loss = -mean(log(l_ts) - Int)

Neural temporal and spatial densities are input-derivatives of the network
CDF, matching ``tf.keras.backend.gradients``. ``create_graph=True`` keeps the
path into the kernel parameters.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from layers import (
    BackgroundFunction,
    KappaNetwork,
    OmoriFunction,
    ProductionFunction,
    SpatialFunction,
    SpatialKernelNetwork,
    TemporalKernelNetwork,
)
from utils import process_data

logger = logging.getLogger(__name__)

_MODES = ("empirical", "neural")


def _grad_wrt(output: torch.Tensor, inp: torch.Tensor, create_graph: bool) -> torch.Tensor:
    """Elementwise d(output)/d(inp) for a pointwise network or CDF."""
    grad = torch.autograd.grad(
        output.sum(),
        inp,
        create_graph=create_graph,
        retain_graph=True,
    )[0]
    if grad is None:
        raise RuntimeError("CDF does not depend on the differentiated input.")
    return grad


class KernelPointProcess(nn.Module):
    """Limited-history ETAS or ST-NKF, or a mix of empirical and neural kernels."""

    def __init__(
        self,
        time_step_train: int,
        time_step_val: int,
        time_step_test: int,
        temporal_id: str,
        spatial_id: str,
        kappa_id: str,
        global_m0: float,
        area: float,
        size_layer: int,
        size_nn: int,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        for name, value in (
            ("temporal_id", temporal_id),
            ("spatial_id", spatial_id),
            ("kappa_id", kappa_id),
        ):
            if value not in _MODES:
                raise ValueError(f"Invalid {name}: {value}")
        self.time_step_train = int(time_step_train)
        self.time_step_val = int(time_step_val)
        self.time_step_test = int(time_step_test)
        self.area_scalar = float(area)
        self.size_layer = int(size_layer)
        self.size_nn = int(size_nn)
        self.temporal_id = temporal_id
        self.spatial_id = spatial_id
        self.kappa_id = kappa_id
        self.global_m0 = float(global_m0)
        self.seed = seed
        self.lr = 0.005
        self.max_time: float | None = None
        self.max_mag: float | None = None
        self.max_dis: float | None = None

        self.bac_func = BackgroundFunction()
        self.produc_func = ProductionFunction(global_m0)
        self.omori_func = OmoriFunction()
        self.spatial_func = SpatialFunction(global_m0)
        self.temporal_net = TemporalKernelNetwork(size_nn, size_layer, seed=seed)
        self.spatial_net = SpatialKernelNetwork(size_nn, size_layer, seed=seed)
        self.kappa_net = KappaNetwork(size_nn, size_layer, seed=seed)

    def set_train_data(self, times, mags, locs_x, locs_y) -> "KernelPointProcess":
        (
            self.hist_t_train,
            self.hist_m_train,
            self.hist_x_train,
            self.hist_y_train,
            self.cur_t_train,
            self.cur_x_train,
            self.cur_y_train,
            self.dis_xy_train,
            self.dis_t1_train,
            self.dis_t2_train,
            self.elapsed_t_train,
        ) = process_data(self.time_step_train, times, mags, locs_x, locs_y)
        return self

    def set_val_data(self, times, mags, locs_x, locs_y) -> "KernelPointProcess":
        (
            self.hist_t_val,
            self.hist_m_val,
            self.hist_x_val,
            self.hist_y_val,
            self.cur_t_val,
            self.cur_x_val,
            self.cur_y_val,
            self.dis_xy_val,
            self.dis_t1_val,
            self.dis_t2_val,
            self.elapsed_t_val,
        ) = process_data(self.time_step_val, times, mags, locs_x, locs_y)
        return self

    def set_test_data(self, times, mags, locs_x, locs_y) -> "KernelPointProcess":
        (
            self.hist_t_test,
            self.hist_m_test,
            self.hist_x_test,
            self.hist_y_test,
            self.cur_t_test,
            self.cur_x_test,
            self.cur_y_test,
            self.dis_xy_test,
            self.dis_t1_test,
            self.dis_t2_test,
            self.elapsed_t_test,
        ) = process_data(self.time_step_test, times, mags, locs_x, locs_y)
        return self

    def set_model(self) -> "KernelPointProcess":
        """Freeze the training-set scales used to normalize neural inputs."""
        self.max_time = float(np.max(self.dis_t1_train))
        self.max_mag = float(np.max(self.hist_m_train))
        self.max_dis = float(np.max(self.dis_xy_train))
        if self.max_time <= 0 or self.max_mag <= 0 or self.max_dis <= 0:
            raise ValueError(
                "Non-positive normalization scale: "
                f"max_time={self.max_time}, max_mag={self.max_mag}, max_dis={self.max_dis}"
            )
        logger.info(
            "Normalization scales max_time=%.6g max_mag=%.6g max_dis=%.6g area=%.6g",
            self.max_time,
            self.max_mag,
            self.max_dis,
            self.area_scalar,
        )
        return self

    def compile(self, lr: float = 0.005) -> "KernelPointProcess":
        self.lr = float(lr)
        return self

    def adam_parameters(self):
        """Trainable tensors except ``mu``, which the epoch callback overwrites."""
        for name, param in self.named_parameters():
            if name == "bac_func.mu":
                continue
            yield param

    def project_constraints(self) -> None:
        self.bac_func.project()
        self.produc_func.project()
        self.omori_func.project()
        self.spatial_func.project()
        self.temporal_net.project()
        self.spatial_net.project()
        self.kappa_net.project()

    def forward(
        self,
        time_in_1: torch.Tensor,
        time_in_2: torch.Tensor,
        mags_in_1: torch.Tensor,
        dis_in: torch.Tensor,
        create_graph: bool = True,
    ) -> dict[str, torch.Tensor]:
        if self.max_time is None or self.max_mag is None or self.max_dis is None:
            raise RuntimeError("Call set_model() before forward.")

        mags_in_2 = mags_in_1[:, :-1, :]

        if self.spatial_id == "neural":
            dis = dis_in.detach().clone().requires_grad_(True)
            cdf_spatial = self.spatial_net(dis / self.max_dis)
            d_v_r = _grad_wrt(cdf_spatial, dis, create_graph)
            pdf_spatial = d_v_r / (2.0 * math.pi * (dis + 1e-5))
        else:
            pdf_spatial, cdf_spatial = self.spatial_func(dis_in ** 2, mags_in_1)

        t1 = time_in_1.detach().clone().requires_grad_(True)
        if self.temporal_id == "neural":
            cdf_temporal_1 = self.temporal_net(t1 / self.max_time)
            cdf_temporal_2 = self.temporal_net(time_in_2 / self.max_time)
        else:
            cdf_temporal_1 = self.omori_func(t1)
            cdf_temporal_2 = self.omori_func(time_in_2)
        pdf_temporal_1 = _grad_wrt(cdf_temporal_1, t1, create_graph)

        if self.kappa_id == "neural":
            kappa_1 = self.kappa_net(mags_in_1 / self.max_mag)
            kappa_2 = self.kappa_net(mags_in_2 / self.max_mag)
        else:
            kappa_1 = self.produc_func(mags_in_1)
            kappa_2 = self.produc_func(mags_in_2)

        int_tri = (cdf_temporal_1 * kappa_1).sum(dim=1, keepdim=True) - (
            cdf_temporal_2 * kappa_2
        ).sum(dim=1, keepdim=True)
        int_l = int_tri + self.bac_func.mu * self.area_scalar * t1[:, -1:, :]

        lam_tri_ts = (kappa_1 * pdf_temporal_1 * pdf_spatial).sum(dim=1, keepdim=True)
        lam_tri_t = (kappa_1 * pdf_temporal_1).sum(dim=1, keepdim=True)
        l_ts = lam_tri_ts + self.bac_func.mu
        l_t = lam_tri_t + self.bac_func.mu * self.area_scalar
        return {
            "l_ts": l_ts,
            "l_t": l_t,
            "int_l": int_l,
            "pdf_temporal": pdf_temporal_1,
            "cdf_temporal": cdf_temporal_1,
            "pdf_spatial": pdf_spatial,
            "cdf_spatial": cdf_spatial,
            "kappa": kappa_1,
        }

    def _arrays(self, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            getattr(self, f"dis_t1_{split}"),
            getattr(self, f"dis_t2_{split}"),
            getattr(self, f"hist_m_{split}"),
            getattr(self, f"dis_xy_{split}"),
        )

    def _batch_tensors(self, split: str, indices: np.ndarray):
        dis_t1, dis_t2, hist_m, dis_xy = self._arrays(split)
        device = self.bac_func.mu.device

        def take(array: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.ascontiguousarray(array[indices])).to(device)

        return take(dis_t1), take(dis_t2), take(hist_m), take(dis_xy)

    @staticmethod
    def nll(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return -(torch.log(outputs["l_ts"]) - outputs["int_l"]).mean()

    def loss_on(self, split: str, indices: np.ndarray, create_graph: bool) -> torch.Tensor:
        time_1, time_2, mags, dis = self._batch_tensors(split, indices)
        return self.nll(self.forward(time_1, time_2, mags, dis, create_graph=create_graph))

    def _mean_loss(self, split: str, batch_size: int, create_graph: bool) -> float:
        n = int(self._arrays(split)[0].shape[0])
        total = 0.0
        seen = 0
        for start in range(0, n, batch_size):
            idx = np.arange(start, min(start + batch_size, n))
            loss = self.loss_on(split, idx, create_graph=create_graph)
            total += float(loss.detach().cpu()) * len(idx)
            seen += len(idx)
            del loss
        return total / seen

    def update_mu(self, batch_size: int = 512) -> float:
        """Stochastic-declustering update used by ``UpdateMuCallback``.

        ``mu <- max(sum(mu / l_ts) / (sum(dt) * area), 1e-9)`` on the training
        events. Validation loss for the epoch is computed before this write,
        and early-stopping snapshots are taken after it, matching Keras
        callback order (``UpdateMuCallback`` then ``EarlyStopping``).
        """
        n = int(self.dis_t1_train.shape[0])
        acc = torch.zeros((), dtype=torch.float32, device=self.bac_func.mu.device)
        mu = self.bac_func.mu.detach()
        for start in range(0, n, batch_size):
            idx = np.arange(start, min(start + batch_size, n))
            time_1, time_2, mags, dis = self._batch_tensors("train", idx)
            outputs = self.forward(time_1, time_2, mags, dis, create_graph=False)
            lam = outputs["l_ts"].detach()
            acc = acc + (mu / lam).sum()
            del outputs
        background_events = float(acc.detach().cpu())
        total_time = float(np.sum(self.elapsed_t_train, dtype=np.float64))
        mu_new = max(background_events / (total_time * self.area_scalar), 1e-9)
        with torch.no_grad():
            self.bac_func.mu.fill_(mu_new)
        return mu_new

    def _snapshot(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}

    def fit_eval(
        self,
        epochs: int = 1000,
        batch_size: int = 128,
        patience: int = 20,
        mu_batch_size: int = 512,
    ) -> "KernelPointProcess":
        """Adam training with Keras 2.14 early-stopping semantics.

        Best weights are restored only when patience is exceeded. A run that
        reaches ``epochs`` keeps the last epoch, including that epoch's ``mu``
        update. Shuffle is deterministic given ``seed`` but does not reproduce
        the TensorFlow batch order.
        """
        if self.max_time is None:
            raise RuntimeError("Call set_model() before fit_eval().")
        n = int(self.dis_t1_train.shape[0])
        optimizer = torch.optim.Adam(self.adam_parameters(), lr=self.lr)
        rng = np.random.default_rng(0 if self.seed is None else int(self.seed))

        best_val = float("inf")
        best_epoch = 0
        best_state: dict[str, torch.Tensor] | None = None
        wait = 0
        stopped = False

        for epoch in range(int(epochs)):
            order = rng.permutation(n)
            running = 0.0
            seen = 0
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss = self.loss_on("train", idx, create_graph=True)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite training loss at epoch {epoch + 1}. "
                        "Check kernel constraints and catalog times."
                    )
                loss.backward()
                optimizer.step()
                self.project_constraints()
                running += float(loss.detach().cpu()) * len(idx)
                seen += len(idx)
                del loss
            train_loss = running / seen
            val_loss = self._mean_loss("val", batch_size, create_graph=False)
            if not math.isfinite(val_loss):
                raise FloatingPointError(
                    f"Non-finite validation loss at epoch {epoch + 1}."
                )
            mu_value = self.update_mu(batch_size=mu_batch_size)

            wait += 1
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                best_state = self._snapshot()
                wait = 0
            logger.info(
                "Epoch %d/%d loss=%.6f val_loss=%.6f mu=%.6e",
                epoch + 1,
                epochs,
                train_loss,
                val_loss,
                mu_value,
            )
            if wait >= patience and epoch > 0:
                if best_state is not None:
                    self.load_state_dict(best_state)
                stopped = True
                logger.info(
                    "Early stopping at epoch %d. Restored weights from epoch %d (val_loss=%.6f).",
                    epoch + 1,
                    best_epoch + 1,
                    best_val,
                )
                break

        self.stopped_early = stopped
        self.best_epoch = best_epoch + 1
        self.best_val_loss = best_val
        return self

    def _predict_split(self, split: str, batch_size: int) -> dict[str, np.ndarray]:
        n = int(self._arrays(split)[0].shape[0])
        keys = (
            "l_ts",
            "l_t",
            "int_l",
            "pdf_temporal",
            "cdf_temporal",
            "pdf_spatial",
            "cdf_spatial",
            "kappa",
        )
        buckets: dict[str, list[torch.Tensor]] = {key: [] for key in keys}
        for start in range(0, n, batch_size):
            idx = np.arange(start, min(start + batch_size, n))
            time_1, time_2, mags, dis = self._batch_tensors(split, idx)
            outputs = self.forward(time_1, time_2, mags, dis, create_graph=False)
            for key in keys:
                buckets[key].append(outputs[key].detach().cpu())
            del outputs
        return {
            key: torch.cat(parts, dim=0).numpy()
            for key, parts in buckets.items()
        }

    def _store_eval(self, split: str, outputs: dict[str, np.ndarray]) -> None:
        lam_ts = outputs["l_ts"]
        lam_t = outputs["l_t"]
        int_l = outputs["int_l"]
        ll_ts = np.log(lam_ts) - int_l
        ll_t = np.log(lam_t) - int_l
        ll_s = ll_ts - ll_t
        setattr(self, f"lam_ts_{split}", lam_ts)
        setattr(self, f"lam_t_{split}", lam_t)
        setattr(self, f"Int_lam_{split}", int_l)
        setattr(self, f"pdf_temporal_{split}", outputs["pdf_temporal"])
        setattr(self, f"cdf_temporal_{split}", outputs["cdf_temporal"])
        setattr(self, f"pdf_spatial_{split}", outputs["pdf_spatial"])
        setattr(self, f"cdf_spatial_{split}", outputs["cdf_spatial"])
        setattr(self, f"kappa_{split}", outputs["kappa"])
        setattr(self, f"LL_ts_{split}", ll_ts)
        setattr(self, f"LL_t_{split}", ll_t)
        setattr(self, f"LL_s_{split}", ll_s)
        setattr(self, f"LL_ts_average_{split}", float(np.mean(ll_ts)))
        setattr(self, f"LL_t_average_{split}", float(np.mean(ll_t)))
        setattr(self, f"LL_s_average_{split}", float(np.mean(ll_s)))

    def eval_train(self, batch_size: int = 128) -> "KernelPointProcess":
        self._store_eval("train", self._predict_split("train", batch_size))
        return self

    def eval_val(self, batch_size: int = 128) -> "KernelPointProcess":
        self._store_eval("val", self._predict_split("val", batch_size))
        return self

    def eval_test(self, batch_size: int = 128) -> "KernelPointProcess":
        self._store_eval("test", self._predict_split("test", batch_size))
        return self

    def save_weights(self, path: str) -> "KernelPointProcess":
        destination = Path(path)
        if destination.suffix == "":
            destination = destination.with_suffix(".pt")
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "max_time": self.max_time,
                "max_mag": self.max_mag,
                "max_dis": self.max_dis,
                "area": self.area_scalar,
                "temporal_id": self.temporal_id,
                "spatial_id": self.spatial_id,
                "kappa_id": self.kappa_id,
                "global_m0": self.global_m0,
                "size_layer": self.size_layer,
                "size_nn": self.size_nn,
                "seed": self.seed,
                "lr": self.lr,
            },
            destination,
        )
        self.weights_path = str(destination)
        logger.info("Weights saved to %s", destination)
        return self

    def load_weights(self, path: str) -> "KernelPointProcess":
        checkpoint = torch.load(path, map_location=self.bac_func.mu.device)
        self.load_state_dict(checkpoint["state_dict"])
        self.max_time = checkpoint.get("max_time", self.max_time)
        self.max_mag = checkpoint.get("max_mag", self.max_mag)
        self.max_dis = checkpoint.get("max_dis", self.max_dis)
        logger.info("Weights loaded from %s", path)
        return self
