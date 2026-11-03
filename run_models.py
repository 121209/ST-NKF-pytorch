"""Train ETAS or ST-NKF on an EarthquakeNPP catalog.

PyTorch reimplementation of zhancx-edu/NeuralKernelFunction ``run_models.py``.

Command line (README / this port):

    python run_models.py <seed> <gpu_id> <dataset> <kappa> <temporal> <spatial>

``kappa``, ``temporal`` and ``spatial`` are each ``empirical`` or ``neural``.
The upstream script assigns those three positions in the opposite component
order (temporal, spatial, kappa). All-empirical and all-neural commands from
the original README are unchanged. Mixed ablations should follow the order
printed by ``--help``.

``gpu_id`` is a CUDA device index. Use ``-1`` or ``cpu`` when no GPU is
available. Optional flags ``--epochs``, ``--batch-size`` and ``--patience``
default to the paper settings (1000, 128, 20).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

from config import DATASET_CONFIGS
from utils import azimuthal_equidistant_projection, quadrilateral_area

logger = logging.getLogger(__name__)

DATASETS = (
    "ComCat",
    "SaltonSea",
    "SanJac",
    "WHITE",
    "SCEDC_20",
    "SCEDC_25",
    "SCEDC_30",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a KernelPointProcess (ETAS, ST-NKF, or a hybrid)."
    )
    parser.add_argument("seed", type=int, help="NumPy / PyTorch / kernel-init seed")
    parser.add_argument(
        "gpu_id",
        type=str,
        help="CUDA device id, or -1/cpu to force CPU",
    )
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("kappa", choices=("empirical", "neural"), help="productivity")
    parser.add_argument("temporal", choices=("empirical", "neural"))
    parser.add_argument("spatial", choices=("empirical", "neural"))
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument(
        "--mu-batch-size",
        type=int,
        default=512,
        help="Batch size for the end-of-epoch background-rate update",
    )
    parser.add_argument(
        "--no-patterns",
        action="store_true",
        help="Skip the sorted kernel-curve .npy dumps under results/",
    )
    return parser.parse_args(argv)


def configure_device(gpu_id: str):
    """Set CUDA_VISIBLE_DEVICES before torch is imported."""
    if gpu_id.lower() in {"-1", "cpu", "none"}:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id


def prepare_run(datasets_id: str) -> dict:
    """Load one catalog and cut the same index windows as the TensorFlow script."""
    if datasets_id not in DATASET_CONFIGS:
        raise KeyError(datasets_id)
    config = DATASET_CONFIGS[datasets_id]
    raw_catalog = pd.read_csv(config["catalog_path"])
    cat_shape = np.load(config["shape_path"])
    if datasets_id in ("SCEDC_20", "SCEDC_25", "SCEDC_30"):
        min_magnitude = float(datasets_id.split("_")[-1]) / 10.0
        raw_catalog = raw_catalog[raw_catalog["magnitude"] >= min_magnitude].reset_index(
            drop=True
        )

    times = raw_catalog["time_days"].to_numpy()
    if len(times) > 1 and np.any(times[1:] < times[:-1]):
        raise ValueError(f"{datasets_id} catalog is not sorted by time_days")

    auxiliary_num = int(
        np.sum(
            (times >= config["auxiliary_start"]) & (times < config["training_start"])
        )
    )
    training_num = int(
        np.sum(
            (times >= config["training_start"]) & (times < config["validation_start"])
        )
    )
    validation_num = int(
        np.sum(
            (times >= config["validation_start"]) & (times < config["testing_start"])
        )
    )
    testing_num = int(
        np.sum((times >= config["testing_start"]) & (times <= config["testing_end"]))
    )
    if auxiliary_num < 2 or training_num < 1 or validation_num < 1 or testing_num < 1:
        raise ValueError(
            f"Empty split for {datasets_id}: auxiliary={auxiliary_num}, "
            f"training={training_num}, validation={validation_num}, testing={testing_num}"
        )

    # Count-based slices. Valid when in-window events form a prefix, which
    # holds for these catalogs (SCEDC may have a few rows after testing_end).
    input_dim_train = auxiliary_num
    input_dim_val = auxiliary_num
    input_dim_test = auxiliary_num
    train_slice = slice(auxiliary_num - input_dim_train, auxiliary_num + training_num)
    val_slice = slice(
        auxiliary_num + training_num - input_dim_val,
        auxiliary_num + training_num + validation_num,
    )
    test_slice = slice(
        auxiliary_num + training_num + validation_num - input_dim_test,
        auxiliary_num + training_num + validation_num + testing_num,
    )

    def pack(slc: slice) -> dict:
        return {
            "t": raw_catalog["time_days"].to_numpy()[slc],
            "m": raw_catalog["magnitude"].to_numpy()[slc],
            "x": raw_catalog["x"].to_numpy()[slc],
            "y": raw_catalog["y"].to_numpy()[slc],
        }

    center_latitude = float(raw_catalog["latitude"].mean())
    center_longitude = float(raw_catalog["longitude"].mean())
    cat_shape_x, cat_shape_y = azimuthal_equidistant_projection(
        cat_shape[:, 0], cat_shape[:, 1], center_latitude, center_longitude
    )
    cat_shape_xy = np.stack((cat_shape_x, cat_shape_y), axis=1)
    area = quadrilateral_area(cat_shape_xy)

    return {
        "config": config,
        "raw_catalog": raw_catalog,
        "auxiliary_num": auxiliary_num,
        "training_num": training_num,
        "validation_num": validation_num,
        "testing_num": testing_num,
        "input_dim_train": input_dim_train,
        "input_dim_val": input_dim_val,
        "input_dim_test": input_dim_test,
        "train": pack(train_slice),
        "val": pack(val_slice),
        "test": pack(test_slice),
        "area": area,
    }


def _write_log_likelihood(datasets_id, temporal_id, spatial_id, kappa_id, model, seed: int) -> None:
    line = (
        f"{datasets_id}, {temporal_id}, {spatial_id}, {kappa_id}, "
        f"{model.LL_ts_average_test}, {model.LL_t_average_test}, "
        f"{model.LL_s_average_test}, seed={seed}\n"
    )
    with open("ll_seismic_model.log", "a", encoding="utf-8") as log_file:
        log_file.write(line)


def _scalar(tensor) -> float:
    return float(tensor.detach().cpu())


def _write_parameters(datasets_id, temporal_id, spatial_id, kappa_id, model, seed: int) -> None:
    fields = [
        datasets_id,
        f"seed={seed}",
        temporal_id,
        spatial_id,
        kappa_id,
        f"mu={_scalar(model.bac_func.mu):.8g}",
    ]
    if kappa_id == "empirical":
        fields.append(f"A={_scalar(model.produc_func.mag_A):.8g}")
        fields.append(f"alpha={_scalar(model.produc_func.mag_alpha):.8g}")
    if temporal_id == "empirical":
        fields.append(f"c={_scalar(model.omori_func.omori_c):.8g}")
        fields.append(f"p={_scalar(model.omori_func.omori_p):.8g}")
    if spatial_id == "empirical":
        fields.append(f"q={_scalar(model.spatial_func.spatial_q):.8g}")
        fields.append(f"D2={_scalar(model.spatial_func.spatial_D2):.8g}")
        fields.append(f"gamma={_scalar(model.spatial_func.spatial_gamma):.8g}")
    with open("etas_parameters.log", "a", encoding="utf-8") as log_file:
        log_file.write(", ".join(fields) + "\n")


def _write_catalog_csv(prep, model, datasets_id, temporal_id, spatial_id, kappa_id) -> None:
    results = prep["raw_catalog"].copy()
    aux = prep["auxiliary_num"]
    n_train = prep["training_num"]
    n_val = prep["validation_num"]
    n_test = prep["testing_num"]
    results.loc[aux : aux + n_train - 1, "step"] = "train"
    results.loc[aux + n_train : aux + n_train + n_val - 1, "step"] = "val"
    results.loc[
        aux + n_train + n_val : aux + n_train + n_val + n_test - 1,
        "step",
    ] = "test"

    blocks = {
        "train": (
            aux,
            n_train,
            model.LL_t_train,
            model.LL_s_train,
            model.LL_ts_train,
            np.log(model.lam_t_train),
            np.log(model.lam_ts_train),
            model.Int_lam_train,
        ),
        "val": (
            aux + n_train,
            n_val,
            model.LL_t_val,
            model.LL_s_val,
            model.LL_ts_val,
            np.log(model.lam_t_val),
            np.log(model.lam_ts_val),
            model.Int_lam_val,
        ),
        "test": (
            aux + n_train + n_val,
            n_test,
            model.LL_t_test,
            model.LL_s_test,
            model.LL_ts_test,
            np.log(model.lam_t_test),
            np.log(model.lam_ts_test),
            model.Int_lam_test,
        ),
    }
    for _name, (start, count, logli_t, logli_s, logli_ts, loglam_t, loglam_ts, intlam) in blocks.items():
        end = start + count - 1
        results.loc[start:end, "logli_t"] = np.asarray(logli_t).reshape(-1)
        results.loc[start:end, "logli_s"] = np.asarray(logli_s).reshape(-1)
        results.loc[start:end, "logli_ts"] = np.asarray(logli_ts).reshape(-1)
        results.loc[start:end, "loglam_t"] = np.asarray(loglam_t).reshape(-1)
        results.loc[start:end, "loglam_ts"] = np.asarray(loglam_ts).reshape(-1)
        results.loc[start:end, "intlam"] = np.asarray(intlam).reshape(-1)

    os.makedirs("csv", exist_ok=True)
    path = f"csv/{datasets_id}_{temporal_id}_{spatial_id}_{kappa_id}.csv"
    results.to_csv(path, index=False, encoding="utf-8")
    logger.info("Results saved to %s", path)


def _write_patterns(model, tag: str) -> None:
    os.makedirs("results", exist_ok=True)

    def dump(stem: str, values: np.ndarray, reference: np.ndarray) -> None:
        flat_ref = np.asarray(reference[:-1]).reshape(-1)
        order = np.argsort(flat_ref)
        flat_val = np.asarray(values[:-1]).reshape(-1)
        np.save(f"results/{tag}_{stem}.npy", flat_val[order])

    dump("dis_xy_train", model.dis_xy_train, model.dis_xy_train)
    dump("pdf_spatial_train", model.pdf_spatial_train, model.dis_xy_train)
    dump("cdf_spatial_train", model.cdf_spatial_train, model.dis_xy_train)
    dump("dis_t_train", model.dis_t1_train, model.dis_t1_train)
    dump("pdf_temporal_train", model.pdf_temporal_train, model.dis_t1_train)
    dump("cdf_temporal_train", model.cdf_temporal_train, model.dis_t1_train)
    dump("hist_m_train", model.hist_m_train, model.hist_m_train)
    dump("kappa_train", model.kappa_train, model.hist_m_train)
    logger.info("Kernel curves saved under results/%s_*.npy", tag)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = parse_args(argv)
    configure_device(args.gpu_id)

    import torch

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        device = torch.device("cuda")
    else:
        if args.gpu_id.lower() not in {"-1", "cpu", "none"}:
            logger.warning("CUDA is not available; training on CPU.")
        device = torch.device("cpu")

    from models import KernelPointProcess

    prep = prepare_run(args.dataset)
    logger.info(
        "%s: auxiliary=%d, training=%d, validation=%d, testing=%d, area_km2=%.6g, device=%s",
        args.dataset,
        prep["auxiliary_num"],
        prep["training_num"],
        prep["validation_num"],
        prep["testing_num"],
        prep["area"],
        device,
    )

    model = KernelPointProcess(
        time_step_train=prep["input_dim_train"],
        time_step_val=prep["input_dim_val"],
        time_step_test=prep["input_dim_test"],
        temporal_id=args.temporal,
        spatial_id=args.spatial,
        kappa_id=args.kappa,
        global_m0=prep["config"]["global_m0"],
        area=prep["area"],
        size_layer=5,
        size_nn=32,
        seed=args.seed,
    )
    model.to(device)
    (
        model.set_train_data(
            prep["train"]["t"], prep["train"]["m"], prep["train"]["x"], prep["train"]["y"]
        )
        .set_val_data(prep["val"]["t"], prep["val"]["m"], prep["val"]["x"], prep["val"]["y"])
        .set_test_data(
            prep["test"]["t"], prep["test"]["m"], prep["test"]["x"], prep["test"]["y"]
        )
        .set_model()
        .compile(lr=args.lr)
        .fit_eval(
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            mu_batch_size=args.mu_batch_size,
        )
        .eval_train(batch_size=args.batch_size)
        .eval_val(batch_size=args.batch_size)
        .eval_test(batch_size=args.batch_size)
        .save_weights(
            f"weights/{args.dataset}_{args.temporal}_{args.spatial}_{args.kappa}"
        )
    )

    logger.info(
        "Test results: LL_ts=%s, LL_t=%s, LL_s=%s",
        model.LL_ts_average_test,
        model.LL_t_average_test,
        model.LL_s_average_test,
    )
    _write_log_likelihood(
        args.dataset, args.temporal, args.spatial, args.kappa, model, args.seed
    )
    _write_parameters(
        args.dataset, args.temporal, args.spatial, args.kappa, model, args.seed
    )
    _write_catalog_csv(
        prep, model, args.dataset, args.temporal, args.spatial, args.kappa
    )
    if not args.no_patterns:
        tag = f"{args.dataset}_{args.temporal}_{args.spatial}_{args.kappa}"
        _write_patterns(model, tag)


if __name__ == "__main__":
    main()
