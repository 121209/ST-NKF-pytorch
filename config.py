"""Dataset windows for the ST-NKF / ETAS experiments.

Time boundaries are Unix-epoch days (UTC), matching the ``time_days`` column in
the jittered catalogs and the auxiliary-window lengths in the original README
(ComCat 14933, SaltonSea 2144, SanJac 1672, WHITE 2196, SCEDC_20 12373,
SCEDC_25 4242, SCEDC_30 1142).

The TensorFlow ``config.py`` builds the same numbers with a timezone-naive
``Timestamp.timestamp()``, which follows the machine local zone. These literals
are the UTC values so a run in UTC+8 does not move the split.
"""

import pandas as pd


def _utc_days(iso_date: str) -> float:
    """Days since the Unix epoch at UTC midnight of ``YYYY-MM-DD``."""
    return float(pd.Timestamp(iso_date, tz="UTC").timestamp() / 86400.0)


DATASET_CONFIGS = {
    "ComCat": {
        "catalog_path": "./Datasets/ComCat/ComCat_catalog_jitter.csv",
        "shape_path": "./Datasets/ComCat/california_shape.npy",
        "auxiliary_start": _utc_days("1971-01-01"),
        "training_start": _utc_days("1981-01-01"),
        "validation_start": _utc_days("1998-01-01"),
        "testing_start": _utc_days("2007-01-01"),
        "testing_end": _utc_days("2020-01-17"),
        "global_m0": 2.5,
    },
    "SaltonSea": {
        "catalog_path": "./Datasets/QTM/SaltonSea_catalog_jitter.csv",
        "shape_path": "./Datasets/QTM/SaltonSea_shape.npy",
        "auxiliary_start": _utc_days("2008-01-01"),
        "training_start": _utc_days("2009-01-01"),
        "validation_start": _utc_days("2014-01-01"),
        "testing_start": _utc_days("2016-01-01"),
        "testing_end": _utc_days("2018-01-01"),
        "global_m0": 1.0,
    },
    "SanJac": {
        "catalog_path": "./Datasets/QTM/SanJac_catalog_jitter.csv",
        "shape_path": "./Datasets/QTM/SanJac_shape.npy",
        "auxiliary_start": _utc_days("2008-01-01"),
        "training_start": _utc_days("2009-01-01"),
        "validation_start": _utc_days("2014-01-01"),
        "testing_start": _utc_days("2016-01-01"),
        "testing_end": _utc_days("2018-01-01"),
        "global_m0": 1.0,
    },
    "WHITE": {
        "catalog_path": "./Datasets/WHITE/WHITE_catalog_jitter.csv",
        "shape_path": "./Datasets/WHITE/WHITE_shape.npy",
        "auxiliary_start": _utc_days("2008-01-01"),
        "training_start": _utc_days("2009-01-01"),
        "validation_start": _utc_days("2014-01-01"),
        "testing_start": _utc_days("2017-01-01"),
        "testing_end": _utc_days("2021-01-01"),
        "global_m0": 0.6,
    },
    "SCEDC_20": {
        "catalog_path": "./Datasets/SCEDC/SCEDC_catalog_jitter.csv",
        "shape_path": "./Datasets/SCEDC/SCEDC_shape.npy",
        "auxiliary_start": _utc_days("1981-01-01"),
        "training_start": _utc_days("1985-01-01"),
        "validation_start": _utc_days("2005-01-01"),
        "testing_start": _utc_days("2014-01-01"),
        "testing_end": _utc_days("2020-01-01"),
        "global_m0": 2.0,
    },
    "SCEDC_25": {
        "catalog_path": "./Datasets/SCEDC/SCEDC_catalog_jitter.csv",
        "shape_path": "./Datasets/SCEDC/SCEDC_shape.npy",
        "auxiliary_start": _utc_days("1981-01-01"),
        "training_start": _utc_days("1985-01-01"),
        "validation_start": _utc_days("2005-01-01"),
        "testing_start": _utc_days("2014-01-01"),
        "testing_end": _utc_days("2020-01-01"),
        "global_m0": 2.5,
    },
    "SCEDC_30": {
        "catalog_path": "./Datasets/SCEDC/SCEDC_catalog_jitter.csv",
        "shape_path": "./Datasets/SCEDC/SCEDC_shape.npy",
        "auxiliary_start": _utc_days("1981-01-01"),
        "training_start": _utc_days("1985-01-01"),
        "validation_start": _utc_days("2005-01-01"),
        "testing_start": _utc_days("2014-01-01"),
        "testing_end": _utc_days("2020-01-01"),
        "global_m0": 3.0,
    },
}
