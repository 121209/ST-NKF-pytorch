"""Execute all plotting code cells against synthetic training outputs.

Run after installing requirements-plotting.txt:
    python -m unittest discover -s tests -p test_plot_notebooks.py
No training, published scores, or downloaded catalogs are used.
"""

import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    "plot CDF.ipynb",
    "plot comparison.ipynb",
    "plot pattern.ipynb",
    "plot regression.ipynb",
    "plot spatial information gain.ipynb",
    "plot temporal information gain.ipynb",
]
DATASETS = ["SanJac", "SCEDC_30"]
TAGS = ["empirical_empirical_empirical", "neural_neural_neural"]


def write_fixture(root):
    (root / "run_models.py").write_text("# synthetic plotting fixture\n", encoding="utf-8")
    (root / "csv").mkdir()
    (root / "results").mkdir()
    rng = np.random.RandomState(12)
    logs = []
    for ds_index, dataset in enumerate(DATASETS):
        n = 48
        frame = pd.DataFrame({
            "step": [np.nan] * 4 + ["train"] * 20 + ["val"] * 8 + ["test"] * 16,
            "time_days": np.linspace(13879, 17531, n),
            "magnitude": rng.uniform(1.0, 3.0, n),
            "x": rng.normal(size=n),
            "y": rng.normal(size=n),
        })
        temporal = rng.normal(0.5, 0.2, n)
        spatial = rng.normal(-2.0, 0.3, n)
        for model_index, tag in enumerate(TAGS):
            output = frame.copy()
            output["logli_t"] = temporal + model_index * rng.normal(0.15, 0.03, n)
            output["logli_s"] = spatial + model_index * rng.normal(0.1, 0.04, n)
            output["logli_ts"] = output["logli_t"] + output["logli_s"]
            output.loc[:3, ["logli_t", "logli_s", "logli_ts"]] = np.nan
            output.to_csv(root / "csv" / f"{dataset}_{tag}.csv", index=False)
            for seed in (1, 2, 3):
                ll_t = 0.5 + seed * 0.01 + model_index * 0.15 + ds_index * 0.2
                ll_s = -2 + seed * 0.02 + model_index * 0.1
                logs.append(f"{dataset}, {tag.replace('_', ', ')}, "
                            f"{ll_t + ll_s}, {ll_t}, {ll_s}, seed={seed}\n")
            time = np.linspace(0.001, 100, 300)
            distance = np.linspace(0.01, 40, 300)
            arrays = {
                "dis_t_train": time,
                "cdf_temporal_train": 1 - np.exp(-time / 5),
                "pdf_temporal_train": np.exp(-time / 5) / 5,
                "dis_xy_train": distance,
                "cdf_spatial_train": 1 - np.exp(-distance / 3),
                "pdf_spatial_train": np.exp(-distance / 3) / (6 * np.pi * distance),
                "hist_m_train": np.linspace(1, 5, 300),
                "kappa_train": np.exp(np.linspace(0, 2, 300) + model_index * 0.1),
            }
            for stem, values in arrays.items():
                np.save(root / "results" / f"{dataset}_{tag}_{stem}.npy", values)
    (root / "ll_seismic_model.log").write_text("".join(logs), encoding="utf-8")


def execute_cells(name, root, draw=True):
    notebook = nbformat.read(str(ROOT / name), as_version=4)
    nbformat.validate(notebook)
    namespace = {"__name__": "__notebook__"}
    old_cwd = Path.cwd()
    os.chdir(str(root))
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            for index, cell in enumerate(notebook.cells):
                if cell.cell_type != "code":
                    continue
                if not draw and index == len(notebook.cells) - 1:
                    break
                exec(compile(cell.source, name + ":cell" + str(index + 1), "exec"), namespace)
                if index == 2:  # Parameter cell, before any inputs are read.
                    namespace.update(DATASETS=DATASETS, DPI=72, SHOW_FIGURES=False, SAMPLE_STEP=7)
    finally:
        plt.close("all")
        os.chdir(str(old_cwd))
    return namespace


class PlotNotebookTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="stnkf-plots-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        write_fixture(self.root)

    def test_all_six_notebooks_export_expected_figures(self):
        for name in NOTEBOOKS:
            with self.subTest(notebook=name):
                execute_cells(name, self.root)
        expected = ["NKF_Temporal_CDF", "NKF_Spatial_CDF",
                    "log_likelihood_comparison_sci_style"]
        for dataset in DATASETS:
            expected.extend([
                f"{dataset}-spatial_pattern-loglog", f"{dataset}-temporal_pattern-loglog",
                f"{dataset}-kappa_pattern", f"regression-{dataset}_temporal",
                f"regression-{dataset}_spatial", f"IG-{dataset}_spatial-Magnitude",
                f"IG-{dataset}_temporal-Magnitude",
            ])
        for stem in expected:
            image = self.root / "fig" / (stem + ".png")
            self.assertGreater(image.stat().st_size, 1000, str(image))
            self.assertEqual(image.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        for stem in expected[:3]:
            pdf = self.root / "fig" / (stem + ".pdf")
            self.assertEqual(pdf.read_bytes()[:5], b"%PDF-")

    def test_comparison_uses_latest_distinct_seed_and_sample_sd(self):
        namespace = execute_cells("plot comparison.ipynb", self.root, draw=False)
        with (self.root / "ll_seismic_model.log").open("a", encoding="utf-8") as stream:
            stream.write("SanJac, neural, neural, neural, -0.8, 0.7, -1.5, seed=1\n")
        with contextlib.redirect_stdout(io.StringIO()):
            runs = namespace["read_runs"]()
        group = runs[(runs.Dataset == "SanJac") & (runs.Model == "ST-NKF")]
        self.assertEqual(len(group), 3)
        summary = namespace["summarize_runs"](runs)
        row = summary[(summary.Dataset == "SanJac") &
                      (summary.Model == "ST-NKF") & (summary.Metric == "Total")].iloc[0]
        expected = np.array([-0.8, -1.19, -1.16])
        self.assertEqual(row.N, 3)
        self.assertAlmostEqual(row.Mean, expected.mean())
        self.assertAlmostEqual(row.Std, expected.std(ddof=1))
        namespace["SEEDS"] = [1]
        single = namespace["summarize_runs"](namespace["read_runs"]())
        self.assertTrue(single.Std.isna().all())
        namespace["SEEDS"] = [1, 2, 3, 4]
        with self.assertRaisesRegex(ValueError, "SEEDS"):
            namespace["read_runs"]()

    def test_csv_notebooks_reject_misaligned_events(self):
        path = self.root / "csv" / "SanJac_empirical_empirical_empirical.csv"
        data = pd.read_csv(path)
        data.loc[10, "time_days"] += 1
        data.to_csv(path, index=False)
        for name in NOTEBOOKS[3:]:
            namespace = execute_cells(name, self.root, draw=False)
            with self.subTest(notebook=name):
                with self.assertRaisesRegex(ValueError, "CSV"):
                    namespace["load_pair"]("SanJac")

    def test_missing_arrays_explain_how_to_generate_them(self):
        namespace = execute_cells("plot CDF.ipynb", self.root, draw=False)
        with self.assertRaisesRegex(FileNotFoundError, "no-patterns"):
            namespace["load_curve"]("missing", TAGS[1], "dis_t_train", "cdf_temporal_train")

    def test_temporal_year_axis_and_colorbar_do_not_overlap(self):
        namespace = execute_cells("plot temporal information gain.ipynb", self.root, draw=False)
        figures = []
        namespace["save_figure"] = lambda fig, stem: figures.append(fig)
        nkf, etas = namespace["load_pair"]("SanJac")
        namespace["plot_information_gain"](nkf, etas, "SanJac")
        fig = figures[0]
        try:
            fig.canvas.draw()
            main = next(ax for ax in fig.axes if ax.get_xlabel() == "Index of earthquakes")
            top = next(ax for ax in fig.axes if ax.get_xlabel() == "Time (year)")
            bar = next(ax for ax in fig.axes if ax.get_ylabel().startswith("Temporal log-likelihood:"))
            self.assertAlmostEqual(main.get_position().x0, top.get_position().x0, places=5)
            self.assertAlmostEqual(main.get_position().x1, top.get_position().x1, places=5)
            self.assertGreater(bar.get_position().x0, main.get_position().x1)
        finally:
            plt.close(fig)

    def test_committed_notebooks_have_no_outputs(self):
        for name in NOTEBOOKS:
            notebook = nbformat.read(str(ROOT / name), as_version=4)
            for cell in notebook.cells:
                if cell.cell_type == "code":
                    self.assertIsNone(cell.execution_count)
                    self.assertEqual(cell.outputs, [])


if __name__ == "__main__":
    unittest.main()

