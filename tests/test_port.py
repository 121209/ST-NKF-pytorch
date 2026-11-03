"""Behavior checks for the PyTorch ST-NKF port. No published metrics."""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from batch import build_commands, parse_args as parse_batch
from layers import OmoriFunction, TemporalKernelNetwork
from models import KernelPointProcess
from run_models import parse_args, prepare_run
from utils import process_data


def _naive_process(time_step, times, mags, locs_x, locs_y):
    n = mags.shape[0]
    hist_t = np.array([times[i : i + time_step] for i in range(n - time_step)]).reshape(
        n - time_step, time_step, 1
    )
    hist_m = np.array([mags[i : i + time_step] for i in range(n - time_step)]).reshape(
        n - time_step, time_step, 1
    )
    cur_t = times[time_step:].reshape(n - time_step, 1, 1)
    dis_t1 = cur_t - hist_t
    dis_t2 = hist_t[:, -1:, :] - hist_t[:, :-1, :]
    elapsed = np.diff(times[time_step - 1 :]).reshape(n - time_step, 1, 1)
    return hist_m, dis_t1, dis_t2, elapsed


class ProcessDataTests(unittest.TestCase):
    def test_matches_list_windows(self):
        rng = np.random.default_rng(0)
        n, step = 30, 7
        times = np.cumsum(rng.uniform(0.01, 0.5, size=n))
        mags = rng.uniform(1.0, 4.0, size=n)
        x = rng.normal(size=n)
        y = rng.normal(size=n)
        hist_t, hist_m, hist_x, hist_y, cur_t, cur_x, cur_y, dis_xy, dis_t1, dis_t2, elapsed = (
            process_data(step, times, mags, x, y)
        )
        naive_m, naive_t1, naive_t2, naive_elapsed = _naive_process(step, times, mags, x, y)
        np.testing.assert_allclose(hist_m, naive_m.astype(np.float32), rtol=0, atol=0)
        np.testing.assert_allclose(dis_t1, naive_t1.astype(np.float32), rtol=0, atol=0)
        np.testing.assert_allclose(dis_t2, naive_t2.astype(np.float32), rtol=0, atol=0)
        np.testing.assert_allclose(elapsed, naive_elapsed.astype(np.float32), rtol=0, atol=0)
        self.assertEqual(dis_t1.shape, (n - step, step, 1))
        self.assertEqual(dis_t2.shape, (n - step, step - 1, 1))
        self.assertEqual(hist_t.dtype, np.float32)
        self.assertTrue(np.all(dis_xy >= 0))


class KernelInitTests(unittest.TestCase):
    def test_shared_seed_matches_and_nonnegative(self):
        model = KernelPointProcess(
            4, 4, 4, "neural", "neural", "neural", 1.0, 100.0, 3, 8, seed=7
        )
        temporal = model.temporal_net.hidden[0].weight.detach().cpu().numpy()
        spatial = model.spatial_net.hidden[0].weight.detach().cpu().numpy()
        kappa = model.kappa_net.hidden[0].weight.detach().cpu().numpy()
        np.testing.assert_array_equal(temporal, spatial)
        np.testing.assert_array_equal(temporal, kappa)
        for net in (model.temporal_net, model.spatial_net, model.kappa_net):
            for layer in list(net.hidden) + [net.output]:
                self.assertTrue(torch.all(layer.weight >= 0))
        model.temporal_net.hidden[0].weight.data.fill_(-0.2)
        model.project_constraints()
        self.assertTrue(torch.all(model.temporal_net.hidden[0].weight >= 0))


class OmoriDerivativeTests(unittest.TestCase):
    def test_autograd_pdf_matches_closed_form(self):
        omori = OmoriFunction()
        t = torch.tensor([[[0.0], [0.02], [1.5]]], dtype=torch.float32, requires_grad=True)
        cdf = omori(t)
        pdf = torch.autograd.grad(cdf.sum(), t, create_graph=True)[0]
        c = omori.omori_c
        p = omori.omori_p
        analytical = (p - 1) / c * torch.pow(1 + t / c, -p)
        self.assertTrue(torch.allclose(pdf, analytical, rtol=1e-5, atol=1e-6))
        # Parameter gradient of the pdf is defined (create_graph).
        pdf.sum().backward()
        self.assertIsNotNone(omori.omori_c.grad)
        self.assertTrue(torch.isfinite(omori.omori_c.grad))


class SyntheticTrainTests(unittest.TestCase):
    def _catalog(self, n=48, seed=0):
        rng = np.random.default_rng(seed)
        times = np.cumsum(rng.uniform(0.05, 0.4, size=n)) + 10_000.0
        mags = rng.uniform(1.0, 3.5, size=n)
        x = rng.normal(scale=5.0, size=n)
        y = rng.normal(scale=5.0, size=n)
        return times, mags, x, y

    def _fit(self, temporal, spatial, kappa, epochs=3):
        times, mags, x, y = self._catalog()
        model = KernelPointProcess(
            time_step_train=8,
            time_step_val=8,
            time_step_test=8,
            temporal_id=temporal,
            spatial_id=spatial,
            kappa_id=kappa,
            global_m0=1.0,
            area=250.0,
            size_layer=2,
            size_nn=4,
            seed=3,
        )
        model.set_train_data(times, mags, x, y)
        model.set_val_data(times, mags, x, y)
        model.set_test_data(times, mags, x, y)
        model.set_model().compile(lr=0.005).fit_eval(
            epochs=epochs, batch_size=16, patience=20, mu_batch_size=16
        )
        model.eval_test(batch_size=16)
        return model

    def test_empirical_step_has_kernel_grad_and_mu_is_closed_form(self):
        times, mags, x, y = self._catalog()
        model = KernelPointProcess(
            8, 8, 8, "empirical", "empirical", "empirical", 1.0, 250.0, 2, 4, seed=1
        )
        model.set_train_data(times, mags, x, y).set_val_data(times, mags, x, y).set_test_data(
            times, mags, x, y
        ).set_model()
        loss = model.loss_on("train", np.arange(0, 10), create_graph=True)
        loss.backward()
        self.assertIsNotNone(model.omori_func.omori_c.grad)
        self.assertIsNotNone(model.produc_func.mag_A.grad)
        # mu is in the graph, but Adam is not allowed to step it.
        self.assertIsNotNone(model.bac_func.mu.grad)
        mu_value = float(model.bac_func.mu.detach())
        c_value = float(model.omori_func.omori_c.detach())
        opt = torch.optim.Adam(model.adam_parameters(), lr=0.005)
        opt.step()
        self.assertEqual(float(model.bac_func.mu.detach()), mu_value)
        self.assertNotEqual(float(model.omori_func.omori_c.detach()), c_value)
        self.assertTrue(math.isfinite(float(loss.detach())))

        model2 = self._fit("empirical", "empirical", "empirical")
        self.assertNotEqual(mu_value, float(model2.bac_func.mu.detach()))
        self.assertTrue(math.isfinite(model2.LL_ts_average_test))
        self.assertTrue(math.isfinite(model2.LL_t_average_test))
        self.assertTrue(math.isfinite(model2.LL_s_average_test))
        self.assertGreaterEqual(float(model2.omori_func.omori_p.detach()), 1.0)
        self.assertGreaterEqual(float(model2.spatial_func.spatial_q.detach()), 1.0)

    def test_neural_loss_finite(self):
        model = self._fit("neural", "neural", "neural", epochs=2)
        self.assertTrue(math.isfinite(model.LL_ts_average_test))
        self.assertGreater(float(model.bac_func.mu.detach()), 0.0)
        for layer in list(model.temporal_net.hidden) + [model.temporal_net.output]:
            self.assertTrue(torch.all(layer.weight >= 0))


class CatalogTests(unittest.TestCase):
    def test_sanjac_split_matches_readme_window(self):
        prep = prepare_run("SanJac")
        self.assertEqual(prep["auxiliary_num"], 1672)
        self.assertEqual(prep["training_num"], 11770)
        self.assertEqual(prep["validation_num"], 3449)
        self.assertEqual(prep["testing_num"], 4400)
        self.assertEqual(len(prep["train"]["t"]), 1672 + 11770)
        self.assertGreater(prep["area"], 1000.0)
        self.assertLess(prep["area"], 100000.0)

    def test_cli_order_is_kappa_temporal_spatial(self):
        args = parse_args(["4", "-1", "SanJac", "neural", "empirical", "empirical"])
        self.assertEqual(args.seed, 4)
        self.assertEqual(args.kappa, "neural")
        self.assertEqual(args.temporal, "empirical")
        self.assertEqual(args.spatial, "empirical")
        self.assertEqual(args.epochs, 1000)
        self.assertEqual(args.batch_size, 128)
        self.assertEqual(args.patience, 20)

    def test_batch_dry_run_order(self):
        args = parse_batch(
            ["--datasets", "SanJac", "--seeds", "1", "--models", "etas", "nkf", "--gpu", "-1"]
        )
        commands = build_commands(args)
        self.assertEqual(len(commands), 2)
        self.assertEqual(commands[0][4:8], ["SanJac", "empirical", "empirical", "empirical"])
        self.assertEqual(commands[1][4:8], ["SanJac", "neural", "neural", "neural"])


class PatternDumpTests(unittest.TestCase):
    def test_sorted_curves_drop_last_row(self):
        import os
        import tempfile

        from run_models import _write_patterns

        class Holder:
            pass

        holder = Holder()
        rng = np.random.default_rng(0)
        n, width = 5, 3
        holder.dis_xy_train = rng.random((n, width, 1)).astype(np.float32)
        holder.dis_t1_train = rng.random((n, width, 1)).astype(np.float32)
        holder.hist_m_train = rng.random((n, width, 1)).astype(np.float32)
        holder.pdf_spatial_train = holder.dis_xy_train + 1
        holder.cdf_spatial_train = holder.dis_xy_train * 0
        holder.pdf_temporal_train = holder.dis_t1_train + 2
        holder.cdf_temporal_train = holder.dis_t1_train * 0
        holder.kappa_train = holder.hist_m_train + 3
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                _write_patterns(holder, "tiny")
                dis = np.load("results/tiny_dis_xy_train.npy")
                pdf = np.load("results/tiny_pdf_spatial_train.npy")
            finally:
                os.chdir(cwd)
        self.assertEqual(dis.shape, ((n - 1) * width,))
        self.assertTrue(np.all(np.diff(dis) >= -1e-6))
        np.testing.assert_allclose(pdf, dis + 1)


class NetworkShapeTests(unittest.TestCase):
    def test_depth(self):
        net = TemporalKernelNetwork(size_nn=8, size_layer=5, seed=1)
        self.assertEqual(len(net.hidden), 5)
        self.assertEqual(net.output.out_features, 1)
        self.assertIsNone(net.hidden[0].bias)
        y = net(torch.zeros(2, 3, 1))
        self.assertEqual(tuple(y.shape), (2, 3, 1))
        self.assertTrue(torch.allclose(y, torch.zeros_like(y)))


if __name__ == "__main__":
    unittest.main()
