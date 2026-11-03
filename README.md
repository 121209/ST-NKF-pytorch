# ST-NKF PyTorch

这是 [ST-NKF](https://github.com/zhancx-edu/NeuralKernelFunction) 的 **PyTorch** 移植：用有限历史的时空点过程做地震预测。条件强度与 ETAS 相同，生产率、时间核、空间核可以各自换成神经网络（Neural Kernel Function）。

```text
λ(t, x, y | H_t) = μ + Σ κ(m_i) · h(t - t_i) · w(x - x_i, y - y_i)
```

三个分量都用 `empirical` 就是带有限历史的 ETAS；都用 `neural` 就是 ST-NKF。也可以混用，做消融。

本仓库的代码在 CPU 上跑通了 SanJac 的前向、训练步、μ 的 declustering 更新和测试集 log-likelihood 计算。**这里没有填写论文表格里的分数。** 那些数字只属于原 TensorFlow 实验，见原仓库 README。短程 smoke 打印出来的 `LL_ts` 不能拿去和论文对比。

## 出处

方法与 TensorFlow 实现：

- Zhan, C., Wu, S., Zhuang, J., & Gao, S. (2026). *A Spatio-Temporal Neural Kernel Function Model for Earthquake Clustering: Bridging ETAS and Neural Point Processes.* Journal of Geophysical Research: Machine Learning and Computation. <https://doi.org/10.1029/2025JH001110>
- 代码与数据发布：<https://github.com/zhancx-edu/NeuralKernelFunction> ，Zenodo <https://doi.org/10.5281/zenodo.21841102>
- 原实现注明改编自 Stockman, Lawson, Werner 的 Neural Point Process（<https://github.com/ss15859/Neural-Point-Process>）
- 目录来自 EarthquakeNPP 基准（<https://github.com/ss15859/EarthquakeNPP>，arXiv:2410.08226），上游再对重复坐标做了小幅 jitter

原训练栈是 TensorFlow 2.14 + CUDA 11.8，conda 环境 `py311-tf`，入口是 `run_models.py` / `batch.py` / `main.ipynb`。

## 安装

Python 3.10+。在仓库根目录：

```bash
pip install -r requirements.txt
```

`requirements.txt` 里的 `torch` 不绑定 CUDA。没有 GPU 时，PyPI 或 CPU wheel 就够做下面的 SanJac smoke。要用 GPU，按自己的驱动到 <https://pytorch.org> 安装对应 CUDA 版本的 PyTorch，不必改模型代码。`environment.yml` 是一条 **CPU** conda 环境（含 `cpuonly`）；GPU 机器不要照抄其中的 `cpuonly`。

原论文的计时环境是 NVIDIA RTX 3090。本移植不重复那张耗时表。

## 数据

`Datasets/` 里是运行所需的 jitter 目录和区域多边形，以及上游附带的未 jitter 目录。说明见 `Datasets/SOURCE.txt`。

`run_models.py` 读的是 `*_catalog_jitter.csv`。SCEDC 的三个阈值共用 `SCEDC_catalog_jitter.csv`，脚本里按 `SCEDC_20` / `SCEDC_25` / `SCEDC_30` 过滤震级（M ≥ 2.0 / 2.5 / 3.0）。

时间切分用 **UTC 零点** 对应的 `time_days`（Unix epoch 天）。这样得到的辅助窗口长度与原 README 一致：

| 目录 | 辅助窗口长度（history） | M0 |
|------|------------------------:|---:|
| ComCat | 14933 | 2.5 |
| SaltonSea | 2144 | 1.0 |
| SanJac | 1672 | 1.0 |
| WHITE | 2196 | 0.6 |
| SCEDC_20 | 12373 | 2.0 |
| SCEDC_25 | 4242 | 2.5 |
| SCEDC_30 | 1142 | 3.0 |

请在仓库根目录运行。路径写成 `./Datasets/...`。

## 训练

```bash
python run_models.py <seed> <gpu_id> <dataset> <kappa> <temporal> <spatial>
```

| 参数 | 含义 |
|------|------|
| seed | 网络初始化与每个 epoch 打乱样本的随机种子 |
| gpu_id | CUDA 设备号。没有 GPU 时用 `-1` 或 `cpu` |
| dataset | `SanJac` `SaltonSea` `WHITE` `ComCat` `SCEDC_20` `SCEDC_25` `SCEDC_30` |
| kappa | 生产率：`empirical` 或 `neural` |
| temporal | 时间核：`empirical` 或 `neural` |
| spatial | 空间核：`empirical` 或 `neural` |

默认 `epochs=1000`，`batch_size=128`，early stopping `patience=20`，Adam `lr=0.005`，网络 `5` 层隐藏层、每层 `32` 单元。与原 `run_models.py` 的默认实验设置相同。可用 `--epochs`、`--batch-size`、`--patience`、`--lr` 覆盖。

ETAS：

```bash
python run_models.py 1 0 SanJac empirical empirical empirical
```

ST-NKF：

```bash
python run_models.py 1 0 SanJac neural neural neural
```

混合例子（生产率用网络，时间核用 Omori，空间核用 ETAS 公式）：

```bash
python run_models.py 1 0 SanJac neural empirical empirical
```

跑完会写：

- `weights/<dataset>_<temporal>_<spatial>_<kappa>.pt`
- `csv/<dataset>_<temporal>_<spatial>_<kappa>.csv`（逐事件 `logli_t` / `logli_s` / `logli_ts`）
- `results/` 下按距离、时间和震级排序的核曲线 `.npy`（`--no-patterns` 可跳过）
- `ll_seismic_model.log`、`etas_parameters.log`（追加，并带上 seed）

日志列顺序是 `dataset, temporal, spatial, kappa, LL_ts, LL_t, LL_s, seed=...`，和上游日志字段一致，只是多了 seed，避免三次重复运行分不开。

### SanJac smoke

不要用完整 1000 epoch 来检查代码是否接上。在仓库根目录：

```bash
python run_models.py 1 -1 SanJac empirical empirical empirical --epochs 1 --no-patterns
```

```bash
python run_models.py 1 -1 SanJac neural neural neural --epochs 1 --no-patterns
```

正常结束时日志里有三样东西：

1. `SanJac: auxiliary=1672, training=11770, validation=3449, testing=4400`
2. 每个 epoch 一行有限的 `loss`、`val_loss` 和 `mu`
3. `Test results: LL_ts=...`，三个数都是有限值

第 1 个 epoch 的 loss 会很大（背景率 μ 的初值是 1，而积分项含 `μ · 面积 · Δt`，SanJac 的研究区大约一万平方公里）。该 epoch 结束时 μ 会被 declustering 公式改掉，从第 2 个 epoch 起 loss 会落到正常量级。这只说明更新顺序和原 Keras callback 一致，**不是**论文里的测试集分数。

在这台没有 GPU 的环境里，SanJac 全 empirical 的 1 个 epoch（含读目录和测试集评估）大约数秒；全 neural 的 1 个 epoch 大约一到两分钟。完整 neural 训练应按原设置走 GPU。论文报告 SanJac 的 ST-NKF 大约 3.3 分钟、ETAS 大约 0.9 分钟（RTX 3090），那是 TensorFlow 实验的计时，不是本仓库测出来的成绩。

`python -m unittest tests.test_port` 覆盖窗口构造、Omori 导数、非负约束、SanJac 切分和 CLI 顺序，不读取论文指标。

### 批量

上游 `batch.py` 会直接对全部目录、seed 1/2/3 启动 ETAS；NKF 循环在上游是注释掉的。这里默认只打印命令，加上 `--run` 才执行：

```bash
python batch.py --datasets SanJac --seeds 1 --models etas --gpu -1 --epochs 1
python batch.py --datasets SanJac --seeds 1 --models etas nkf --gpu 0 --run
```

`--models` 的默认值是 `etas`，和上游真正会跑的那一截相同。

## 实现要点

`KernelPointProcess`（`models.py`）把每条目标事件前面固定长度的历史展开成张量。长度等于该目录的辅助事件数。损失是

```text
-mean(log λ_ts - ∫λ)
```

其中 `λ_ts` 是时空强度，积分是上一次事件到当前事件之间的补偿项，背景部分为 `μ · area · Δt`。

- **empirical 时间核**：修正 Omori CDF。PDF 用 `autograd.grad` 对 CDF 求导，对应 TensorFlow 的 `tf.gradients`。
- **empirical 空间核**：原 `SpatialFunction` 的闭式 pdf / cdf（对距离平方和震级）。
- **empirical 生产率**：`A exp(α (m - m0))`。
- **neural**：无偏置 `tanh` 全连接（时间、空间），生产率网络带 bias、输出 `softplus`。权重 `abs(Glorot uniform)` 初始化，Adam 更新后再投影到非负，对应 Keras `NonNeg` 约束（约束不进反向传播）。
- 神经网络的输入按训练集上的最大时间差、最大震级、最大距离归一化。空间 PDF 为 `dCDF/dr / (2 π (r + 1e-5))`。
- **μ 不进 Adam**。每个 epoch 先用当前 μ 算完 train/val loss，再按 `sum(μ / λ_ts) / (总时间 · 面积)` 覆盖 μ，并截断到 `≥ 1e-9`。Early stopping 在这之后记录权重。这是为了对齐 Keras 里 `UpdateMuCallback` 排在 `EarlyStopping` 前面的顺序。原代码里 `BackgroundFunction` 没有被 `call`，μ 通常也不在 `model.trainable_weights` 里。
- Early stopping 按 Keras 2.14：`monitor=val_loss`、`patience=20`、严格变小才算提升。只有提前停止时才恢复最好的那一轮权重。若跑满 `--epochs`，保留最后一轮（含该轮的 μ 更新）。

## 和 TensorFlow 数值的差别

不要期待和 TF 2.14 逐位相同的 loss 或测试集 log-likelihood。

1. Neural 核的 Glorot 随机流与 Keras `GlorotUniform` 不同，初始权重不同。三个网络使用同一个 seed 各自从头初始化，因此初始核权重彼此相同，这一点和“每个 initializer 用同一个 seed”一致，但和 TF 的具体随机数不同。
2. 每个 epoch 的样本顺序由 `numpy` 的 `Generator(seed)` 决定，和 TF 的 shuffle 不是同一序列。
3. Adam 使用 PyTorch 默认矩估计（`lr=0.005`, `β=(0.9, 0.999)`, `ε=1e-8`），对应 `tf.compat.v1.train.AdamOptimizer` 的公式，但浮点归约顺序不同。
4. 经验时间核的 PDF 是 Omori CDF 的自动微分，数学上等于原 `tf.gradients`；float32 舍入可能不同。空间经验核是同一闭式。
5. 原 `config.py` 用无时区的 `Timestamp.timestamp()`，切分会随机器时区变化。这里固定为 UTC，并已核对七个目录的辅助窗口长度与原 README 一致。
6. 原 `run_models.py` 把第 4、5、6 个位置参数收成 **temporal、spatial、kappa**，它自己的 README 表格则是 **productivity、temporal、spatial**。本移植按 README 和下面这条命令：

   ```text
   python run_models.py <seed> <gpu_id> <dataset> <kappa> <temporal> <spatial>
   ```

   三个分量都是 `empirical` 或都是 `neural` 时，两种顺序没有区别。混用时以上表为准，不要按上游脚本的位置去套。
7. 上游把 `"empirical"` 写成了 `"empiricl"`，ETAS 参数因此不会写入 `etas_parameters.log`。这里按正确拼写记录 μ 以及当前确实是 empirical 的那些参数。

## 还没做的部分

- 未移植 `main.ipynb` 和 `plot *.ipynb`（CDF、信息增益、回归图）。评估需要的逐事件 log-likelihood 已经写在 `csv/`。
- `batch.py` 可以排队跑完全部目录，但默认是 dry-run。ComCat 与 `SCEDC_20` 会把很长的历史窗口展开成稠密数组（训练窗上单份 `float32` 距离矩阵大约数 GB，再乘上时间、震级和验证/测试切分）。SanJac、SCEDC_30 小得多。内存不够时不要直接 `--run` 全网格。
- 没有在本仓库里重跑论文的 3 个 seed × 全部目录，因此没有新的成绩表。

## 许可

仓库根目录的 MIT License 适用于这里的 PyTorch 代码。地震目录的来源和引用方式见 `Datasets/SOURCE.txt`。上游 GitHub 仓库没有 SPDX 许可证；使用目录和结果时请引用 Zhan 等人与 EarthquakeNPP。
