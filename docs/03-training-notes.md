# 训练时踩过的坑，以及我实际记下的数字

代码：`train.py`、`src/my_training.py`、`src/training_loop.py`、`generate.py`。硬件是 RTX 4060 8GB。正式跑了 170,000 step，checkpoint 里有 22,696,448 个参数。

先说一件尴尬的：170k 那次的 stdout **没有存下来**。仓库里的 `train_run.log` 是更早的一次 3000 step 试跑。所以下面凡是「loss 从多少到多少」，只用 3000 那次；170k 只谈配置、checkpoint 和代码里能看出来的决定。这不是谦虚，是记录丢了。

## 8GB 上能训起来，靠的是缩小体积和 AMP

模型 22.7M，上下文 256，batch 32。一步看 `32 × 256 = 8192` 个 token。170k step 按这个乘，计划看到大约 13.9 亿 token。这是用超参乘出来的，不是从 log 里读的。

4060 8GB 不够任性。CUDA 上开了 `autocast` + `GradScaler`。前向在 fp16 里算，反向用 scaler 防止梯度下溢。CPU 上这两样是关掉的。

`torch.compile` 我包了一层 try。注释里写「这个小模型大约快一倍」，那是当时的体感，没有单独的 benchmark 文件，所以不当正式数字报。它带来一个真实的坑，见下一节。

## 坑：compile 之后不能直接存那个 wrapper

`torch.compile(model)` 得到的模块，`state_dict()` 的键会带 `_orig_mod.` 前缀。采样脚本按普通 `Transformer_LM` 去 `load_state_dict`，键对不上，直接报错。

所以训练时有两份引用：`model_raw` 是原模块，`model` 可能是编译后的。前向走 `model`，`save_checkpoint` 必须存 `model_raw`。`train.py` 里这段注释是后来补的，说明我已经被这个坑咬过一次。

## 坑：手写交叉熵不要在 fp16 里做

作业要求自己写 `cross_entropy`：减 max、`log(sum(exp))`、再减目标位置的 logit。这在 fp32 里没问题。AMP 下 logits 若保持 fp16，`exp` 很容易 inf，loss 变 NaN，看起来就像「loss 不降」甚至直接崩。

处理是前向仍在 autocast 里，**算 loss 之前把 logits 转成 float32**：

```python
logits_flat = logits.view(-1, vocab_size).float()
loss = cross_entropy(logits_flat, targets_flat)
```

验证 loss 同样转 fp32。这不是性能优化，是让手写公式和半精度共存。

梯度裁剪也要注意顺序：先 `scaler.unscale_(optimizer)`，再 `gradient_clipping`。在缩放过的梯度上裁剪，阈值 1.0 没有物理意义。

## 学习率：warmup 再余弦，而不是固定 3e-4

配置里 `lr` 和 `max_lr` 都是 `3e-4`，`min_lr` 是 `3e-5`，warmup 2000 step（3000 那次试跑的配置不同：warmup 400，总步数 3000）。实现是 `get_lr_cos_schedule`：

- `t < warmup`：从 0 线性升到 `max_lr`
- 然后余弦降到 `min_lr`
- 超过 `T_c` 就钉在 `min_lr`

一开始不用全额学习率，是因为 embedding 和注意力还是随机的，太大的步子会把 logits 打飞。我没有做「关掉 warmup 对比」的消融，所以只讲动机，不讲「关掉会怎样」。

AdamW 是手写的：β `(0.9, 0.95)`，weight decay `0.1`。每个 step 把算出来的 lr 写进 `param_groups`。`SGD` 那个类是课程示例，训练没用。

## 验证 loss 为什么抖得厉害

每 10 步从 val 切片里 **随机抽一个 batch** 算一次 val loss，不是扫完整验证集。所以曲线很吵。

`train_run.log`（3000 step 那次）开头和结尾是：

| step | train | val |
|---|---|---|
| 0 | 6.9753 | 6.9684 |
| 2999 | 3.3108 | 3.1918 |

图在 `results/loss_curve.png`。中间 val 会上下跳：有的 step 到 2.7，旁边又回到 3.5。这不能解释成过拟合或学崩了，首先是估计本身方差大。词表 10,000，均匀乱猜的交叉熵是 `log(10000) ≈ 9.21`。从 7 降到 3 左右，说明 3000 step 已经比乱猜好很多，但离「写得像故事」还早。

后来又跑了一次带 csv 的 170k（`results/metrics_170k.csv`）：5.8 小时，step 0 为 9.27，最后 1000 step 平均 train 1.299 / val 1.319。全程图：`results/loss_curve_170k.png`。

更早那份没留 log 的 `checkpoint.pt`，事后抽了 32 个随机 batch（`seed=0`）：train 1.31 / val 1.32。记录在 `results/eval_170k.txt`。两份 170k 权重数字接近。

数据划分是 token 数组切 90%/10%，不是按故事切。边界上一个故事可能被劈开。对作业级训练够用，统计上不是最干净的 val。

`get_batch` 在 `0 .. n-context_length` 上均匀随机起点，输入是 `x[i:i+L]`，目标是 `x[i+1:i+L+1]`。没有把数据先搬到 GPU 常驻成 tensor，每次从 numpy 再 `torch.tensor`。能跑，不是最快的写法。

## 采样：温度和 top-p，我还没有对照样例

`generate.py` 里默认 `temperature=0.8`，`top_p=0.9`，最多 250 个新 token，碰到 `<|endoftext|>` 停。温度除在 softmax 之前：`softmax(logits / T)`。

直觉（机制，不是这台模型上的观测）：

- `T → 0`：几乎总是 argmax，句子稳，容易重复
- `T = 1`：按模型学到的分布采样
- `T > 1`：分布被拍平，更胡编

top-p 是把概率从大到小累加，切掉累计超过 p 的尾巴，剩下的再归一化。`p=0.9` 时，很小的那些 logit 基本抽不到。`mask[0]=True` 是为了保证至少留下最大的那一个，避免全被裁光。

**对着 170k checkpoint、temperature ∈ {0.6, 0.8, 1.0} 的原文：** 见 [results/samples.md](../results/samples.md)。同一 seed 下，0.6/0.8 能写完一篇 TinyStories 腔的短故事；1.0 出现 `steals was not good`、角色先是 squirrel 再是 mouse，以及重复句子。这是机制对上的观测，不是另做的消融实验。

还有一个实现细节：序列超过 256 时只保留末尾 256 个 id。变量当时起名叫 `logit`，其实是 token id。不影响对错，读的时候别被名字骗了。

## 日志这次留下了

第一次 170k 只 `print`，曲线丢了。补训这一次写了 `results/metrics_170k.csv`，权重在 `checkpoints/run170k.pt`，没有覆盖原来的 `checkpoint.pt`。

墙上时间 **20833 秒（5.8 小时）**。step 0：train/val 9.275。最后一点：train 1.328 / val 1.315。最后 1000 step 平均：train 1.299 / val 1.319。曲线：`results/loss_curve_170k.png`。

## 目前能诚实说的

- 模型结构、超参、22.70M、170k step、4060 8GB：有
- 有日志的 170k：5.8 小时，loss 9.27 → ~1.32，曲线 `results/loss_curve_170k.png`：有
- 3000 step 试跑：train 6.98 → 3.31，曲线 `results/loss_curve.png`：有
- 旧 `checkpoint.pt` 事后 32-batch 评估：train 1.31 / val 1.32：有
- 生成原文（两个 prompt × 三个温度，来自旧 checkpoint）：有，见 `results/samples.md`
- 手写 CE + AMP、compile 的键名前缀：是代码里留下的真实修复
