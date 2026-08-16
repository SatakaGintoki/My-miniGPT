# Transformer 组件：我实际写下来的那些块

代码全在 `src/my_model.py`。配置在 `config.py`：4 层，`d_model=512`，16 头，`d_ff=1344`，上下文 256，词表 10,000。从 `checkpoint.pt` 数出来 22,696,448 个参数。

课程作业常见写法是 LayerNorm + GELU。我这份不是。归一化是 **RMSNorm**，MLP 是 **SwiGLU**，位置是 **RoPE**。下面按数据流走一遍，只讲我为什么这么写。

## 先有一张图

一个 token id 进来，先变成 512 维向量。然后过 4 次同样的块。每个块是：

```
x = x + Attention(RMSNorm(x))
x = x + SwiGLU(RMSNorm(x))
```

最后再 RMSNorm 一次，乘一个 `d_model → vocab` 的线性层，得到每个位置上 10,000 维的 logits。

没有 `nn.Transformer`。`Linear` 和 `Embedding` 也是自己写的：权重是 `nn.Parameter`，前向用 `einsum`。线性层没有 bias。初始化是截断正态，标准差跟 fan-in/fan-out 有关。这些选择来自作业规范，不是我发明的。

## Q、K、V 在干什么

注意力要回答：当前位置该从序列里哪些位置抄信息。抄的对象叫 V（value）。去哪里抄、抄多少，由 Q 和 K 的点积决定。

多头里我没有用四个 `Linear` 模块，而是四块权重：`W_Q`、`W_K`、`W_V`、`W_O`，形状都是 `(512, 512)`。`x` 乘完之后，把最后一维拆成 16 个头，每头 `d_k = 32`：

```
Q, K, V: (..., seq, 512) → (..., 16, seq, 32)
```

每个头看 32 维的一个子空间。16 个头看完，拼回 512，再乘 `W_O`。直觉：一个头可能盯着上一句的主语，另一个盯着标点或对话轮次。我没有可视化过这些头，所以到此为止，不编「第 3 个头在看什么」。

Q 和 K 还会过 RoPE。绝对位置 embedding 是「第 5 个位置加一个固定向量」。RoPE 是把这对 32 维向量按二维一组转一个角度，角度跟位置有关。相对位置近的 Q、K，转过之后点积更大一些。实现上：偶数维和奇数维拆开，`(x_even, x_odd)` 变成 `(-x_odd, x_even)`，再和缓存好的 `cos`、`sin` 组合。`theta=10000`，和配置里的 `rope_theta` 一致。

## 为什么要除以 sqrt(d_k)

点积是 32 个数相乘再相加。如果 Q、K 的每一维都是大约方差 1 的数，点积的方差大约是 32，数值会到 ±好几。softmax 对大数很敏感：差一点点，概率就塌成 one-hot。训练初期注意力会「假自信」。

除以 `sqrt(d_k)=sqrt(32)` 把点积的尺度拉回来。代码就一行：

```python
QK = QK / math.sqrt(d_k)
```

没有更神秘的东西。`d_k` 取 Q 的最后一维，避免手写写错 32。

## mask：语言模型不能看未来

训练时输入是 `x[t]`，目标是 `x[t+1]`。如果位置 5 的注意力能看到位置 6、7，模型会抄答案。所以 mask 是下三角：

```python
mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=Q.device))
QK = torch.where(mask, QK, -torch.inf)
```

`True` 的位置保留分数，`False` 填 `-inf`。softmax 之后那些位置的权重是 0。对角线是 `True`，当前位置可以看自己。

softmax 我自己写了：先减 max 再 `exp`，防止溢出。和后面交叉熵里减 max 是同一类手法。

## RMSNorm：为什么除以「根号」

LayerNorm 会减均值、再除标准差。RMSNorm 不减均值，只除均方根：

```
rms = sqrt(mean(x^2) + eps)
y = (x / rms) * weight
```

「除以 root」不是随便除一个根号。RMS 这三个字母就是 Root Mean Square：先平方，再平均，再开方。得到的是这条向量的尺度。除掉它，不同层、不同位置的激活不会越跑越大。`eps=1e-5` 防止全零向量把除法打爆。

我在 fp32 里算 `x^2` 的均值和 `sqrt`，再除回去。AMP 训练时激活可能是 fp16，在半精度里开方、再除，容易变成 NaN。这是后一篇会提到的同一类问题：数值要在 fp32 里做。

`weight` 是长度为 `d_model` 的可学习向量，初始化全 1。没有 bias。块里用两次：注意力前一次，MLP 前一次（pre-norm）。残差在 norm 外面加，梯度比较好走。

## MLP 不是 GELU，是 SwiGLU

很多 GPT 教程写 `Linear → GELU → Linear`。我写的是 SwiGLU：

```
silu(x @ W1) * (x @ W3)  →  再乘 W2
```

`SiLU(z) = z * sigmoid(z)`。一边是门（过完 SiLU 的那条），一边是内容。门接近 0 就把这条通道关掉。多出来的 `W3` 就是这个门。

`d_ff=1344`。如果调用时不传 `d_ff`，代码会用 `8/3 * d_model` 再向上取到 64 的倍数。配置里写死了 1344，和 4 层、512 维凑在一起，整网大约 22.7M，能塞进 4060 的 8GB（还要开 AMP）。

## 一块、整网

`Transformer_Block` 就是上面两行残差。`Transformer_LM` 是 embedding → 4 个 block → 最终 RMSNorm → `lm_head`。`lm_head` 是我写的 `Linear(512, 10000)`，没有和 embedding 绑权重。

前向接受的是 token id。采样时如果序列比 256 长，`generate.py` 只喂最后 256 个 id。模型本身没有相对更长上下文的外推实验，RoPE 的 cache 也只建到 `max_seq_len=256`。

## 对照着读的顺序

1. `softmax`、`Attention`：缩放和 mask 都在这里
2. `RMSNorm`：开方在哪一行
3. `RoPE._rotate_every_two`：二维旋转
4. `MultiHeadSelfAttention.forward`：拆头、加 RoPE、因果 mask、拼回去
5. `SwiGLU.forward`：三块权重
6. `Transformer_Block` / `Transformer_LM`：组装

下一篇是训练：8GB 显存、手写交叉熵在 fp16 下溢出、`torch.compile` 把 checkpoint 键名改掉、以及采样温度。有数字的地方只用仓库里真实有的 log。
