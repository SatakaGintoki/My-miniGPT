# Demo：对着 checkpoint 采样

需要本地文件：

- `checkpoint.pt`（170000 step，约 260MB，不在 git 里）
- `data/vocab.pkl`、`data/merges.pkl`

没有 checkpoint 时，先训练（RTX 4060 8GB，这次 170k 跑了 **5.8 小时**）：

```bash
python prepare_data.py
python train.py
```

有 checkpoint 之后：

```bash
python generate.py --prompt "Once upon a time" --temperature 0.8 --seed 42
```

也可以：`python generate.py "Once upon a time"`。`--temperature` 默认 0.8，`--top_p` 默认 0.9。

## 温度对比（已跑过）

固定 prompt `Once upon a time`，`seed=42`，`top_p=0.9`。全文见 [results/samples.md](results/samples.md)。

| T | 开头（原文） | 是否遇到 eos |
|---|---|---|
| 0.6 | Lily，房间里的盒子和娃娃 | 是 |
| 0.8 | Tim，会说话的魔法帽 | 是 |
| 1.0 | 老鼠 Tim，奶酪；Sam 既是松鼠又是老鼠 | 是 |

第二个 prompt `My name is Tom. I live in a beautiful place.` 不像训练集里的故事开头。模型会先补一句引号里的话，再往 TinyStories 上靠。T=1.0 时 250 token 仍未结束。

这些是 170k checkpoint 的真实输出，不是手写示例。
