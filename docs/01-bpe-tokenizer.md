# 从零实现 BPE：词表是怎么长出来的

代码：`src/my_bpe.py`、`src/my_tokenizer.py`。词表大小 10,000，特殊 token 只有 `<|endoftext|>`。BPE 是在 TinyStories 的 50MB 子集上训的，不是完整 2.1GB——全量会在这台机器上 OOM。

## 先有一个直觉

分词要解决的事很具体：模型的输入是整数 id，文本是字节流。最笨的办法是一个字节一个 id，词表 256，序列会很长。最猛的办法是每个英文单词一个 id，词表爆炸，罕见词直接没有。

BPE 走中间。先把文本拆成字节。然后反复问一个问题：**哪两个相邻符号一起出现得最多？把它们焊成一个新符号。** 焊一次，词表多一项。焊到 10,000 就停。

所以词表不是「我事先列了 10,000 个单词」，而是「从 256 个字节出发，长出来的 9743 次合并记录，外加几个特殊 token」。`vocab.pkl` 是成品，`merges.pkl` 是生长史。编码的时候按这部生长史原样再演一遍。

## 词表一开始长什么样

`train_bpe()` 开头几乎没有魔法：

```python
vocab = {idx: bytes([idx]) for idx in range(256)}
new_idx = 256
```

id 0 到 255 就是单个字节。任何 UTF-8 文本都能被表示，不会有「这个字不在词表里」。然后才把 `<|endoftext|>` 塞进去，占 id 256。之后每次 merge 的新 id 从 257 往上加。

目标词表 10,000，减掉 256 个字节，再减掉 1 个特殊 token，循环跑 `10000 - 256 - 1 = 9743` 次 merge。每一次选全局出现次数最高的 pair，焊成新 token，写进 `merges`。

平票怎么破：我用的是 `(次数, 左 token 的字节, 右 token 的字节)` 取 max。次数相同就比字节内容。这是为了结果确定，换一台机器 merge 顺序不会漂。

## 为什么不能直接在整段文本上 merge

如果把整本书当成一条超长 byte 序列来 merge，空格两边的词会被焊在一起，`"the cat"` 可能变成一个 token。那不是我们要的。BPE 先做 **pre-tokenization**：用 GPT-2 那条 regex 把文本切成「词」——带不带前导空格的字母串、数字串、标点、空白。merge **只发生在一个词内部**。

`<|endoftext|>` 更特殊。它必须保持整块。如果 regex 先切，`<|`、`endoftext`、`|>` 会变成三个普通片段，后面的 merge 可能把它们焊得四不像，故事边界就丢了。所以切的时候先按特殊 token 劈开，劈出来的那一段整块放进列表，并且 **不参与 pair 统计**。

多进程切文件时，边界也尽量落在 `<|endoftext|>` 上（`find_chunk_boundaries` 来自课程代码）。这样一篇故事不会被切成两半，半个 token 也不会出现在 chunk 交界处。

## merge 一轮实际在干什么

预分词之后，语料是很多个「词」，每个词是一个 id 列表，一开始就是它的字节。

我维护两张表：

- `counts[(a, b)]`：pair `(a, b)` 在全语料里出现多少次
- `word_idx[(a, b)]`：哪些词里出现过这个 pair

每轮：

1. 取出当前 `counts` 最大的 pair
2. 只去改包含它的那些词（`word_idx`），不要扫全部语料
3. 词里相邻的 `(a, b)` 换成新 id
4. 把这个词旧的 pair 计数减掉，新的 pair 计数加上
5. `vocab[new_idx] = vocab[a] + vocab[b]`，`merges` 里记一笔 `(bytes_a, bytes_b)`
6. `new_idx += 1`

早期 merge 都是高频字母对，比如 `t`+`h`、`i`+`n`。再往后才是 `the`、`ing` 这种。TinyStories 里小孩、动物、once upon a time 会出现得特别勤，对应的 merge 会排得很靠前。我没有把 merge 列表打印成文章——那是可以做的课后作业，打开 `merges.pkl` 看前 50 项就能对上语感。

第一版我扫全量词来更新，太慢。改成倒排索引之后，每轮只碰真正含有该 pair 的词。`my_bpe.py` 里还留着一大段注释掉的旧代码，是这条路没走通时的痕迹，没删。

## 编码：把生长史再演一遍

训练只做一次。之后每次 `encode` 都是：

1. 同样的特殊 token 切分 + regex
2. 每个词先变成字节 id（0–255）
3. 按 `merges` 的 **顺序**（也就是 rank）不断合并

实现上我没有真的 for 循环 9743 次 merge。我把 merge 做成 `ranks[(a,b)] = 第几次被学到`。然后在一个词的 id 列表上，反复找 **当前 rank 最小** 的相邻 pair，焊掉，直到找不到还能焊的。这和训练时「先学到的 merge 先应用」是同一件事。

decode 反向：id → `vocab[id]` 的 bytes → 拼起来 → UTF-8。坏字节用 `errors="replace"`，不会因为生成了奇怪 id 就崩。

## 数据和一个实际限制

完整 TinyStories 训练集大约 2.1GB。在 50MB 子集上训 BPE，再拿同一套 vocab/merges 去编码全文。`prepare_data.py` 的注释写了原因：全量训 BPE 会 OOM。子集和全文是同一个分布，词表够用，只是极罕见的拼写可能被拆得更碎。

编码全文也不能把所有 id 先装进一个 Python list。流程是：按换行切 chunk → 每个 worker 写成 `uint16` 小文件 → 再拼成 `data/train_tokens.npy`。词表 10,000，最大 id 小于 65535，`uint16` 够了。`verify_data.py` 就是检查这件事：`max id < 10000`，以及 `<|endoftext|>`（id 256）出现的次数。

## 可以对照着看的点

- 词表「长出来」= 循环 `vocab_size - 256 - 特殊 token 数` 次
- 特殊 token 有 id，但不进 merge
- 编码必须用同一份 `merges` 顺序，否则和训练数据对不上
- 50MB 训词表、全文编码，是显存/内存不够时的折中，不是理论上更优

下一篇写模型内部：QKV、缩放、mask、RMSNorm 为什么除以根号、以及为什么 MLP 是 SwiGLU 而不是 GELU。
