# 生成样例（真实输出）

来源：本地 `checkpoint.pt`（iteration **170000**，22.70M），RTX 4060。  
脚本：`python generate.py`。固定 `seed=42`，`top_p=0.9`，`max_new_tokens=250`。  
下面全文是模型原文，没有改写、没有删句。`<|endoftext|>` 是停止符。

复现同一条：

```bash
python generate.py --prompt "Once upon a time" --temperature 0.6 --seed 42
python generate.py --prompt "Once upon a time" --temperature 0.8 --seed 42
python generate.py --prompt "Once upon a time" --temperature 1.0 --seed 42
```

---

## Prompt: `Once upon a time`

### temperature = 0.6

```
Once upon a time, there was a little girl named Lily. She loved to play with her toys and have fun with her friends. One day, Lily found a big box in her room. She was very excited and wanted to see what was inside.
Lily opened the box and found a pretty doll. She loved the doll very much and wanted to play with it. But the doll was not a toy, and it was a gift for her. Lily was sad and did not know what to do.
Lily's mom saw her sad face and asked what was wrong. Lily told her mom about the doll and the gift. Her mom smiled and said, "Don't worry, Lily. I will help you." Together, they opened the box and found a new doll inside. Lily was so happy and played with her new doll all day long.
<|endoftext|>
```

### temperature = 0.8

```
Once upon a time, there was a little boy named Tim. Tim was a happy boy who loved to play with his toys. One day, Tim found a big box in his room. He was very excited to see what was inside.
Tim opened the box and found a funny hat. He put the hat on his head and started to dance. Suddenly, the hat started to talk! "Hello, Tim!" said the hat. Tim was very surprised. He did not know that the hat could talk.
The hat told Tim that it was a magic hat. It could make things happen if you put on a show. Tim was very excited. He put on a show for his friends, and they all danced and had fun. In the end, Tim's show was the best one ever.
<|endoftext|>
```

### temperature = 1.0

```
Once upon a time, there was a little mouse named Tim. Tim was a happy mouse who loved to play with his friends. One day, Tim found a big piece of cheese. He was very excited and wanted to eat it all by himself.
Tim's friend, Sam the squirrel, saw the cheese too. Sam was a smart mouse who knew how to steal. So, Sam took the cheese and ran away. Tim was sad because he knew that steals was not good.
Tim saw his friend, the wise owl, sitting under a tree. Tim told the owl about the cheese. The owl said, "This cheese belongs to all the animals. We should share it." Tim learned that it is better to share and be kind to others. From that day on, Tim shared his cheese with all his friends, and they all played happily together.
<|endoftext|>
```

---

## Prompt: `My name is Tom. I live in a beautiful place.`

这个 prompt 不像 TinyStories 的开头（常见是 `Once upon a time`）。模型会把它接成对话或故事中段。

### temperature = 0.6

```
My name is Tom. I live in a beautiful place. I like to make friends."
Lily and Tom became friends. They played together every day. They liked to run, jump, and laugh. They were very happy.
One day, Lily and Tom found a big box. They did not know what was inside. They opened the box and found a lot of toys. They were so happy. They played with the toys all day long. The big box was their new favorite place to play.
<|endoftext|>
```

### temperature = 0.8

```
My name is Tom. I live in a beautiful place. I like to make friends." The other animals were happy to meet Tom. They all played together and had a great time.
The moral of the story is to be kind and make friends with everyone.
<|endoftext|>
```

### temperature = 1.0

250 个新 token 用完仍未碰到 `<|endoftext|>`，下面在截断处结束。

```
My name is Tom. I live in a beautiful place. I have many things to make friends. Do you want to see my magic wand?" Anna says.
Tom likes magic. He laughs and thanks Anna.
"Thank you, Anna. Your magic is very nice. I want to see your flower. I want to see your flower. It is very pretty. Can you recommend something to it?" Tom says.
Anna thinks for a moment. She is curious. She likes her flower. She likes flowers. She wants to see Tom's flower.
"Tom, can you show me your flower? It is a very nice flower. It is very popular. We can be friends. Do you want to go to the garden with me?" Anna says.
Tom thinks for a moment. He likes Anna's answer. He does not know how to say hello. He feels scared. He thinks Anna is not always best. He thinks his flower is not nice.
"OK, Anna. I want to go to the garden. But can I have my flower? You can show me your flower. Maybe it will make you feel better," Tom says.
Anna smiles. She is happy. She hugs Tom. She is glad he is a good friend. She is glad she had
```

---

## 从这些原文能看出的差异（不是额外实验）

同一套权重、同一个 seed，只改温度：

- **0.6**：句子短、结构像儿童故事（发现盒子 → 打开 → 结局）。第一段里出现「娃娃不是玩具又是礼物」这种自相矛盾，但整体仍收在 `<|endoftext|>`。
- **0.8**：同样能写完一篇，内容换成会说话的帽子、表演；还是 TinyStories 腔。
- **1.0**：角色更容易跳（Sam 先是 squirrel 再是 mouse），出现 `steals was not good` 这种语法错误；Tom 那条变得更长、重复（`I want to see your flower` 连说两遍），250 token 还不停。

这不是「温度越高越有创意」的广告，只是这六条样本上能直接读到的差别。
