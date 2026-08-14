from .my_bpe import train_bpe

import os
import regex as re

class Tokenizer:

    def __init__(self,vocab,merges,special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        if special_tokens is None:
            self.special_tokens=[]
        else:
            self.special_tokens=sorted(special_tokens, key=len, reverse=True)

        self.special_tokens_bytes = [i.encode("utf-8") for i in self.special_tokens]

        self.bytes_to_id = {b: i for i, b in vocab.items()}

        self.merges_int = [(self.bytes_to_id[i[0]],self.bytes_to_id[i[1]]) for i in merges]

        self.pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        self._pretok_re = re.compile(self.pattern)
        if self.special_tokens:
            self._special_re = re.compile(
                "(" + "|".join(re.escape(t) for t in self.special_tokens) + ")"
            )
        else:
            self._special_re = None

        self.ranks = {pair: i for i, pair in enumerate(self.merges_int)}
        self.pair_to_id = {
            (a, b): self.bytes_to_id[self.vocab[a] + self.vocab[b]]
            for a, b in self.merges_int
        }

    def _merge_ids(self,ids):
        while len(ids)>=2:
            best_i = None
            best_rank = float("inf")

            for i in range(len(ids)-1):
                rank = self.ranks.get((ids[i],ids[i+1]),float("inf"))
                if rank < best_rank:
                    best_rank = rank
                    best_i = i
            if best_i is None:
                break

            a,b = ids[best_i],ids[best_i+1]
            ids = ids[:best_i]+[self.pair_to_id[(a, b)]]+ ids[best_i + 2 :]

        return ids

    def encode(self,text:str):
        chunks=[]

        if self.special_tokens:
                special_pattern = re.compile(
                    "(" + "|".join(re.escape(t) for t in self.special_tokens) + ")"
                )
        
                for part in re.split(special_pattern, text):
                    if not part:
                        continue
                    if part in self.special_tokens:
                        chunks.append(part.encode("utf-8"))
                    else:
                        for chunk in re.findall(self.pattern, part):
                            chunks.append(list(chunk.encode("utf-8")))
        else:
            for chunk in re.findall(self.pattern, text):
                chunks.append(list(chunk.encode("utf-8")))
        

        ans = []

        for part in chunks:
            if isinstance(part,bytes) and part in self.special_tokens_bytes:
                ans.append(self.bytes_to_id[part])
            else:
                ids = [self.bytes_to_id[bytes([b])] for b in part]
                ans.extend(self._merge_ids(ids))

        return ans


    def encode_iterable(self, iterable):
        """给定一个字符串的可迭代对象（如 Python 文件句柄），
        返回一个惰性产出 token ID 的生成器。
        用于内存高效地 tokenize 无法直接加载到内存的大文件。"""
        for line in iterable:
            if not line:
                continue
            yield from self.encode(line)


    def decode(self, ids: list[int]):
        ans_text = []
        for i in ids:
            ans_text.append(self.vocab[i])

        full_bytes = b''.join(ans_text)

        text = full_bytes.decode("utf-8", errors="replace")

        return text