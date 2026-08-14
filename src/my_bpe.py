import regex as re
import os 
from multiprocessing import Pool

from .pretokenization_example import find_chunk_boundaries

def merge(ids,pair,idx,special_list):

    if ids in special_list:
        return ids

    i=0

    ans = []
    
    while i < len(ids):
        if ids[i]==pair[0] and i < len(ids)-1 and ids[i+1] == pair[1]:
            ans.append(idx)
            i+=2
        else:
            ans.append(ids[i])
            i+=1
            continue
        
    return ans


def get_state(ids,counts,special_list):
    '''
    用来返回连续对出现次数
    '''
    if ids in special_list:
        return counts

    for pair in zip(ids,ids[1:]):
        counts[pair] = counts.get(pair,0)+1
        
    return counts

def _process_chunk(args):
    """多进程 Worker：读取文件切片，并保护 special_tokens 不被正则拆碎"""
    input_path, start, end, pattern_str, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk_text = f.read(end - start).decode("utf-8", errors="ignore")

    pattern = re.compile(pattern_str)
    chunk_bytes_list = []

    if special_tokens:
        # 构造特殊字符匹配正则
        special_pattern = re.compile(
            "(" + "|".join(re.escape(t) for t in special_tokens) + ")"
        )

        for part in re.split(special_pattern, chunk_text):
            if not part:
                continue
            if part in special_tokens:
                # 关键：特殊 token 作为一个整体放入，保护其不被正则切割，不参与后续 merge
                chunk_bytes_list.append(list(part.encode("utf-8")))
            else:
                for chunk in re.findall(pattern, part):
                    chunk_bytes_list.append(list(chunk.encode("utf-8")))
    else:
        for chunk in re.findall(pattern, chunk_text):
            chunk_bytes_list.append(list(chunk.encode("utf-8")))

    return chunk_bytes_list


def train_bpe(input_path, vocab_size, special_tokens=None,**kwargs):
    if special_tokens is None:
        special_tokens=[]

    vocab = {idx: bytes([idx]) for idx in range(256)}
    new_idx=256
    merges=[]
    special_list=[list(i.encode("utf-8")) for i in special_tokens]

    special_vocab={}
    for st in special_tokens:
        special_vocab[st] = new_idx
        vocab[new_idx] = st.encode("utf-8")
        new_idx+=1

    pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    text_byte = []

    if input_path is not None and os.path.exists(input_path):
        num_processes = kwargs.get("num_processes",os.cpu_count() or 4)

        split_token = (
            special_tokens[0].encode("utf-8") if special_tokens else b"\n"
        )

        with open(input_path,"rb") as f:
            boundaries = find_chunk_boundaries(f,num_processes,split_token)

        task = [
            (input_path,start,end,pattern,special_tokens)
            for start,end in zip(boundaries[:-1],boundaries[1:])
        ]

        with Pool(processes=num_processes) as pool:
            results = pool.map(_process_chunk,task)
            for res in results:
                text_byte.extend(res)
    else:
        raise ValueError("invalid path")

    # word_state = [] #用来记录每个word的有什么对,每个对的个数

    # word_idx = {} #用来记录每个对在那个word出现

    # for idx,word in enumerate(text_byte):
    #     if word not in special_list:
    #         temp = {}
    #         for pair in zip(word,word[1:]):
    #             if pair not in word_idx:
    #                 word_idx[pair] = set()

    #             if idx not in word_idx[pair]:
    #                 word_idx[pair].add(idx)
    #             temp[pair] = temp.get(pair,0)+1

    #         word_state.append(temp)
    #     else:
    #         word_state.append({})

    # counts = {}

    # for word in text_byte:
    #     counts = get_state(word,counts,special_list)


    word_idx = {} #用来记录每个对在那个word出现

    counts = {} #全局pair

    for idx,word in enumerate(text_byte):
        if word not in special_list:
            for pair in zip(word,word[1:]):
                if pair not in word_idx:
                    word_idx[pair]=set()
                word_idx[pair].add(idx)
                counts[pair] = counts.get(pair,0)+1


    for _ in range(vocab_size-256-len(special_tokens)):

        if not counts:
            break

        max_pair = max(counts, key=lambda p: (counts[p], vocab[p[0]], vocab[p[1]]))

        max_idx = word_idx[max_pair]


        # for _ in sorted(max_idx):
        #     temp = []
        #     i=0
        #     while i < len(text_byte[_]):
        #         if text_byte[_][i]==max_pair[0] and i < len(text_byte[_])-1 and text_byte[_][i+1] == max_pair[1]:
        #             temp.append(new_idx)
        #             i+=2

        #             # text_byte[_].pop(i)
        #             # text_byte[_].pop(i)
        #             # text_byte[_].insert(i,new_idx)
        #             # i+=1
        #         else:
        #             temp.append(text_byte[_][i])
        #             i+=1

        #     text_byte[_] = temp

        # for _ in sorted(max_idx):
        #     for key,val in word_state[_].items():
        #         counts[key]=counts[key]-val
        #         word_idx[key].remove(_)

        #     temp={}
        #     for pair in zip(text_byte[_],text_byte[_][1:]):
        #         counts[pair] = counts.get(pair,0)+1

        #         if pair not in word_idx:
        #             word_idx[pair] = set()
                
        #         if _ not in word_idx[pair]:
        #             word_idx[pair].add(_)
        #         temp[pair] = temp.get(pair,0)+1
                
        #     word_state[_]=temp

        # for key in list(counts.keys()):
        #     if counts[key] <= 0:
        #         del counts[key]

        # #text_byte=[merge(ids,max_pair,new_idx,special_list) for ids in text_byte]
        # vocab[new_idx]=vocab[max_pair[0]]+vocab[max_pair[1]]
        # merges.append((vocab[max_pair[0]],vocab[max_pair[1]]))
        # new_idx+=1

        for _ in sorted(max_idx):
            temp = []

            i=0
            while i<len(text_byte[_]):
                if text_byte[_][i]==max_pair[0] and i < len(text_byte[_])-1 and text_byte[_][i+1] == max_pair[1]:
                    temp.append(new_idx)
                    i+=2
                else:
                    temp.append(text_byte[_][i])
                    i+=1


            old_counts = {}
            for pair in zip(text_byte[_],text_byte[_][1:]):
                old_counts[pair] = old_counts.get(pair,0)+1

            text_byte[_] = temp

            for key,val in old_counts.items():
                counts[key] = counts[key]-val
                word_idx[key].remove(_)

            for pair in zip(text_byte[_],text_byte[_][1:]):
                counts[pair] = counts.get(pair,0)+1
                if pair not in word_idx:
                    word_idx[pair] = set()

                word_idx[pair].add(_)
                
        for key in list(counts.keys()):
            if counts[key] <= 0:
                del counts[key]

        vocab[new_idx]=vocab[max_pair[0]]+vocab[max_pair[1]]
        merges.append((vocab[max_pair[0]],vocab[max_pair[1]]))
        new_idx+=1

    return (vocab,merges)
    