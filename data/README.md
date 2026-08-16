# Data

Tracked in git:

- `vocab.pkl`, `merges.pkl` — BPE trained on a 50MB TinyStories slice (`train_50m.txt`)

Not in git (download / generate locally):

- `TinyStoriesV2-GPT4-train.txt` — full train corpus (~2.1GB)
- `train_tokens.npy` — `python prepare_data.py` encodes the corpus with the pickled BPE

TinyStories V2 (GPT-4) is the dataset used here. Put the train file at `data/TinyStoriesV2-GPT4-train.txt`, then:

```bash
python prepare_data.py
python train.py
```

If you only want to sample and already have `checkpoint.pt`, you only need `vocab.pkl` and `merges.pkl`.
