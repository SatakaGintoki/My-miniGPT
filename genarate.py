import torch
import pickle
from src.my_model import Transformer_LM, get_device, softmax
from src.my_tokenizer import Tokenizer
from src.training_loop import load_checkpoint
from config import MODEL_CONFIG, TOKENIZER_CONFIG, TRAIN_CONFIG, PATHS

device = get_device()

with open(PATHS["vocab"],"rb") as f:
    vocab = pickle.load(f)
with open(PATHS["merges"],"rb") as f:
    merges = pickle.load(f)


vocab_size = MODEL_CONFIG["vocab_size"]
context_length = MODEL_CONFIG["context_length"]
num_layers = MODEL_CONFIG["num_layers"]
d_model = MODEL_CONFIG["d_model"]
num_heads = MODEL_CONFIG["num_heads"]
d_ff = MODEL_CONFIG["d_ff"]
rope_theta = MODEL_CONFIG["rope_theta"]

batch_size = TRAIN_CONFIG["batch_size"]
max_steps = TRAIN_CONFIG["max_steps"]
max_lr = TRAIN_CONFIG["max_lr"]
min_lr = TRAIN_CONFIG["min_lr"]
warmup_steps = TRAIN_CONFIG["warmup_steps"]
grad_clip_val = TRAIN_CONFIG["grad_clip_val"]


weight_path = PATHS["checkpoint"]

tokenizer = Tokenizer(
    vocab=vocab,
    merges=merges,
    special_tokens=TOKENIZER_CONFIG["special_tokens"]
)


model = Transformer_LM(
    vocab_size=vocab_size,
    context_length=context_length,
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    rope_theta=rope_theta
).to(device)

load_checkpoint(weight_path,model=model)

end_token_ids = tokenizer.encode("<|endoftext|>")[0]

def top_p_filter(pro,p):
    sorted_p,indices = torch.sort(pro,descending = True)

    cumsum_p = torch.cumsum(sorted_p,dim=0)

    mask = cumsum_p <= p

    mask[0]=True

    sorted_p = sorted_p*mask

    sorted_p = sorted_p / sorted_p.sum()

    ans= torch.zeros_like(pro)

    ans[indices] = sorted_p

    return ans

    
@torch.no_grad()
def sample_next_token(model,token_ids,temperatrue = 0.5,top_p = 0.9,device = device):

    torch_token_ids = torch.tensor(token_ids).to(device)
    if len(torch_token_ids) > context_length:
        logit = torch_token_ids[-context_length:]
    else:
        logit = torch_token_ids

    output_list = model(logit)

    last_ids = output_list[-1]

    pro = softmax(last_ids/temperatrue,i=0)

    if top_p < 1.0:
        pro = top_p_filter(pro, top_p)

    chose_id = torch.multinomial(pro, 1).item()

    return chose_id



prompt = "Once upon a time"

token_ids = tokenizer.encode(prompt)

new_token = 100



for _ in range(new_token):
    new_id = sample_next_token(model,token_ids)

    token_ids.append(new_id)

    if new_id == end_token_ids:
        break


text = tokenizer.decode(token_ids)
print(text)





