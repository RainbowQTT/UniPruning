from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

import torch
import numpy as np
import os
import re
from datasets import load_dataset, DatasetDict
import random
import argparse
from trl import SFTConfig, SFTTrainer 

from adamw_spp_prune import AdamWSPP
from TrainerSFT_SPP import SFTTrainerWithSPP
from typing import Tuple
from inner_loop_without_grad_decay import spp_inner_training_loop

from transformers import set_seed

set_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default= "meta-llama/Llama-2-7b", 
                    help='Provide the model path for finetuning')                
parser.add_argument('--dataset_path', type=str, default="", 
                    help='Calibration dataset path')
parser.add_argument('--dataset_name', type=str, default="en", 
                    help='Calibration dataset name')
parser.add_argument('--saving_path', type=str, default="", 
                    help='saving path')
parser.add_argument('--mask', type=bool, default=False, 
                    help='using mask or not')
parser.add_argument('--search', type=bool, default=True, 
                    help='spp search')
parser.add_argument('--ctx_len', type=int, default="4096", 
                    help='ctx length ratio')
parser.add_argument('--samples', type=int, default="128", 
                    help='samples nr')
parser.add_argument('--batch_size', type=int, default="1", 
                    help='batch size')
parser.add_argument('--learning_rate', type=float, default=1e-4, 
                    help='lr')

# compression parameters
parser.add_argument('--w_sp_attn', default=4.8e-6, type=float, help='regularization coefficient for attn')
parser.add_argument('--w_sp_mlp', default=2e-7, type=float, help='regularization coefficient for mlp')
parser.add_argument('--lambda_param', default=0.01, type=float, help='lambda parameter for proximal operator')
parser.add_argument('--lambda2_param', default=0.01, type=float, help='lambda parameter for proximal operator')

parser.add_argument('--p', default=0.5, type=float, help='total compression ratio')
parser.add_argument('--interval', default=20, type=int, help='interval of updating compression mask')
parser.add_argument('--use_ceph', action='store_true')  

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
seed_everything(42)

args = parser.parse_args()
model_path = args.model_path
dataset_path = args.dataset_path
dataset_name = args.dataset_name
lr = args.learning_rate
dataset = load_dataset("json", data_files=args.dataset_path)
samples = args.samples
batch_size = args.batch_size
ctx_len = args.ctx_len
w_sp_attn = args.w_sp_attn
w_sp_mlp = args.w_sp_mlp
lambda_param = args.lambda_param
lambda2_param = args.lambda2_param
saving_path = args.saving_path

train_testvalid = dataset["train"].train_test_split(test_size=0.95, seed=42) # train 3563 on 0.99 # when data size is large, there might be memory outage, mostly 95
valid_test = train_testvalid["test"].train_test_split(test_size=0.999, seed=42) # valid 352
dataset = DatasetDict({
    'train': train_testvalid['train'].select(range(samples)), # 5x data, but 16x, in case of huggingface bug
    'validation': valid_test['train'].select(range(32)), # we prune at every checkpoint
    'test': valid_test['test']})
del train_testvalid, valid_test, dataset["test"]

# load model
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16, # float 32
    device_map= "auto",
)

model.train()

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token 
tokenizer.padding_side = "left" 

dataset_id = dataset_name

repository_id=saving_path.split('/')[-1]
sft_config = SFTConfig(
    dataset_text_field="text",
    output_dir=saving_path,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    max_seq_length=ctx_len,
    learning_rate=lr,
    num_train_epochs=1,
    optim="adamw_torch", 
    warmup_ratio = 0.1, 
    logging_dir=f"{repository_id}/logs",
    logging_strategy="steps",
    logging_steps=0.01,
    logging_first_step=True,
    save_strategy="no", 
    save_total_limit=0,  
    load_best_model_at_end=False,  
    eval_accumulation_steps=2,
    eval_steps=0.5,
    save_only_model=False,  
    seed=42,
)
trainer = SFTTrainerWithSPP(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    args=sft_config,
    w_sp_attn=w_sp_attn,
    w_sp_mlp=w_sp_mlp,
    lambda_param=lambda_param,
    lambda2_param=lambda2_param,
    training_steps_per_epoch= samples,
    event_dir=os.path.join(saving_path, "gamma_record"), 
)

SFTTrainerWithSPP._inner_training_loop = spp_inner_training_loop

trainer.train()
print("Searching Finished!")