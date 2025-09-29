#Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from trl import SFTTrainer
from adamw_spp_prune import AdamWSPP 
import torch
from transformers import get_scheduler
from transformers.trainer import TrainerState, TRAINER_STATE_NAME, TrainOutput
import os, math, time, shutil
import torch
from transformers.trainer_utils import speed_metrics
from transformers.utils import logging
logger = logging.get_logger(__name__)

class SFTTrainerWithSPP(SFTTrainer):
    def __init__(self, *args, **kwargs):

        self.w_sp_attn = kwargs.pop("w_sp_attn", 4.8e-4)
        self.w_sp_mlp = kwargs.pop("w_sp_mlp", 2e-5)
        self.lambda_param = kwargs.pop("lambda_param", 0.01)
        self.lambda2_param = kwargs.pop("lambda2_param", 0.01)
        self.training_steps_per_epoch = kwargs.pop("training_steps_per_epoch", 128)
        self.event_dir = kwargs.pop("event_dir", None)  

        super().__init__(*args, **kwargs) 

    def is_attention_layer(self, name):
        return any(k in name for k in ["q_proj", "k_proj", "v_proj", "o_proj"]) and "bias" not in name

    def is_mlp_layer(self, name):
        return any(k in name for k in ["fc1", "fc2", "up_proj", "down_proj", "gate_proj"]) and "bias" not in name
        
    def create_optimizer(self):
        attn_params = []
        mlp_params = []
        attn_names = []
        mlp_names = []
        for name, param in self.model.named_parameters():
            is_attn = self.is_attention_layer(name)
            is_mlp = self.is_mlp_layer(name)
            param.is_attn = is_attn
            param.is_mlp = is_mlp
            if is_attn:
                attn_params.append(param)
                attn_names.append(name)
            elif is_mlp:
                mlp_params.append(param)
                mlp_names.append(name)

        self.optimizer = AdamWSPP(
            [
                {
                    "params": attn_params,
                    "lr": self.args.learning_rate,
                    'weight_decay':0, 
                    'eps':1e-10, 
                    'sp_role':"attn",
                    'names': attn_names,
                },
                {
                    "params": mlp_params,
                    "lr": self.args.learning_rate,
                    'weight_decay':0, 
                    'eps':1e-10, 
                    'sp_role':"mlp",
                    'names': mlp_names,
                }
            ],
            event_dir=self.event_dir,
            lambda_param=self.lambda_param,
            lambda2_param=self.lambda2_param,
            sapmle_num=self.training_steps_per_epoch
        )
        return self.optimizer
   