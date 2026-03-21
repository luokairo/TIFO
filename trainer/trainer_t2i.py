import torch
import functools
import numpy as np
from itertools import repeat
from easydict import EasyDict
import torch.distributed as dist
from torch.cuda.amp import autocast
import os
from utils import logger, AverageMeter
from trainer.utils import build_optimizer
from dataset.tifo_dataset2 import TextToImageDataloader
from trainer.utils import TrainerBase, print_model_param_num
from models import VLChatProcessor, MultiModalityCausalLM, MultiModalityConfig
import torch.nn as nn

import traceback

def rprint(msg):
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else -1
    print(f"[rank {rank}] {msg}", flush=True)


def init_tifo(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

def repeater(data_loader):
    for i, loader in enumerate(repeat(data_loader)):
        sampler = getattr(loader, "sampler", None)
        if sampler is not None:
            sampler.set_epoch(i)
        for data in loader:
            yield data

def train_setup(model: MultiModalityCausalLM):
    for n, p in model.language_model.named_parameters():
        p.requires_grad = False
    model.language_model.eval()
    model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    for n, p in model.gen_embed.named_parameters():
        p.requires_grad = False
    model.gen_embed.eval()
    for n, p in model.gen_head.named_parameters():
        p.requires_grad = False
    model.gen_head.eval()
    for n, p in model.gen_aligner.named_parameters():
        p.requires_grad = False
    model.gen_aligner.eval()
    for n, p in model.aligner.named_parameters():
        p.requires_grad = False
    model.aligner.eval()
    for n, p in model.vision_model.named_parameters():
        p.requires_grad = False
    model.vision_model.eval()
    for n, p in model.gen_vision_model.named_parameters():
        p.requires_grad = False
    model.gen_vision_model.eval()
    # only training tifo's module
    for n, p in model.vision_slots_adapter.named_parameters():
        p.requires_grad = True
    model.vision_slots_adapter.train()
    for n, p in model.text_slots_adapter.named_parameters():
        p.requires_grad = True
    model.text_slots_adapter.train()
    for n, p in model.text_conductor.named_parameters():
        p.requires_grad = True
    model.text_conductor.train()


def find_latest_directory(base_path):
    if not os.path.exists(base_path):
        return None
    subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    subdirs.sort(key=lambda x: float(x))
    for subdir in reversed(subdirs):
        full_path = os.path.join(base_path, subdir)
        ckpt_path = os.path.join(full_path, 'ckpt')
        if os.path.isdir(ckpt_path):
            latest_checkpoint = find_latest_checkpoint(ckpt_path)
            return latest_checkpoint
    return None

def find_latest_checkpoint(ckpt_path):
    pth_files = [f for f in os.listdir(ckpt_path) if f.endswith('.pth')]
    pth_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    return os.path.join(ckpt_path, pth_files[-1]) if pth_files else None

class TextToImageTrainer(TrainerBase):
    def __init__(self, cfg):
        super().__init__()
        try:
            rprint("enter trainer __init__")

            self.cfg = cfg
            rprint("after self.cfg = cfg")

            self.data_loader, self.prob = TextToImageDataloader(cfg, tasks=cfg.dataloader.tasks)
            rprint("after TextToImageDataloader")

            # self.cfg.dataloader_len = len(self.data_loader[0]) // self.prob[0]
            self.cfg.dataloader_len = len(self.data_loader[0])
            print(self.cfg.dataloader_len)
            self._data_loader_iter = [iter(repeater(dl)) for dl in self.data_loader]
            rprint("after build data loader iter")

            self.totals = EasyDict()
            self.totals.epochs = self.cfg.optimize.max_epochs
            # self.totals.iter_per_epoch = len(self.data_loader[0]) // self.prob[0]
            self.totals.iter_per_epoch = len(self.data_loader[0])
            self.totals.total_iters = self.cfg.optimize.max_epochs * self.totals.iter_per_epoch
            rprint("after totals setup")

            self.dist = EasyDict()
            self.dist.rank = dist.get_rank()
            self.dist.world_size = dist.get_world_size()
            rprint(f"after dist setup, world_size={self.dist.world_size}")

            # !!! 这里我建议你顺手也打印一下 search_path
            # self.search_path = f'{self.cfg.common.pre_path}/configs/t2i_generation.yml'
            self.search_path = self.cfg.common.pre_path
            rprint(f"search_path = {self.search_path}")

            self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(self.cfg.model.processor_path)
            rprint("after VLChatProcessor.from_pretrained")

            self.tokenizer = self.vl_chat_processor.tokenizer
            rprint("after tokenizer setup")

            self.model_before_ddp = MultiModalityCausalLM.from_pretrained(
                cfg.model.model_path, trust_remote_code=True
            ).to(torch.bfloat16).cuda()
            rprint("after MultiModalityCausalLM.from_pretrained + cuda")

            self.model_before_ddp.text_conductor.apply(init_tifo)
            rprint("after init text_conductor")

            self.model_before_ddp.text_slots_adapter.apply(init_tifo)
            rprint("after init text_slots_adapter")

            self.model_before_ddp.vision_slots_adapter.apply(init_tifo)
            rprint("after init vision_slots_adapter")

            train_setup(self.model_before_ddp)
            rprint("after train_setup")

            self.pretrain_model = find_latest_directory(self.search_path)
            rprint(f"pretrain_model = {self.pretrain_model}")

            if self.pretrain_model is not None:
                rprint("before torch.load pretrain_model")
                state_dict = torch.load(self.pretrain_model, "cpu")
                rprint("after torch.load pretrain_model")

                if "ema" in state_dict:
                    module_dict = state_dict["ema"]
                    rprint("use ema in checkpoint")
                elif "module" in state_dict:
                    module_dict = state_dict["module"]
                    rprint("use module in checkpoint")
                else:
                    module_dict = state_dict
                    rprint("use raw state_dict in checkpoint")

                rprint("before load_state_dict")
                missing, unexpected = self.model_before_ddp.load_state_dict(module_dict, strict=False)
                rprint("after load_state_dict")

                del state_dict
                rprint(f"GPT: Restored from {self.pretrain_model} with {len(missing)} missing and {len(unexpected)} unexpected keys")
                if len(missing) > 0:
                    rprint(f"Missing Keys: {missing}")
                    rprint(f"Unexpected Keys: {unexpected}")

            rprint("before print_model_param_num")
            print_model_param_num(cfg.model, self.model_before_ddp)
            rprint("after print_model_param_num")

            if not self.cfg.common.use_fsdp:
                raise NotImplementedError
            rprint("USING FSDP from Pytorch...")

            from torch.distributed.fsdp import (
                FullyShardedDataParallel as FSDP,
                MixedPrecision,
                BackwardPrefetch,
                ShardingStrategy,
            )
            from torch.distributed.fsdp.fully_sharded_data_parallel import BackwardPrefetch
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
            from transformers.models.llama.modeling_llama import LlamaDecoderLayer
            rprint("after fsdp imports")

            my_auto_wrap_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={LlamaDecoderLayer},
            )
            rprint("after my_auto_wrap_policy")

            self.use_bf16 = cfg.common.use_bf16
            self.use_fp16 = cfg.common.use_fp16
            rprint(f"use_bf16={self.use_bf16}, use_fp16={self.use_fp16}")

            if not self.use_bf16:
                raise NotImplementedError

            logger.info("Use bfloat16 training...")
            fpSixteen = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
            rprint("after MixedPrecision")

            rprint("before FSDP wrap")
            self.model = FSDP(
                self.model_before_ddp,
                auto_wrap_policy=my_auto_wrap_policy,
                mixed_precision=fpSixteen if self.use_bf16 else None,
                device_id=torch.cuda.current_device(),
                sharding_strategy=ShardingStrategy.HYBRID_SHARD,
                backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                use_orig_params=True,
                limit_all_gathers=True
            )
            rprint("after FSDP wrap")

            self.optimizer = build_optimizer(cfg, self.model)
            rprint("finish build optimizer")

            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
            rprint("before ShardedGradScaler")
            self._scaler = ShardedGradScaler(enabled=self.use_fp16)
            rprint("after ShardedGradScaler")

            rprint("before barrier")
            dist.barrier()
            rprint("after barrier")

            self.model.train()
            rprint("begin train")

        except Exception as e:
            rprint(f"FAILED with error: {repr(e)}")
            traceback.print_exc()
            raise
        self.gradient_accumulation_steps = self.cfg.optimize.get("gradient_accumulation_steps", 1)
        self.input_token_max_len = [1000, 3000, 3000]

        self.meters = EasyDict()
        self.meters["batch_time"] = AverageMeter(self.cfg.common.log_interval, fstr="%.3f")
        self.meters["loss1"] = AverageMeter(self.cfg.common.log_interval, fstr="%.5f")
        self.meters["grad_norm1"] = AverageMeter(self.cfg.common.log_interval, fstr="%.5f")
        self.meters["loss2"] = AverageMeter(self.cfg.common.log_interval, fstr="%.5f")
        self.meters["grad_norm2"] = AverageMeter(self.cfg.common.log_interval, fstr="%.5f")
        self.meters["loss3"] = AverageMeter(self.cfg.common.log_interval, fstr="%.5f")
        self.meters["loss4"] = AverageMeter(self.cfg.common.log_interval, fstr="%.5f")
        self.meters["lr"] = AverageMeter(self.cfg.common.log_interval, fstr="%.3e")

    

    

    def get_next_data(self):
        # x = np.array(range(len(self._data_loader_iter)))
        # idx = np.random.choice(a=x, size=1, replace=True, p=self.prob)[0]
        # input_data = next(self._data_loader_iter[idx])
        # batch_size = len(input_data)
        # input_token_max_len = self.input_token_max_len[idx]
        idx = 0
        input_data = next(self._data_loader_iter[0])
        batch_size = len(input_data)
        input_token_max_len = self.input_token_max_len[0]
        if idx == 0:
            batched_input_ids = torch.full(
                (batch_size, input_token_max_len), self.vl_chat_processor.pad_id
            ).long()
            batched_attention_mask = torch.zeros((batch_size, input_token_max_len)).long()
            image1 = torch.stack([input_data[k]['image'] for k in range(batch_size)], dim=0)
            batched_front = []
            batched_end = []
            batched_use_attn = []
            for k in range(batch_size):
                input_ids = input_data[k]['input_ids']
                front = input_data[k]['front']
                end = input_data[k]['end']
                use_attn = input_data[k]['use_attn']
                seq_len = len(input_ids)
                if seq_len >= input_token_max_len:
                    batched_attention_mask[k, :] = 1
                    batched_input_ids[k, :] = torch.LongTensor(input_ids)[:input_token_max_len]  
                    front = [0, input_token_max_len-1]
                else:
                    batched_attention_mask[k, -seq_len:] = 1
                    batched_input_ids[k, -seq_len:] = torch.LongTensor(input_ids)  
                    front = [input_token_max_len-1-seq_len, input_token_max_len-1]
                end = [input_token_max_len, input_token_max_len+575]
                batched_front.append(front)
                batched_end.append(end)
                batched_use_attn.append(use_attn)
                    
            return {'input_ids': batched_input_ids.cuda(), 'attention_mask': batched_attention_mask.cuda(), 'image1':image1.cuda(), 'task_type': 0, 
                    "front": batched_front, "end": batched_end, "use_attn": batched_use_attn}, idx
        
        # if idx == 1:
        #     batched_input_ids = torch.full(
        #         (batch_size, input_token_max_len), self.vl_chat_processor.pad_id
        #     ).long()
        #     batched_attention_mask = torch.zeros((batch_size, input_token_max_len)).long()
        #     batched_labels = torch.ones((batch_size, input_token_max_len)).long() * -100
        #     batched_images_seq_mask = torch.zeros((batch_size, input_token_max_len)).bool()
        #     image1 = torch.stack([input_data[k]['image'] for k in range(batch_size)], dim=0)
        #     batched_front = []
        #     batched_end = []
        #     batched_use_attn = []
        #     for k in range(batch_size):
        #         input_ids = input_data[k]['input_ids']
        #         labels = input_data[k]['labels']
        #         front = input_data[k]['front']
        #         end = input_data[k]['end']
        #         use_attn = input_data[k]['use_attn']
        #         seq_len = len(input_ids)

        #         if seq_len >= input_token_max_len:
        #             batched_attention_mask[k, :] = 1
        #             batched_input_ids[k, :] = torch.LongTensor(input_ids)[:input_token_max_len]  
        #             batched_labels[k, :] = torch.LongTensor(labels)[:input_token_max_len]  
        #             batched_images_seq_mask[k, :] = (input_ids == self.vl_chat_processor.image_id)[:input_token_max_len]  
        #             use_attn = 0
        #         else:
        #             batched_attention_mask[k, -seq_len:] = 1
        #             batched_input_ids[k, -seq_len:] = torch.LongTensor(input_ids)  
        #             batched_labels[k, -seq_len:] = torch.LongTensor(labels)   
        #             batched_images_seq_mask[k, -seq_len:] = input_ids == self.vl_chat_processor.image_id
        #             offset = input_token_max_len - seq_len
        #             front = [x + offset for x in front]
        #             end = [[a + offset for a in x] for x in end]
        #         batched_front.append(front)
        #         batched_end.append(end)
        #         batched_use_attn.append(use_attn)

            return {'input_ids': batched_input_ids.cuda(), 'attention_mask': batched_attention_mask.cuda(), 'image1':image1.cuda(), 'labels':batched_labels.cuda(), 'image_seq_mask': batched_images_seq_mask.cuda(), 'task_type': 1, 
                    "front": batched_front, "end": batched_end, "use_attn": batched_use_attn}, idx

      
    def run_step(self):
        model_input, idx = self.get_next_data()
        self.optimizer.zero_grad(set_to_none=True)
        with autocast(dtype=torch.bfloat16, cache_enabled=False):
            self._loss, self._loss_attn = self.model(**model_input)
        self._loss.backward()

        total_norm = self._clip_grad_norm(max_norm=1.0)
        self.optimizer.step()
        reduced_loss = self._loss.clone().detach() / self.dist.world_size
        reduced_loss_attn = self._loss_attn.clone().detach() / self.dist.world_size
        reduced_grad_norm = total_norm.clone().detach() / self.dist.world_size
        # if idx==0:
        #     self.meters.loss1.reduce_update(reduced_loss)
        #     self.meters.loss2.reduce_update(reduced_loss_attn)
        #     self.meters.grad_norm1.reduce_update(reduced_grad_norm)
        # elif idx==1:
        #     self.meters.loss3.reduce_update(reduced_loss)
        #     self.meters.loss4.reduce_update(reduced_loss_attn)
        #     self.meters.grad_norm2.reduce_update(reduced_grad_norm)
        self.meters.loss1.reduce_update(reduced_loss)
        self.meters.loss2.reduce_update(reduced_loss_attn)
        self.meters.grad_norm1.reduce_update(reduced_grad_norm)
        # self.meters.lr.reduce_update(torch.tensor(self.optimizer.param_groups[0]['lr']).cuda() / self.dist.world_size)
        self.meters.lr.reduce_update(torch.tensor(self.optimizer.param_groups[0]['lr']).cuda() / self.dist.world_size)

    def _clip_grad_norm(self, max_norm):
        if hasattr(self.cfg.common, "use_fsdp") and self.cfg.common.use_fsdp:
            total_norm = self.model.clip_grad_norm_(max_norm)
        else:
            total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)
        return total_norm