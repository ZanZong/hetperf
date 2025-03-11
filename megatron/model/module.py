# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Megatron Module"""

import torch
from torch.autograd import Variable
import torch.distributed
from torch.nn.parameter import Parameter

from megatron import get_args
from megatron.core import mpu, tensor_parallel


_FLOAT_TYPES = (torch.FloatTensor, torch.cuda.FloatTensor)
_HALF_TYPES = (torch.HalfTensor, torch.cuda.HalfTensor)
_BF16_TYPES = (torch.BFloat16Tensor, torch.cuda.BFloat16Tensor)



def param_is_not_shared(param):
    return not hasattr(param, 'shared') or not param.shared



class MegatronModule(torch.nn.Module):
    """Megatron specific extensions of torch Module with support
    for pipelining."""

    def __init__(self, config=None, share_embeddings_and_output_weights=True):
        super(MegatronModule, self).__init__()
        self.config = config
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights


    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        """Use this function to override the state dict for
        saving checkpoints."""
        return self.state_dict(prefix=prefix, keep_vars=keep_vars)


    def shared_embedding_or_output_weight(self):
        if self.pre_process:
            return self.language_model.embedding.word_embeddings.weight
        else:
            if not self.share_embeddings_and_output_weights:
                raise Exception('shared_embedding_or_output_weight() called for last '
                                'stage, but share_embeddings_and_output_weights is false')
            return self.word_embeddings.weight


    def initialize_word_embeddings(self):
        args = get_args()
        if not self.share_embeddings_and_output_weights:
            raise Exception('initialize_word_embeddings() was called but '
                            'share_embeddings_and_output_weights is false')

        # This function just initializes the word embeddings in the final stage
        # when we are using pipeline parallelism. Nothing to do if we aren't
        # using pipeline parallelism.
        if args.pipeline_model_parallel_size == 1:
            return

        # Parameters are shared between the word embeddings layers, and the
        # heads at the end of the model. In a pipelined setup with more than
        # one stage, the initial embedding layer and the head are on different
        # workers, so we do the following:
        # 1. Create a second copy of word_embeddings on the last stage, with
        #    initial parameters of 0.0.
        # 2. Do an all-reduce between the first and last stage to ensure that
        #    the two copies of word_embeddings start off with the same
        #    parameter values.
        # 3. In the training loop, before an all-reduce between the grads of
        #    the two word_embeddings layers to ensure that every applied weight
        #    update is the same on both stages.
        if mpu.is_pipeline_last_stage() and not self.pre_process:
            assert not mpu.is_pipeline_first_stage()
            self._word_embeddings_for_head_key = 'word_embeddings_for_head'
            # set word_embeddings weights to 0 here, then copy first
            # stage's weights using all_reduce below.
            self.word_embeddings = tensor_parallel.VocabParallelEmbedding(
                args.padded_vocab_size, self.config.hidden_size,
                config=self.config, init_method=self.config.init_method)
            self.word_embeddings.weight.data.fill_(0)
            self.word_embeddings.weight.shared = True

        # Zero out initial weights for decoder embedding.
        # NOTE: We don't currently support T5 with the interleaved schedule.
        if not mpu.is_pipeline_first_stage(ignore_virtual=True) and \
                self.pre_process:
            self.language_model.embedding.zero_parameters()

        if not torch.distributed.is_initialized():
            if not getattr(MegatronModule, "embedding_warning_printed", False):
                print("WARNING! Distributed processes aren't initialized, so "
                      "word embeddings in the last layer are not initialized. "
                      "If you are just manipulating a model this is fine, but "
                      "this needs to be handled manually. If you are training "
                      "something is definitely wrong.")
                MegatronModule.embedding_warning_printed = True
            return

        # Ensure that first and last stages have the same initial parameter
        # values.
        # print(f"rank={torch.distributed.get_rank()}, mpu.is_rank_in_embedding_group()={mpu.is_rank_in_embedding_group()}", flush=True)

        if mpu.is_rep_rank_in_embedding_group():
            rank = torch.distributed.get_rank()
            embedding_weight = self.shared_embedding_or_output_weight().data
            tp_world_size = mpu.get_tensor_model_parallel_world_size()
            if tp_world_size > 1: # gather from tp group
                rep_rank = mpu._TENSOR_MODEL_PARALLEL_GLOBAL_RANKS[0]
                if rank == rep_rank:
                    # gather in tp group
                    gather_list = [torch.zeros_like(embedding_weight) for _ in range(tp_world_size)]
                    torch.distributed.gather(embedding_weight, gather_list=gather_list, dst=rep_rank, \
                                             group=mpu.get_tensor_model_parallel_group())
                    gathered_embedding_weight = torch.cat(gather_list, dim=0)
                    
                    # all-reduce in embedding group
                    torch.distributed.all_reduce(gathered_embedding_weight, group=mpu.get_embedding_group())

                    # scatter in tp group
                    split_embedding_weight = torch.chunk(gathered_embedding_weight, tp_world_size, dim=0)
                    scatter_list = list(split_embedding_weight)
                    torch.distributed.scatter(embedding_weight, scatter_list=scatter_list, src=rep_rank, \
                                            group=mpu.get_tensor_model_parallel_group())
                else:
                    # gather in tp group
                    torch.distributed.gather(embedding_weight, gather_list=None, dst=rep_rank, \
                                             group=mpu.get_tensor_model_parallel_group())
                    # scatter in tp group
                    torch.distributed.scatter(embedding_weight, scatter_list=None, src=rep_rank, \
                                            group=mpu.get_tensor_model_parallel_group())
            else:
                torch.distributed.all_reduce(embedding_weight, group=mpu.get_embedding_group())
        else:
            if mpu.is_rank_in_embedding_group():
                torch.distributed.all_reduce(self.shared_embedding_or_output_weight().data, group=mpu.get_embedding_group())

        # Ensure that encoder(first stage) and decoder(split stage) position
        # embeddings have the same initial parameter values
        # NOTE: We don't currently support T5 with the interleaved schedule.
        
        if mpu.is_rep_rank_in_position_embedding_group():
            if mpu._POSITION_EMBEDDING_GLOBAL_RANKS is not None and \
                len(mpu._POSITION_EMBEDDING_GLOBAL_RANKS) > 1:
                
                self.language_model.embedding.cuda()
                position_embeddings = self.language_model.embedding.position_embeddings
                
                rank = torch.distributed.get_rank()
                embedding_weight = position_embeddings.data
                tp_world_size = mpu.get_tensor_model_parallel_world_size()
                if tp_world_size > 1: # gather from tp group
                    rep_rank = mpu._TENSOR_MODEL_PARALLEL_GLOBAL_RANKS[0]
                    if rank == rep_rank:
                        # gather in tp group
                        gather_list = [torch.zeros_like(embedding_weight) for _ in range(tp_world_size)]
                        torch.distributed.gather(embedding_weight, gather_list=gather_list, dst=rep_rank, \
                                                group=mpu.get_tensor_model_parallel_group())
                        gathered_embedding_weight = torch.cat(gather_list, dim=0)
                        
                        # all-reduce in positio embedding group
                        torch.distributed.all_reduce(gathered_embedding_weight, group=mpu.get_position_embedding_group())

                        # scatter in tp group
                        split_embedding_weight = torch.chunk(gathered_embedding_weight, tp_world_size, dim=0)
                        scatter_list = list(split_embedding_weight)
                        torch.distributed.scatter(embedding_weight, scatter_list=scatter_list, src=rep_rank, \
                                                group=mpu.get_tensor_model_parallel_group())
                    else:
                        # gather in tp group
                        torch.distributed.gather(embedding_weight, gather_list=None, dst=rep_rank, \
                                                group=mpu.get_tensor_model_parallel_group())
                        # scatter in tp group
                        torch.distributed.scatter(embedding_weight, scatter_list=None, src=rep_rank, \
                                                group=mpu.get_tensor_model_parallel_group())
                else:
                    torch.distributed.all_reduce(embedding_weight, group=mpu.get_position_embedding_group())
        else:
            if mpu.is_rank_in_position_embedding_group() and \
                    args.pipeline_model_parallel_split_rank is not None:
                # TODO: Support tokentype embedding.
                self.language_model.embedding.cuda()
                position_embeddings = self.language_model.embedding.position_embeddings
                torch.distributed.all_reduce(position_embeddings.weight.data,
                                            group=mpu.get_position_embedding_group())
        print("--init word embedding finish", flush=True)


def conversion_helper(val, conversion):
    """Apply conversion to val. Recursively apply conversion if `val`
    #is a nested tuple/list structure."""
    if not isinstance(val, (tuple, list)):
        return conversion(val)
    rtn = [conversion_helper(v, conversion) for v in val]
    if isinstance(val, tuple):
        rtn = tuple(rtn)
    return rtn


def fp32_to_float16(val, float16_convertor):
    """Convert fp32 `val` to fp16/bf16"""
    def half_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if isinstance(val_typecheck, _FLOAT_TYPES):
            val = float16_convertor(val)
        return val
    return conversion_helper(val, half_conversion)


def float16_to_fp32(val):
    """Convert fp16/bf16 `val` to fp32"""
    def float_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if isinstance(val_typecheck, (_BF16_TYPES, _HALF_TYPES)):
            val = val.float()
        return val
    return conversion_helper(val, float_conversion)



class Float16Module(MegatronModule):

    def __init__(self, module, args):
        super(Float16Module, self).__init__()

        if args.fp16:
            self.add_module('module', module.half())
            def float16_convertor(val):
                return val.half()
        elif args.bf16:
            self.add_module('module', module.bfloat16())
            def float16_convertor(val):
                return val.bfloat16()
        else:
            raise Exception('should not be here')

        self.float16_convertor = float16_convertor


    def set_input_tensor(self, input_tensor):
        return self.module.set_input_tensor(input_tensor)


    def forward(self, *inputs, **kwargs):
        if mpu.is_pipeline_first_stage():
            inputs = fp32_to_float16(inputs, self.float16_convertor)
        outputs = self.module(*inputs, **kwargs)
        if mpu.is_pipeline_last_stage():
            outputs = float16_to_fp32(outputs)
        return outputs


    def state_dict(self, prefix='', keep_vars=False):
        return self.module.state_dict(prefix=prefix, keep_vars=keep_vars)


    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        return self.module.state_dict_for_save_checkpoint(prefix=prefix,
                                                          keep_vars=keep_vars)


    def load_state_dict(self, state_dict, strict=True):
        self.module.load_state_dict(state_dict, strict=strict)
