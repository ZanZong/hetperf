# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Model and data parallel groups."""

import os
from typing import Optional

import torch
import networkx as nx
import torch.distributed

from .utils import GlobalMemoryBuffer
from megatron import get_args

# Intra-layer model parallel group that the current rank belongs to.
_TENSOR_MODEL_PARALLEL_GROUP = None
# Inter-layer model parallel group that the current rank belongs to.
_PIPELINE_MODEL_PARALLEL_GROUP = None
_PIPELINE_MODEL_PARALLEL_REP_GROUP = None
# Group ID of the current node's pipeline.
_PIPELINE_GROUP_ID = None
# Model parallel group (both intra- and pipeline) that the current rank belongs to.
_MODEL_PARALLEL_GROUP = None
# Embedding group.
_EMBEDDING_GROUP = None
# Position embedding group.
_POSITION_EMBEDDING_GROUP = None
# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None
_DATA_PARALLEL_GROUP_GLOO = None
# tensor model parallel group and data parallel group combined
# used for fp8 and moe training
_TENSOR_AND_DATA_PARALLEL_GROUP = None
# Expert parallel group that the current rank belongs to.
_TENSOR_AND_EXPERT_PARALLEL_GROUP = None
_DATA_MODULO_EXPERT_PARALLEL_GROUP = None


_VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None
_VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
_PIPELINE_MODEL_PARALLEL_SPLIT_RANK = None

# These values enable us to change the mpu sizes on the fly.
_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_TENSOR_MODEL_PARALLEL_RANK = None
_MPU_PIPELINE_MODEL_PARALLEL_RANK = None

# A list of ranks that have a copy of the embedding.
_EMBEDDING_GLOBAL_RANKS = None

# A list of ranks that have a copy of the position embedding.
_POSITION_EMBEDDING_GLOBAL_RANKS = None

# A list of ranks for current tensor model group
_TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = None

# A list of global ranks for each pipeline group to ease calculation of the source
# rank when broadcasting from the first or last pipeline stage.
_PIPELINE_GLOBAL_RANKS = None

# A list of global ranks for each data parallel group to ease calculation of the source
# rank when broadcasting weights from src to all other data parallel ranks
_DATA_PARALLEL_GLOBAL_RANKS = None

# Context parallel group that the current rank belongs to
_CONTEXT_PARALLEL_GROUP = None
# A list of global ranks for each context parallel group to ease calculation of the
# destination rank when exchanging KV/dKV between context parallel_ranks
_CONTEXT_PARALLEL_GLOBAL_RANKS = None

# Data parallel group information with context parallel combined.
_DATA_PARALLEL_GROUP_WITH_CP = None
_DATA_PARALLEL_GROUP_WITH_CP_GLOO = None
_DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = None

# combined parallel group of TP, DP, and CP used for fp8
_TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = None

# Memory buffers to avoid dynamic memory allocation
_GLOBAL_MEMORY_BUFFER = None

# The device type of all ordered ranks.
_HETERO_DEVICE_TYPES = None

# List of ranks in current pipeline first/last stage (sorted by ranks)
_PIPELINE_FIRST_STAGE_RANKS = None
_PIPELINE_LAST_STAGE_RANKS = None
# List of micro batch sizes of first/last stage (sorted by ranks)
_PIPELINE_FIRST_STAGE_MICRO_BATCH_SIZES = None
_PIPELINE_LAST_STAGE_MICRO_BATCH_SIZES = None

# List of predecessor/successor nodes in pipeline parallel send communication
# each element as (micro_batch_size, [node0, node1, ...])
_SEND_SUCC_NODES = None
_SEND_PRED_NODES = None

# List of predecessor/successor nodes in pipeline parallel recv communication
# each element as (micro_batch_size, node)
_RECV_PRED_NODES = None
_RECV_SUCC_NODES = None

def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    pipeline_model_parallel_split_rank: Optional[int] = None,
    use_sharp: bool = False,
    context_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
) -> None:
    """Initialize model data parallel groups.

    Arguments:
        tensor_model_parallel_size (int, default = 1):
            The number of GPUs to split individual tensors across.

        pipeline_model_parallel_size (int, default = 1):
            The number of tensor parallel GPU groups to split the
            Transformer layers across. For example, if
            tensor_model_parallel_size is 4 and
            pipeline_model_parallel_size is 2, the model will be split
            into 2 groups of 4 GPUs.

        virtual_pipeline_model_parallel_size (int, optional):
            The number of stages that each pipeline group will have,
            interleaving as necessary. If None, no interleaving is
            performed. For example, if tensor_model_parallel_size is 1,
            pipeline_model_parallel_size is 4,
            virtual_pipeline_model_parallel_size is 2, and there are
            16 transformer layers in the model, the model will be
            split into 8 stages with two layers each and each GPU
            would get 2 stages as such (layer number starting with 1):

            GPU 0: [1, 2] [9, 10]
            GPU 1: [3, 4] [11, 12]
            GPU 2: [5, 6] [13, 14]
            GPU 3: [7, 8] [15, 16]

        pipeline_model_parallel_split_rank (int, optional):
            For models with both an encoder and decoder, the rank in
            pipeline to switch between encoder and decoder (i.e. the
            first rank of the decoder). This allows the user to set
            the pipeline parallel size of the encoder and decoder
            independently. For example, if
            pipeline_model_parallel_size is 8 and
            pipeline_model_parallel_split_rank is 3, then ranks 0-2
            will be the encoder and ranks 3-7 will be the decoder.

        use_sharp (bool, default = False):
            Set the use of SHARP for the collective communications of
            data-parallel process groups. When `True`, run barrier
            within each data-parallel process group, which specifies
            the SHARP application target groups.

        context_parallel_size (int, default = 1):
            The number of tensor parallel GPU groups to split the
            network input sequence length across. Compute of attention
            module requires tokens of full sequence length, so GPUs
            in a context parallel group need to communicate with each
            other to exchange information of other sequence chunks.
            Each GPU and its counterparts in other tensor parallel
            groups compose a context parallel group.

            For example, assume we have 8 GPUs, if tensor model parallel
            size is 4 and context parallel size is 2, the network input
            will be split into two sequence chunks, which are processed
            by 2 different groups of 4 GPUs. One chunk is processed by
            GPU0-3, the other chunk is processed by GPU4-7. Four groups
            are build to do context parallel communications: [GPU0, GPU4],
            [GPU1, GPU5], [GPU2, GPU6], and [GPU3, GPU7].

            Context parallelism partitions sequence length, so it has no
            impact on weights, which means weights are duplicated among
            GPUs in a context parallel group. Hence, weight gradients
            all-reduce is required in backward. For simplicity, we piggyback
            GPUs of context parallelism on data parallel group for
            weight gradient all-reduce.

    Let's say we have a total of 16 GPUs denoted by g0 ... g15 and we
    use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
    the model pipeline. The present function will
    create 8 tensor model-parallel groups, 4 pipeline model-parallel groups
    and 8 data-parallel groups as:
        8 data_parallel groups:
            [g0, g2], [g1, g3], [g4, g6], [g5, g7], [g8, g10], [g9, g11], [g12, g14], [g13, g15]
        8 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7], [g8, g9], [g10, g11], [g12, g13], [g14, g15]
        4 pipeline model-parallel groups:
            [g0, g4, g8, g12], [g1, g5, g9, g13], [g2, g6, g10, g14], [g3, g7, g11, g15]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.

    """
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()
    args = get_args()
    # generate this config through parallelism degree?
    parallel_groups = args.parallel_config["parallel_groups"]
    pipe_deps = [args.parallel_config["pipe_deps"][pipe_id] for pipe_id in sorted(args.parallel_config["pipe_deps"].keys())]
    print("call parallel state init", flush=True)
    # graph analyze
    pipe_graph = []
    pipe_depth = []
    pipe_stage_device = []
    tp_groups = [tp_group for tp_group in parallel_groups["tp"]]
    for deps in pipe_deps:
        devices = sorted(list(set({device for tup in deps for device in tup})))
        g = nx.DiGraph()
        g.add_nodes_from(devices, tp_group=[])
        g.add_edges_from(deps)

        # set tp group
        for tp_group in tp_groups:
            if tp_group and g.has_node(tp_group[0]):
                g.nodes[tp_group[0]]["tp_group"] = tp_group

        depth = {node: 0 for node in devices}
        for node in nx.topological_sort(g):
            # if node < 0:
            #     break
            for succ in g.successors(node):
                depth[succ] = max(depth[succ], depth[node] + 1)
        # virtual node and edge for input data
        g.add_node(-1)
        for node, dep in depth.items():
            if dep == 0:
                g.add_edge(-1, node)
        pipe_graph.append(g)
        pipe_depth.append(depth)
    args.pipe_depth = pipe_depth
    args.rep_ranks = [ranks[0] for ranks in tp_groups] # fisrt rank of each tp group
    print(f"args.pipe_depth={args.pipe_depth}\n\n", flush=True)
    args.pipe_graph = pipe_graph
    for graph in args.pipe_graph:
        set_micro_batch_dp_dispatcher(graph, args.micro_batch_size)
    init_group_from_config = True
    
    global _DATA_PARALLEL_GROUP
    global _DATA_PARALLEL_GROUP_GLOO
    global _DATA_PARALLEL_GLOBAL_RANKS
    global _DATA_PARALLEL_GROUP_WITH_CP
    global _DATA_PARALLEL_GROUP_WITH_CP_GLOO
    global _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    global _CONTEXT_PARALLEL_GROUP
    global _CONTEXT_PARALLEL_GLOBAL_RANKS
    global _TENSOR_MODEL_PARALLEL_GROUP
    global _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS
    global _MODEL_PARALLEL_GROUP
    global _PIPELINE_MODEL_PARALLEL_GROUP
    global _PIPELINE_MODEL_PARALLEL_REP_GROUP
    global _PIPELINE_GLOBAL_RANKS
    global _EMBEDDING_GROUP
    global _EMBEDDING_GLOBAL_RANKS
    global _POSITION_EMBEDDING_GROUP
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    global _TENSOR_AND_DATA_PARALLEL_GROUP
    global _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
    global _TENSOR_AND_EXPERT_PARALLEL_GROUP
    global _DATA_MODULO_EXPERT_PARALLEL_GROUP
    if init_group_from_config:
        global _PIPELINE_GROUP_ID
        if virtual_pipeline_model_parallel_size is not None:
            raise RuntimeError("virtual pipeline model paralllel is to be supported.")
        num_tensor_model_parallel_groups = len(parallel_groups["tp"])
        num_pipeline_model_parallel_groups = len(parallel_groups["pp"])
        
        if pipeline_model_parallel_split_rank is not None:
            _PIPELINE_MODEL_PARALLEL_SPLIT_RANK = pipeline_model_parallel_split_rank

        rank = torch.distributed.get_rank()

        # Build the data-parallel groups.
        assert _DATA_PARALLEL_GROUP is None, 'data parallel group is already initialized'
        all_data_parallel_group_ranks_with_cp = []
        
        for ranks in parallel_groups["dp"]:
            group = torch.distributed.new_group(ranks)
            group_gloo = torch.distributed.new_group(ranks, backend="gloo")
            if rank in ranks:
                _DATA_PARALLEL_GROUP = group
                _DATA_PARALLEL_GROUP_GLOO = group_gloo
                _DATA_PARALLEL_GLOBAL_RANKS = ranks
                _DATA_PARALLEL_GROUP_WITH_CP = group
                _DATA_PARALLEL_GROUP_WITH_CP_GLOO = group_gloo
                _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = ranks
        
        assert _CONTEXT_PARALLEL_GROUP is None, 'context parallel group is already initialized'
        for ranks in parallel_groups["cp"]:
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                _CONTEXT_PARALLEL_GROUP = group
                _CONTEXT_PARALLEL_GLOBAL_RANKS = ranks
        
        # Build the tensor model-parallel groups.
        assert (
            _TENSOR_MODEL_PARALLEL_GROUP is None
        ), 'tensor model parallel group is already initialized'
        for ranks in parallel_groups["tp"]:
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                _TENSOR_MODEL_PARALLEL_GROUP = group
                _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = ranks
                args.tensor_model_parallel_size = len(ranks)
        
        assert _TENSOR_MODEL_PARALLEL_GROUP is not None, 'tensor model parallel group is not initialized'
        # Build the model-parallel groups. TODO: corner case
        _MODEL_PARALLEL_GROUP = _TENSOR_MODEL_PARALLEL_GROUP
        
        # Build the pipeline model-parallel groups and embedding groups
        # (first and last rank in each pipeline model-parallel group).
        assert (
            _PIPELINE_MODEL_PARALLEL_GROUP is None
        ), 'pipeline model parallel group is already initialized'
        
        assert _EMBEDDING_GROUP is None, 'embedding group is already initialized'
        assert _POSITION_EMBEDDING_GROUP is None, 'position embedding group is already initialized'
        for ranks in parallel_groups["pp"]:
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                assert _PIPELINE_MODEL_PARALLEL_GROUP is None, 'pipeline model parallel group is initialized twice'
                _PIPELINE_MODEL_PARALLEL_GROUP = group
                _PIPELINE_GLOBAL_RANKS = ranks
                rep_rank = _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS[0]
                for index, pipe in enumerate(args.pipe_depth):
                    if rep_rank in pipe.keys():
                        _PIPELINE_GROUP_ID = index
                        set_pred_and_succ_nodes(args.pipe_graph[index], rep_rank, rank)
                        break
                assert _PIPELINE_GROUP_ID is not None, 'pipeline group id not found'
            # Setup embedding group (to exchange gradients between
            # first and last stages).
            # TODO set right embedding/position_embedding group
            if len(ranks) > 1:
                embedding_ranks = [ranks[0], ranks[-1]]
                position_embedding_ranks = [ranks[0]]
                if pipeline_model_parallel_split_rank is not None:
                    if ranks[pipeline_model_parallel_split_rank] not in embedding_ranks:
                        embedding_ranks = [
                            ranks[0],
                            ranks[pipeline_model_parallel_split_rank],
                            ranks[-1],
                        ]
                    if ranks[pipeline_model_parallel_split_rank] not in position_embedding_ranks:
                        position_embedding_ranks = [ranks[0], ranks[pipeline_model_parallel_split_rank]]
            else:
                embedding_ranks = ranks
                position_embedding_ranks = ranks

            group = torch.distributed.new_group(embedding_ranks)
            if rank in embedding_ranks:
                _EMBEDDING_GROUP = group
            if rank in ranks:
                _EMBEDDING_GLOBAL_RANKS = embedding_ranks

            group = torch.distributed.new_group(position_embedding_ranks)
            if rank in position_embedding_ranks:
                _POSITION_EMBEDDING_GROUP = group
            if rank in ranks:
                _POSITION_EMBEDDING_GLOBAL_RANKS = position_embedding_ranks

        pipeline_model_parallel_rep_group = torch.distributed.new_group(args.rep_ranks)
        if rank in args.rep_ranks:
            _PIPELINE_MODEL_PARALLEL_REP_GROUP = pipeline_model_parallel_rep_group

        cur_pipe_graph = args.pipe_graph[_PIPELINE_GROUP_ID]
        cur_pipe_depth = args.pipe_depth[_PIPELINE_GROUP_ID]
        rep_rank = _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS[0]
        args.real_micro_batch_size = cur_pipe_graph.nodes[rep_rank]["micro_batch_size"]

        global _PIPELINE_FIRST_STAGE_RANKS
        global _PIPELINE_LAST_STAGE_RANKS
        global _PIPELINE_FIRST_STAGE_MICRO_BATCH_SIZES
        global _PIPELINE_LAST_STAGE_MICRO_BATCH_SIZES

        _PIPELINE_FIRST_STAGE_RANKS = sorted([node for node in cur_pipe_graph.successors(-1)])
        _PIPELINE_FIRST_STAGE_MICRO_BATCH_SIZES = \
            [cur_pipe_graph.nodes[node]["micro_batch_size"] for node in _PIPELINE_FIRST_STAGE_RANKS]
        
        max_depth = max(cur_pipe_depth.values())
        _PIPELINE_LAST_STAGE_RANKS = sorted([node for node, depth in cur_pipe_depth.items() if depth == max_depth])
        _PIPELINE_LAST_STAGE_MICRO_BATCH_SIZES = \
            [cur_pipe_graph.nodes[node]["micro_batch_size"] for node in _PIPELINE_LAST_STAGE_RANKS]
        
        print(f"rank:{rank} | last_list {_PIPELINE_LAST_STAGE_RANKS} cur_pipe_depth {cur_pipe_depth}")

        # if isinstance(_PIPELINE_MODEL_PARALLEL_GROUP, list) and len(_PIPELINE_MODEL_PARALLEL_GROUP) == 1:
        #     _PIPELINE_MODEL_PARALLEL_GROUP = _PIPELINE_MODEL_PARALLEL_GROUP[0]
        #     _PIPELINE_GLOBAL_RANKS = _PIPELINE_GLOBAL_RANKS[0]
        
        # Build the tensor + data parallel groups. TODO: fix this
        assert (
            _TENSOR_AND_DATA_PARALLEL_GROUP is None
        ), 'Tensor + data parallel group is already initialized'
        _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = _TENSOR_MODEL_PARALLEL_GROUP
        _TENSOR_AND_DATA_PARALLEL_GROUP = _TENSOR_MODEL_PARALLEL_GROUP

        # Build the tensor + expert parallel groups, TODO: fix this
        assert (
            _TENSOR_AND_EXPERT_PARALLEL_GROUP is None
        ), 'Tensor + expert parallel group is already initialized'
        assert (
            _DATA_MODULO_EXPERT_PARALLEL_GROUP is None
        ), 'Data modulo expert group is already initialized'
        # tensor_and_data_group_size: int = tensor_model_parallel_size * data_parallel_size
        # num_tensor_and_data_groups: int = world_size // tensor_and_data_group_size
        # tensor_and_expert_group_size: int = tensor_model_parallel_size * expert_model_parallel_size
        # num_expert_groups: int = data_parallel_size // expert_model_parallel_size
        # for i in range(num_tensor_and_data_groups):
        #     for j in range(num_expert_groups):
        #         start_rank = i * tensor_and_data_group_size + j * tensor_and_expert_group_size
        #         end_rank = i * tensor_and_data_group_size + (j + 1) * tensor_and_expert_group_size
        #         ranks = range(start_rank, end_rank)
        #         group = torch.distributed.new_group(ranks)
        #         if rank in ranks:
        #             _TENSOR_AND_EXPERT_PARALLEL_GROUP = group

        # for i in range(num_tensor_and_data_groups):
        #     start_rank = i * tensor_and_data_group_size
        #     end_rank = (i + 1) * tensor_and_data_group_size
        #     for j in range(tensor_and_expert_group_size):
        #         ranks = range(start_rank + j, end_rank, tensor_and_expert_group_size)
        #         group = torch.distributed.new_group(ranks)
        #         if rank in ranks:
        #             _DATA_MODULO_EXPERT_PARALLEL_GROUP = group
        _TENSOR_AND_EXPERT_PARALLEL_GROUP = _TENSOR_MODEL_PARALLEL_GROUP
        _DATA_MODULO_EXPERT_PARALLEL_GROUP = _DATA_PARALLEL_GROUP
        
        # Initialize global memory buffer
        # This isn't really "parallel state" but there isn't another good place to
        # put this. If we end up with a more generic initialization of megatron-core
        # we could stick it there
        _set_global_memory_buffer()
        
    else:
        if (
            world_size
            % (tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size)
            != 0
        ):
            raise RuntimeError(
                f"world_size ({world_size}) is not divisible by tensor_model_parallel_size "
                f"({tensor_model_parallel_size}) x pipeline_model_parallel_size ({pipeline_model_parallel_size}) "
                f"x context_parallel_size ({context_parallel_size})"
            )

        data_parallel_size: int = world_size // (
            tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
        )

        if data_parallel_size % expert_model_parallel_size != 0:
            raise RuntimeError(
                f"data_parallel_size ({data_parallel_size}) is not divisible by expert_model_parallel_size "
            )

        if expert_model_parallel_size > 1 and context_parallel_size > 1:
            raise RuntimeError(
                f"combination of expert model prallellism and context parallelism is not supported"
            )

        num_tensor_model_parallel_groups: int = world_size // tensor_model_parallel_size
        num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size

        if virtual_pipeline_model_parallel_size is not None:
            if not pipeline_model_parallel_size > 2:
                raise RuntimeError(
                    "pipeline-model-parallel size should be greater than 2 with interleaved schedule"
                )
            global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
            global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
            _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = 0
            _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = virtual_pipeline_model_parallel_size

        if pipeline_model_parallel_split_rank is not None:
            # global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
            _PIPELINE_MODEL_PARALLEL_SPLIT_RANK = pipeline_model_parallel_split_rank

        rank = torch.distributed.get_rank()

        # Build the data-parallel groups.
        # global _DATA_PARALLEL_GROUP
        # global _DATA_PARALLEL_GROUP_GLOO
        # global _DATA_PARALLEL_GLOBAL_RANKS
        # global _DATA_PARALLEL_GROUP_WITH_CP
        # global _DATA_PARALLEL_GROUP_WITH_CP_GLOO
        # global _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP
        assert _DATA_PARALLEL_GROUP is None, 'data parallel group is already initialized'
        all_data_parallel_group_ranks_with_cp = []
        
        for i in range(pipeline_model_parallel_size):
            start_rank = i * num_pipeline_model_parallel_groups
            end_rank = (i + 1) * num_pipeline_model_parallel_groups
            for j in range(context_parallel_size * tensor_model_parallel_size):
                ranks = range(
                    start_rank + j, end_rank, context_parallel_size * tensor_model_parallel_size
                )
                group = torch.distributed.new_group(ranks)
                group_gloo = torch.distributed.new_group(ranks, backend="gloo")
                if rank in ranks:
                    _DATA_PARALLEL_GROUP = group
                    _DATA_PARALLEL_GROUP_GLOO = group_gloo
                    _DATA_PARALLEL_GLOBAL_RANKS = ranks
            for j in range(tensor_model_parallel_size):
                ranks_with_cp = range(start_rank + j, end_rank, tensor_model_parallel_size)
                all_data_parallel_group_ranks_with_cp.append(list(ranks_with_cp))
                group_with_cp = torch.distributed.new_group(ranks_with_cp)
                group_with_cp_gloo = torch.distributed.new_group(ranks_with_cp, backend="gloo")
                if rank in ranks_with_cp:
                    _DATA_PARALLEL_GROUP_WITH_CP = group_with_cp
                    _DATA_PARALLEL_GROUP_WITH_CP_GLOO = group_with_cp_gloo
                    _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = ranks_with_cp

        # Apply SHARP to DP process groups
        if use_sharp:
            if rank == 0:
                print(
                    "The number of process groups to use SHARP with depends on the type "
                    "of the network switch. Nvidia QM1 switch supports SAHRP up to 8 "
                    "process groups and QM2 supports up to 256 process groups. We apply "
                    "SHARP to the communications of the data-parallel domain. If the "
                    "number of data-parallel process groups is larger than the max "
                    "process groups that the network switch supports, the communication "
                    "will fall back to non-SHARP operators. To enable SHARP, "
                    "`#SBATCH_NETWORK=sharp` should be set in the sbatch script."
                )
            torch.distributed.barrier(
                group=get_data_parallel_group(with_context_parallel=context_parallel_size > 1),
                device_ids=[torch.cuda.current_device()],
            )
            # Set `NCCL_SHARP_DISABLE=1` to restrict SHARP application to DP process groups
            os.environ["NCCL_SHARP_DISABLE"] = "1"

        # Build the context-parallel groups.
        # global _CONTEXT_PARALLEL_GROUP
        # global _CONTEXT_PARALLEL_GLOBAL_RANKS
        assert _CONTEXT_PARALLEL_GROUP is None, 'context parallel group is already initialized'
        for i in range(pipeline_model_parallel_size):
            for j in range(data_parallel_size):
                start_rank = (
                    i * num_pipeline_model_parallel_groups
                    + j * tensor_model_parallel_size * context_parallel_size
                )
                end_rank = (
                    i * num_pipeline_model_parallel_groups
                    + (j + 1) * tensor_model_parallel_size * context_parallel_size
                )
                for k in range(tensor_model_parallel_size):
                    ranks = range(start_rank + k, end_rank, tensor_model_parallel_size)
                    group = torch.distributed.new_group(ranks)
                    if rank in ranks:
                        _CONTEXT_PARALLEL_GROUP = group
                        _CONTEXT_PARALLEL_GLOBAL_RANKS = ranks

        # Build the model-parallel groups.
        # global _MODEL_PARALLEL_GROUP
        assert _MODEL_PARALLEL_GROUP is None, 'model parallel group is already initialized'
        # if args.hetero_cluster:
        #     pass
        # else:
        for i in range(data_parallel_size * context_parallel_size):
            ranks = [
                data_parallel_group_ranks_with_cp[i]
                for data_parallel_group_ranks_with_cp in all_data_parallel_group_ranks_with_cp
            ]
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                _MODEL_PARALLEL_GROUP = group

        # Build the tensor model-parallel groups.
        # global _TENSOR_MODEL_PARALLEL_GROUP
        assert (
            _TENSOR_MODEL_PARALLEL_GROUP is None
        ), 'tensor model parallel group is already initialized'
        for i in range(num_tensor_model_parallel_groups):
            ranks = range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size)
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                _TENSOR_MODEL_PARALLEL_GROUP = group

        # Build the pipeline model-parallel groups and embedding groups
        # (first and last rank in each pipeline model-parallel group).
        # global _PIPELINE_MODEL_PARALLEL_GROUP
        # global _PIPELINE_GLOBAL_RANKS
        assert (
            _PIPELINE_MODEL_PARALLEL_GROUP is None
        ), 'pipeline model parallel group is already initialized'
        # global _EMBEDDING_GROUP
        # global _EMBEDDING_GLOBAL_RANKS
        assert _EMBEDDING_GROUP is None, 'embedding group is already initialized'
        # global _POSITION_EMBEDDING_GROUP
        # global _POSITION_EMBEDDING_GLOBAL_RANKS
        assert _POSITION_EMBEDDING_GROUP is None, 'position embedding group is already initialized'
       
        for i in range(num_pipeline_model_parallel_groups):
            ranks = range(i, world_size, num_pipeline_model_parallel_groups)
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                _PIPELINE_MODEL_PARALLEL_GROUP = group
                _PIPELINE_GLOBAL_RANKS = ranks
            # Setup embedding group (to exchange gradients between
            # first and last stages).
            if len(ranks) > 1:
                embedding_ranks = [ranks[0], ranks[-1]]
                position_embedding_ranks = [ranks[0]]
                if pipeline_model_parallel_split_rank is not None:
                    if ranks[pipeline_model_parallel_split_rank] not in embedding_ranks:
                        embedding_ranks = [
                            ranks[0],
                            ranks[pipeline_model_parallel_split_rank],
                            ranks[-1],
                        ]
                    if ranks[pipeline_model_parallel_split_rank] not in position_embedding_ranks:
                        position_embedding_ranks = [ranks[0], ranks[pipeline_model_parallel_split_rank]]
            else:
                embedding_ranks = ranks
                position_embedding_ranks = ranks

            group = torch.distributed.new_group(embedding_ranks)
            if rank in embedding_ranks:
                _EMBEDDING_GROUP = group
            if rank in ranks:
                _EMBEDDING_GLOBAL_RANKS = embedding_ranks

            group = torch.distributed.new_group(position_embedding_ranks)
            if rank in position_embedding_ranks:
                _POSITION_EMBEDDING_GROUP = group
            if rank in ranks:
                _POSITION_EMBEDDING_GLOBAL_RANKS = position_embedding_ranks

        # Build the tensor + data parallel groups.
        # global _TENSOR_AND_DATA_PARALLEL_GROUP
        # global _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
        assert (
            _TENSOR_AND_DATA_PARALLEL_GROUP is None
        ), 'Tensor + data parallel group is already initialized'
        tensor_and_data_group_size_with_cp: int = tensor_model_parallel_size * data_parallel_size * context_parallel_size
        num_tensor_and_data_groups_with_cp: int = world_size // tensor_and_data_group_size_with_cp
        for i in range(num_tensor_and_data_groups_with_cp):
            start_rank = i * tensor_and_data_group_size_with_cp
            end_rank = start_rank + tensor_and_data_group_size_with_cp
            ranks = range(start_rank, end_rank)
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = group

            for j in range(context_parallel_size):
                ranks = []
                for k in range(data_parallel_size):
                    start_rank = (
                        i * tensor_and_data_group_size_with_cp
                        + j * tensor_model_parallel_size
                        + k * tensor_model_parallel_size * context_parallel_size
                    )
                    end_rank = start_rank + tensor_model_parallel_size
                    ranks = ranks + list(range(start_rank, end_rank))
                group = torch.distributed.new_group(ranks)
                if rank in ranks:
                    _TENSOR_AND_DATA_PARALLEL_GROUP = group

        # Build the tensor + expert parallel groups
        # global _TENSOR_AND_EXPERT_PARALLEL_GROUP
        assert (
            _TENSOR_AND_EXPERT_PARALLEL_GROUP is None
        ), 'Tensor + expert parallel group is already initialized'
        # global _DATA_MODULO_EXPERT_PARALLEL_GROUP
        assert (
            _DATA_MODULO_EXPERT_PARALLEL_GROUP is None
        ), 'Data modulo expert group is already initialized'
        tensor_and_data_group_size: int = tensor_model_parallel_size * data_parallel_size
        num_tensor_and_data_groups: int = world_size // tensor_and_data_group_size
        tensor_and_expert_group_size: int = tensor_model_parallel_size * expert_model_parallel_size
        num_expert_groups: int = data_parallel_size // expert_model_parallel_size
        for i in range(num_tensor_and_data_groups):
            for j in range(num_expert_groups):
                start_rank = i * tensor_and_data_group_size + j * tensor_and_expert_group_size
                end_rank = i * tensor_and_data_group_size + (j + 1) * tensor_and_expert_group_size
                ranks = range(start_rank, end_rank)
                group = torch.distributed.new_group(ranks)
                if rank in ranks:
                    _TENSOR_AND_EXPERT_PARALLEL_GROUP = group

        for i in range(num_tensor_and_data_groups):
            start_rank = i * tensor_and_data_group_size
            end_rank = (i + 1) * tensor_and_data_group_size
            for j in range(tensor_and_expert_group_size):
                ranks = range(start_rank + j, end_rank, tensor_and_expert_group_size)
                group = torch.distributed.new_group(ranks)
                if rank in ranks:
                    _DATA_MODULO_EXPERT_PARALLEL_GROUP = group

        # Initialize global memory buffer
        # This isn't really "parallel state" but there isn't another good place to
        # put this. If we end up with a more generic initialization of megatron-core
        # we could stick it there
        _set_global_memory_buffer()
        
        # Get heterogeneous cluster info
        self_device_type = os.environ.get('DEVICE_TYPE', None)
        global _HETERO_DEVICE_TYPES
        if args.hetero_cluster and args.stage_recompute_num_layers is not None:
            print(f'_HETERO_DEVICE_TYPES={_HETERO_DEVICE_TYPES}', flush=True)
            args.recompute_num_layers = args.stage_recompute_num_layers[get_pipeline_model_parallel_rank()]
            print(f"Recompute {args.recompute_num_layers} for stage {get_pipeline_model_parallel_rank()}")
        if self_device_type is not None:
            self_device_type = int(self_device_type)
            self_device_type = torch.tensor([self_device_type], device=torch.cuda.current_device(), dtype=torch.int32)
            all_device_types = [torch.zeros([1], device=torch.cuda.current_device(), dtype=torch.int32) for _ in range(world_size)]
            torch.distributed.all_gather(all_device_types, self_device_type)
            _HETERO_DEVICE_TYPES = list(t.tolist()[0] for t in all_device_types)
            print(f"device memory={torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory/1024/1024/1024}GB", flush=True)
        
    # print(f"localrank={rank}, pipeline global ranks={get_pipeline_model_parallel_rank()}, prev rank={get_pipeline_model_parallel_prev_rank()}, next rank={get_pipeline_model_parallel_next_rank()}, first rank={get_pipeline_model_parallel_first_rank()}, last rank={get_pipeline_model_parallel_last_rank()}, is first stage={is_pipeline_first_stage()}, is last stage={is_pipeline_last_stage()}", flush=True)


def set_micro_batch_dp_dispatcher(graph: nx.DiGraph, micro_batch_size: int):
    """Determine the flow of micro-batch data for the given graph and number of micro-batches."""
    total_mbs = {node: 0 for node in graph.nodes}
    assert -1 in total_mbs.keys(), 'Guard node -1 not exist'
    total_mbs[-1] = micro_batch_size
    for node in nx.topological_sort(graph):
        graph.nodes[node]["micro_batch_size"] = total_mbs[node]
        succ_nodes = list(graph.successors(node))
        num_successors = len(succ_nodes)
        if num_successors == 0:
            continue
        assert total_mbs[node] > num_successors, 'micro_batch_size < num_successors, invalid pipeline diagram'
        base_mbs = total_mbs[node] // num_successors
        remain_mbs = total_mbs[node] % num_successors
        for index, succ in enumerate(succ_nodes):
            edge_mbs = base_mbs + 1 if index < remain_mbs else base_mbs
            graph[node][succ]['micro_batch_size'] = edge_mbs
            total_mbs[succ] += edge_mbs
    
    # stages = sorted(list(set(graph_depth.values())))
    # stage_device_list = []
    # for stage in stages:
    #     ranks = []
    #     for node in sorted(graph_depth.keys()):
    #         if graph_depth[node] == stage:
    #             ranks.append(node)
    #     stage_device_list.append(ranks)
    # # Mode 1: uniform sharding
    # per_device_micro_batch_sizes = {}
    # for stage, ranks in enumerate(stage_device_list):
    #     if micro_batch_size % len(ranks) == 0:
    #         per_device_micro_batch_sizes[stage] = {d: micro_batch_size // len(ranks) for d in ranks}
    #     else:            
    #         rounded_bs = int(micro_batch_size // len(ranks))
    #         per_device_micro_batch_sizes[stage] = {d: rounded_bs for d in ranks[:-1]}
    #         per_device_micro_batch_sizes[stage][ranks[-1]] = micro_batch_size - (len(ranks) - 1) * rounded_bs
    
    # # TODO Mode 2: manual sharding
    # # Set micro-batch numbers to edge weights
    # for stage in per_device_micro_batch_sizes.keys():
    #     for rank, input_bs in per_device_micro_batch_sizes[stage].items():
    #         predes = [pred for pred in graph.predecessors(rank)]
    #         for i, pred in enumerate(predes):
    #             if input_bs % len(predes) == 0:
    #                 print(f"update rank={rank}, pred={pred}, weight = {input_bs // len(predes)}", flush=True)
    #                 graph[pred][rank]['weight'] = input_bs // len(predes)
    #             else:
    #                 rounded_bs = int(input_bs // len(predes))
    #                 if i != len(predes) - 1:
    #                     graph[pred][rank]['weight'] = rounded_bs
    #                 else:
    #                     graph[pred][rank]['weight'] = input_bs - (len(predes) - 1) * rounded_bs

def set_pred_and_succ_nodes(graph: nx.DiGraph, rep_node: int, cur_node: int):
    """
        Determine the predecessor and successor nodes for communication 
        based on the given graph, current node, and representative node.
    """
    global _SEND_SUCC_NODES
    global _RECV_PRED_NODES
    global _SEND_PRED_NODES
    global _RECV_SUCC_NODES
    cur_tp_group_ranks = _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS
    cur_tp_group_size = len(cur_tp_group_ranks)
    assert cur_node in cur_tp_group_ranks, f'current node {cur_node} not in TP group'
    cur_tp_rank = cur_tp_group_ranks.index(cur_node)
    assert cur_tp_group_size > 0, 'current TP group is empty'
    recv_pred_nodes = []
    send_pred_nodes = []
    for node in graph.predecessors(rep_node):
        if node == -1:
            continue
        pred_tp_group_ranks = graph.nodes[node]["tp_group"]
        pred_tp_group_size = len(pred_tp_group_ranks)
        assert pred_tp_group_size > 0, 'predecessor TP group is empty'
        micro_batch_size = graph[node][rep_node]['micro_batch_size']
        # forward (recv pred)
        if pred_tp_group_size < cur_tp_group_size:
            base = cur_tp_group_size // pred_tp_group_size
            remainder = cur_tp_group_size % pred_tp_group_size
            if cur_tp_rank < remainder * (base + 1):
                pred_tp_rank = cur_tp_rank // (base + 1)
            else:
                pred_tp_rank = remainder + (cur_tp_rank - remainder * (base + 1)) // base
        else:
            pred_tp_rank = cur_tp_rank
        recv_pred_nodes.append((micro_batch_size, pred_tp_group_ranks[pred_tp_rank]))
        # backward (send pred)
        if cur_tp_group_size < pred_tp_group_size:
            base = pred_tp_group_size // cur_tp_group_size
            remainder = pred_tp_group_size % cur_tp_group_size
            if cur_tp_rank < remainder:
                begin_index = cur_tp_rank * (base + 1)
                end_index = begin_index + (base + 1)
            else:
                begin_index = cur_tp_rank * base + remainder
                end_index = begin_index + base
            send_pred_nodes.append((micro_batch_size, [pred_tp_group_ranks[index] for index in range(begin_index, end_index)]))
        else:
            if cur_tp_rank < pred_tp_group_size:
                send_pred_nodes.append((micro_batch_size, [pred_tp_group_ranks[cur_tp_rank]]))
    send_succ_nodes = []
    recv_succ_nodes = []
    for node in graph.successors(rep_node):
        succ_tp_group_ranks = graph.nodes[node]['tp_group']
        succ_tp_group_size = len(succ_tp_group_ranks)
        assert succ_tp_group_size > 0, 'successor TP group is empty'
        micro_batch_size = graph[rep_node][node]['micro_batch_size']
        # forward (send succ)
        if cur_tp_group_size < succ_tp_group_size:
            base = succ_tp_group_size // cur_tp_group_size
            remainder = succ_tp_group_size % cur_tp_group_size
            if cur_tp_rank < remainder:
                begin_index = cur_tp_rank * (base + 1)
                end_index = begin_index + (base + 1)
            else:
                begin_index = cur_tp_rank * base + remainder
                end_index = begin_index + base
            send_succ_nodes.append((micro_batch_size, [succ_tp_group_ranks[index] for index in range(begin_index, end_index)]))
        else:
            if cur_tp_rank < succ_tp_group_size:
                send_succ_nodes.append((micro_batch_size, [succ_tp_group_ranks[cur_tp_rank]]))
        # backward (recv succ)
        if succ_tp_group_size < cur_tp_group_size:
            base = cur_tp_group_size // succ_tp_group_size
            remainder = cur_tp_group_size % succ_tp_group_size
            if cur_tp_rank < remainder * (base + 1):
                succ_tp_rank = cur_tp_rank // (base + 1)
            else:
                succ_tp_rank = remainder + (cur_tp_rank - remainder * (base + 1)) // base
        else:
            succ_tp_rank = cur_tp_rank
        recv_succ_nodes.append((micro_batch_size, succ_tp_group_ranks[succ_tp_rank]))

    _SEND_SUCC_NODES = send_succ_nodes
    _RECV_PRED_NODES = recv_pred_nodes
    _SEND_PRED_NODES = send_pred_nodes
    _RECV_SUCC_NODES = recv_succ_nodes
    print(f"rank:{cur_node} | r_p{recv_pred_nodes}, s_s{send_succ_nodes}, r_s{recv_succ_nodes}, s_p{send_pred_nodes}", flush=True)


def is_unitialized():
    """Useful for code segments that may be accessed with or without mpu initialization"""
    return _DATA_PARALLEL_GROUP is None


def model_parallel_is_initialized():
    """Check if model and data parallel groups are initialized."""
    if (
        _TENSOR_MODEL_PARALLEL_GROUP is None
        or _PIPELINE_MODEL_PARALLEL_GROUP is None
        or _DATA_PARALLEL_GROUP is None
    ):
        return False
    return True


def get_model_parallel_group():
    """Get the model parallel group the caller rank belongs to."""
    assert _MODEL_PARALLEL_GROUP is not None, 'model parallel group is not initialized'
    return _MODEL_PARALLEL_GROUP

def get_send_successor_ranks():
    """Get a list of (micro_batch_size, [rank0, rank1, ...]) for successor nodes needs to be sent"""
    assert _SEND_SUCC_NODES is not None, 'send successor nodes is not initialized'
    return _SEND_SUCC_NODES

def get_send_predecessor_ranks():
    """Get a list of (micro_batch_size, [rank0, rank1, ...]) for predecessor nodes needs to be sent"""
    assert _SEND_PRED_NODES is not None, 'send predecessor nodes is not initialized'
    return _SEND_PRED_NODES

def get_recv_successor_ranks():
    """Get a list of (micro_batch_size, rank) for successor nodes needs to be received"""
    assert _RECV_SUCC_NODES is not None, 'recv successor nodes is not initialized'
    return _RECV_SUCC_NODES
    
def get_recv_predecessor_ranks():
    """Get a list of (micro_batch_size, rank) for predecessor nodes needs to be received"""
    assert _RECV_PRED_NODES is not None, 'recv predecessor nodes is not initialized'
    return _RECV_PRED_NODES

def get_tensor_model_parallel_group(check_initialized=True):
    """Get the tensor model parallel group the caller rank belongs to."""
    if check_initialized:
        assert (
            _TENSOR_MODEL_PARALLEL_GROUP is not None
        ), 'tensor model parallel group is not initialized'
    return _TENSOR_MODEL_PARALLEL_GROUP


def get_pipeline_model_parallel_group():
    """Get the pipeline model parallel group the caller rank belongs to."""
    assert (
        _PIPELINE_MODEL_PARALLEL_GROUP is not None
    ), 'pipeline_model parallel group is not initialized'
    return _PIPELINE_MODEL_PARALLEL_GROUP

def get_pipeline_model_parallel_rep_group():
    return _PIPELINE_MODEL_PARALLEL_REP_GROUP
    
def get_pipeline_model_parallel_group_id():
    """Get the pipeline model parallel group the caller rank belongs to."""
    assert (
        _PIPELINE_GROUP_ID is not None
    ), 'pipeline_model parallel group is not initialized'
    return _PIPELINE_GROUP_ID

def get_data_parallel_group(with_context_parallel=False):
    """Get the data parallel group the caller rank belongs to."""
    if with_context_parallel:
        assert (
            _DATA_PARALLEL_GROUP_WITH_CP is not None
        ), 'data parallel group with context parallel combined is not initialized'
        return _DATA_PARALLEL_GROUP_WITH_CP
    else:
        assert _DATA_PARALLEL_GROUP is not None, 'data parallel group is not initialized'
        return _DATA_PARALLEL_GROUP


def get_data_parallel_group_gloo(with_context_parallel=False):
    """Get the data parallel group-gloo the caller rank belongs to."""
    if with_context_parallel:
        assert (
            _DATA_PARALLEL_GROUP_WITH_CP_GLOO is not None
        ), 'data parallel group-gloo with context parallel combined is not initialized'
        return _DATA_PARALLEL_GROUP_WITH_CP_GLOO
    else:
        assert _DATA_PARALLEL_GROUP_GLOO is not None, 'data parallel group-gloo is not initialized'
        return _DATA_PARALLEL_GROUP_GLOO


def get_context_parallel_group(check_initialized=True):
    """Get the context parallel group the caller rank belongs to."""
    if check_initialized:
        assert _CONTEXT_PARALLEL_GROUP is not None, 'context parallel group is not initialized'
    return _CONTEXT_PARALLEL_GROUP


def get_context_parallel_global_ranks(check_initialized=True):
    """Get all global ranks of the context parallel group that the caller rank belongs to."""
    if check_initialized:
        assert (
            _CONTEXT_PARALLEL_GLOBAL_RANKS is not None
        ), 'context parallel group is not initialized'
    return _CONTEXT_PARALLEL_GLOBAL_RANKS


def get_embedding_group():
    """Get the embedding group the caller rank belongs to."""
    assert _EMBEDDING_GROUP is not None, 'embedding group is not initialized'
    return _EMBEDDING_GROUP


def get_position_embedding_group():
    """Get the position embedding group the caller rank belongs to."""
    assert _POSITION_EMBEDDING_GROUP is not None, 'position embedding group is not initialized'
    return _POSITION_EMBEDDING_GROUP


def get_amax_reduction_group(with_context_parallel=False):
    """Get the FP8 amax reduction group the caller rank belongs to."""
    if with_context_parallel:
        assert (
            _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP is not None
        ), 'FP8 amax reduction group is not initialized'
        return _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
    else:
        assert (
            _TENSOR_AND_DATA_PARALLEL_GROUP is not None
        ), 'FP8 amax reduction group is not initialized'
        return _TENSOR_AND_DATA_PARALLEL_GROUP


def get_tensor_and_data_parallel_group(with_context_parallel=False):
    """Get the tensor and data parallel group the caller rank belongs to."""
    if with_context_parallel:
        assert (
            _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP is not None
        ), 'tensor and data parallel group is not initialized'
        return _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
    else:
        assert (
            _TENSOR_AND_DATA_PARALLEL_GROUP is not None
        ), 'tensor and data parallel group is not initialized'
        return _TENSOR_AND_DATA_PARALLEL_GROUP


def get_tensor_and_expert_parallel_group():
    assert (
        _TENSOR_AND_EXPERT_PARALLEL_GROUP is not None
    ), 'tensor and expert parallel group is not initialized'
    return _TENSOR_AND_EXPERT_PARALLEL_GROUP


def get_data_modulo_expert_parallel_group():
    assert (
        _DATA_MODULO_EXPERT_PARALLEL_GROUP is not None
    ), 'data modulo expert parallel group is not initialized'
    return _DATA_MODULO_EXPERT_PARALLEL_GROUP


def set_tensor_model_parallel_world_size(world_size):
    """Set the tensor model parallel size"""
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = world_size


def set_pipeline_model_parallel_world_size(world_size):
    """Set the pipeline model parallel size"""
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = world_size


def set_virtual_pipeline_model_parallel_world_size(world_size):
    """Set the pipeline model parallel size"""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = world_size


def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_tensor_model_parallel_group())


def get_pipeline_model_parallel_world_size():
    """Return world size for the pipeline model parallel group."""
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    if _PIPELINE_GROUP_ID is not None:
        if _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE is None:
            args = get_args()
            _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = max(args.pipe_depth[_PIPELINE_GROUP_ID].values()) + 1
        return _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    else:
        if _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE is not None:
            return _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
        return torch.distributed.get_world_size(group=get_pipeline_model_parallel_group())


def set_tensor_model_parallel_rank(rank):
    """Set tensor model parallel rank."""
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    _MPU_TENSOR_MODEL_PARALLEL_RANK = rank


def set_pipeline_model_parallel_rank(rank):
    """Set pipeline model parallel rank."""
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    _MPU_PIPELINE_MODEL_PARALLEL_RANK = rank


def set_pipeline_model_parallel_split_rank(rank):
    """Set pipeline model parallel split rank."""
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    _PIPELINE_MODEL_PARALLEL_SPLIT_RANK = rank


def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    if _MPU_TENSOR_MODEL_PARALLEL_RANK is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_tensor_model_parallel_group())


def get_pipeline_model_parallel_rank():
    """Return my rank for the pipeline model parallel group."""
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    if _PIPELINE_GROUP_ID is not None:
        if _MPU_PIPELINE_MODEL_PARALLEL_RANK is not None:
            return _MPU_PIPELINE_MODEL_PARALLEL_RANK
        args = get_args()
        rep_rank = _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS[0]
        assert rep_rank in args.pipe_depth[_PIPELINE_GROUP_ID], f'{rep_rank} not in pipe {args.pipe_depth[_PIPELINE_GROUP_ID]}'
        return args.pipe_depth[_PIPELINE_GROUP_ID][rep_rank]
    else:
        if _MPU_PIPELINE_MODEL_PARALLEL_RANK is not None:
            return _MPU_PIPELINE_MODEL_PARALLEL_RANK
        return torch.distributed.get_rank(group=get_pipeline_model_parallel_group())


def get_pipeline_model_parallel_split_rank():
    """Return pipeline model parallel split rank."""
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    return _PIPELINE_MODEL_PARALLEL_SPLIT_RANK


def is_pipeline_first_stage(ignore_virtual=False):
    """Return True if in the first pipeline model-parallel stage, False otherwise."""
    if not ignore_virtual:
        if (
            get_virtual_pipeline_model_parallel_world_size() is not None
            and get_virtual_pipeline_model_parallel_rank() != 0
        ):
            return False
    return get_pipeline_model_parallel_rank() == 0


def is_pipeline_last_stage(ignore_virtual=False):
    """Return True if in the last pipeline model-parallel stage, False otherwise."""
    if not ignore_virtual:
        virtual_pipeline_model_parallel_world_size = (
            get_virtual_pipeline_model_parallel_world_size()
        )
        if virtual_pipeline_model_parallel_world_size is not None and get_virtual_pipeline_model_parallel_rank() != (
            virtual_pipeline_model_parallel_world_size - 1
        ):
            return False
    rank = get_pipeline_model_parallel_rank()
    return rank == (get_pipeline_model_parallel_world_size() - 1)


def is_rank_in_embedding_group(ignore_virtual=False):
    """Return true if current rank is in embedding group, False otherwise."""
    rank = torch.distributed.get_rank()
    global _EMBEDDING_GLOBAL_RANKS
    if ignore_virtual:
        return rank in _EMBEDDING_GLOBAL_RANKS
    if rank in _EMBEDDING_GLOBAL_RANKS:
        if rank == _EMBEDDING_GLOBAL_RANKS[0]:
            return is_pipeline_first_stage(ignore_virtual=False)
        elif rank == _EMBEDDING_GLOBAL_RANKS[-1]:
            return is_pipeline_last_stage(ignore_virtual=False)
        else:
            return True
    return False


def is_rank_in_position_embedding_group():
    """Return true if current rank is in position embedding group, False otherwise."""
    rank = torch.distributed.get_rank()
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    return rank in _POSITION_EMBEDDING_GLOBAL_RANKS


def is_pipeline_stage_before_split(rank=None):
    """Return True if pipeline stage executes encoder block for a model
    with both encoder and decoder."""
    if get_pipeline_model_parallel_world_size() == 1:
        return True
    if rank is None:
        rank = get_pipeline_model_parallel_rank()
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    if _PIPELINE_MODEL_PARALLEL_SPLIT_RANK is None:
        return True
    if rank < _PIPELINE_MODEL_PARALLEL_SPLIT_RANK:
        return True
    return False


def is_pipeline_stage_after_split(rank=None):
    """Return True if pipeline stage executes decoder block for a model
    with both encoder and decoder."""
    if get_pipeline_model_parallel_world_size() == 1:
        return True
    if rank is None:
        rank = get_pipeline_model_parallel_rank()
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    if _PIPELINE_MODEL_PARALLEL_SPLIT_RANK is None:
        return True
    if rank >= _PIPELINE_MODEL_PARALLEL_SPLIT_RANK:
        return True
    return False


def is_pipeline_stage_at_split():
    """Return true if pipeline stage executes decoder block and next
    stage executes encoder block for a model with both encoder and
    decoder."""
    rank = get_pipeline_model_parallel_rank()
    return is_pipeline_stage_before_split(rank) and is_pipeline_stage_after_split(rank + 1)


def get_virtual_pipeline_model_parallel_rank():
    """Return the virtual pipeline-parallel rank."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK


def set_virtual_pipeline_model_parallel_rank(rank):
    """Set the virtual pipeline-parallel rank."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = rank


def get_virtual_pipeline_model_parallel_world_size():
    """Return the virtual pipeline-parallel world size."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE


def get_tensor_model_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the tensor model parallel group."""
    assert _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS is not None, "Tensor model parallel group is not initialized"
    return _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS[0]


def get_data_parallel_src_rank(with_context_parallel=False):
    """Calculate the global rank corresponding to the first local rank
    in the data parallel group."""
    if with_context_parallel:
        assert (
            _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP is not None
        ), "Data parallel group with context parallel combined is not initialized"
        return _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP[0]
    else:
        assert _DATA_PARALLEL_GLOBAL_RANKS is not None, "Data parallel group is not initialized"
        return _DATA_PARALLEL_GLOBAL_RANKS[0]


def get_pipeline_model_parallel_first_rank():
    """Return the global rank of the first process in the pipeline for the
    current tensor parallel group"""    
    # For all devices in the given group, they have the same first or last rank(s).
    if _PIPELINE_GROUP_ID is not None:
        args = get_args()
        ranks = [k for k, v in args.pipe_depth[_PIPELINE_GROUP_ID] if v == 0]
        return ranks[0] if len(ranks) == 1 else ranks
    else:
        assert _PIPELINE_GLOBAL_RANKS is not None, "Pipeline parallel group is not initialized"
        return _PIPELINE_GLOBAL_RANKS[0]


def get_pipeline_model_parallel_last_rank():
    """Return the global rank of the last process in the pipeline for the
    current tensor parallel group"""
    if _PIPELINE_GROUP_ID is not None:
        max_depth = get_pipeline_model_parallel_world_size() - 1
        args = get_args()
        ranks = [rank for rank, depth in args.pipe_depth[_PIPELINE_GROUP_ID] if depth == max_depth]
        return ranks[0] if len(ranks) == 1 else ranks
    else:
        assert _PIPELINE_GLOBAL_RANKS is not None, "Pipeline parallel group is not initialized"
        last_rank_local = get_pipeline_model_parallel_world_size() - 1
        return _PIPELINE_GLOBAL_RANKS[last_rank_local]


def get_pipeline_model_parallel_next_rank():
    """Return the global rank that follows the caller in the pipeline"""
    assert _PIPELINE_GLOBAL_RANKS is not None, "Pipeline parallel group is not initialized"
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline + 1) % world_size]


def get_pipeline_model_parallel_prev_rank():
    """Return the global rank that preceeds the caller in the pipeline"""
    assert _PIPELINE_GLOBAL_RANKS is not None, "Pipeline parallel group is not initialized"
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline - 1) % world_size]


def get_data_parallel_world_size(with_context_parallel=False):
    """Return world size for the data parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size(
            group=get_data_parallel_group(with_context_parallel=with_context_parallel)
        )
    else:
        return 0


def get_data_parallel_rank(with_context_parallel=False):
    """Return my rank for the data parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(
            group=get_data_parallel_group(with_context_parallel=with_context_parallel)
        )
    else:
        return 0


def get_context_parallel_world_size():
    """Return world size for the context parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size(group=get_context_parallel_group())
    else:
        return 0


def get_context_parallel_rank():
    """Return my rank for the context parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(group=get_context_parallel_group())
    else:
        return 0


def get_expert_model_parallel_world_size():
    """Return my rank for the expert parallel group"""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        tensor_and_expert_parallel_world_size = torch.distributed.get_world_size(
            group=get_tensor_and_expert_parallel_group()
        )
        return tensor_and_expert_parallel_world_size // get_tensor_model_parallel_world_size()
    else:
        return 0


def get_expert_model_parallel_rank():
    """Return my rank for the expert parallel group"""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        tensor_and_expert_parallel_rank = torch.distributed.get_rank(
            group=get_tensor_and_expert_parallel_group()
        )
        return tensor_and_expert_parallel_rank // get_tensor_model_parallel_world_size()
    else:
        return 0


def get_data_modulo_expert_parallel_rank():
    """Return my rank for the context parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(group=get_data_modulo_expert_parallel_group())
    else:
        return 0


def _set_global_memory_buffer():
    """Initialize global buffer"""
    global _GLOBAL_MEMORY_BUFFER
    assert _GLOBAL_MEMORY_BUFFER is None, 'global memory buffer is already initialized'
    _GLOBAL_MEMORY_BUFFER = GlobalMemoryBuffer()


def get_global_memory_buffer():
    """Return the global GlobalMemoryBuffer object"""
    assert _GLOBAL_MEMORY_BUFFER is not None, 'global memory buffer is not initialized'
    return _GLOBAL_MEMORY_BUFFER


def destroy_global_memory_buffer():
    """Sets the global memory buffer to None"""
    global _GLOBAL_MEMORY_BUFFER
    _GLOBAL_MEMORY_BUFFER = None


def destroy_model_parallel():
    """Set the groups to none."""
    global _MODEL_PARALLEL_GROUP
    _MODEL_PARALLEL_GROUP = None
    global _TENSOR_MODEL_PARALLEL_GROUP
    _TENSOR_MODEL_PARALLEL_GROUP = None
    global _PIPELINE_MODEL_PARALLEL_GROUP
    _PIPELINE_MODEL_PARALLEL_GROUP = None
    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = None
    global _DATA_PARALLEL_GROUP_WITH_CP
    _DATA_PARALLEL_GROUP_WITH_CP = None
    global _CONTEXT_PARALLEL_GROUP
    _CONTEXT_PARALLEL_GROUP = None
    global _CONTEXT_PARALLEL_GLOBAL_RANKS
    _CONTEXT_PARALLEL_GLOBAL_RANKS = None
    global _EMBEDDING_GROUP
    _EMBEDDING_GROUP = None
    global _POSITION_EMBEDDING_GROUP
    _POSITION_EMBEDDING_GROUP = None
    global _TENSOR_AND_DATA_PARALLEL_GROUP
    _TENSOR_AND_DATA_PARALLEL_GROUP = None
    global _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
    _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = None
    global _TENSOR_AND_EXPERT_PARALLEL_GROUP
    _TENSOR_AND_EXPERT_PARALLEL_GROUP = None
    global _DATA_MODULO_EXPERT_PARALLEL_GROUP
    _DATA_MODULO_EXPERT_PARALLEL_GROUP = None
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    _MPU_TENSOR_MODEL_PARALLEL_RANK = None
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    _MPU_PIPELINE_MODEL_PARALLEL_RANK = None
    global _GLOBAL_MEMORY_BUFFER
    _GLOBAL_MEMORY_BUFFER = None
