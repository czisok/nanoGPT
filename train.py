"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""
# 中文说明：
# 本脚本是 nanoGPT 的训练入口，既支持单卡/CPU 调试，也支持多卡 DDP 分布式训练。
# 超参不是用 argparse 传入，而是直接写成下面的全局变量，再由 configurator.py 通过
# "exec 执行配置文件 / 命令行 --key=value" 的方式覆盖（优先级：命令行 > 配置文件 > 默认值）。

import os
# 让 CUDA 显存分配器使用可扩展段（expandable segments），减少显存碎片，缓解大词表/大 batch 下的 OOM。
# 必须在 import torch（CUDA 上下文初始化）之前设置才生效；setdefault 不覆盖外部已设置的值。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import time
import math
import pickle
from contextlib import nullcontext  # 空上下文管理器：CPU 上不需要 autocast 时用它占位

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP  # 分布式数据并行包装器
from torch.distributed import init_process_group, destroy_process_group  # DDP 进程组的初始化/销毁

from model import GPTConfig, GPT  # 模型定义在 model.py 中

# -----------------------------------------------------------------------------
# 默认配置：这套默认值是为在 OpenWebText 上训练 GPT-2 (124M) 设计的
# 实际使用时一般用 config/ 下的配置文件（如 train_shakespeare_char.py）覆盖其中大部分
# -----------------------------------------------------------------------------
# I/O 相关
out_dir = 'out'                    # checkpoint（ckpt.pt）输出目录
eval_interval = 2000               # 每隔多少个训练 iteration 做一次验证评估
log_interval = 1                   # 每隔多少个 iteration 打印一次训练日志
eval_iters = 200                   # 每次评估时在 train/val 上各取多少个 batch 求平均 loss（降低噪声）
eval_only = False                  # 若为 True，则在第一次评估后立即退出（用于只看 loss 不训练）
always_save_checkpoint = True      # 若为 True，每次评估后都存 checkpoint；否则仅在 val loss 创新低时存
init_from = 'scratch'              # 模型初始化方式：'scratch' 从零训练 / 'resume' 从 ckpt 续训 / 'gpt2*' 加载 OpenAI 预训练权重
# wandb 实验日志（默认关闭）
wandb_log = False                  # 是否启用 weights & biases 云端日志
wandb_project = 'owt'              # wandb 项目名
wandb_run_name = 'gpt2'            # wandb 本次运行名；也可用 'run' + str(time.time()) 自动生成
# 数据相关
dataset = 'openwebtext'            # 数据集名，对应 data/ 下的子目录（里面有 prepare.py 生成的 train.bin/val.bin/meta.pkl）
gradient_accumulation_steps = 5 * 8  # 梯度累积步数：用多个 micro-batch 累积梯度后再更新一次，等效放大 batch size
batch_size = 12                    # 每个 micro-batch 的样本数；若 gradient_accumulation_steps>1，这是 micro-batch 大小
block_size = 1024                  # 上下文长度（序列长度 T），即模型一次能看到的最大 token 数
# 模型结构相关（GPT 默认配置，对应 GPT-2 small）
n_layer = 12                       # Transformer Block 的层数
n_head = 12                        # 多头注意力的头数
n_embd = 768                       # 词向量/隐状态维度（每个 token 的向量长度）
dropout = 0.0                      # dropout 概率：预训练用 0 即可，微调/小数据防过拟合可设 0.1~0.2
bias = False                       # Linear 和 LayerNorm 中是否使用偏置项；GPT-2 用 True，去掉偏置（False）略快且效果相当
# AdamW 优化器相关
learning_rate = 6e-4               # 最大学习率（warmup 后的峰值）
max_iters = 600000                 # 训练的总 iteration 数
weight_decay = 1e-1                # 权重衰减系数（L2 正则），只作用于矩阵类权重，偏置/LayerNorm 不衰减（见 model.py）
beta1 = 0.9                        # AdamW 一阶矩衰减系数
beta2 = 0.95                       # AdamW 二阶矩衰减系数
grad_clip = 1.0                    # 梯度裁剪阈值：把梯度总范数限制在该值以内，防止训练初期梯度爆炸；设为 0.0 可关闭
# 学习率衰减策略（余弦退火 + 线性 warmup）
decay_lr = True                    # 是否对学习率做衰减；False 则全程用恒定 learning_rate
warmup_iters = 2000                # 学习率线性预热的步数（训练初期从小 lr 慢慢升到峰值，稳定训练）
lr_decay_iters = 600000            # 余弦衰减的总步数，按 Chinchilla 经验应约等于 max_iters
min_lr = 6e-5                      # 余弦衰减的最低学习率，按 Chinchilla 经验约为 learning_rate/10
# DDP 分布式相关
backend = 'nccl'                   # 分布式后端：GPU 间通信用 'nccl'，CPU 用 'gloo'
# 系统/运行环境相关
device = 'cuda'                    # 运行设备：'cpu'、'cuda'、'cuda:0' 等；macOS 可用 'mps'
# 混合精度数据类型：GPU 支持 bfloat16 就用它（数值稳定、无需 GradScaler），否则退化为 float16（会自动启用 GradScaler）
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'  # 可选 'float32'/'bfloat16'/'float16'
compile = True                     # 是否用 PyTorch 2.0 的 torch.compile 编译模型（算子融合，提速但首次编译较慢）
# -----------------------------------------------------------------------------
# 收集所有"值为 int/float/bool/str"的全局变量名，作为可被配置覆盖的超参清单
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())  # 执行配置器：解析命令行，exec 配置文件或用 --key=value 覆盖上面的全局变量
config = {k: globals()[k] for k in config_keys}  # 把最终生效的超参收集成字典，供 wandb 日志和 checkpoint 保存
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 分布式（DDP）判断与初始化
# -----------------------------------------------------------------------------
# 通过环境变量 RANK 判断是否处于 torchrun 启动的分布式环境（单卡运行时没有该变量）
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)       # 初始化进程组，建立多卡/多机通信
    ddp_rank = int(os.environ['RANK'])        # 全局进程编号（跨所有机器）
    ddp_local_rank = int(os.environ['LOCAL_RANK'])  # 本机上的进程编号（用于绑定对应 GPU）
    ddp_world_size = int(os.environ['WORLD_SIZE'])  # 总进程数（通常 = 总 GPU 数）
    device = f'cuda:{ddp_local_rank}'         # 每个进程绑定一张本地 GPU
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0            # 只有 rank 0 主进程负责打印日志、保存 checkpoint，避免重复
    seed_offset = ddp_rank                    # 每个进程用不同随机种子偏移，保证各卡数据不同
    # 多进程同时训练，等效 batch 会随进程数放大，因此按进程数等比例减少每进程的梯度累积步数
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # 非分布式：单卡/CPU，只有一个进程
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
# 每次参数更新实际"看到"的 token 数 = 梯度累积步数 × 进程数 × 每 micro-batch 样本数 × 序列长度
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)  # 只有主进程创建输出目录，避免多进程竞争
torch.manual_seed(1337 + seed_offset)   # 设置随机种子（DDP 下各进程加不同偏移）
torch.backends.cuda.matmul.allow_tf32 = True  # 允许矩阵乘法使用 TF32（Ampere+ GPU 上显著加速，精度损失极小）
torch.backends.cudnn.allow_tf32 = True        # 允许 cudnn 使用 TF32
device_type = 'cuda' if 'cuda' in device else 'cpu'  # 供 torch.amp.autocast 使用的设备类型
# 注意：float16 会自动配合 GradScaler 使用（见下文），bfloat16 动态范围大则不需要
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
# 自动混合精度上下文：GPU 上用 autocast 让算子自动选择低精度；CPU 上用 nullcontext 不做任何事
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
# 极简数据加载器：从 prepare.py 生成的 .bin 文件中随机取一个 batch
# -----------------------------------------------------------------------------
data_dir = os.path.join('data', dataset)  # 数据目录，如 data/shakespeare_char

# 数据 dtype 自动识别：cl100k 中文数据 id 可达 10 万+，必须 uint32；gpt2/字符级数据为 uint16
# 由数据目录下 meta.pkl 的 'dtype' 字段声明（prepare 脚本写入）；无 meta.pkl 或无该字段时默认 uint16
_data_dtype = np.uint16
_meta_path_local = os.path.join(data_dir, 'meta.pkl')
if os.path.exists(_meta_path_local):
    try:
        with open(_meta_path_local, 'rb') as f:
            _local_meta = pickle.load(f)
        _data_dtype = np.dtype(_local_meta.get('dtype', 'uint16'))
    except Exception as e:
        print(f"warning: failed to read dtype from {_meta_path_local}: {e}, fallback to uint16")

# SFT response-only loss 标记：若数据目录存在 {split}.loss.bin（uint8 的 0/1 掩码，与 token 流对齐），
# 则只对 mask=1 的目标 token（回答部分）计算 loss；预训练语料没有该文件，行为与原来完全一致
_has_loss_mask = {s: os.path.exists(os.path.join(data_dir, f'{s}.loss.bin')) for s in ('train', 'val')}
print(f"data dir = {data_dir}, dtype = {_data_dtype}, loss-mask = {_has_loss_mask}")

def get_batch(split):
    # 每次都重新创建 np.memmap，避免长期持有导致内存泄漏
    # （memmap 是内存映射，不会把整个文件读进内存，适合几十 GB 的语料）
    data = np.memmap(os.path.join(data_dir, f'{split}.bin'), dtype=_data_dtype, mode='r')
    # 随机选取 batch_size 个起始位置（保证起点之后还能取满 block_size 个 token）
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # x：输入序列，形状 (batch_size, block_size)
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    # y：目标序列 = 输入整体右移一位（自监督：用前一个 token 预测下一个 token）
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if _has_loss_mask[split]:
        # SFT：y[t] 预测的是 token[t+1]，故 loss 掩码也从 i+1 起取；
        # mask=0（prompt 指令段）的目标置为 -1，cross_entropy 的 ignore_index=-1 会跳过这些位置
        mdata = np.memmap(os.path.join(data_dir, f'{split}.loss.bin'), dtype=np.uint8, mode='r')
        m = torch.stack([torch.from_numpy((mdata[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        y = torch.where(m.bool(), y, torch.full_like(y, -1))
    if device_type == 'cuda':
        # 先 pin_memory（锁页内存）再 non_blocking 异步拷贝到 GPU，可与 GPU 计算重叠、隐藏传输延迟
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# 这些变量先初始化；若 init_from='resume' 会从 checkpoint 中覆盖
iter_num = 0           # 当前训练 iteration（可能从 checkpoint 恢复）
best_val_loss = 1e9    # 历史最优验证 loss，用于决定是否保存 checkpoint

# 尝试从数据集的 meta.pkl 中读取词表大小（字符级数据由 prepare.py 生成）
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# -----------------------------------------------------------------------------
# 模型初始化：三种来源 scratch / resume / gpt2 预训练
# -----------------------------------------------------------------------------
# 先用命令行/配置里的模型超参组装 model_args（vocab_size 暂时留空，下面确定）
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)
if init_from == 'scratch':
    # 从零随机初始化一个新模型
    print("Initializing a new model from scratch")
    # 确定词表大小，优先级：①数据集 meta.pkl（字符级数据）②配置/命令行里的 vocab_size（如中文 cl100k）③GPT-2 默认 50304
    # 50304 = 50257（GPT-2 真实词表）向上取整到 64 的倍数，对 GPU 矩阵运算更高效
    # 注意：vocab_size 不在 train.py 顶层默认值中，故用 globals().get 读取（未配置时为 None，保持原行为）
    config_vocab_size = globals().get('vocab_size', None)
    if meta_vocab_size is not None:
        chosen_vocab = meta_vocab_size
    elif config_vocab_size is not None:
        chosen_vocab = config_vocab_size
        print(f"using vocab_size from config = {config_vocab_size}")
    else:
        chosen_vocab = 50304
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = chosen_vocab
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    # 从 checkpoint 断点续训
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # 这些结构相关超参必须与 checkpoint 完全一致，否则权重形状对不上、无法加载
    # 其余超参（如 dropout、学习率）仍以命令行/配置为准，允许调整
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # 用恢复后的结构参数创建模型
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # 修复 state_dict 的键名前缀：torch.compile 后权重键名可能带 '_orig_mod.' 前缀，这里去掉
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)        # 加载模型权重
    iter_num = checkpoint['iter_num']        # 恢复训练步数
    best_val_loss = checkpoint['best_val_loss']  # 恢复历史最优验证 loss
elif init_from.startswith('gpt2'):
    # 加载 HuggingFace 上的 OpenAI GPT-2 预训练权重（gpt2 / gpt2-medium / gpt2-large / gpt2-xl）
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    override_args = dict(dropout=dropout)  # 预训练权重只允许覆盖 dropout，结构超参由模型类型决定
    model = GPT.from_pretrained(init_from, override_args)
    # 从加载好的模型中读回结构超参，保证后续保存 checkpoint 时记录的是真实配置
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# 模型"外科手术"：若所需 block_size 小于模型自带的（如加载 1024 的 GPT-2 但只想用 256），则裁剪位置嵌入
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size  # 同步更新，保证 checkpoint 记录正确值
model.to(device)  # 把模型参数搬到 GPU/CPU

# 统计模型参数量：区分 embedding 参数与网络参数
# 注意 wte（词嵌入）与 lm_head 权重共享（weight tying），named_parameters() 去重后只计一次
embedding_params = 0
network_params = 0
for name, p in model.named_parameters():
    if name.endswith('wte.weight') or name.endswith('wpe.weight'):
        embedding_params += p.numel()  # numel() = 该参数张量的元素总数
    else:
        network_params += p.numel()
total_params = embedding_params + network_params
print("模型参数量统计:")
print(f"  embedding 参数 (wte 词嵌入 + wpe 位置嵌入): {embedding_params:,} ({embedding_params / 1e6:.2f}M)")
print(f"    - wte 词嵌入 (与 lm_head 权重共享): {model.transformer.wte.weight.numel():,}")
print(f"    - wpe 位置嵌入:                     {model.transformer.wpe.weight.numel():,}")
print(f"  网络参数 (transformer blocks + ln_f):     {network_params:,} ({network_params / 1e6:.2f}M)")
print(f"  模型总参数量:                             {total_params:,} ({total_params / 1e6:.2f}M)")

# 梯度缩放器：仅 float16 训练时启用（防止低精度下梯度下溢为 0）；bfloat16/float32 时 enabled=False，相当于空操作
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# 优化器：在 model.py 中构建，内部会把"矩阵权重（weight_decay）"与"偏置/LayerNorm（不衰减）"分组
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    # 续训时连同优化器状态（Adam 动量等）一起恢复；
    # 但 SFT 场景下 ckpt 可能被刻意重置（删除 optimizer、iter 归零），此时全新初始化优化器
    if 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print("checkpoint has no optimizer state (SFT reset): initializing a fresh optimizer")
checkpoint = None  # 释放 checkpoint 字典占用的内存

# 用 PyTorch 2.0 编译模型：捕获计算图、融合算子、优化显存，通常提速 20%~40%（首次编译约需一分钟）
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model       # 保留未编译版本的引用
    model = torch.compile(model)    # requires PyTorch 2.0

# 把模型包进 DDP 容器：多卡训练时自动在反向传播阶段做梯度 all-reduce 同步
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# -----------------------------------------------------------------------------
# 评估函数：在 train/val 上各跑 eval_iters 个 batch，返回平均 loss（更平滑、噪声更小）
# -----------------------------------------------------------------------------
@torch.no_grad()  # 评估阶段不需要梯度，节省显存与计算
def estimate_loss():
    out = {}
    model.eval()  # 切换到评估模式（关闭 dropout 等）
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:  # 评估同样在混合精度上下文里前向
                logits, loss = model(X, Y)
            losses[k] = loss.item()  # .item() 会触发 GPU→CPU 同步
        out[split] = losses.mean()
    model.train()  # 切回训练模式（重新启用 dropout）
    return out

# -----------------------------------------------------------------------------
# 学习率调度器：线性 warmup + 余弦衰减
# -----------------------------------------------------------------------------
def get_lr(it):
    # 1) warmup 阶段：学习率从接近 0 线性增长到峰值 learning_rate
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) 超过衰减周期后：恒定使用最低学习率 min_lr
    if it > lr_decay_iters:
        return min_lr
    # 3) 中间阶段：按余弦曲线从 learning_rate 平滑衰减到 min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)  # 0→1 的衰减进度
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # 余弦系数：decay_ratio=0 时为 1，=1 时为 0
    return min_lr + coeff * (learning_rate - min_lr)

# 初始化 wandb 日志（仅主进程）
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# -----------------------------------------------------------------------------
# 训练主循环
# -----------------------------------------------------------------------------
X, Y = get_batch('train')  # 预先取第一个 batch（进入循环前备好数据）
t0 = time.time()           # 计时起点，用于统计每 iteration 耗时
local_iter_num = 0         # 本进程内的 iteration 计数（DDP 续训时与全局 iter_num 不同）
raw_model = model.module if ddp else model  # 解包 DDP/compile 容器，拿到原始 GPT 模型（用于存权重、算 MFU）
running_mfu = -1.0         # 滑动平均的 MFU（模型算力利用率），-1 表示尚未计算
while True:

    # 1) 计算并设置当前 iteration 的学习率
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # 2) 定期评估 + 保存 checkpoint（仅主进程执行）
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100,  # 转成百分比
            })
        # 仅当 val loss 创新低、或配置为总是保存时，才写 checkpoint
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:  # iter 0 时模型还没训练，跳过保存
                checkpoint = {
                    'model': raw_model.state_dict(),    # 模型权重（用解包后的原始模型）
                    'optimizer': optimizer.state_dict(),  # 优化器状态（续训需要）
                    'model_args': model_args,           # 模型结构超参（重建模型需要）
                    'iter_num': iter_num,               # 当前步数
                    'best_val_loss': best_val_loss,     # 历史最优 val loss
                    'config': config,                   # 全部生效超参
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    # eval_only 模式：第一次评估完就退出
    if iter_num == 0 and eval_only:
        break

    # 3) 前向 + 反向（带梯度累积），float16 时通过 GradScaler 缩放损失
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # DDP 默认每次反向都同步梯度；梯度累积时只需在最后一个 micro-step 同步，
            # 其余步骤关闭同步以减少通信。官方做法是 model.no_sync() 上下文，这里直接切换该标志位。
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:  # 混合精度前向
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps  # 缩放 loss：多个 micro-batch 的梯度累加后等效于对大 batch 求平均
        # 在 GPU 做前向/反向的同时，异步预取下一个 batch 的数据（掩盖数据加载时间）
        X, Y = get_batch('train')
        # 反向传播（float16 时 scaler 会先放大 loss 再反传，防止小梯度下溢）
        scaler.scale(loss).backward()
    # 4) 梯度裁剪：先 unscale_ 把梯度还原回真实尺度，再把梯度总范数裁剪到 grad_clip 以内
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # 5) 更新模型参数（float16 时 scaler 会判断梯度是否溢出，溢出则跳过本次更新），并更新缩放因子
    scaler.step(optimizer)
    scaler.update()
    # 6) 清空梯度（set_to_none=True 直接置 None 比写 0 更省显存）
    optimizer.zero_grad(set_to_none=True)

    # 7) 计时与日志
    t1 = time.time()
    dt = t1 - t0  # 本 iteration 耗时（秒）
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # 取 loss 标量（这里是 GPU→CPU 同步点）；乘回 gradient_accumulation_steps 以还原真实 loss 量级
        # （严格来说累积的是平均而非求和，这里只是近似还原）
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:  # 前几步编译/预热不稳定，跳过 MFU 统计，等训练平稳后再算
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)  # 估算模型算力利用率
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu  # 指数滑动平均，平滑 MFU
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # 终止条件：超过最大训练步数就退出
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()  # 清理分布式进程组
