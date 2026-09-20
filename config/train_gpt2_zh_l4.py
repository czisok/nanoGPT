# =============================================================================
# GPU 档位开关（L4 / V100 两套参数，二选一）
# -----------------------------------------------------------------------------
# gpu_profile 可选：
#   'auto' : 运行时自动探测当前 GPU（默认）
#   'l4'   : NVIDIA L4，Ada 架构(sm_89)，24GB，原生 bf16，开 torch.compile
#   'v100' : Tesla V100，Volta 架构(sm_70)，常见 16GB，无 bf16→用 fp16+GradScaler，关 compile
#
# 命令行临时覆盖（configurator.py 是「先 exec 本文件、后解析 argv」，普通 --k=v 覆盖发生在
# 下方分支执行之后、无法触发分支，故这里自行扫描 sys.argv 提前取出 gpu_profile）：
#   python train.py config/train_gpt2_zh_l4.py --gpu_profile=v100
#   python train.py config/train_gpt2_zh_l4.py --gpu_profile=l4
# =============================================================================
import os
gpu_profile = 'auto'

def _detect_gpu_profile():
    # 自动探测：优先按显卡名判断；未知型号再按「是否支持 bf16 + 显存大小」推断
    try:
        import torch
        if not torch.cuda.is_available():
            return 'l4'  # 无 CUDA（CPU 训练）时给默认值，dtype 在 train.py 里会被 nullcontext 忽略
        name = torch.cuda.get_device_name(0)
        if 'V100' in name:
            return 'v100'
        if 'L4' in name:
            return 'l4'
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        # 大显存 + 支持 bf16 视为新卡（l4 档），否则按保守的 v100 档
        return 'l4' if (torch.cuda.is_bf16_supported() and total_gb >= 20) else 'v100'
    except Exception:
        return 'l4'

# 命令行 --gpu_profile=xxx 提前解析（早于 configurator.py 的统一 argv 解析），保证下方分支即时生效
import sys as _sys
for _arg in _sys.argv[1:]:
    if _arg.startswith('--gpu_profile='):
        gpu_profile = _arg.split('=', 1)[1].strip().strip('"\'')

if gpu_profile == 'auto':
    gpu_profile = _detect_gpu_profile()

assert gpu_profile in ('l4', 'v100'), f"gpu_profile 只能是 auto/l4/v100，收到: {gpu_profile}"
print(f"[gpu_profile] 当前使用档位: {gpu_profile}")

# 输出与评估
out_dir = 'out-base-gpt2-zh'
eval_interval = 500
log_interval = 20
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'  # 从零开始预训练

# 数据配置
# 注意：dataset 对应 data/ 下的子目录名，train.py 会读 data/{dataset}/train.bin
# prepare_zh.py 的 bin 输出在 data/minimind_data/，故这里必须写 minimind_data（写 'zh' 会找不到文件）
# 数据为 cl100k_base 分词、uint32 存储（train.py 的 get_batch 已改为按 meta.pkl 自动识别 dtype）
data_dir = "/home/zhangbo.999/jupyter_workspace/dataset/llm_dataset"
dataset = os.path.join(data_dir, "minimind_data")
block_size = 1024      # 上下文窗口（两档一致）

# 模型结构（124M主体 + cl100k词表；两档一致）
n_layer = 12
n_head = 12
n_embd = 768
vocab_size = 100352    # cl100k词表共100277，100352=向上取整到128的倍数，GPU矩阵运算更高效（数据最大id为100254）
dropout = 0.0
bias = False
# 说明：flash attention 无需手动开关——PyTorch>=2.0 时 model.py 自动启用 scaled_dot_product_attention

# 优化器配置（中文预训练学习率略低于英文；两档一致）
learning_rate = 5e-4
max_iters = 6000        # tokens/iter=batch*accum*block=64*1024=65536，6000步约3.9亿token ≈ 0.9个epoch（train.bin约4.49亿token）
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# 学习率衰减（两档一致）
decay_lr = True
warmup_iters = 300      # 中文warmup步数略多
lr_decay_iters = 6000
min_lr = 5e-5

# =============================================================================
# 随 GPU 档位切换的差异参数（其余超参两档完全相同）
# 两档都保持 batch_size * gradient_accumulation_steps = 64
#   => tokens/iter = batch*accum*block = 64*1024 = 65536 恒定，训练动态一致
# cl100k 大词表(100352)使 logits 显存 ∝ batch*block*vocab，交叉熵前向/反向的 logits 是显存大头。
# 若仍偶发 OOM/碎片，启动前可加环境变量 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True（train.py 已默认设置）
# =============================================================================
if gpu_profile == 'l4':
    # ---- NVIDIA L4（Ada sm_89，24GB）----
    dtype = 'bfloat16'                    # L4 原生支持 BF16：动态范围大、数值稳定、无需 GradScaler
    compile = True                        # torch.compile 算子融合提速（新卡 Triton 支持完善）
    batch_size = 8
    gradient_accumulation_steps = 8       # 等效总 batch = 8*8 = 64
else:
    # ---- Tesla V100（Volta sm_70，常见 16GB PCIe）----
    # V100 硬件不支持 bfloat16（bf16 需 Ampere sm_80+，如 L4/A100），必须用 float16：
    # train.py 在 dtype=='float16' 时自动启用 torch.cuda.amp.GradScaler，放大 loss 防小梯度下溢、
    # 检测到溢出则跳过该步更新。fp16 也正是 V100 Tensor Core 的强项。
    dtype = 'float16'
    # Volta(sm_70) 的 Triton 支持不如新卡稳定，且编译期 Triton autotune 会克隆 buffer 额外吃显存
    # （此前 OOM 正发生在该阶段），默认关闭。V100 32GB(SXM2) 且 torch/triton 较新时可尝试改 True。
    compile = False
    batch_size = 4                        # 按 16GB 保守取值；32GB SXM2 可改为 8（accum 同步改 8）
    gradient_accumulation_steps = 16      # 等效总 batch = 4*16 = 64

# 说明：nanoGPT 不支持梯度检查点（grad_checkpointing），此配置项无效
# 补充：train.py 里的 allow_tf32 在 V100 上会被自动忽略（TF32 同样是 Ampere sm_80+ 特性），无需处理
