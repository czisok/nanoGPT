"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""
# 中文说明：本文件是 GPT 语言模型的完整定义（GPT-2 结构的精简复现），全部代码都在这一个文件里。
# 整体结构自底向上为：LayerNorm → CausalSelfAttention（因果自注意力）→ MLP（前馈网络）
# → Block（一个 Transformer 块 = 注意力 + MLP，各带残差和 LayerNorm）→ GPT（堆叠 n_layer 个 Block）。

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """
    # 中文：层归一化。PyTorch 自带的 nn.LayerNorm 不支持 bias=False（bias 是必建参数），
    # 这里自定义一个可选择是否带偏置的版本：weight 始终存在，bias 可选。

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))           # 可学习的缩放参数 gamma，初始化为 1
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None  # 可学习的偏移参数 beta，bias=False 时不创建

    def forward(self, input):
        # 对最后一个维度做 LayerNorm，eps=1e-5 防止除零；weight/bias 做仿射变换
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    # 中文：因果（causal）多头自注意力。"因果"指位置 t 只能关注 ≤t 的位置，不能看到未来 token，
    # 这是 decoder-only 语言模型（GPT）的核心约束。

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0  # 隐状态维度必须能被头数整除，才能均分到每个头
        # key, query, value projections for all heads, but in a batch
        # 中文：把 Q/K/V 三个投影合并成一个 Linear：输入 n_embd 维，输出 3*n_embd 维，一次算出全部头的 Q、K、V
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        # 中文：注意力输出后的投影层，把多个头的结果融合回 n_embd 维
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)   # 注意力权重上的 dropout
        self.resid_dropout = nn.Dropout(config.dropout)  # 残差/输出上的 dropout
        self.n_head = config.n_head   # 注意力头数
        self.n_embd = config.n_embd   # 隐状态维度
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        # 中文：检测是否可用 Flash Attention（PyTorch>=2.0 自带），它是融合算子，既快又省显存
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # 中文：老版本 PyTorch 没有 Flash Attention，需手动建因果掩码：
            # 下三角矩阵（tril），reshape 成 (1,1,T,T) 以便广播到 batch 和 head 维度
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # 中文：B=批大小，T=序列长度，C=隐状态维度(n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # 中文：一次线性变换得到拼接的 QKV，再沿最后一维切成三份；view 成多头形状并转置，
        # 最终 q/k/v 形状均为 (B, nh, T, hs)，其中 hs = C // n_head 是每个头的维度
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # 中文：自注意力 = softmax(QK^T / sqrt(hs)) · V，注意力分数矩阵形状为 (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            # 中文：优先走 Flash Attention，is_causal=True 自动加因果掩码；训练时才启用 dropout
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            # 中文：手动实现注意力（教学/兼容用）：
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # 缩放点积：QK^T / sqrt(hs)，防止点积过大导致 softmax 饱和
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))  # 因果掩码：把"未来位置"填 -inf，softmax 后其权重变 0
            att = F.softmax(att, dim=-1)   # 沿 key 维度归一化成注意力权重（每个 query 对所有位置的权重和为 1）
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)  # 用注意力权重对 V 加权求和
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # 中文：把多头结果转置回 (B, T, nh, hs) 并拼接成 (B, T, C)，contiguous 保证内存连续以便 view

        # output projection
        # 中文：过输出投影层融合多头信息，再做 dropout
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    # 中文：前馈网络（Feed-Forward Network）。每个 Block 中注意力层之后接一个 MLP，
    # 对每个位置独立地做非线性变换：先升维 4 倍，再降回原维度。

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)  # 升维：n_embd → 4*n_embd
        self.gelu    = nn.GELU()                          # GELU 激活函数（平滑版 ReLU，GPT 系列使用）
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)  # 降维：4*n_embd → n_embd
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    # 中文：一个 Transformer Block。GPT-2 采用 Pre-LN 结构（LayerNorm 在注意力/MLP 之前），
    # 相比原始 Transformer 的 Post-LN 梯度更稳定、更容易训练深层网络。每个子层都有残差连接。

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)  # 注意力前的 LayerNorm
        self.attn = CausalSelfAttention(config)                 # 因果自注意力
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)  # MLP 前的 LayerNorm
        self.mlp = MLP(config)                                  # 前馈网络

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))  # 残差连接：先归一化 → 注意力 → 加回输入
        x = x + self.mlp(self.ln_2(x))   # 残差连接：先归一化 → MLP → 加回输入
        return x

@dataclass
class GPTConfig:
    # 中文：GPT 模型的配置（用 dataclass 集中管理超参）
    block_size: int = 1024   # 上下文长度（模型一次能处理的最大 token 数）
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    # 中文：词表大小。GPT-2 真实词表是 50257，向上取整到 64 的倍数 50304，对 GPU 矩阵运算更高效
    n_layer: int = 12       # Transformer Block 层数（GPT-2 small）
    n_head: int = 12        # 注意力头数
    n_embd: int = 768       # 隐状态/词向量维度
    dropout: float = 0.0    # dropout 概率
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    # 中文：是否在 Linear 和 LayerNorm 中使用偏置。GPT-2 用 True；设为 False 略快且效果相当

class GPT(nn.Module):
    # 中文：GPT 模型主体。整体数据流：
    # token id → 词嵌入(wte) + 位置嵌入(wpe) → n_layer 个 Block → 最终 LayerNorm(ln_f) → lm_head 输出每个词的 logits

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # 用 ModuleDict 组织 transformer 的各个组件
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),   # 词嵌入（word token embedding）：token id → 向量
            wpe = nn.Embedding(config.block_size, config.n_embd),   # 位置嵌入（position embedding）：位置 id → 向量（可学习，非正弦）
            drop = nn.Dropout(config.dropout),                      # 嵌入相加后的 dropout
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # n_layer 个 Transformer Block
            ln_f = LayerNorm(config.n_embd, bias=config.bias),      # 所有 Block 之后的最终 LayerNorm
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # 输出头：把隐状态投影成词表大小的 logits
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying
        # 中文：权重绑定（weight tying）：让输入词嵌入 wte 与输出投影 lm_head 共享同一个权重矩阵。
        # 好处：减少参数量（省去一个 vocab_size×n_embd 的大矩阵），且被证明有助于泛化。

        # init all weights
        # 中文：用 self.apply 对所有子模块递归调用 _init_weights 做权重初始化
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        # 中文：按 GPT-2 论文，对残差路径上的 c_proj 权重做特殊缩放初始化：
        # 每个 Block 有 2 处残差相加，共 2*n_layer 次，标准差除以 sqrt(2*n_layer)，
        # 防止残差流的方差随层数累积而增大
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        # 中文：打印模型参数量（默认扣掉位置嵌入 wpe，与 GPT-2 论文口径一致）
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        # 中文：统计模型参数量。non_embedding=True（默认）时扣掉位置嵌入 wpe；
        # 词嵌入 wte 不扣，因为它通过权重绑定同时充当了最终输出层 lm_head 的权重，属于"网络参数"。
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        # 中文：权重初始化规则（GPT-2 的设定）：Linear 和 Embedding 的权重都用均值 0、标准差 0.02 的正态分布；
        # Linear 的偏置初始化为 0
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # 中文：前向传播。idx 形状 (b, t)，是 token id 序列；targets 形状同为 (b, t)，是右移一位的目标（训练时传入）。
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)  # 位置索引 0..t-1

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)  # 词嵌入
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)  # 位置嵌入（广播到 batch）
        x = self.transformer.drop(tok_emb + pos_emb)  # 词嵌入与位置嵌入相加，再 dropout
        for block in self.transformer.h:  # 依次通过 n_layer 个 Block
            x = block(x)
        x = self.transformer.ln_f(x)      # 最终 LayerNorm

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            # 中文：训练模式：对所有位置计算 logits，并与 targets 算交叉熵损失。
            # logits 形状 (b, t, vocab_size)，reshape 成 (b*t, vocab_size)；
            # 这是一个"在每个位置预测下一个 token"的 vocab_size 分类问题。ignore_index=-1 用于忽略填充位。
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            # 中文：推理模式（生成时）：只需预测"最后一个位置"的下一个 token，
            # 因此只对最后一个时间步做 lm_head，省去整序列的输出投影计算。用 [-1] 切片以保留时间维度。
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        # 中文：模型"外科手术"：把上下文长度裁剪到更小的值。
        # 例如加载 block_size=1024 的 GPT-2 预训练权重，但想用 256 的上下文训练小模型时，
        # 直接截取位置嵌入的前 block_size 行，并裁剪注意力因果掩码。
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])  # 截取位置嵌入
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):  # 仅手动实现注意力时存在 bias 掩码 buffer
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        # 中文：类方法，从 HuggingFace 加载 OpenAI 官方 GPT-2 预训练权重到本模型。
        # 支持 gpt2 / gpt2-medium / gpt2-large / gpt2-xl 四种规模。
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)  # 只允许覆盖 dropout，结构超参由模型类型决定
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        # 中文：不同规模 GPT-2 的结构超参（层数/头数/维度）及对应参数量
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints  # GPT-2 词表固定 50257（不取整）
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints  # GPT-2 上下文固定 1024
        config_args['bias'] = True # always True for GPT model checkpoints         # GPT-2 官方权重带偏置
        # we can override the dropout rate, if desired
        # 中文：微调时可覆盖 dropout 率
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        # 中文：先按配置创建一个本仓库结构的 GPT 模型（权重已随机初始化）
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param
        # 中文：排除注意力因果掩码（它是 buffer 不是可学习参数，不需要从预训练模型拷贝）

        # init a huggingface/transformers model
        # 中文：加载 HuggingFace 版 GPT-2，拿到它的 state_dict
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        # 中文：把 HuggingFace 权重复制到本模型，同时校验键名和形状对齐
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        # 中文：OpenAI 检查点用的是 "Conv1D" 模块（权重形状与 Linear 相反），本模型用标准 nn.Linear，
        # 因此这几个权重矩阵拷贝时需要转置（.t()）
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                # 中文：Conv1D 权重需转置后拷贝（形状反转后应与目标一致）
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                # 中文：其余参数（LayerNorm、Embedding、偏置等）直接按相同形状拷贝
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # 中文：构建 AdamW 优化器，并按参数类型做权重衰减分组。
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}  # 只优化需要梯度的参数
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        # 中文：分组规则——维度 ≥2 的参数（Linear 权重矩阵、Embedding 矩阵）施加权重衰减（L2 正则）；
        # 维度 <2 的参数（偏置 bias、LayerNorm 的 weight）不衰减（对这类缩放/偏移参数衰减无意义）。
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},   # 衰减组
            {'params': nodecay_params, 'weight_decay': 0.0}           # 不衰减组
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        # 中文：创建 AdamW 优化器；CUDA 且 PyTorch 支持时使用 fused 版本（把多个参数的更新融合，更快）
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # 中文：估算 MFU（Model FLOPs Utilization，模型算力利用率），
        # 即模型实际达到的 FLOPS 占 GPU 峰值 FLOPS 的比例，是衡量训练效率的指标。
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        # 中文：按 PaLM 论文附录 B 的公式估算每步浮点运算量
        N = self.get_num_params()  # 模型参数量
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size  # 层数、头数、每头维度、序列长度
        flops_per_token = 6*N + 12*L*H*Q*T       # 每个 token 的 FLOPs：6N 来自权重相关的矩阵乘，12LHQT 是注意力的逐 token 项
        flops_per_fwdbwd = flops_per_token * T   # 一次前向+反向（fwdbwd）处理 T 个 token 的 FLOPs
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter  # 每次 iteration 的总 FLOPs（乘以 batch 大小等）
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second  # 实际达到的 FLOPS（每秒运算量）
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS  # A100 bfloat16 峰值算力 312 TFLOPS
        mfu = flops_achieved / flops_promised  # 利用率 = 实际 / 峰值
        return mfu

    @torch.no_grad()  # 生成过程不需要梯度
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        # 中文：自回归生成。给定前缀 idx（形状 (b,t) 的 token id），逐 token 地续写 max_new_tokens 次；
        # 每步把模型预测出的新 token 接回输入，再预测下一个（自回归）。
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            # 中文：序列会越来越长，超过 block_size 时只保留最近的 block_size 个 token（滑动窗口截断）
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)  # 前向，得到各位置 logits
            # pluck the logits at the final step and scale by desired temperature
            # 中文：只取最后一个时间步的 logits（预测下一个 token）；除以 temperature：
            # temperature<1 分布更尖锐（更确定/保守），>1 更平坦（更随机/多样）
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            # 中文：top-k 采样：只保留概率最高的 k 个候选，其余 logits 置 -inf，避免采样到低概率的离谱 token
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)  # logits → 概率分布
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # 按概率分布随机采样 1 个 token（而非取 argmax，以产生多样性）
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)  # 把新 token 拼到序列末尾，继续下一轮

        return idx
