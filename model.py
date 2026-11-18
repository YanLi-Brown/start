import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# 环境：wls2+ubuntu
# 运行前启动虚拟环境:
# source .venv/bin/activate
# 文件夹目录：/mnt/d/current_work/hands/trans/start


class ModelConfig:
    def __init__(self, **kwargs):
        # 使用了gpt2 small的超参
        # 最大上下文长度
        self.block_size = 1024
        # 词汇表大小
        self.vocab_size = 50257
        # block数
        self.n_layer = 12
        # 注意力头数
        self.n_head = 12
        # embedding维度
        self.n_embd = 768
        # 推理时关闭
        self.dropout = 0.0
        for k, v in kwargs.items():
            setattr(self, k, v)
        assert self.n_embd % self.n_head == 0


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        # 线性层，输入768维，输出768*3维。同时产出Q,K,V三份向量，之后split()切成三份
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # 注意力输出投影层
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # dropout层
        self.attn_dropout = nn.Dropout(config.dropout)
        # 注意力头数12，embedding维数768
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x, kv_cache=None, pos=0):
        # x:[B,T,768] embedding向量，B=batch size,T=序列长度
        # pos本次输入第一个token的位置，用于计算掩码
        # C为768，也是embedding维数
        B, T, C = x.size()
        # c_attn(x)输出[B,T,768*3],沿dim=2切分为三份，依次是Q,K,V
        # c_attn()线性层，将[B,T,768]映射到[B,T,768*3],和将embedding向量乘以三份矩阵Wq,Wk,Wv得到q,k,v等效
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # embedding维数 / 头数, 得到的结果作为q,k,v维数head_dim=64
        head_dim = C // self.n_head
        # 将q[B,T,768]重新切出12段,每一段64维,形状[B,T,12,64],之后交换第1,2维,形状变为[B,12,T,64]
        # @ 矩阵乘法运算，对于张量维度>2的张量，会对保持前面的维度不变，只对最后两个维度进行2维矩阵乘法
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        if kv_cache is not None:
            # 如果带了历史缓存，将历史K/V拼接到新k/v前面
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        kv_cache = (k, v)

        # Tq是q的长度，Tk是K的序列长度。
        # 如果不带历史缓存，Tq,Tk值均为T；如果带历史缓存K/V，Tk会较大，Tq为1
        Tq, Tk = q.size(2), k.size(2)
        # torch.ons()构造全1向量[Tq,Tk]，torch.tril构造下三角矩阵
        # diagonal=0时，构造下三角矩阵，j<=i位置为1；diagonal=pos时，j<=i+pos位置为1，即分界线偏移pos位
        mask = torch.tril(
            torch.ones(Tq, Tk, dtype=torch.bool, device=x.device), diagonal=pos
        )

        # 矩阵乘法，k.transpose(-2,-1)交互k的最后两个维度
        # q @ k.transpose(-2,-2) q[B,12,T,64]与k[B，12，64,T]最后两个维度进行2维矩阵乘法
        # 点积值随维度增大而变大，除以head_dim**0.5以防止softmax饱和，梯度消失
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
        # att.masked_fill()将值为False的位置填为负无穷
        att = att.masked_fill(~mask, float("-inf"))
        # 沿最后一个维度进行softmax
        att = F.softmax(att, dim=-1)
        # 经过dropout层
        att = self.attn_dropout(att)
        # att,v最后两个维度做矩阵乘法
        y = att @ v

        # transpose(1,2)交换y的第1，2个维度。contiguous()将y的内存排布连续。
        # view(B,T,C)将y[B,T,12,64]拼回[B,T,768]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # 输出[B,T,768]经过线性层，输出仍为[B,T,768]
        return self.c_proj(y), kv_cache


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # c_fc 全连接层 第一个线性层，将768维向量放大到768*4维
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        # proj 投影层 第2个线性层，将768*4维向量投影到768维
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        # 训练时随机丢部分值防止过拟合
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # 先将x投影到4*768维，再做非线性（tanh近似版GELU）,再缩回768维，最后经过dropout
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x), approximate="tanh")))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, kv_cache=None, pos=0):
        # pre-LayerNorm + 残差连接
        a, kv_cache = self.attn(self.ln_1(x), kv_cache, pos)
        x = x + a
        x = x + self.mlp(self.ln_2(x))
        return x, kv_cache


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # nn.ModuleDict 带名字的子模块字典
        #
        self.transformer = nn.ModuleDict(
            dict(
                # wte 查找表[50257, 768]，输入token id，输出embedding向量
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                # wpe 位置查找表 根据位置输出向量，较旧版本，新版本使用RoPE
                wpe=nn.Embedding(config.block_size, config.n_embd),
                # 正则化 训练时随机丢一部分数据，防止过拟合
                drop=nn.Dropout(config.dropout),
                # 列表推导式 表示重复12次Block
                # nn.ModuleList 保证12个Block的参数都被追踪
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                # 最后一层LayerNorm
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        # 输出头LM 把上层输出映射为[1,50257]的向量，每个数代表下一个token是这个词的分数（logit）
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 权值共享 使wte和lm_head共用一个权重张量
        self.transformer.wte.weight = self.lm_head.weight
        # 递归的把函数应用到模型的每一个子模块上 目的是随机初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # 判断传入的module是否是 nn.Linear.如果是，权重填均值0，标准差0.02的正态分布随机数
        # torch.nn.init.normal_ 末尾的下划线在pytorch通常表示原地修改这个张量，不返回新张量
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            # 判断传入的module是否是 nn.Embedding.如果是，权重填均值0，标准差0.02的正态分布随机数
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, kv_cache=None, pos=0):
        # forward 定义数据流动路径
        # idx [B,T]的token id矩阵 B=batch大小，T=序列长度
        # kv_cache KV缓存，None表示首次前向
        # pos 本次输入第一个token的位置

        # 解包出idx的两个维度 如idx.size=[1,5] -> B=1,T=5
        B, T = idx.size()
        # 构造一个位置下标序列[pos,pos+1,.....,pos+T-1]
        pos_idx = torch.arange(pos, pos + T, device=idx.device)
        # 把每个tokeng根据其id和位置，替换为embedding 768维向量，再相加；drop 经过一层dropout
        x = self.transformer.drop(
            self.transformer.wte(idx) + self.transformer.wpe(pos_idx)
        )

        if kv_cache is None:
            # 如果首次前向没有缓存，就造一个列表，对应12个Block各自的缓存槽位
            kv_cache = [None] * self.config.n_layer

        new_caches = []
        # zip将12个Block和12个缓存槽一一配对，x,cache接收每个Block输出的x,cache
        for block, cache in zip(self.transformer.h, kv_cache):
            x, cache = block(x, cache, pos)
            new_caches.append(cache)

        # 最后ln_f归一化，输出投影为[B,T,50257]的logits
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits, new_caches
