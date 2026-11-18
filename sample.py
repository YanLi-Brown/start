import torch
import torch.nn.functional as F

# git
# !git status                 查看状态
# !git add .                  增加提交到缓冲区
# !git commit -m "add files"  缓存区内容提交
# !gitpush                    网络上传


@torch.no_grad()
# 装饰器，修饰下面的函数 在函数执行期间关闭梯度梯度计算
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
    # idx 输入token id ,shape[B,T],max_new_token最多生成多少个新token
    # temperature 控制新token选取随机率

    # 把模型切换到推理模式
    model.eval()
    kv_cache = None
    pos = 0

    for _ in range(max_new_tokens):
        # 循环max_new_tokens次，每次1个token

        # 如果kv_cache is None, idx_cond = idx
        # 如果kc_cache is not None, idx的最后一个token传给idx_cond(形状[B,1])
        idx_cond = idx if kv_cache is None else idx[:, -1:]

        # 前向传播，得到logits(形状[B,T,vocab_size])和更新后的kc_cache
        logits, kv_cache = model(idx_cond, kv_cache, pos=pos)
        # 更新位置偏移 prefill时加上prompt长度，decode时每次加1
        pos += idx_cond.size(1)

        # logits[:,-1,:] 取最后一个位置的logits(形状[B,vocab_size])
        # temperature <1让分布更加集中，>1时更加平滑。max防止除以0
        logits = logits[:, -1, :] / max(temperature, 1e-8)

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("-inf")

        if top_p is not None:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumprobs = torch.cumsum(sorted_probs, dim=-1)
            to_remove = cumprobs > top_p
            to_remove[..., 1:] = to_remove[..., :-1].clone()
            to_remove[..., 0] = 0
            # 把"是否移除"按原下标散射回去
            remove_mask = torch.zeros_like(logits, dtype=torch.bool)
            remove_mask.scatter_(1, sorted_idx, to_remove)
            logits[remove_mask] = -float("inf")

        # F.softmax() 对logits的最后一个维度进行softmax
        probs = F.softmax(logits, dim=-1)
        # 按照概率分布随机抽取1个token
        idx_next = torch.multinomial(probs, num_samples=1)
        # 把新采出的token拼接到序列末尾，形成新的idx
        idx = torch.cat([idx, idx_next], dim=1)

    # 返回完整序列
    return idx
