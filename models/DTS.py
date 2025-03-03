from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath


def get_unique_indices(indices: Tensor, max_value: int) -> Tensor:
    """
    确保采样的索引唯一，避免重复
    :param indices: 待去重的索引 [B, n_tokens]
    :param max_value: 最大索引值（用于填充重复位置）
    :return: 唯一索引 [B, n_tokens]
    """
    # 按行排序索引
    sorted_indices = torch.sort(indices, dim=1)[0]
    # 左移并填充1.0以检测重复项
    shift_left = F.pad(sorted_indices[:, 1:], (0, 1), value=1.0)
    # 标记重复项为max_value
    unique_indices = torch.where(
        (shift_left - sorted_indices) == 0,
        max_value * torch.ones_like(indices),
        sorted_indices,
    )
    # 重新排序并将重复项放到末尾
    unique_indices = torch.sort(unique_indices, dim=1)[0]
    return unique_indices


def create_ys(normalized_cdf: Tensor, n_tokens: int) -> Tensor:
    """
    :param normalized_cdf: 归一化的累积分布函数 [B, T-1]
    :param n_tokens: 目标token数量
    :return: 采样点坐标 [B, n_tokens-1]
    """

    B = normalized_cdf.shape[0]
    # 生成均匀间隔点（0到1之间）
    ys = (
        torch.linspace(
            start=0,
            end=1.0,
            steps=n_tokens - 1,
            device=normalized_cdf.device,
        )
        .unsqueeze(0)
        .repeat(B, 1)
    )
    ys_start = (
        torch.min(normalized_cdf + (normalized_cdf == 0).float() * 1e8, dim=1)[0]
        .unsqueeze(-1)
        .expand_as(ys)
    )
    # 调整采样点间隔
    steps = (
        torch.range(0, n_tokens - 2, device=normalized_cdf.device)
        .unsqueeze(0)
        .expand_as(ys_start)
    )
    ys = ys_start + (((ys * (n_tokens - 2)) - ys_start * steps) / (n_tokens - 2))

    return ys


def score_assignment_step(attn: Tensor) -> (Tensor, Tensor):
    """
    Token Score Assignment Step.
    :param attn: attention matrix
    :param v: values
    :return: sorted significance scores and their corresponding indices
    """
    significance_score = attn[:, :, 0].sum(
        dim=1
    )
    significance_score = significance_score / significance_score.sum(
        dim=1, keepdim=True
    )
    sorted_scores, sorted_indices = torch.sort(
        significance_score, descending=False, dim=1
    )

    return sorted_scores, sorted_indices

def inverse_transform_sampling(
        sorted_scores: Tensor,
        sorted_indices: Tensor,
        attn: Tensor,
        n_tokens: int,
        raw_x: Tensor,
        n_ref_tokens: int,
) -> (Tensor, Tensor):
    """
    逆变换采样实现token选择
    :param sorted_scores: 排序后的分数 [B, T-1]
    :param sorted_indices: 对应索引 [B, T-1]
    :param attn: 原始注意力矩阵 [B, H, T, T]
    :param n_tokens: 目标token数量
    :param raw_x: 原始输入token [B, T, C]
    :param n_ref_tokens: 参考token数（如197）
    :return: 选中的token及新注意力矩阵
    """
    B, N, C = raw_x.shape
    # 计算累积分布函数
    cdf = torch.cumsum(sorted_scores, dim=1)  # [B x T-1]
    # 归一化CDF到[0,1]
    normalized_cdf = (  # normalized cdf
                             cdf - cdf.min(dim=1)[0].unsqueeze(dim=1)
                     ) / ((cdf.max(dim=1)[0] - cdf.min(dim=1)[0]) / 1.0).unsqueeze(dim=1)
    # 生成y轴采样点
    ys = create_ys(normalized_cdf, n_ref_tokens).unsqueeze(
        dim=2
    )  # sampled values from y-axis of size [B, n-1, 1]
    normalized_cdf = normalized_cdf.unsqueeze(dim=1)  # [B, 1, N - 1]

    # 寻找最接近采样点的CDF索引
    expanded_ys = torch.Tensor.expand(ys, (B, ys.shape[1], ys.shape[1]))
    diff_tokens = ys.shape[1] - (N - 1)
    tokens_to_pick_ind = torch.min(
        torch.abs(expanded_ys - F.pad(normalized_cdf, (diff_tokens, 0))),
        dim=2,
    )[1]  # [B x n-1]

    # 处理索引偏移
    tokens_to_pick_ind = tokens_to_pick_ind - diff_tokens

    # 重组注意力矩阵和token
    # Sort attention matrix and add CLS weights.
    attn_sorted = torch.gather(
        attn[:, :, 1:],
        2,
        sorted_indices.unsqueeze(1)
        .unsqueeze(-1)
        .expand(B, 12, N - 1, N),
    )  # [B x h x T-1 x T]

    attn_tmp = F.pad(attn_sorted, (0, 0, 0, 1), value=0.0)  # [B x h x T x T]

    # # Sort tokens and add CLS token.
    raw_x_tmp = torch.gather(
        raw_x[:, 1:], 1, sorted_indices.unsqueeze(-1).expand(B, N - 1, C)
    )
    raw_x_tmp = F.pad(raw_x_tmp, (0, 0, 0, 1), value=0.0)  # [B x n x C]

    unique_indices = get_unique_indices(
        indices=tokens_to_pick_ind, max_value=N - 1
    )[:, : N - 1]

    # 根据唯一索引选择token
    # Prune the attention matrix and input tokens.
    attn_tmp = torch.gather(
        attn_tmp,
        2,
        unique_indices.unsqueeze(1)
        .unsqueeze(3)
        .expand(B, 12, n_tokens - 1, N),
    )
    raw_x_tmp = torch.gather(
        raw_x_tmp, 1, unique_indices.unsqueeze(2).expand(B, n_tokens - 1, C)
    )

    # 拼接CLS注意力
    attn_tmp = torch.cat([attn[:, :, 0:1], attn_tmp], dim=2)
    # 拼接CLS token
    raw_x_tmp = torch.cat([raw_x[:, 0:1], raw_x_tmp], dim=1)

    # 生成保留token的掩码（policy）
    policy = (unique_indices != (N - 1)).unsqueeze(-1).float()
    policy = F.pad(policy, (0, 0, 1, 0), value=1.0)
    selected_x = raw_x_tmp
    attn = attn_tmp

    sampler = torch.nonzero(policy)

    return selected_x, attn, policy, sampler


def forward(
        self,
        x: Tensor,
        policy: Tensor,
        sampler: Tensor,
        n_tokens: float,
        raw_x: Tensor,
        n_ref_tokens: int,
):
    B, N, C = x.shape

    if isinstance(N, Tensor):
        N = N.cpu().item()

    if n_tokens > N:  # Number of tokens to be sampled can't be larger than N.
        n_tokens = N
    if n_tokens <= 1.0:  # When n_tokens is a ratio.
        n_tokens = n_tokens * N
    if n_tokens < 8:  # Number of tokens to be sampled can't be less than 8.
        n_tokens = 8

    n_tokens = round(n_tokens)
    if N < n_tokens:
        n_tokens = N

    qkv = self.qkv(x, policy, sampler)
    qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(
        2, 0, 3, 1, 4
    )
    qkv = qkv * policy.unsqueeze(0).unsqueeze(
        2
    )  # Get rid of previously removed tokens.
    q, k, v = (
        qkv[0],
        qkv[1],
        qkv[2],
    )

    attn_no_softmax = (q @ k.transpose(-2, -1)) * self.scale
    attn = self.softmax_with_policy(attn_no_softmax, policy)  # [B x H x T x T]

    # --------------------------
    # Token Score Assignment
    # --------------------------

    sorted_scores, sorted_indices = self.score_assignment_step(attn, v)

    # --------------------------
    # Inverse Transform Sampling
    # --------------------------

    selected_x, attn, policy, sampler = self.inverse_transform_sampling(
        sorted_scores, sorted_indices, attn, n_tokens, raw_x, n_ref_tokens
    )

    x = (attn @ v).transpose(1, 2).reshape(B, attn.shape[2], C)

    # Pruning
    if self.drop_tokens:
        out_mask_size = policy.sum(1).max().int()

        sampler_out = sampler[:, 0] * out_mask_size + sampler[:, 1]
        sampler = sampler[:, 0] * n_tokens + sampler[:, 1]
        sampler_input = sampler.unsqueeze(-1).expand(-1, C)
        sampler_output = sampler_out.unsqueeze(-1).expand(-1, C)
        flatten_x = x.reshape(-1, C)
        flatten_selected_x = selected_x.reshape(-1, C)

        x_prunned = torch.gather(flatten_x, 0, sampler_input)
        selected_x_prunned = torch.gather(flatten_selected_x, 0, sampler_input)

        out_zero_mask = self.out_zero_mask.expand(B * out_mask_size, -1)

        x = out_zero_mask.scatter(
            0, sampler_output, x_prunned, reduce="add"
        ).reshape((B, out_mask_size, C))
        selected_x = out_zero_mask.scatter(
            0, sampler_output, selected_x_prunned, reduce="add"
        ).reshape((B, out_mask_size, C))

        policy = (
            out_zero_mask[:, 0]
            .scatter(0, sampler_out, 1, reduce="add")
            .reshape(B, out_mask_size, 1)
        )

    x = self.proj(x, policy, sampler)
    x = x * policy
    x = self.proj_drop(x)
    return x, selected_x, policy, sampler