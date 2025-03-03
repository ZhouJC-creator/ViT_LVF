import time

from torch import Tensor
from .transformer_block import Attention, Mlp  # 导入基础注意力模块和MLP模块
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath  # 导入DropPath正则化方法


class AdaptiveTokenSampler(Attention):
    """
    自适应令牌采样器，继承自基础Attention类
    主要功能：通过自适应策略减少Transformer中的token数量
    """

    def __init__(
            self,
            dim,  # 输入维度
            num_heads=8,  # 注意力头数
            qkv_bias=False,  # 是否在QKV线性变换中使用偏置
            qk_scale=None,  # 缩放因子
            attn_drop=0.0,  # 注意力dropout率
            proj_drop=0.0,  # 投影层dropout率
            drop_path=0.0,  # DropPath概率
            drop_tokens=False,  # 是否启用token剪枝
    ):
        # 初始化父类（基础Attention模块）
        super(AdaptiveTokenSampler, self).__init__(
            dim,
            num_heads,
            qkv_bias,
            qk_scale,
            attn_drop,
            proj_drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        # 用于mask输出的零参数（不可训练）
        self.out_zero_mask = nn.Parameter(torch.zeros(1, dim), requires_grad=False)
        self.drop_tokens = drop_tokens  # 是否剪枝token

    @staticmethod
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

    @staticmethod
    def create_ys(normalized_cdf: Tensor, n_tokens: int) -> Tensor:
        """
        在y轴上生成均匀采样点（用于逆变换采样）
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
        ) # 扩展到batch维度
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



    @staticmethod
    def score_assignment_step(attn: Tensor, v: Tensor) -> (Tensor, Tensor):
        """
        token重要性评分计算
        :param attn: 注意力矩阵 [B, H, T, T]
        :param v: 值向量 [B, H, T, C/H]
        :return: 排序后的分数和索引 [B, T-1]
        """
        B, H, _, _ = attn.shape
        C = v.shape[-1] * H
        # 计算值向量的L2范数 [B, T]
        v_norm = torch.norm(v.transpose(1, 2).reshape(B, -1, C), p=2, dim=2)
        # 计算CLS token的注意力权重之和 [B, T]
        significance = attn[:, :, 0].sum(dim=1)  # 取CLS对各token的注意力
        significance = significance * v_norm[:, 1:]  # 排除CLS自身
        significance = significance / significance.sum(dim=1, keepdim=True)  # 归一化
        # 按分数升序排列（后续逆采样使用）
        sorted_scores, sorted_indices = torch.sort(significance, dim=1, descending=False)
        return sorted_scores, sorted_indices


    def inverse_transform_sampling(self, sorted_scores, sorted_indices,
                                   attn, n_tokens, raw_x, n_ref_tokens):
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
        cdf = torch.cumsum(sorted_scores, dim=1)  # [B, T-1]
        # 归一化CDF到[0,1]
        min_cdf = cdf.min(dim=1)[0].unsqueeze(1)
        normalized_cdf = (cdf - min_cdf) / (cdf.max(dim=1)[0] - min_cdf).unsqueeze(1)

        # 生成y轴采样点
        ys = self.create_ys(normalized_cdf, n_ref_tokens).unsqueeze(2)  # [B, n-1, 1]
        normalized_cdf = normalized_cdf.unsqueeze(1)  # [B, 1, T-1]

        # 寻找最接近采样点的CDF索引
        expanded_ys = ys.expand(B, ys.size(1), ys.size(1))
        diff = expanded_ys - F.pad(normalized_cdf, (diff_tokens, 0))
        tokens_to_pick_ind = torch.argmin(torch.abs(diff), dim=2)  # [B, n-1]

        # 处理索引偏移
        tokens_to_pick_ind -= diff_tokens
        unique_indices = self.get_unique_indices(tokens_to_pick_ind, N - 1)[:, :n_tokens - 1]

        # 重组注意力矩阵和token
        attn_sorted = torch.gather(attn[:, :, 1:], 2,
                                   sorted_indices.unsqueeze(1).expand(-1, self.num_heads, -1, -1))
        attn_tmp = F.pad(attn_sorted, (0, 0, 0, 1))  # 末尾填充0
        raw_x_tmp = torch.gather(raw_x[:, 1:], 1, sorted_indices.unsqueeze(-1).expand(-1, -1, C))
        raw_x_tmp = F.pad(raw_x_tmp, (0, 0, 0, 1))  # 添加CLS

        # 根据唯一索引选择token
        attn_tmp = torch.gather(attn_tmp, 2, unique_indices.unsqueeze(1).expand(-1, self.num_heads, -1, N))
        raw_x_tmp = torch.gather(raw_x_tmp, 1, unique_indices.unsqueeze(2).expand(-1, -1, C))
        attn = torch.cat([attn[:, :, :1], attn_tmp], dim=2)  # 拼接CLS注意力
        selected_x = torch.cat([raw_x[:, :1], raw_x_tmp], dim=1)  # 拼接CLS token

        # 生成保留token的掩码（policy）
        policy = (unique_indices != (N - 1)).unsqueeze(-1).float()
        policy = F.pad(policy, (0, 0, 1, 0), value=1.0)  # CLS位置设为1
        sampler = torch.nonzero(policy)  # 记录非零位置索引

        return selected_x, attn, policy, sampler


    def forward(self, x, policy, sampler, n_tokens, raw_x, n_ref_tokens):
        B, N, C = x.shape
        # 动态调整目标token数量
        if n_tokens > N: n_tokens = N
        if n_tokens <= 1.0: n_tokens *= N  # 处理比例输入
        n_tokens = max(round(n_tokens), 8)  # 确保最小8个token

        # 生成QKV并应用当前policy
        qkv = self.qkv(x, policy, sampler).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 计算注意力（带policy mask）
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.softmax_with_policy(attn, policy)  # 带掩码的softmax

        # 重要性评分与逆采样
        sorted_scores, sorted_indices = self.score_assignment_step(attn, v)
        selected_x, attn, policy, sampler = self.inverse_transform_sampling(
            sorted_scores, sorted_indices, attn, n_tokens, raw_x, n_ref_tokens)

        # 注意力加权求和
        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)

        # 可选token剪枝（减少计算量）
        if self.drop_tokens:
            out_mask_size = policy.sum(1).max().int()
            # 使用scatter/gather高效处理稀疏数据
            sampler_input = sampler.unsqueeze(-1).expand(-1, C)
            flatten_x = x.reshape(-1, C)
            x_pruned = torch.gather(flatten_x, 0, sampler_input)
            # 用零掩码重构输出
            out_zero_mask = self.out_zero_mask.expand(B * out_mask_size, -1)
            x = out_zero_mask.scatter(0, sampler_output, x_pruned).view(B, out_mask_size, C)

        # 投影层与dropout
        x = self.proj(x, policy, sampler)
        x = self.proj_drop(x) * policy
        return x, selected_x, policy, sampler


class ATSBlock(nn.Module):
    """
    Transformer块+自适应token采样
    包含：LayerNorm -> AdaptiveTokenSampler -> MLP
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, drop_tokens=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = AdaptiveTokenSampler(
            dim, num_heads, qkv_bias, qk_scale, attn_drop, drop,
            drop_path, drop_tokens)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x, n_tokens, policy=None, sampler=None, n_ref_tokens=197):
        # 注意力层
        x_attn, selected_x, policy, sampler = self.attn(
            self.norm1(x), policy, sampler, n_tokens, x, n_ref_tokens)
        x = selected_x + self.drop_path(x_attn)  # 残差连接
        x = x * policy  # 应用保留掩码

        # MLP层
        x_mlp = self.mlp(self.norm2(x), policy, sampler)
        x = x + self.drop_path(x_mlp)
        x = x * policy
        return x, policy