import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import Softmax

class DynamicSelect(nn.Module):
    def __init__(self, config, fix=True):
        super(DynamicSelect, self).__init__()
        self.fix = fix
        self.num_heads = config.transformer["num_heads"]
        self.ff_select = Part_Attention()
        self.vote_num = config.vote_num

        if self.fix:
            self.kernel = torch.tensor([[1, 1, 1],
                                        [1, 4, 1],
                                        [1, 1, 1]], device='cuda').unsqueeze(0).unsqueeze(0).half()
            self.conv = F.conv2d
        else:
            self.conv = nn.Conv2d(1, 1, 3, 1, 1)

    def forward(self, attn_weights, hidden_states, select_num=None, last=False):
        # (32, 784)
        B, patch_num = attn_weights[0].shape[0], attn_weights[0].shape[3] - 1

        # (32, 784)
        count_sum = torch.zeros((B, patch_num), dtype=torch.int, device='cuda').half()

        layer_selected_tokens = [[] for i in range(B)]
        layer_selected_inx = [[] for _ in range(B)]
        selected_inx_all = None

        for i, weight in enumerate(attn_weights):
            max_inx = self.ff_select(weight)
            for b in range(B):
                selected_inx = max_inx[:, :select_num[b][i]]
                layer_selected_inx[b].append(selected_inx[b])
                layer_selected_tokens[b].extend(hidden_states[i][b, selected_inx[b, :]])
            if i == 0:
                selected_inx_all = selected_inx
            else:
                selected_inx_all = torch.cat((selected_inx_all, selected_inx), dim=-1)

        for i, b in enumerate(selected_inx_all):
            scores = torch.bincount(b, minlength=784)
            count_sum[i, :] += scores

        count = self.enhance_local(count_sum)

        # (32, 784) (32, 784)
        patch_value, patch_idx = torch.sort(count, dim=-1, descending=True)
        # 选取前select_num个token的索引
        last_select_idx = patch_idx[:, :self.vote_num]

        tokens = [torch.stack(token) for token in layer_selected_tokens]
        tokens = torch.stack(tokens).squeeze(1)

        # 每一层存活下来的数量
        alive_num = torch.zeros((B, 10))
        for b in range(B):
            for i, select_indices in enumerate(layer_selected_inx[b]):
                alive_num[b][i] = torch.sum(torch.isin(select_indices, last_select_idx[b]))

        return tokens, alive_num

    def enhance_local(self, count):
        # B:32 H:28
        B, H = count.shape[0], math.ceil(math.sqrt(count.shape[1]))
        # (32, 28, 28)
        count = count.reshape(B, H, H)
        if self.fix:
            # (32, 1, 28, 28) 卷积后：(32, 784)
            count = self.conv(count.unsqueeze(1), self.kernel, stride=1, padding=1).reshape(B, -1)
        else:
            count = self.conv(count.unsqueeze(1)).reshape(B, -1)
        return count

class Part_Attention(nn.Module):
    def __init__(self):
        super(Part_Attention, self).__init__()
        self.softmax = Softmax(dim=-1)

    def forward(self, x,):
        # x:(32, 12, 197, 197)

        # (32, 197) 取第一行并且对所有头进行平均得到 a0
        a0 = x[:, :, 0, :].mean(1)
        b0 = x[:, :, :, 0].mean(1)

        # (32, 197)
        scores = a0 * b0

        scores = self.softmax(scores)

        scores = scores[:,1:]

        # (32, 197) 返回的是按从大到小的值的索引排序
        max_inx = torch.argsort(scores, dim=1, descending=True)

        # max_inx: (32, 197)
        return  max_inx