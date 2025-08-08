import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F

class MultiHeadChannelAttention(nn.Module):
    """多头通道注意力机制"""
    def __init__(self, in_channels, num_heads=8):
        super(MultiHeadChannelAttention, self).__init__()
        self.num_heads = num_heads
        self.in_channels = in_channels
        
        # 确保通道数能被头数整除
        assert in_channels % num_heads == 0
        self.head_dim = in_channels // num_heads
        
        # 多个注意力头的线性层
        self.query = nn.Linear(in_channels, in_channels)
        self.key = nn.Linear(in_channels, in_channels)
        self.value = nn.Linear(in_channels, in_channels)
        
        # 输出投影层
        self.out_proj = nn.Linear(in_channels, in_channels)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x: [batch, channel, 1, 1]
        b, c, _, _ = x.size()
        x_flat = x.view(b, c)  # [batch, channel]
        
        # 生成 Q, K, V
        Q = self.query(x_flat)  # [batch, channel]
        K = self.key(x_flat)    # [batch, channel]
        V = self.value(x_flat)  # [batch, channel]
        
        # 重塑为多头形式: [batch, num_heads, head_dim]
        Q = Q.view(b, self.num_heads, self.head_dim)
        K = K.view(b, self.num_heads, self.head_dim)
        V = V.view(b, self.num_heads, self.head_dim)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [batch, num_heads, head_dim]
        
        # 对每个头独立计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)  # [batch, num_heads, head_dim]
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        attended = attn_weights * V  # [batch, num_heads, head_dim]
        
        # 重新组合多头输出
        attended = attended.view(b, c)  # [batch, channel]
        
        # 输出投影
        output = self.out_proj(attended)  # [batch, channel]
        
        return output

class AudioClassifier(nn.Module):
    """音频分类器 - 用于AI音乐检测"""
    def __init__(self):
        super(AudioClassifier, self).__init__()
        
        conv_layers = []

        self.conv1 = nn.Conv2d(1, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        self.conv = nn.Sequential(*conv_layers)

        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        # 使用多头注意力机制，64个通道，8个头
        self.attn = MultiHeadChannelAttention(in_channels=64, num_heads=8)
        self.lin = nn.Linear(in_features=64, out_features=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.ap(x)  # [batch, 64, 1, 1]
        x = self.attn(x)  # [batch, 64]
        x = self.lin(x)
        return x