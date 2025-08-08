import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.models as models

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
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [batch, num_heads, head_dim, head_dim]
        
        # 对每个头独立计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)  # [batch, num_heads, head_dim, head_dim]
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        attended = torch.matmul(attn_weights, V)  # [batch, num_heads, head_dim]
        
        # 重新组合多头输出
        attended = attended.transpose(1, 2).contiguous().view(b, c)  # [batch, channel]
        
        # 输出投影
        output = self.out_proj(attended)  # [batch, channel]
        output = output.view(b, c, 1, 1)  # 恢复形状 [batch, channel, 1, 1]
        
        return output
    

class ResNeXtBottleneck(nn.Module):
    '''ResNeXt Bottleneck Block - 用于音频分类器的基础模块'''
    def __init__(self, in_channels, out_channels, stride=1, cardinality=8, base_width=4):
        super(ResNeXtBottleneck, self).__init__()
        width_ratio = out_channels / (64.0)  # 和 ResNeXt 一致的宽度比例公式
        D = int(base_width * width_ratio) * cardinality
        
        self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1, bias=False)
        self.bn_reduce = nn.BatchNorm2d(D)
        
        self.conv_group = nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1, 
                                    groups=cardinality, bias=False)
        self.bn_group = nn.BatchNorm2d(D)
        
        self.conv_expand = nn.Conv2d(D, out_channels, kernel_size=1, bias=False)
        self.bn_expand = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn_reduce(self.conv_reduce(x)))
        out = torch.relu(self.bn_group(self.conv_group(out)))
        out = self.bn_expand(self.conv_expand(out))
        out += self.shortcut(x)
        return torch.relu(out)

class AudioClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(AudioClassifier, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Stage 1
        self.layer1 = self._make_layer(16, 32, num_blocks=1, stride=1)
        # Stage 2
        self.layer2 = self._make_layer(32, 64, num_blocks=1, stride=2)
        # Stage 3
        self.layer3 = self._make_layer(64, 128, num_blocks=1, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # 多头通道注意力
        self.channel_attn = MultiHeadChannelAttention(in_channels=128, num_heads=8)
        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(128, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResNeXtBottleneck(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResNeXtBottleneck(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # 全局池化
        x = self.avgpool(x)  # [B, 256, 1, 1]

        # 多头通道注意力
        attn_out = self.channel_attn(x)  # [B, 256, 1, 1]

        # 残差连接
        x = x + attn_out  # [B, 256, 1, 1]

        # 展平并分类
        x = x.view(x.size(0), -1)  # [B, 256]
        x = self.fc(x)
        return x

