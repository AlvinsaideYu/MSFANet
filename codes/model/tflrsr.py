# ############
# # Memory consumption: 5.53 MB
# # Number of parameters: 1.4455M


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision.models import vgg19
# from torch.cuda.amp import autocast, GradScaler
# from torch.optim.lr_scheduler import CosineAnnealingLR
# import sys

# # Utility convolution function
# def default_conv(in_channels, out_channels, kernel_size, bias=True):
#     return nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2, bias=bias)

# # Improved Window-based Attention
# class WindowAttention(nn.Module):
#     def __init__(self, dim, window_size, num_heads):
#         super().__init__()
#         self.num_heads = num_heads
#         self.window_size = window_size
#         self.scale = (dim // num_heads) ** -0.5
        
#         self.qkv = nn.Linear(dim, dim * 3)
#         self.proj = nn.Linear(dim, dim)

#     def forward(self, x):
#         B, H, W, C = x.shape
#         qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
#         q, k, v = qkv.unbind(0)
        
#         q = q.reshape(B, self.num_heads, H * W // self.window_size**2, self.window_size**2, -1)
#         k = k.reshape(B, self.num_heads, H * W // self.window_size**2, self.window_size**2, -1)
#         v = v.reshape(B, self.num_heads, H * W // self.window_size**2, self.window_size**2, -1)
        
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
        
#         x = (attn @ v).reshape(B, self.num_heads, H, W, C // self.num_heads)
#         x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
#         x = self.proj(x)
#         return x

# # Transformer Encoder with Improved Attention
# class TransformerEncoder(nn.Module):
#     def __init__(self, num_fea, args):
#         super().__init__()
#         self.window_size = 8
#         self.num_heads = 4
        
#         self.norm1 = nn.LayerNorm(num_fea)
#         self.attn = WindowAttention(num_fea, self.window_size, self.num_heads)
        
#         self.norm2 = nn.LayerNorm(num_fea)
#         self.mlp = nn.Sequential(
#             nn.Linear(num_fea, num_fea * 2),
#             nn.GELU(),
#             nn.Linear(num_fea * 2, num_fea)
#         )

#     def forward(self, x):
#         B, C, H, W = x.shape
#         x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        
#         # Attention
#         x_flat = self.norm1(x_flat)
#         x_flat = x_flat.view(B, H, W, C)
#         x_flat = self.attn(x_flat)
#         x_flat = x_flat.view(B, H * W, C)
        
#         # MLP
#         out = self.norm2(x_flat + self.mlp(x_flat))
        
#         return out.reshape(B, C, H, W)

# # Multi-Scale Feature Extractor
# class MultiScaleFeatureExtractor(nn.Module):
#     def __init__(self, num_fea):
#         super().__init__()
#         self.conv1 = nn.Conv2d(num_fea, num_fea, 3, 1, 1)
#         self.conv2 = nn.Conv2d(num_fea, num_fea, 5, 1, 2)
#         self.conv3 = nn.Conv2d(num_fea, num_fea, 7, 1, 3)
#         self.fuse = nn.Conv2d(num_fea * 3, num_fea, 1)

#     def forward(self, x):
#         x1 = self.conv1(x)
#         x2 = self.conv2(x)
#         x3 = self.conv3(x)
#         return self.fuse(torch.cat([x1, x2, x3], dim=1))

# # Edge Guided Module
# class EdgeGuidedModule(nn.Module):
#     def __init__(self, num_fea):
#         super().__init__()
#         self.edge_conv = nn.Conv2d(num_fea, 1, 3, 1, 1)
#         self.fusion_conv = nn.Conv2d(num_fea + 1, num_fea, 1)

#     def forward(self, x):
#         edge = self.edge_conv(x)
#         edge = torch.sigmoid(edge)
#         return self.fusion_conv(torch.cat([x, edge], dim=1))

# # Channel Attention Layer
# class CALayer(nn.Module):
#     def __init__(self, num_fea, reduction=8):
#         super(CALayer, self).__init__()
#         self.conv_du = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(num_fea, num_fea // reduction, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(num_fea // reduction, num_fea, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         return x * self.conv_du(x)


# # Enhanced Pixel Shuffle Upsample
# class EnhancedPixelShuffle(nn.Module):
#     def __init__(self, in_channels, out_channels, upscale_factor):
#         super().__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), 3, 1, 1)
#         self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
#         self.ca = CALayer(out_channels)

#     def forward(self, x):
#         x = self.pixel_shuffle(self.conv(x))
#         return self.ca(x)


# # Texture Enhancement Module
# class TextureEnhancementModule(nn.Module):
#     def __init__(self, num_fea):
#         super().__init__()
#         # Multi-scale convolution to capture different textures
#         self.multi_scale = MultiScaleFeatureExtractor(num_fea)
#         # Channel attention to focus on important channels
#         self.ca = CALayer(num_fea)
#         # Spatial attention to refine spatial features
#         self.sa = nn.Sequential(
#             nn.Conv2d(num_fea, num_fea, 3, 1, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         # Extract multi-scale features
#         x_multiscale = self.multi_scale(x)
#         # Apply channel attention
#         x_ca = self.ca(x_multiscale)
#         # Apply spatial attention
#         x_sa = self.sa(x_ca)
#         # Combine the features
#         return x_multiscale + x * x_sa


# #新加注意力机制
# class LSKA(nn.Module):
#     # Large-Separable-Kernel-Attention
#     # <url id="d06palaebk45b81qe5o0" type="url" status="parsed" title="GitHub - StevenLauHKHK/Large-Separable-Kernel-Attention" wc="2129">https://github.com/StevenLauHKHK/Large-Separable-Kernel-Attention/tree/main</url>
#     def __init__(self, dim, k_size=11):
#         super().__init__()

#         self.k_size = k_size

#         if k_size == 7:
#             self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1,1), padding=(0,(3-1)//2), groups=dim)
#             self.conv0v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1,1), padding=((3-1)//2,0), groups=dim)
#             self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1,1), padding=(0,2), groups=dim, dilation=2)
#             self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1,1), padding=(2,0), groups=dim, dilation=2)
#         elif k_size == 11:
#             self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1,1), padding=(0,(3-1)//2), groups=dim)
#             self.conv0v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1,1), padding=((3-1)//2,0), groups=dim)
#             self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,4), groups=dim, dilation=2)
#             self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=(4,0), groups=dim, dilation=2)
#         elif k_size == 23:
#             self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,(5-1)//2), groups=dim)
#             self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=((5-1)//2,0), groups=dim)
#             self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 7), stride=(1,1), padding=(0,9), groups=dim, dilation=3)
#             self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(7, 1), stride=(1,1), padding=(9,0), groups=dim, dilation=3)
#         elif k_size == 35:
#             self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,(5-1)//2), groups=dim)
#             self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=((5-1)//2,0), groups=dim)
#             self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 11), stride=(1,1), padding=(0,15), groups=dim, dilation=3)
#             self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(11, 1), stride=(1,1), padding=(15,0), groups=dim, dilation=3)
#         elif k_size == 41:
#             self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,(5-1)//2), groups=dim)
#             self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=((5-1)//2,0), groups=dim)
#             self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 13), stride=(1,1), padding=(0,18), groups=dim, dilation=3)
#             self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(13, 1), stride=(1,1), padding=(18,0), groups=dim, dilation=3)
#         elif k_size == 53:
#             self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,(5-1)//2), groups=dim)
#             self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=((5-1)//2,0), groups=dim)
#             self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 17), stride=(1,1), padding=(0,24), groups=dim, dilation=3)
#             self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(17, 1), stride=(1,1), padding=(24,0), groups=dim, dilation=3)

#         self.conv1 = nn.Conv2d(dim, dim, 1)


#     # LSKA类的forward方法
#     def forward(self, x):
#         u = x.clone()
#         attn = self.conv0h(x)
#         attn = self.conv0v(attn)
#         attn = self.conv_spatial_h(attn)
#         attn = self.conv_spatial_v(attn)
#         attn = self.conv1(attn)
#         return u + attn

# # Improved Transformer Encoder with Local Window Attention
# class TransformersLocalWindow(nn.Module):
#     def __init__(self, dim, args):
#         super().__init__()
#         self.window_size = 8
#         self.num_heads = 4
        
#         self.norm1 = nn.LayerNorm(dim)
#         self.lw_attn = LSKA(dim=dim)
        
#         self.norm2 = nn.LayerNorm(dim)
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, dim * 2),
#             nn.GELU(),
#             nn.Linear(dim * 2, dim)
#         )

    
#     def forward(self, x):
#         B, C, H, W = x.shape
#         x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        
#         # Local Window Attention
#         x_flat = self.norm1(x_flat)
#         x_flat = x_flat.view(B, H, W, C)
#         x_flat = self.lw_attn(x_flat.permute(0, 3, 1, 2))
#         x_flat = x_flat.reshape(B, H * W, C)  # 使用 reshape 替代 view
        
#         # MLP
#         out = self.norm2(x_flat + self.mlp(x_flat))
        
#         return out.reshape(B, C, H, W)
    
# # Enhanced Pixel Shuffle Upsample
# class EnhancedPixelShuffle(nn.Module):
#     def __init__(self, in_channels, out_channels, upscale_factor):
#         super().__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), 3, 1, 1)
#         self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
#         self.ca = CALayer(out_channels)

#     def forward(self, x):
#         x = self.pixel_shuffle(self.conv(x))
#         return self.ca(x)



# #Improved TFLRSR Model with Local Window Attention
# class TFLRSR(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         num_fea = 64
#         scale = args.scale[0] if isinstance(args.scale, (list, tuple)) else args.scale
#         self.scale = scale #store the scale
#         self.conv_in = default_conv(args.n_colors, num_fea, 3)

#         # Feature Extraction Modules
#         self.feature_extract1 = MultiScaleFeatureExtractor(num_fea)
#         self.feature_extract2 = MultiScaleFeatureExtractor(num_fea)

#         # Transformer Modules with Local Window Attention
#         self.transformer = nn.Sequential(*[TransformersLocalWindow(num_fea, args) for _ in range(2)])

#         # Edge Guided Module
#         self.edge_guide = EdgeGuidedModule(num_fea)

#         # Texture Enhancement Module
#         self.texture_enhance = TextureEnhancementModule(num_fea)

#         # Upsample Modules
#         self.upsample1 = EnhancedPixelShuffle(num_fea, num_fea, upscale_factor=2)
#         self.upsample2 = EnhancedPixelShuffle(num_fea, num_fea, upscale_factor=2)

#         # Output Module
#         self.conv_out = default_conv(num_fea, args.n_colors, 3)
#         self.upsample_final = nn.Upsample(scale_factor=float(scale), mode='bicubic', align_corners=False)

#     def forward(self, x):
#         residual = self.upsample_final(x)
#         b, c, h, w = residual.shape # Get the shape of the upsampled input
        
#         x = self.conv_in(x)
        
#         # Feature Extraction
#         feat1 = self.feature_extract1(x)
#         feat2 = self.feature_extract2(feat1)
        
#         # Transformer Processing with Local Window Attention
#         x = self.transformer(feat2)
        
#         # Edge Guidance
#         x = self.edge_guide(x)
        
#         # Texture Enhancement
#         x = self.texture_enhance(x)
        
#         # Upsampling
#         x = self.upsample1(x)
#         x = x[:, :, :feat1.size(2), :feat1.size(3)]  # 裁剪上采样输出
#         x = x + feat1
#         x = self.upsample2(x)
#         x = x[:, :, :feat2.size(2), :feat2.size(3)]  # 裁剪上采样输出
#         x = x + feat2
        
#         x = self.conv_out(x)
#         # Resize the output to match the residual
#         x = F.interpolate(x, size=(h, w), mode='bicubic', align_corners=False)

#         return x + residual

# # Make Model Function
# def make_model(args, parent=False):
#     return TFLRSR(args)


# if __name__ == '__main__':
#     class Args:
#         n_colors = 3
#         scale = [4]
#         patch_size = 64

#     args = Args()
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     net = TFLRSR(args).to(device)
#     input_tensor = torch.rand(1, 3, 64, 64).to(device)
#     output = net(input_tensor)
#     print(output.shape)


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision.models import vgg19
# from torch.cuda.amp import autocast, GradScaler
# from torch.optim.lr_scheduler import CosineAnnealingLR
# import sys

# # Utility convolution function
# def default_conv(in_channels, out_channels, kernel_size, bias=True):
#     return nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2, bias=bias)

# # Improved Window-based Attention
# class WindowAttention(nn.Module):
#     def __init__(self, dim, window_size, num_heads):
#         super().__init__()
#         self.num_heads = num_heads
#         self.window_size = window_size
#         self.scale = (dim // num_heads) ** -0.5
        
#         self.qkv = nn.Linear(dim, dim * 3)
#         self.proj = nn.Linear(dim, dim)

#     def forward(self, x):
#         B, H, W, C = x.shape
#         qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
#         q, k, v = qkv.unbind(0)
        
#         q = q.reshape(B, self.num_heads, H * W // self.window_size**2, self.window_size**2, -1)
#         k = k.reshape(B, self.num_heads, H * W // self.window_size**2, self.window_size**2, -1)
#         v = v.reshape(B, self.num_heads, H * W // self.window_size**2, self.window_size**2, -1)
        
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
        
#         x = (attn @ v).reshape(B, self.num_heads, H, W, C // self.num_heads)
#         x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
#         x = self.proj(x)
#         return x

# # Transformer Encoder with Improved Attention
# class TransformerEncoder(nn.Module):
#     def __init__(self, num_fea, args):
#         super().__init__()
#         self.window_size = 8
#         self.num_heads = 6
        
#         self.norm1 = nn.LayerNorm(num_fea)
#         self.attn = WindowAttention(num_fea, self.window_size, self.num_heads)
        
#         self.norm2 = nn.LayerNorm(num_fea)
#         self.mlp = nn.Sequential(
#             nn.Linear(num_fea, num_fea * 2),
#             nn.GELU(),
#             nn.Linear(num_fea * 2, num_fea)
#         )

#     def forward(self, x):
#         B, C, H, W = x.shape
#         x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        
#         # Attention
#         x_flat = self.norm1(x_flat)
#         x_flat = x_flat.view(B, H, W, C)
#         x_flat = self.attn(x_flat)
#         x_flat = x_flat.view(B, H * W, C)
        
#         # MLP with residual
#         identity = x_flat
#         x_flat = self.mlp(x_flat)
#         x_flat = self.norm2(identity + x_flat)
        
#         return x_flat.reshape(B, C, H, W)

# # Multi-Scale Feature Extractor
# class MultiScaleFeatureExtractor(nn.Module):
#     def __init__(self, num_fea):
#         super().__init__()
#         self.conv1 = nn.Conv2d(num_fea, num_fea, 3, 1, 1)
#         self.conv2 = nn.Conv2d(num_fea, num_fea, 5, 1, 2)
#         self.conv3 = nn.Conv2d(num_fea, num_fea, 7, 1, 3)
#         self.fuse = nn.Conv2d(num_fea * 3, num_fea, 1)

#     def forward(self, x):
#         x1 = self.conv1(x)
#         x2 = self.conv2(x)
#         x3 = self.conv3(x)
#         return self.fuse(torch.cat([x1, x2, x3], dim=1))

# # Edge Guided Module
# class EdgeGuidedModule(nn.Module):
#     def __init__(self, num_fea):
#         super().__init__()
#         self.edge_conv = nn.Conv2d(num_fea, 1, 3, 1, 1)
#         self.fusion_conv = nn.Conv2d(num_fea + 1, num_fea, 1)

#     def forward(self, x):
#         edge = self.edge_conv(x)
#         edge = torch.sigmoid(edge)
#         return self.fusion_conv(torch.cat([x, edge], dim=1))

# # Channel Attention Layer
# class CALayer(nn.Module):
#     def __init__(self, num_fea, reduction=8):
#         super(CALayer, self).__init__()
#         self.conv_du = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(num_fea, num_fea // reduction, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(num_fea // reduction, num_fea, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         return x * self.conv_du(x)


# # Enhanced Pixel Shuffle Upsample
# class EnhancedPixelShuffle(nn.Module):
#     def __init__(self, in_channels, out_channels, upscale_factor):
#         super().__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), 3, 1, 1)
#         self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
#         self.ca = CALayer(out_channels)

#     def forward(self, x):
#         x = self.pixel_shuffle(self.conv(x))
#         return self.ca(x)


# # Texture Enhancement Module
# class TextureEnhancementModule(nn.Module):
#     def __init__(self, num_fea):
#         super().__init__()
#         # Multi-scale convolution to capture different textures
#         self.multi_scale = MultiScaleFeatureExtractor(num_fea)
#         # Channel attention to focus on important channels
#         self.ca = CALayer(num_fea)
#         # Spatial attention to refine spatial features
#         self.sa = nn.Sequential(
#             nn.Conv2d(num_fea, num_fea, 3, 1, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         # Extract multi-scale features
#         x_multiscale = self.multi_scale(x)
#         # Apply channel attention
#         x_ca = self.ca(x_multiscale)
#         # Apply spatial attention
#         x_sa = self.sa(x_ca)
#         # Combine the features
#         return x_multiscale + x * x_sa


# #新加注意力机制
# class LSKA(nn.Module):
#     # Large-Separable-Kernel-Attention
#     def __init__(self, dim, k_size=11):
#         super().__init__()

#         self.k_size = k_size

#         if k_size == 7:
#             self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1,1), padding=(0,(3-1)//2), groups=dim)
#             self.conv0v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1,1), padding=((3-1)//2,0), groups=dim)
#             self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1,1), padding=(0,2), groups=dim, dilation=2)
#             self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1,1), padding=(2,0), groups=dim, dilation=2)
#         elif k_size == 11:
#             self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1,1), padding=(0,(3-1)//2), groups=dim)
#             self.conv0v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1,1), padding=((3-1)//2,0), groups=dim)
#             self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,4), groups=dim, dilation=2)
#             self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=(4,0), groups=dim, dilation=2)
#         elif k_size == 23:
#             self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,(5-1)//2), groups=dim)
#             self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=((5-1)//2,0), groups=dim)
#             self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 7), stride=(1,1), padding=(0,9), groups=dim, dilation=3)
#             self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(7, 1), stride=(1,1), padding=(9,0), groups=dim, dilation=3)
#         elif k_size == 35:
#             self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,(5-1)//2), groups=dim)
#             self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=((5-1)//2,0), groups=dim)
#             self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 11), stride=(1,1), padding=(0,15), groups=dim, dilation=3)
#             self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(11, 1), stride=(1,1), padding=(15,0), groups=dim, dilation=3)
#         elif k_size == 41:
#             self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,(5-1)//2), groups=dim)
#             self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=((5-1)//2,0), groups=dim)
#             self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 13), stride=(1,1), padding=(0,18), groups=dim, dilation=3)
#             self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(13, 1), stride=(1,1), padding=(18,0), groups=dim, dilation=3)
#         elif k_size == 53:
#             self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,(5-1)//2), groups=dim)
#             self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=((5-1)//2,0), groups=dim)
#             self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 17), stride=(1,1), padding=(0,24), groups=dim, dilation=3)
#             self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(17, 1), stride=(1,1), padding=(24,0), groups=dim, dilation=3)

#         self.conv1 = nn.Conv2d(dim, dim, 1)


#     # LSKA类的forward方法
#     def forward(self, x):
#         u = x.clone()
#         attn = self.conv0h(x)
#         attn = self.conv0v(attn)
#         attn = self.conv_spatial_h(attn)
#         attn = self.conv_spatial_v(attn)
#         attn = self.conv1(attn)
#         return u + attn

# # Improved Transformer Encoder with Local Window Attention
# class TransformersLocalWindow(nn.Module):
#     def __init__(self, dim, args):
#         super().__init__()
#         self.window_size = 8
#         self.num_heads = 6
        
#         self.norm1 = nn.LayerNorm(dim)
#         self.lw_attn = LSKA(dim=dim)
        
#         self.norm2 = nn.LayerNorm(dim)
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, dim * 2),
#             nn.GELU(),
#             nn.Linear(dim * 2, dim)
#         )

    
#     def forward(self, x):
#         B, C, H, W = x.shape
#         x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        
#         # Local Window Attention
#         x_flat = self.norm1(x_flat)
#         x_flat = x_flat.view(B, H, W, C)
#         x_flat = self.lw_attn(x_flat.permute(0, 3, 1, 2))
#         x_flat = x_flat.reshape(B, H * W, C)  # 使用 reshape 替代 view
        
#         # MLP with residual
#         identity = x_flat
#         x_flat = self.mlp(x_flat)
#         x_flat = self.norm2(identity + x_flat)
        
#         return x_flat.reshape(B, C, H, W)
    
# # Enhanced Pixel Shuffle Upsample
# class EnhancedPixelShuffle(nn.Module):
#     def __init__(self, in_channels, out_channels, upscale_factor):
#         super().__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), 3, 1, 1)
#         self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
#         self.ca = CALayer(out_channels)

#     def forward(self, x):
#         x = self.pixel_shuffle(self.conv(x))
#         return self.ca(x)

# # Improved TFLRSR Model with Local Window Attention and Enhanced Features
# class EnhancedTFLRSR(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         num_fea = 96  # 增加通道数
#         scale = args.scale[0] if isinstance(args.scale, (list, tuple)) else args.scale
#         self.scale = scale #存储缩放比例
#         self.conv_in = default_conv(args.n_colors, num_fea, 3)

#         # Feature Extraction Modules with LSKA
#         self.feature_extract1 = nn.Sequential(
#             MultiScaleFeatureExtractor(num_fea),
#             LSKA(num_fea)
#         )
#         self.feature_extract2 = nn.Sequential(
#             MultiScaleFeatureExtractor(num_fea),
#             LSKA(num_fea)
#         )

#         # Transformer Modules with Local Window Attention (增加层数)
#         self.transformer = nn.Sequential(*[TransformersLocalWindow(num_fea, args) for _ in range(4)])

#         # Edge Guided Module
#         self.edge_guide = EdgeGuidedModule(num_fea)

#         # Texture Enhancement Module with LSKA
#         self.texture_enhance = nn.Sequential(
#             TextureEnhancementModule(num_fea),
#             LSKA(num_fea)
#         )

#         # Upsample Modules with LSKA
#         self.upsample1 = nn.Sequential(
#             EnhancedPixelShuffle(num_fea, num_fea, upscale_factor=2),
#             LSKA(num_fea)
#         )
#         self.upsample2 = nn.Sequential(
#             EnhancedPixelShuffle(num_fea, num_fea, upscale_factor=2),
#             LSKA(num_fea)
#         )

#         # Output Module
#         self.conv_out = default_conv(num_fea, args.n_colors, 3)
#         self.upsample_final = nn.Upsample(scale_factor=float(scale), mode='bicubic', align_corners=False)

#     def forward(self, x):
#         residual = self.upsample_final(x)
#         b, c, h, w = residual.shape # 获取上采样后输入的形状
        
#         x = self.conv_in(x)
        
#         # Feature Extraction
#         feat1 = self.feature_extract1(x)
#         feat2 = self.feature_extract2(feat1)
        
#         # Transformer Processing with Local Window Attention
#         x = self.transformer(feat2)
        
#         # Edge Guidance
#         x = self.edge_guide(x)
        
#         # Texture Enhancement
#         x = self.texture_enhance(x)
        
#         # Upsampling
#         x = self.upsample1(x)
#         x = x[:, :, :feat1.size(2), :feat1.size(3)]  # 裁剪上采样输出
#         x = x + feat1
#         x = self.upsample2(x)
#         x = x[:, :, :feat2.size(2), :feat2.size(3)]  # 裁剪上采样输出
#         x = x + feat2
        
#         x = self.conv_out(x)
#         # 调整输出大小以匹配残差
#         x = F.interpolate(x, size=(h, w), mode='bicubic', align_corners=False)

#         return x + residual

# # Make Model Function
# def make_model(args, parent=False):
#     return EnhancedTFLRSR(args)


# if __name__ == '__main__':
#     class Args:
#         n_colors = 3
#         scale = [4]
#         patch_size = 64

#     args = Args()
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     net = EnhancedTFLRSR(args).to(device)
#     input_tensor = torch.rand(1, 3, 64, 64).to(device)
#     output = net(input_tensor)
#     print(output.shape)



import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
import sys

# Utility convolution function
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2, bias=bias)

# Improved Window-based Attention
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, H, W, C = x.shape
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv.unbind(0)
        
        q = q.reshape(B, self.num_heads, H * W // self.window_size**2, self.window_size**2, -1)
        k = k.reshape(B, self.num_heads, H * W // self.window_size**2, self.window_size**2, -1)
        v = v.reshape(B, self.num_heads, H * W // self.window_size**2, self.window_size**2, -1)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).reshape(B, self.num_heads, H, W, C // self.num_heads)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        x = self.proj(x)
        return x

# Transformer Encoder with Improved Attention
class TransformerEncoder(nn.Module):
    def __init__(self, num_fea, args):
        super().__init__()
        self.window_size = 8
        self.num_heads = 6
        
        self.norm1 = nn.LayerNorm(num_fea)
        self.attn = WindowAttention(num_fea, self.window_size, self.num_heads)
        
        self.norm2 = nn.LayerNorm(num_fea)
        self.mlp = nn.Sequential(
            nn.Linear(num_fea, num_fea * 2),
            nn.GELU(),
            nn.Linear(num_fea * 2, num_fea)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        
        # Attention
        x_flat = self.norm1(x_flat)
        x_flat = x_flat.view(B, H, W, C)
        x_flat = self.attn(x_flat)
        x_flat = x_flat.view(B, H * W, C)
        
        # MLP with residual
        identity = x_flat
        x_flat = self.mlp(x_flat)
        x_flat = self.norm2(identity + x_flat)
        
        return x_flat.reshape(B, C, H, W)

# Multi-Scale Feature Extractor
class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, num_fea):
        super().__init__()
        self.conv1 = nn.Conv2d(num_fea, num_fea, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_fea, num_fea, 5, 1, 2)
        self.conv3 = nn.Conv2d(num_fea, num_fea, 7, 1, 3)
        self.fuse = nn.Conv2d(num_fea * 3, num_fea, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        return self.fuse(torch.cat([x1, x2, x3], dim=1))

# Edge Guided Module
class EdgeGuidedModule(nn.Module):
    def __init__(self, num_fea):
        super().__init__()
        self.edge_conv = nn.Conv2d(num_fea, 1, 3, 1, 1)
        self.fusion_conv = nn.Conv2d(num_fea + 1, num_fea, 1)

    def forward(self, x):
        edge = self.edge_conv(x)
        edge = torch.sigmoid(edge)
        return self.fusion_conv(torch.cat([x, edge], dim=1))

# Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, num_fea, reduction=8):
        super(CALayer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_fea, num_fea // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_fea // reduction, num_fea, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.conv_du(x)

# Enhanced Pixel Shuffle Upsample
class EnhancedPixelShuffle(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.ca = CALayer(out_channels)

    def forward(self, x):
        x = self.pixel_shuffle(self.conv(x))
        return self.ca(x)

# Texture Enhancement Module
class TextureEnhancementModule(nn.Module):
    def __init__(self, num_fea):
        super().__init__()
        # Multi-scale convolution to capture different textures
        self.multi_scale = MultiScaleFeatureExtractor(num_fea)
        # Channel attention to focus on important channels
        self.ca = CALayer(num_fea)
        # Spatial attention to refine spatial features
        self.sa = nn.Sequential(
            nn.Conv2d(num_fea, num_fea, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Extract multi-scale features
        x_multiscale = self.multi_scale(x)
        # Apply channel attention
        x_ca = self.ca(x_multiscale)
        # Apply spatial attention
        x_sa = self.sa(x_ca)
        # Combine the features
        return x_multiscale + x * x_sa

# Large-Separable-Kernel-Attention
class LSKA(nn.Module):
    def __init__(self, dim, k_size=11):
        super().__init__()
        self.k_size = k_size

        if k_size == 7:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1,1), padding=(0,1), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1,1), padding=(1,0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1,1), padding=(0,2), groups=dim, dilation=2)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1,1), padding=(2,0), groups=dim, dilation=2)
        elif k_size == 11:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1,1), padding=(0,1), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1,1), padding=(1,0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,4), groups=dim, dilation=2)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=(4,0), groups=dim, dilation=2)
        elif k_size == 23:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=(2,0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 7), stride=(1,1), padding=(0,9), groups=dim, dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(7, 1), stride=(1,1), padding=(9,0), groups=dim, dilation=3)
        elif k_size == 35:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=(2,0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 11), stride=(1,1), padding=(0,15), groups=dim, dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(11, 1), stride=(1,1), padding=(15,0), groups=dim, dilation=3)
        elif k_size == 41:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=(2,0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 13), stride=(1,1), padding=(0,18), groups=dim, dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(13, 1), stride=(1,1), padding=(18,0), groups=dim, dilation=3)
        elif k_size == 53:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=(2,0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 17), stride=(1,1), padding=(0,24), groups=dim, dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(17, 1), stride=(1,1), padding=(24,0), groups=dim, dilation=3)

        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0h(x)
        attn = self.conv0v(attn)
        attn = self.conv_spatial_h(attn)
        attn = self.conv_spatial_v(attn)
        attn = self.conv1(attn)
        return u + attn

# Improved Transformer Encoder with Local Window Attention
class TransformersLocalWindow(nn.Module):
    def __init__(self, dim, args):
        super().__init__()
        self.window_size = 8
        self.num_heads = 6
        
        self.norm1 = nn.LayerNorm(dim)
        self.lw_attn = LSKA(dim=dim)
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        
        # Local Window Attention
        x_flat = self.norm1(x_flat)
        x_flat = x_flat.view(B, H, W, C)
        x_flat = self.lw_attn(x_flat.permute(0, 3, 1, 2))
        x_flat = x_flat.reshape(B, H * W, C)
        
        # MLP with residual
        identity = x_flat
        x_flat = self.mlp(x_flat)
        x_flat = self.norm2(identity + x_flat)
        
        return x_flat.reshape(B, C, H, W)

# Enhanced TFLRSR Model with Local Window Attention and Enhanced Features
class EnhancedTFLRSR(nn.Module):
    def __init__(self, args):
        super().__init__()
        num_fea = 128  # 增加特征通道数
        scale = args.scale[0] if isinstance(args.scale, (list, tuple)) else args.scale
        self.scale = scale
        self.conv_in = default_conv(args.n_colors, num_fea, 3)

        # Feature Extraction Modules with LSKA
        self.feature_extract1 = nn.Sequential(
            MultiScaleFeatureExtractor(num_fea),
            LSKA(num_fea)
        )
        self.feature_extract2 = nn.Sequential(
            MultiScaleFeatureExtractor(num_fea),
            LSKA(num_fea)
        )

        # Transformer Modules with Local Window Attention
        self.transformer = nn.Sequential(*[TransformersLocalWindow(num_fea, args) for _ in range(6)])  # 增加Transformer层数

        # Edge Guided Module
        self.edge_guide = EdgeGuidedModule(num_fea)

        # Texture Enhancement Module with LSKA
        self.texture_enhance = nn.Sequential(
            TextureEnhancementModule(num_fea),
            LSKA(num_fea)
        )

        # Upsample Modules with LSKA
        self.upsample1 = nn.Sequential(
            EnhancedPixelShuffle(num_fea, num_fea, upscale_factor=2),
            LSKA(num_fea)
        )
        self.upsample2 = nn.Sequential(
            EnhancedPixelShuffle(num_fea, num_fea, upscale_factor=2),
            LSKA(num_fea)
        )

        # Output Module
        self.conv_out = default_conv(num_fea, args.n_colors, 3)
        self.upsample_final = nn.Upsample(scale_factor=float(scale), mode='bicubic', align_corners=False)

    def forward(self, x):
        residual = self.upsample_final(x)
        b, c, h, w = residual.shape

        x = self.conv_in(x)
        
        # Feature Extraction
        feat1 = self.feature_extract1(x)
        feat2 = self.feature_extract2(feat1)
        
        # Transformer Processing with Local Window Attention
        x = self.transformer(feat2)
        
        # Edge Guidance
        x = self.edge_guide(x)
        
        # Texture Enhancement
        x = self.texture_enhance(x)
        
        # Upsampling
        x = self.upsample1(x)
        x = x[:, :, :feat1.size(2), :feat1.size(3)]  # 裁剪上采样输出
        x = x + feat1
        x = self.upsample2(x)
        x = x[:, :, :feat2.size(2), :feat2.size(3)]  # 裁剪上采样输出
        x = x + feat2
        
        x = self.conv_out(x)
        x = F.interpolate(x, size=(h, w), mode='bicubic', align_corners=False)

        return x + residual

# 训练代码
def train_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnhancedTFLRSR(args).to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler()

    # 模拟数据加载
    for epoch in range(args.epochs):
        model.train()
        for _ in range(300):  # 模拟100个批次
            with autocast():
                input_tensor = torch.rand(16, 3, 48, 48).to(device)  # LR图像
                target_tensor = torch.rand(16, 3, 192, 192).to(device)  # HR图像
                output = model(input_tensor)
                loss = criterion(output, target_tensor)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        scheduler.step()
        
        # 验证和保存模型
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_input = torch.rand(1, 3, 64, 64).to(device)
                val_output = model(val_input)
                print(f"Epoch {epoch}, Val Output Shape: {val_output.shape}")
            torch.save(model.state_dict(), f"model_epoch_{epoch}.pth")
            print(f"Epoch {epoch} completed, model saved")

# Make Model Function
def make_model(args, parent=False):
    return EnhancedTFLRSR(args)


if __name__ == '__main__':
    class Args:
        n_colors = 3
        scale = [4]
        patch_size = 64

    args = Args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = EnhancedTFLRSR(args).to(device)
    input_tensor = torch.rand(1, 3, 64, 64).to(device)
    output = net(input_tensor)
    print(output.shape)