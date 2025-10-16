from model import common
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.cuda.amp as amp  # 引入混合精度训练模块

def make_model(args, parent=False):
    return SFRMamba(args)

class OSAG(nn.Module):
    def __init__(self, channel_num, window_size):
        super(OSAG, self).__init__()
        self.conv1 = nn.Conv2d(channel_num, channel_num, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channel_num, channel_num, kernel_size=3, stride=1, padding=1, bias=True)
        self.window_size = window_size

    def forward(self, x):
        _, _, h, w = x.size()
        pad_h = (self.window_size - h % self.window_size) % self.window_size
        pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, pad_w, 0, pad_h), 'constant', 0)
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return out

class pixelshuffle_block(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor, bias=True):
        super(pixelshuffle_block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), kernel_size=3, stride=1, padding=1, bias=bias)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        return self.pixel_shuffle(self.conv(x))

class FENetModule(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(FENetModule, self).__init__()

        n_feats = 128  # 增加特征通道
        kernel_size = 3

        self.head = nn.Sequential(
            nn.Conv2d(args.n_colors, n_feats, kernel_size, padding=kernel_size//2, bias=True)
        )

        self.body = nn.Sequential(
            *[common.BasicBlock(conv, n_feats, n_feats, kernel_size, bias=True, bn=True) for _ in range(30)]  # 增加层数
        )

        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, args.n_colors, kernel_size, padding=kernel_size//2, bias=True)
        )

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x

class SRDDModule(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(SRDDModule, self).__init__()

        n_feats = 128  # 增加特征通道
        kernel_size = 3

        self.head = nn.Sequential(
            nn.Conv2d(args.n_colors, n_feats, kernel_size, padding=kernel_size//2, bias=True)
        )

        res_scale = getattr(args, 'res_scale', 1.0)

        self.body = nn.Sequential(
            *[common.ResBlock(conv, n_feats, kernel_size, act=nn.ReLU(True), res_scale=res_scale) for _ in range(30)]  # 增加层数
        )

        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, args.n_colors, kernel_size, padding=kernel_size//2, bias=True)
        )

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x

class AttentionModule(nn.Module):
    def __init__(self, channels):
        super(AttentionModule, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.conv(x)
        attn = self.sigmoid(attn)
        return x * attn

class RCAB(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, reduction=16, bias=True, act=nn.ReLU(True), res_scale=1):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if i == 0: modules_body.append(act)
        self.body = nn.Sequential(*modules_body)

        # Channel Attention (CA) Layer
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(n_feats, n_feats // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats // reduction, n_feats, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res = res * self.ca(res)
        res += x
        return res

class MultiScaleBlock(nn.Module):
    def __init__(self, in_channels):
        super(MultiScaleBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out = out1 + out2 + out3
        out = self.relu(out)
        return out

class RDNModule(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(RDNModule, self).__init__()
        n_feats = 128  # 增加特征通道
        kernel_size = 3

        self.head = nn.Sequential(
            nn.Conv2d(args.n_colors, n_feats, kernel_size, padding=kernel_size//2, bias=True)
        )

        self.body = nn.Sequential(
            *[RCAB(conv, n_feats, kernel_size, act=nn.ReLU(True), res_scale=1.0) for _ in range(30)]  # 增加层数
        )

        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, args.n_colors, kernel_size, padding=kernel_size//2, bias=True)
        )

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x

class SFRMamba(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(SFRMamba, self).__init__()

        self.fenet = FENetModule(args, conv)
        self.srdd = SRDDModule(args, conv)
        self.rdn = RDNModule(args, conv)
        self.attention = AttentionModule(128)  # 更新通道数
        self.conv_in = nn.Conv2d(args.n_colors, 128, kernel_size=1)  # 更新通道数
        self.conv_out = nn.Conv2d(128, args.n_colors, kernel_size=1)  # 更新通道数
        self.multi_scale = MultiScaleBlock(128)

        scale = args.scale if isinstance(args.scale, (int, float)) else args.scale[0]
        self.upsample = nn.Upsample(scale_factor=float(scale), mode='bicubic', align_corners=False)

    def forward(self, x):
        residual = self.upsample(x)

        # FENet output
        fenet_out = self.fenet(x)
        fenet_out = nn.functional.interpolate(fenet_out, size=(residual.size(2), residual.size(3)), mode='bicubic', align_corners=False)

        # SRDD output
        srdd_out = self.srdd(x)
        srdd_out = nn.functional.interpolate(srdd_out, size=(residual.size(2), residual.size(3)), mode='bicubic', align_corners=False)

        # RDN output
        rdn_out = self.rdn(x)
        rdn_out = nn.functional.interpolate(rdn_out, size=(residual.size(2), residual.size(3)), mode='bicubic', align_corners=False)

        # Combine outputs and apply attention
        combined_out = fenet_out + srdd_out + rdn_out
        combined_out = self.conv_in(combined_out)  # Convert to 128 channels
        combined_out = self.multi_scale(combined_out)  # Apply multi-scale block
        combined_out = self.attention(combined_out)
        combined_out = self.conv_out(combined_out)  # Convert back to original number of channels

        out = combined_out + residual

        return out

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = torchvision.models.vgg19(pretrained=True).features
        self.loss_network = nn.Sequential(*list(vgg)[:36]).eval()
        for param in self.loss_network.parameters():
            param.requires_grad = False
        self.mse_loss = nn.MSELoss()

    def forward(self, high_resolution, fake_high_resolution):
        perception_loss = self.mse_loss(self.loss_network(high_resolution), self.loss_network(fake_high_resolution))
        return perception_loss

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0.
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

def count_parameters(net):
    params = list(net.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    print("total parameters:" + str(k))

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Total number of parameters: %d' % num_params)

if __name__ == '__main__':
    import psutil
    import time
    from option import args
    net = SFRMamba(args).cuda()
    perceptual_loss = PerceptualLoss().cuda()
    l1_loss = nn.L1Loss().cuda()

    from thop import profile

    torch.cuda.reset_max_memory_allocated()
    x = torch.rand(1, 3, 64 * 4, 64 * 4).cuda()
    y = net(x)
    max_memory_reserved = torch.cuda.max_memory_reserved(device='cuda') / (1024 ** 2)

    print(f"模型最大内存消耗: {max_memory_reserved:.2f} MB")
    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameters: %.5fM" % (total / 1e6))
    timer = Timer()
    timer.tic()
    for i in range(100):
        timer.tic()
        y = net(x)
        timer.toc()

    print('Do once forward need {:.3f}ms '.format(timer.total_time * 1000 / 100.0))
    flops, params = profile(net, (x,))
    print('flops: %.4f G, params: %.4f M' % (flops / 1e9, params / 1000000.0))

    # 示例训练循环
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    scaler = amp.GradScaler()  # 使用混合精度训练

    for epoch in range(100):  # 假设训练1000个epoch
        for i in range(10):  # 假设每个epoch有100个批次
            # 示例数据生成
            inputs = torch.rand(1, 3, 64, 64).cuda()
            targets = torch.rand(1, 3, 256, 256).cuda()

            optimizer.zero_grad()
            with amp.autocast():  # 开启混合精度训练
                outputs = net(inputs)
                loss = l1_loss(outputs, targets) + 0.006 * perceptual_loss(outputs, targets)
            scaler.scale(loss).backward()  # 使用混合精度缩放梯度
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
