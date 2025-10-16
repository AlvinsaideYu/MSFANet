from model import common
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_model(args, parent=False):
    return FSMamba(args)

class VDSRModule(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(VDSRModule, self).__init__()

        n_feats = 32  # 减少特征数量以降低显存占用
        kernel_size = 3

        self.head = nn.Sequential(
            nn.Conv2d(args.n_colors, n_feats, kernel_size, padding=kernel_size//2, bias=True)
        )

        self.body = nn.Sequential(
            *[common.BasicBlock(conv, n_feats, n_feats, kernel_size, bias=True, bn=True) for _ in range(8)]  # 减少卷积层数
        )

        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size//2, bias=True)
        )

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x

class EDSRModule(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSRModule, self).__init__()

        n_feats = 32  # 减少特征数量以降低显存占用
        kernel_size = 3

        self.head = nn.Sequential(
            nn.Conv2d(args.n_colors, n_feats, kernel_size, padding=kernel_size//2, bias=True)
        )

        res_scale = getattr(args, 'res_scale', 1.0)

        self.body = nn.Sequential(
            *[common.ResBlock(conv, n_feats, kernel_size, act=nn.LeakyReLU(0.2, inplace=True), res_scale=res_scale) for _ in range(8)]  # 减少残差块数量
        )

        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size//2, bias=True)
        )

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x

class ESRGANModule(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(ESRGANModule, self).__init__()

        self.generator = nn.Sequential(
            nn.Conv2d(args.n_colors, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 32, kernel_size=3, stride=1, padding=1)  # 调整输出通道数以匹配其他模块
        )

    def forward(self, x):
        return self.generator(x)

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out

class FSMamba(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(FSMamba, self).__init__()

        self.vdsr = VDSRModule(args, conv)
        self.edsr = EDSRModule(args, conv)
        self.esrgan = ESRGANModule(args, conv)
        self.attention = SelfAttention(99)  # 自注意力机制，调整通道数以匹配输入

        self.final_conv = nn.Conv2d(99, 3, kernel_size=3, stride=1, padding=1)  # 将通道数调整回3

        scale = args.scale if isinstance(args.scale, (int, float)) else args.scale[0]
        self.upsample = nn.Upsample(scale_factor=float(scale), mode='bicubic', align_corners=False)

    def forward(self, x):
        residual = self.upsample(x)

        # VDSR output
        vdsr_out = self.vdsr(x)
        vdsr_out = nn.functional.interpolate(vdsr_out, size=(residual.size(2), residual.size(3)), mode='bicubic', align_corners=False)

        # EDSR output
        edsr_out = self.edsr(x)
        edsr_out = nn.functional.interpolate(edsr_out, size=(residual.size(2), residual.size(3)), mode='bicubic', align_corners=False)

        # ESRGAN output
        esrgan_out = self.esrgan(x)
        esrgan_out = nn.functional.interpolate(esrgan_out, size=(residual.size(2), residual.size(3)), mode='bicubic', align_corners=False)

        # Combine outputs
        combined = torch.cat([vdsr_out, edsr_out, esrgan_out, residual], dim=1)  # 在通道维度拼接
        attention_out = self.attention(combined)  # 应用自注意力机制

        out = self.final_conv(attention_out)  # 将通道数调整回3

        return out

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
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
    net = FSMamba(args).cuda()
    from thop import profile

    torch.cuda.reset_max_memory_allocated()
    x = torch.rand(1, 3, 64, 64).cuda()  # 调整输入尺寸以减少显存占用
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
