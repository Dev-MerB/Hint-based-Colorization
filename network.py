import torch
import torch.nn as nn
from torch.nn import init
def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class conv_block(nn.Module): # kernel_size 3 stride 1 padding 1
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, num_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(num_channels, num_channels // ratio, kernel_size=1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(num_channels // ratio, num_channels, kernel_size=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, num_channels):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x)
        return self.sigmoid(out)

class CAUnet(nn.Module):
    def __init__(self, img_ch=4, output_ch=2):
        super(CAUnet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = SpatialAttention(num_channels=512)
        self.Cha5 = ChannelAttention(num_channels=512, ratio=16)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = SpatialAttention(num_channels=256)
        self.Cha4 = ChannelAttention(num_channels=256, ratio=16)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = SpatialAttention(num_channels=128)
        self.Cha3 = ChannelAttention(num_channels=128, ratio=16)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = SpatialAttention(num_channels=64)
        self.Cha2 = ChannelAttention(num_channels=64, ratio=16)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        a4 = self.Att5(x4)
        c4 = self.Cha5(x4)
        x4 = (x4 * a4) * c4 # channel att -> spatial attention
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        a3 = self.Att4(x3)
        c3 = self.Cha4(x3)
        x3 = (x3 * a3) * c3
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        a2 = self.Att3(x2)
        c2 = self.Cha3(x2)
        x2 = (x2 * a2) * c2
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        a1 = self.Att2(x1)
        c1 = self.Cha2(x1)
        x1 = (x1 * a1) * c1
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class conv_block_3(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_block_5(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_5, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_block_7(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_7, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=7, stride=1, padding=3, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=7, stride=1, padding=3, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_fusion(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_fusion, self).__init__()
        self.conv_3 = conv_block_3(ch_in, ch_out)
        #self.conv_5 = conv_block_5(ch_in, ch_out)
        self.conv_7 = conv_block_7(ch_in, ch_out)

        self.conv = nn.Conv2d(ch_out * 2, ch_out, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x3 = self.conv_3(x)
        #x5 = self.conv_5(x)
        x7 = self.conv_7(x)

        x = torch.cat((x3, x7), dim=1)
        x = self.conv(x)

        return x

class up_conv_fusion(nn.Module):
    def __init__(self, ch_in, ch_out, bilinear = False):
        super(up_conv_fusion, self).__init__()
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(ch_in),
                nn.ReLU(inplace=True)
            )
        else:
            self.up = nn.ConvTranspose2d(ch_in, ch_out, kernel_size=2, stride=2)

    def forward(self, x):

        x = self.up(x)

        return x

class MSU_Net(nn.Module):
    def __init__(self, img_ch=4, output_ch=2):
        super(MSU_Net, self).__init__()

        filters_number = [32, 64, 128, 256, 512]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_fusion(ch_in=img_ch, ch_out=filters_number[0])
        self.Conv2 = conv_fusion(ch_in=filters_number[0], ch_out=filters_number[1])
        self.Conv3 = conv_fusion(ch_in=filters_number[1], ch_out=filters_number[2])
        self.Conv4 = conv_fusion(ch_in=filters_number[2], ch_out=filters_number[3])
        self.Conv5 = conv_fusion(ch_in=filters_number[3], ch_out=filters_number[4])

        self.Up5 = up_conv_fusion(ch_in=filters_number[4], ch_out=filters_number[3])
        self.Att5 = SpatialAttention(num_channels=filters_number[3])
        self.Cha5 = ChannelAttention(num_channels=filters_number[3], ratio=16)
        self.Up_conv5 = conv_fusion(ch_in=filters_number[4], ch_out=filters_number[3])

        self.Up4 = up_conv_fusion(ch_in=filters_number[3], ch_out=filters_number[2])
        self.Att4 = SpatialAttention(num_channels=filters_number[2])
        self.Cha4 = ChannelAttention(num_channels=filters_number[2], ratio=16)
        self.Up_conv4 = conv_fusion(ch_in=filters_number[3], ch_out=filters_number[2])

        self.Up3 = up_conv_fusion(ch_in=filters_number[2], ch_out=filters_number[1])
        self.Att3 = SpatialAttention(num_channels=filters_number[1])
        self.Cha3 = ChannelAttention(num_channels=filters_number[1], ratio=16)
        self.Up_conv3 = conv_fusion(ch_in=filters_number[2], ch_out=filters_number[1])

        self.Up2 = up_conv_fusion(ch_in=filters_number[1], ch_out=filters_number[0])
        self.Att2 = SpatialAttention(num_channels=filters_number[0])
        self.Cha2= ChannelAttention(num_channels=filters_number[0], ratio=16)
        self.Up_conv2 = conv_fusion(ch_in=filters_number[1], ch_out=filters_number[0])

        self.Conv_1x1 = nn.Conv2d(filters_number[0], output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4= self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        
        d5 = self.Up5(x5)
        a4 = self.Att5(x4)
        c4 = self.Cha5(x4)
        x4 = (x4 * a4) * c4 # channel att -> spatial attention
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        a3 = self.Att4(x3)
        c3 = self.Cha4(x3)
        x3 = (x3 * a3) * c3
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        a2 = self.Att3(x2)
        c2 = self.Cha3(x2)
        x2 = (x2 * a2) * c2
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        a1 = self.Att2(x1)
        c1 = self.Cha2(x1)
        x1 = (x1 * a1) * c1
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1