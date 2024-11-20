import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, n_channels, n_classes, dw_kernel_size, dw_padding, pw_kernel_size, pw_padding, dw_dilation=1,
                 pw_dilation=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise_conv = nn.Conv2d(
            n_channels,
            n_channels,
            kernel_size=dw_kernel_size,
            padding=dw_padding,
            dilation=dw_dilation,
            groups=n_channels
        )
        self.pointwise_conv = nn.Conv2d(
            n_channels,
            n_classes,
            kernel_size=pw_kernel_size,
            padding=pw_padding,
            groups=1
        )

    def forward(self, x):
        out = self.depthwise_conv(x)
        out = self.pointwise_conv(out)
        return out


class DilatedDepthwiseSeparableConv(nn.Module):
    def __init__(self, n_channels, n_classes, dw_kernel_size=3, dw_padding=1, dilation=1, pw_kernel_size=1,
                 pw_padding=0):
        super(DilatedDepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(n_channels, n_channels, kernel_size=dw_kernel_size, padding=dw_padding,
                                   dilation=dilation, groups=n_channels)
        self.pointwise = nn.Conv2d(n_channels, n_classes, kernel_size=pw_kernel_size, padding=pw_padding)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x




class coordinate_attention(nn.Module):
    def __init__(self, inp, oup, groups=4):
        super(coordinate_attention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 适应性平均池化到 (H, 1)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 适应性平均池化到 (1, W)

        mip = max(8, inp // groups)  # 中间通道数，防止通道数过小

        # 定义卷积层和批归一化
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()  # 或者 h_swish()，根据需求选择激活函数

    def forward(self, x):
        identity = x  # 保存输入以便于跳跃连接
        n, c, h, w = x.size()  # 获取输入的尺寸

        # 计算 x 方向和 y 方向的特征图
        x_h = self.pool_h(x)  # 在 H 维度上进行池化
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # 在 W 维度上进行池化并转置

        # 将两个方向的特征图拼接在一起
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)  # 经过卷积层
        y = self.bn1(y)    # 批归一化
        y = self.relu(y)   # 激活函数

        # 将拼接后的特征图分割回 H 和 W 方向
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # 恢复原始维度

        # 计算注意力权重
        x_h = self.conv2(x_h).sigmoid()  # x 方向的注意力
        x_w = self.conv3(x_w).sigmoid()  # y 方向的注意力
        x_h = x_h.expand(-1, -1, h, w)    # 扩展到原始尺寸
        x_w = x_w.expand(-1, -1, h, w)

        # 应用注意力权重
        y = identity * x_w * x_h

        return y




class REBNCONV(nn.Module):
    def __init__(self, n_channels=3, n_classes=4, dirate=1):
        super(REBNCONV, self).__init__()

        self.conv_s1 = nn.Conv2d(
            n_channels, n_classes, 3, padding=1 * dirate, dilation=1 * dirate
        )
        self.bn_s1 = nn.BatchNorm2d(n_classes)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout


## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src, tar):
    src = F.interpolate(src, size=tar.shape[2:], mode="bilinear", align_corners=False)

    return src


### RSU-7 ###
class RSU7(nn.Module):
    def __init__(self, n_channels=3, mid_ch=12, n_classes=4):
        super(RSU7, self).__init__()

        self.rebnconvin = REBNCONV(n_channels, n_classes, dirate=1)

        self.rebnconv1 = REBNCONV(n_classes, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv6d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, n_classes, dirate=1)

        # 使用通道注意力
        self.coordinate_attention = coordinate_attention(n_classes, n_classes)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx6dup = _upsample_like(hx6d, hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        # print('h1:{}'.format(hx1d.shape))
        # 使用坐标注意力
        hx1d = self.coordinate_attention(hx1d)
        # print('h2:{}'.format(hx1d.shape))

        """
        del hx1, hx2, hx3, hx4, hx5, hx6, hx7
        del hx6d, hx5d, hx3d, hx2d
        del hx2dup, hx3dup, hx4dup, hx5dup, hx6dup
        """

        return hx1d + hxin


### RSU-6 ###
class RSU6(nn.Module):  # UNet06DRES(nn.Module):
    def __init__(self, n_channels=3, mid_ch=12, n_classes=4):
        super(RSU6, self).__init__()

        self.rebnconvin = REBNCONV(n_channels, n_classes, dirate=1)

        self.rebnconv1 = REBNCONV(n_classes, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, n_classes, dirate=1)
        self.coordinate_attention = coordinate_attention(n_classes, n_classes)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        hx1d = self.coordinate_attention(hx1d)

        """
        del hx1, hx2, hx3, hx4, hx5, hx6
        del hx5d, hx4d, hx3d, hx2d
        del hx2dup, hx3dup, hx4dup, hx5dup
        """

        return hx1d + hxin


### RSU-5 ###
class RSU5(nn.Module):  # UNet05DRES(nn.Module):
    def __init__(self, n_channels=3, mid_ch=12, n_classes=4):
        super(RSU5, self).__init__()

        self.rebnconvin = REBNCONV(n_channels, n_classes, dirate=1)

        self.rebnconv1 = REBNCONV(n_classes, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, n_classes, dirate=1)
        self.coordinate_attention = coordinate_attention(n_classes, n_classes)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        hx1d = self.coordinate_attention(hx1d)

        """
        del hx1, hx2, hx3, hx4, hx5
        del hx4d, hx3d, hx2d
        del hx2dup, hx3dup, hx4dup
        """

        return hx1d + hxin


### RSU-4 ###
class RSU4(nn.Module):  # UNet04DRES(nn.Module):
    def __init__(self, n_channels=3, mid_ch=12, n_classes=4):
        super(RSU4, self).__init__()

        self.rebnconvin = REBNCONV(n_channels, n_classes, dirate=1)

        self.rebnconv1 = REBNCONV(n_classes, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, n_classes, dirate=1)
        self.coordinate_attention = coordinate_attention(n_classes, n_classes)
    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        hx1d = self.coordinate_attention(hx1d)

        """
        del hx1, hx2, hx3, hx4
        del hx3d, hx2d
        del hx2dup, hx3dup
        """

        return hx1d + hxin


### RSU-4F ###
class RSU4F(nn.Module):
    def __init__(self, n_channels=3, mid_ch=12, n_classes=4):
        super(RSU4F, self).__init__()

        # 使用普通的深度可分离卷积
        self.rebnconvin = DepthwiseSeparableConv(n_channels, n_classes, dw_kernel_size=3, dw_padding=1,
                                                 pw_kernel_size=1, pw_padding=0)

        # 使用空洞卷积替换原始卷积层
        self.rebnconv1 = DilatedDepthwiseSeparableConv(n_classes, mid_ch, dw_kernel_size=3, dw_padding=2, dilation=2)
        self.rebnconv2 = DilatedDepthwiseSeparableConv(mid_ch, mid_ch, dw_kernel_size=3, dw_padding=2,
                                                       dilation=2)  # 空洞卷积，dilation=2
        self.rebnconv3 = DilatedDepthwiseSeparableConv(mid_ch, mid_ch, dw_kernel_size=3, dw_padding=4,
                                                       dilation=4)  # 空洞卷积，dilation=4

        # 更深的空洞卷积
        self.rebnconv4 = DilatedDepthwiseSeparableConv(mid_ch, mid_ch, dw_kernel_size=3, dw_padding=8,
                                                       dilation=8)  # 空洞卷积，dilation=8

        # 逆向卷积部分同样使用空洞卷积
        self.rebnconv3d = DilatedDepthwiseSeparableConv(mid_ch * 2, mid_ch, dw_kernel_size=3, dw_padding=4,
                                                        dilation=4)  # 空洞卷积，dilation=4
        self.rebnconv2d = DilatedDepthwiseSeparableConv(mid_ch * 2, mid_ch, dw_kernel_size=3, dw_padding=2,
                                                        dilation=2)  # 空洞卷积，dilation=2
        self.rebnconv1d = DilatedDepthwiseSeparableConv(mid_ch * 2, n_classes, dw_kernel_size=3, dw_padding=2,
                                                        dilation=2)
        self.coordinate_attention = coordinate_attention(n_classes, n_classes)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))
        # print('h3:{}'.format(hx1d.shape))
        hx1d = self.coordinate_attention(hx1d)
        # print('h4:{}'.format(hx1d.shape))
        return hx1d + hxin


### U^2-Net small ###
class U2NETP(nn.Module):
    def __init__(self, n_channels=3, n_classes=4):
        super(U2NETP, self).__init__()

        self.stage1 = RSU7(n_channels, 16, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 16, 64)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(64, 16, 64)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(64, 16, 64)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(64, 16, 64)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(64, 16, 64)

        # decoder
        self.stage5d = RSU4F(128, 16, 64)
        self.stage4d = RSU4(128, 16, 64)
        self.stage3d = RSU5(128, 16, 64)
        self.stage2d = RSU6(128, 16, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = nn.Conv2d(64, n_classes, 3, padding=1)
        self.side2 = nn.Conv2d(64, n_classes, 3, padding=1)
        self.side3 = nn.Conv2d(64, n_classes, 3, padding=1)
        self.side4 = nn.Conv2d(64, n_classes, 3, padding=1)
        self.side5 = nn.Conv2d(64, n_classes, 3, padding=1)
        self.side6 = nn.Conv2d(64, n_classes, 3, padding=1)

        self.outconv = nn.Conv2d(6 * n_classes, n_classes, 1)

    def forward(self, x):
        hx = x

        # stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)

        # decoder
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        """
        del hx1, hx2, hx3, hx4, hx5, hx6
        del hx5d, hx4d, hx3d, hx2d, hx1d
        del hx6up, hx5dup, hx4dup, hx3dup, hx2dup
        """

        return d0, d1, d2, d3, d4, d5, d6
        # return torch.softmax(d0, dim=1), torch.softmax(d1, dim=1), \
        #        torch.softmax(d2, dim=1), torch.softmax(d3, dim=1), \
        #        torch.softmax(d4, dim=1), torch.softmax(d1, dim=1), torch.softmax(d5, dim=1)


if __name__ == '__main__':
    from torchsummary import summary

    model = U2NETP(3)
    # print(model)
    inputs = torch.randn([8, 3, 256, 256])
    # outputs = model(inputs)
    '''
    Total params 是总参数量
    Trainable params 是用于训练的参数量
    Non-trainable params 是未用于训练的参数量
    Input size (MB) 是输入尺寸的 img 大小
    Forward/backward pass size (MB) 是进行了前向计算的计算量大小
    Estimated Total Size (MB) 是预估的整个过程用掉的计算量
    '''
    summary(model.to('cuda:0'), input_size=(3, 256, 256), batch_size=1)
