import torch
import torch.nn as nn
import torch.nn.functional as F
from .MoGE import DecMoE
from .MoLE import MoELocal

from ..builder import build_loss
from ..registry import HEADS


class ConvBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1,
                 padding=1):
        super(ConvBlock, self).__init__()
        self.conv_r = nn.Conv2d(input_size, output_size, kernel_size, stride, padding)
        self.conv_i = nn.Conv2d(input_size, output_size, kernel_size, stride, padding)

    def forward(self, x_r, x_i):
        out_r = self.conv_r(x_r)
        out_i = self.conv_i(x_i)
        
        return out_r, out_i

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()

        self.Conv1 = ConvBlock(1,16,3,1,1)
        self.Relu_rgb  = nn.ReLU(inplace=True)
        self.Relu_tir  = nn.ReLU(inplace=True)
        self.moe_fg = MoELocal(ds_inputsize=614400, input_size=16, output_size=16, num_experts=4, hidden_size=8, noisy_gating=True, k=2, trainingmode=True) 
        self.moe_bg = MoELocal(ds_inputsize=614400, input_size=16, output_size=16, num_experts=4, hidden_size=8, noisy_gating=True, k=2, trainingmode=True)
        self.DenseConv1 = ConvBlock(16,16,3,1,1)
        self.DenseConv2 = ConvBlock(32,16,3,1,1)
        self.DenseConv3 = ConvBlock(48,16,3,1,1)

    def atten_downsample(self, x1, x2, att):
        x1_fg = x1 * att
        x1_bg = x1 * (1-att)
        x2_fg = x2 * att
        x2_bg = x2 * (1-att)
        
        x12_channel_fg = torch.cat((x1_fg, x2_fg), dim=1)  
        x12_channel_bg = torch.cat((x1_bg, x2_bg), dim=1) 
        
        xc_fg_ds = F.interpolate(x12_channel_fg, scale_factor=0.25, mode='bilinear', align_corners=True, recompute_scale_factor=True)
        xc_bg_ds = F.interpolate(x12_channel_bg, scale_factor=0.25, mode='bilinear', align_corners=True, recompute_scale_factor=True) 
        return x1_fg, x1_bg, x2_fg, x2_bg, xc_fg_ds, xc_bg_ds

    def forward(self, x_r, x_i, up_att_w):

        x_r, x_i = self.Conv1(x_r, x_i)
        x_r = self.Relu_rgb(x_r)
        x_i = self.Relu_tir(x_i)

        x1_r, x1_i = self.DenseConv1(x_r, x_i)
        x1_r = torch.cat([x_r, x1_r], 1)
        x1_i = torch.cat([x_i, x1_i], 1)

        x2_r, x2_i = self.DenseConv2(x1_r, x1_i)
        x2_r = torch.cat([x1_r, x2_r], 1)
        x2_i = torch.cat([x1_i, x2_i], 1)

        x3_r, x3_i = self.DenseConv3(x2_r, x2_i) 

        xr_fg, xr_bg, xi_fg, xi_bg, xc_fg_ds, xc_bg_ds = self.atten_downsample(x3_r, x3_i, up_att_w)
        xf_fg, aux_loss_fg = self.moe_fg(xc_fg_ds.view(xc_fg_ds.shape[0], -1), xr_fg, xi_fg)
        xf_bg, aux_loss_bg = self.moe_bg(xc_bg_ds.view(xc_bg_ds.shape[0], -1), xr_bg, xi_bg)
        
        x_f = torch.cat((xf_fg, xf_bg), dim=1)
        
        x3_r = torch.cat([x2_r, x3_r], 1)
        x3_i = torch.cat([x2_i, x3_i], 1)

        enc_f = torch.cat((x3_r, x3_i, x_f), dim=1)
        
        aux_loss_fb = aux_loss_fg + aux_loss_bg
        
        return enc_f, aux_loss_fb

@HEADS.register_module
class MoE_fusion_head(nn.Module):
    """Fusion RGB and TIR images in this head """
    def __init__(self,
                 loss_grad=dict(
                     type='NewGradLoss', loss_weight=1.0),
                 loss_detdriven=dict(
                     type='DetcropPixelLoss', loss_weight=1.0)):
        super(MoE_fusion_head, self).__init__()

        self.encoder = Encoder()
        self.DecoderMoE = DecMoE(ds_inputsize=768000, input_size=160, output_size=1, num_experts=4, hidden_size=16, noisy_gating=True, k=2, trainingmode=True)

        self.loss_grad = build_loss(loss_grad)
        self.loss_detdriven = build_loss(loss_detdriven)

    def forward(self, im_r, im_i, att_rgb, att_tir):

        att_w = torch.max(att_rgb[0], att_tir[0]) 
        up_att_w = F.interpolate(att_w, scale_factor=4, mode='bilinear', align_corners=True)

        enc_fusion, aux_loss_fgbg = self.encoder(im_r, im_i, up_att_w)

        enc_fusion_ds = F.interpolate(enc_fusion, scale_factor=0.125, mode='bilinear', align_corners=True, recompute_scale_factor=True)
        de_x, aux_loss = self.DecoderMoE(enc_fusion_ds.view(enc_fusion_ds.shape[0], -1), enc_fusion, loss_coef=0.02)

        aux_loss_total = aux_loss_fgbg + aux_loss
        return de_x, aux_loss_total

    def gard_loss(self,
             im_fusion,
             im_rgb,
             im_tir):
        losses = dict()
        losses['max_loss_grad'] = self.loss_grad(im_fusion, im_rgb, im_tir)
        return losses

    def pixel_loss(self,
             im_fusion,
             im_rgb,
             im_tir,
             detect_box):
        losses = dict()
        losses['detcrop_pixel_loss'] = self.loss_detdriven(im_fusion, im_rgb, im_tir, detect_box)
        return losses
