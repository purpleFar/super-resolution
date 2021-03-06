# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import model.ops as ops
# import torch.nn.functional as F

def make_model(args, parent=False):
    return DRLN(args)

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.c1 = ops.BasicBlock(channel , channel // reduction, 3, 1, 3, 3)
        self.c2 = ops.BasicBlock(channel , channel // reduction, 3, 1, 5, 5)
        self.c3 = ops.BasicBlock(channel , channel // reduction, 3, 1, 7, 7)
        self.c4 = ops.BasicBlockSig((channel // reduction)*3, channel , 3, 1, 1)

    def forward(self, x):
        y = self.avg_pool(x)
        c1 = self.c1(y)
        c2 = self.c2(y)
        c3 = self.c3(y)
        c_out = torch.cat([c1, c2, c3], dim=1)
        y = self.c4(c_out)
        return x * y

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, group=1):
        super(Block, self).__init__()

        self.r1 = ops.ResidualBlock(in_channels, out_channels)
        self.r2 = ops.ResidualBlock(in_channels*2, out_channels*2)
        self.r3 = ops.ResidualBlock(in_channels*4, out_channels*4)
        self.g = ops.BasicBlock(in_channels*8, out_channels, 1, 1, 0)
        self.ca = CALayer(in_channels)

    def forward(self, x):
        c0 =  x

        r1 = self.r1(c0)
        c1 = torch.cat([c0, r1], dim=1)
                
        r2 = self.r2(c1)
        c2 = torch.cat([c1, r2], dim=1)
               
        r3 = self.r3(c2)
        c3 = torch.cat([c2, r3], dim=1)

        g = self.g(c3)
        out = self.ca(g)
        return out
        

class DRLN(nn.Module):
    def __init__(self, args):
        super(DRLN, self).__init__()
        
        self.stop_block = 1
        self.scale = 3#args.scale[0]
        chs=64

        self.sub_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        
        self.head = nn.Conv2d(3, chs, 3, 1, 1)

        self.b1 = Block(chs, chs)
        self.b2 = Block(chs, chs)
        self.b3 = Block(chs, chs)
        self.b4 = Block(chs, chs)
        self.b5 = Block(chs, chs)
        self.b6 = Block(chs, chs)
        self.b7 = Block(chs, chs)
        self.b8 = Block(chs, chs)
        self.b9 = Block(chs, chs)
        self.b10 = Block(chs, chs)
        self.b11 = Block(chs, chs)
        self.b12 = Block(chs, chs)
        self.b13 = Block(chs, chs)
        self.b14 = Block(chs, chs)
        self.b15 = Block(chs, chs)
        self.b16 = Block(chs, chs)
        self.b17 = Block(chs, chs)
        self.b18 = Block(chs, chs)
        self.b19 = Block(chs, chs)
        self.b20 = Block(chs, chs)
        
        self.b21 = Block(chs, chs)
        self.b22 = Block(chs, chs)
        self.b23 = Block(chs, chs)
        self.b24 = Block(chs, chs)

        self.c1 = ops.BasicBlock(chs*2, chs, 3, 1, 1)
        self.c2 = ops.BasicBlock(chs*3, chs, 3, 1, 1)
        self.c3 = ops.BasicBlock(chs*4, chs, 3, 1, 1)
        self.c4 = ops.BasicBlock(chs*2, chs, 3, 1, 1)
        self.c5 = ops.BasicBlock(chs*3, chs, 3, 1, 1)
        self.c6 = ops.BasicBlock(chs*4, chs, 3, 1, 1)
        self.c7 = ops.BasicBlock(chs*2, chs, 3, 1, 1)
        self.c8 = ops.BasicBlock(chs*3, chs, 3, 1, 1)
        self.c9 = ops.BasicBlock(chs*4, chs, 3, 1, 1)
        self.c10 = ops.BasicBlock(chs*2, chs, 3, 1, 1)
        self.c11 = ops.BasicBlock(chs*3, chs, 3, 1, 1)
        self.c12 = ops.BasicBlock(chs*4, chs, 3, 1, 1)
        self.c13 = ops.BasicBlock(chs*2, chs, 3, 1, 1)
        self.c14 = ops.BasicBlock(chs*3, chs, 3, 1, 1)
        self.c15 = ops.BasicBlock(chs*4, chs, 3, 1, 1)
        self.c16 = ops.BasicBlock(chs*5, chs, 3, 1, 1)
        self.c17 = ops.BasicBlock(chs*2, chs, 3, 1, 1)
        self.c18 = ops.BasicBlock(chs*3, chs, 3, 1, 1)
        self.c19 = ops.BasicBlock(chs*4, chs, 3, 1, 1)
        self.c20 = ops.BasicBlock(chs*5, chs, 3, 1, 1)
        
        self.c21 = ops.BasicBlock(chs*2, chs, 3, 1, 1)
        self.c22 = ops.BasicBlock(chs*3, chs, 3, 1, 1)
        self.c23 = ops.BasicBlock(chs*4, chs, 3, 1, 1)
        self.c24 = ops.BasicBlock(chs*5, chs, 3, 1, 1)

        self.upsample = ops.UpsampleBlock(chs, self.scale , multi_scale=False)
    
        self.tail = nn.Conv2d(chs, 3, 3, 1, 1)
                
    def forward(self, x, mask):
        x = self.sub_mean(x)
        x = self.head(x)
        c0 = o0 = x
        
        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        
        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        
        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)
        a1 = o3 + c0

        if self.stop_block==1:
            return self.last_block(a1, x, mask)
        
        b4 = self.b4(a1)
        c4 = torch.cat([o3, b4], dim=1)
        o4 = self.c4(c4)
 
        b5 = self.b5(o4) # self.b5(o4)
        c5 = torch.cat([c4, b5], dim=1)
        o5 = self.c5(c5)

        b6 = self.b6(o5)
        c6 = torch.cat([c5, b6], dim=1)
        o6 = self.c6(c6)
        a2 = o6 + a1
        
        if self.stop_block==2:
            return self.last_block(a2, x, mask)

        b7 = self.b7(a2)
        c7 = torch.cat([o6, b7], dim=1)
        o7 = self.c7(c7)

        b8 = self.b8(o7)
        c8 = torch.cat([c7, b8], dim=1)
        o8 = self.c8(c8)

        b9 = self.b9(o8)
        c9 = torch.cat([c8, b9], dim=1)
        o9 = self.c9(c9)
        a3 = o9 + a2

        if self.stop_block==3:
            return self.last_block(a3, x, mask)

        
        b10 = self.b10(a3)
        c10 = torch.cat([o9, b10], dim=1)
        o10 = self.c10(c10)


        b11 = self.b11(o10)
        c11 = torch.cat([c10, b11], dim=1)
        o11 = self.c11(c11)

        b12 = self.b12(o11)
        c12 = torch.cat([c11, b12], dim=1)
        o12 = self.c12(c12)
        a4 = o12 + a3

        if self.stop_block==4:
            return self.last_block(a4, x, mask)


        b13 = self.b13(a4)
        c13 = torch.cat([o12, b13], dim=1)
        o13 = self.c13(c13)

        b14 = self.b14(o13)
        c14 = torch.cat([c13, b14], dim=1)
        o14 = self.c14(c14)


        b15 = self.b15(o14)
        c15 = torch.cat([c14, b15], dim=1)
        o15 = self.c15(c15)

        b16 = self.b16(o15)
        c16 = torch.cat([c15, b16], dim=1)
        o16 = self.c16(c16)
        a5 = o16 + a4

        if self.stop_block==5:
            return self.last_block(a5, x, mask)


        b17 = self.b17(a5)
        c17 = torch.cat([o16, b17], dim=1)
        o17 = self.c17(c17)

        b18 = self.b18(o17)
        c18 = torch.cat([c17, b18], dim=1)
        o18 = self.c18(c18)


        b19 = self.b19(o18)
        c19 = torch.cat([c18, b19], dim=1)
        o19 = self.c19(c19)

        b20 = self.b20(o19)
        c20 = torch.cat([c19, b20], dim=1)
        o20 = self.c20(c20)
        
        a6 = o20 + a5
        
        if self.stop_block==6:
            return self.last_block(a6, x, mask)
        
        b21 = self.b21(a6)
        c21 = torch.cat([o20, b21], dim=1)
        o21 = self.c21(c21)

        b22 = self.b22(o21)
        c22 = torch.cat([c21, b22], dim=1)
        o22 = self.c22(c22)


        b23 = self.b23(o22)
        c23 = torch.cat([c22, b23], dim=1)
        o23 = self.c22(c22)

        b24 = self.b24(o23)
        c24 = torch.cat([c23, b24], dim=1)
        o24 = self.c24(c24)
        
        a7 = o24 + a6

        return self.last_block(a7, x, mask)
        
    def last_block(self, output, x, mask):
        b_out = output + x
        out = self.upsample(b_out, scale=self.scale)
        out = self.tail(out)
        f_out = torch.clamp(self.add_mean(out), 0, 1)
        return f_out*mask

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0 or name.find('upsample') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
