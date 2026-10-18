"""Deformable convolution compatibility for ADNet inference.

Uses torchvision's real DCNv1 operator, with state-dict layout matching MMCV.
Cross-library numerical equivalence remains a documented validation limitation.
"""
from torchvision.ops import DeformConv2d as TorchvisionDeformConv2d

class DeformConv2d(TorchvisionDeformConv2d):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,
                 dilation=1,groups=1,deform_groups=1,bias=False,im2col_step=32):
        super().__init__(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias)
        self.deform_groups=deform_groups
    def forward(self,x,offset):
        if offset.shape[1]!=2*self.kernel_size[0]*self.kernel_size[1]*self.deform_groups:
            raise ValueError('Deformable group count does not match offsets')
        return super().forward(x,offset)
