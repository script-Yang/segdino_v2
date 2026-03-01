import torch
import torch.nn as nn
import torch.nn.functional as F

class DWRes(nn.Module):
    def __init__(self, c, use_gn=True):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)
        self.pw = nn.Conv2d(c, c, 1, bias=False)
        self.norm = nn.GroupNorm(min(32, c), c) if use_gn else nn.BatchNorm2d(c)
        self.act = nn.GELU()
        self.gamma = nn.Parameter(torch.zeros(1)) 

    def forward(self, x):
        r = self.act(self.norm(self.pw(self.dw(x))))
        return x + self.gamma * r

class MiniDPTHead(nn.Module):
    def __init__(self, in_dims, embed_dim=128, num_classes=2, use_gn=True):
        super().__init__()
        assert len(in_dims) == 4
        
        # TPA Projections
        self.proj = nn.ModuleList([nn.Conv2d(c, embed_dim, 1, bias=False) for c in in_dims])
        
        # TPA Resizing
        self.rsz1 = nn.ConvTranspose2d(embed_dim, embed_dim, 4, stride=4, padding=0, bias=False) # 1/4
        self.rsz2 = nn.ConvTranspose2d(embed_dim, embed_dim, 2, stride=2, padding=0, bias=False) # 1/8
        self.rsz3 = nn.Identity()                                                                # 1/16
        self.rsz4 = nn.Conv2d(embed_dim, embed_dim, 3, stride=2, padding=1, bias=False)          # 1/32

        # SAD Intra-scale
        self.intra_r1 = DWRes(embed_dim, use_gn=use_gn)
        self.intra_r2 = DWRes(embed_dim, use_gn=use_gn)
        self.intra_r3 = DWRes(embed_dim, use_gn=use_gn)
        self.intra_r4 = DWRes(embed_dim, use_gn=use_gn)

        # SAD Inter-scale
        self.inter_r4 = DWRes(embed_dim, use_gn=use_gn)
        self.inter_r3 = DWRes(embed_dim, use_gn=use_gn)
        self.inter_r2 = DWRes(embed_dim, use_gn=use_gn)
        self.inter_r1 = DWRes(embed_dim, use_gn=use_gn)
        
        self.out_conv = nn.Conv2d(embed_dim, num_classes, 1)

    def forward(self, feats_4, patch_h, patch_w):
        # --- Token Pyramid Adaptation (TPA) ---
        p = []
        for i, x in enumerate(feats_4):
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.proj[i](x)
            p.append(x)

        p1 = self.rsz1(p[0])  # H/4
        p2 = self.rsz2(p[1])  # H/8
        p3 = self.rsz3(p[2])  # H/16
        p4 = self.rsz4(p[3])  # H/32

        # --- Scale-Aware Decoding (SAD) ---
        l1 = self.intra_r1(p1)
        l2 = self.intra_r2(p2)
        l3 = self.intra_r3(p3)
        l4 = self.intra_r4(p4)

        # Inter-scale fusion
        x4 = self.inter_r4(l4)

        x3_up = F.interpolate(x4, size=l3.shape[-2:], mode="bilinear", align_corners=False)
        x3 = self.inter_r3(x3_up + l3)

        x2_up = F.interpolate(x3, size=l2.shape[-2:], mode="bilinear", align_corners=False)
        x2 = self.inter_r2(x2_up + l2)

        x1_up = F.interpolate(x2, size=l1.shape[-2:], mode="bilinear", align_corners=False)
        x1 = self.inter_r1(x1_up + l1)

        # Final prediction
        logits = self.out_conv(x1)   
        return logits

class DPT(nn.Module):
    def __init__(
        self, 
        encoder_size='base', 
        nclass=2,
        features=128, 
        patch_size=16,
        use_bn=False,
        backbone=None
    ):
        super(DPT, self).__init__()
        
        self.intermediate_layer_idx = {
            'small': [2, 5, 8, 11],
            'base': [2, 5, 8, 11], 
            'large': [4, 11, 17, 23],
        }
        
        self.encoder_size = encoder_size
        self.patch_size = patch_size
        self.backbone = backbone
        self.nclass = nclass
        self.in_dims = [self.backbone.embed_dim] * 4
        self.head = MiniDPTHead(self.in_dims, self.backbone.embed_dim, self.nclass, use_bn)

    def lock_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x, return_feats=False):
        patch_h, patch_w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        feats = self.backbone.get_intermediate_layers(
            x, n=self.intermediate_layer_idx[self.encoder_size]
        )

        out = self.head(feats, patch_h, patch_w)
        out = F.interpolate(out, (patch_h * self.patch_size, patch_w * self.patch_size),
                            mode='bilinear', align_corners=True)
        if return_feats:
            return out, feats[-1]
        return out