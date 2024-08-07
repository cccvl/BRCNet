from functools import partial
# from model.common import AlignSeg_CM, AlignSeg_FA
from timm.models import xception
# from model.common import SeparableConv2d, Block
# from model.common import GuidedAttention, GraphReasoning, GuidedAttention2
from model.common import *
from model.m2tr_transform import *

import torch
import torch.nn as nn
import torch.nn.functional as F

encoder_params = {
    "xception": {
        "features": 2048,
        "init_op": partial(xception, pretrained=True)
    }
}


class BRCNet(nn.Module):
    """ End-to-End Reconstruction-Classification Learning for Face Forgery Detection """

    def __init__(self, num_classes, drop_rate=0.2):
        super(BRCNet, self).__init__()
        self.name = "xception"
        self.loss_inputs = dict()
        self.encoder = encoder_params[self.name]["init_op"]()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(drop_rate)
        self.fc = nn.Linear(encoder_params[self.name]["features"], num_classes)

        self.attention = GuidedAttention(depth=728, drop_rate=drop_rate)
        self.attention2 = GuidedAttention2(depth=728, drop_rate=drop_rate)
        self.reasoning = GraphReasoning(728, 256, 256, 256, 128, 256, [2, 4], drop_rate)

        self.aligncm0 = AlignSeg_CM(728)
        self.aligncm1 = AlignSeg_CM(256)
        self.aligncm2 = AlignSeg_CM(128)

        self.feature_aggregation = Feature_Agg()

        # self.alignfa1 = AlignSeg_FA(256)
        # self.alignfa2 = AlignSeg_FA(128)

        self.decoder1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(728, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.decoder2 = Block(256, 256, 3, 1)
        self.decoder3 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(256, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.decoder4 = Block(128, 128, 3, 1)
        self.decoder5 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.decoder6 = nn.Sequential(
            nn.Conv2d(64, 3, 1, 1, bias=False),
            nn.Tanh()
        )

    def norm_n_corr(self, x):
        norm_embed = F.normalize(self.global_pool(x), p=2, dim=1)
        corr = (torch.matmul(norm_embed.squeeze(), norm_embed.squeeze().T) + 1.) / 2.
        return norm_embed, corr

    @staticmethod
    def add_white_noise(tensor, mean=0., std=1e-6):
        rand = torch.rand([tensor.shape[0], 1, 1, 1])
        rand = torch.where(rand > 0.5, 1., 0.).to(tensor.device)
        white_noise = torch.normal(mean, std, size=tensor.shape, device=tensor.device)
        noise_t = tensor + white_noise * rand
        noise_t = torch.clip(noise_t, -1., 1.)
        return noise_t

    def forward(self, x1, x2):
        # clear the loss inputs
        # 有两个decoder，因此度量学习的loss会有两个
        self.loss_inputs = dict(recons=[], contra=[])
        noise_x = self.add_white_noise(x1) if self.training else x1
        out = self.encoder.conv1(noise_x)
        out = self.encoder.bn1(out)
        out = self.encoder.act1(out)
        out = self.encoder.conv2(out)
        out = self.encoder.bn2(out)
        out = self.encoder.act2(out)
        out = self.encoder.block1(out)
        out = self.encoder.block2(out)
        out = self.encoder.block3(out)
        embedding = self.encoder.block4(out)

        norm_embed, corr = self.norm_n_corr(embedding)
        self.loss_inputs['contra'].append(corr)

       # decoder
        out_decoder = self.dropout(embedding)
        out_decoder = self.decoder1(out_decoder)
        out_d2_decoder = self.decoder2(out_decoder)

        norm_embed, corr = self.norm_n_corr(out_d2_decoder)
        self.loss_inputs['contra'].append(corr*2)

        out_decoder = self.decoder3(out_d2_decoder)
        out_d4_decoder = self.decoder4(out_decoder)

        norm_embed, corr = self.norm_n_corr(out_d4_decoder)
        self.loss_inputs['contra'].append(corr*2)

        out_decoder = self.decoder5(out_d4_decoder)
        pred_decoder = self.decoder6(out_decoder)

        recons_x_decoder = F.interpolate(pred_decoder, size=x1.shape[-2:], mode='bilinear', align_corners=True)
        self.loss_inputs['recons'].append(recons_x_decoder)

        # Feature alignment module (FAM)
        embedding = self.aligncm0(embedding)
        out_d2    = self.aligncm1(out_d2_decoder)
        out_d4    = self.aligncm2(out_d4_decoder)

        embedding = self.encoder.block5(embedding)
        embedding = self.encoder.block6(embedding)
        embedding = self.encoder.block7(embedding)

        # Multi-scale feature aggregation module (MFAM)
        # Multi-scale attention module (MAM)
        fusion = self.feature_aggregation(out_d4, out_d2, embedding, 10) + embedding

        embedding = self.encoder.block8(fusion)
        img_att = self.attention2(x1, x2, recons_x_decoder, recons_x_decoder, embedding)

        embedding = self.encoder.block9(img_att)
        embedding = self.encoder.block10(embedding)
        embedding = self.encoder.block11(embedding)
        embedding = self.encoder.block12(embedding)

        embedding = self.encoder.conv3(embedding)
        embedding = self.encoder.bn3(embedding)
        embedding = self.encoder.act3(embedding)
        embedding = self.encoder.conv4(embedding)
        embedding = self.encoder.bn4(embedding)
        embedding = self.encoder.act4(embedding)

        embedding = self.global_pool(embedding).squeeze()

        out = self.dropout(embedding)
        return self.fc(out)