import torch.nn as nn
import torch.nn.functional as F

from modules.resnet import resnet18
from modules.segmentation_head import DBHead
from modules.segmentation_body import FPN

backbone_dict = {'resnet18': {'models': resnet18, 'out': [64, 128, 256, 512]}}
segmentation_body_dict = {'FPN': FPN}
segmentation_head_dict = {'DBHead': DBHead}


class DBTextModel(nn.Module):
    def __init__(self):

        super().__init__()
        pretrained = True
        backbone_name = "resnet18"
        segmentation_body_name = "FPN"
        segmentation_head_name = "DBHead"

        backbone_model, backbone_out = \
            backbone_dict[backbone_name]['models'], backbone_dict[backbone_name]['out']  # noqa
        self.backbone = backbone_model(pretrained=pretrained)
        self.segmentation_body = segmentation_body_dict[
            segmentation_body_name](backbone_out, inner_channels=256)
        self.segmentation_head = segmentation_head_dict[
            segmentation_head_name](self.segmentation_body.out_channels,
                                    out_channels=2)

        self.name = '{}_{}_{}'.format(backbone_name, segmentation_body_name,
                                      segmentation_head_name)

    def forward(self, x):
        """
        :return: TRAIN mode: prob_map, threshold_map, appro_binary_map
        :return: EVAL mode: prob_map, threshold_map
        """
        _, _, H, W = x.size()
        backbone_out = self.backbone(x)
        segmentation_body_out = self.segmentation_body(backbone_out)
        segmentation_head_out = self.segmentation_head(segmentation_body_out)
        y = F.interpolate(segmentation_head_out,
                          size=(H, W),
                          mode='bilinear',
                          align_corners=True)

        return y


if __name__ == '__main__':
    dbnet = DBTextModel().to('cpu')
    print(dbnet)
