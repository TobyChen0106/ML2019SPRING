import torch.nn as nn
import torch
import math
import time
import torch.utils.model_zoo as model_zoo
from utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
from anchors import Anchors
import losses
from lib.nms.pth_nms import pth_nms

def nms(dets, thresh):
    """Dispatch to either CPU or GPU NMS implementations.\
    Accept dets as tensor"""
    return pth_nms(dets, thresh)

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
}

class PyramidFeatures(nn.Module):
    def __init__(self, C2_size, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()
        
        # upsample C5 to get P5 from the FPN paper
        self.P5_1           = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled   = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2           = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1           = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled   = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2           = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1           = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled   = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_2           = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P3 elementwise to C2 (customized by Yu Han Huang)
        self.P2_1 = nn.Conv2d(C2_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P2_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, C2, C3, C4, C5):

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)
        
        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_upsampled_x = self.P3_upsampled(P3_x)
        P3_x = self.P3_2(P3_x)
        
        # add P3 elementwise to C2 (customized by Yu Han Huang)
        P2_x = self.P2_1(C2)
        P2_x = P2_x + P3_upsampled_x
        P2_x = self.P2_2(P2_x)
        
        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        return [P2_x, P3_x, P4_x, P5_x, P6_x, P7_x]


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()
        
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors*4, kernel_size=3, padding=1)

    def forward(self, x):

        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)

class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256, dropout1=0.25, dropout2=0.25):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        #add dropout1 to ClassificationModel (customized by Yu Han Hunag)
        self.dropout1 = nn.Dropout(p=dropout1)

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()
        #add dropout2 to ClassificationModel (customized by Yu Han Hunag)
        self.dropout2 = nn.Dropout(p=dropout2)

        self.output = nn.Conv2d(feature_size, num_anchors*num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):

        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)
        out = self.dropout1(out)

        out = self.conv4(out)
        out = self.act4(out)
        out = self.dropout2(out)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)

class ResNet(nn.Module):

    def __init__(self, num_classes, block, layers, groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 dropout1=0.25, dropout2=0.25, alpha=0.25, gamma=2.0,
                 loss_with_no_bboxes=False, no_bboxes_alpha=0.5, no_bboxes_gamma=2.0):
        #Has been changed to ResNext(customized by Yu Han Huang)
        self.inplanes = 64
        super(ResNet, self).__init__()
        #add self.dilation, width_per_group, replace_stride_with_dilation (customized by Yu Han Huang)
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        #add dilate=replace_stride_with_dilation (customized by Yu Han Huang)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        #add C2 layer_size to fpn_sizes (customized by Yu Han Huang)
        if block == BasicBlock:
            fpn_sizes = [self.layer1[layers[0]-1].conv2.out_channels, self.layer2[layers[1]-1].conv2.out_channels,
             self.layer3[layers[2]-1].conv2.out_channels, self.layer4[layers[3]-1].conv2.out_channels]
        elif block == BasicBlock:
            fpn_sizes = [self.layer1[layers[0]-1].conv3.out_channels, self.layer2[layers[1]-1].conv3.out_channels,
             self.layer3[layers[2]-1].conv3.out_channels, self.layer4[layers[3]-1].conv3.out_channels]
        #add fpn_sizes[0] into PyramidFeatures (customized by Yu Han Huang)
        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2], fpn_sizes[3])
        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=num_classes, dropout1=dropout1, dropout2=dropout2)
        self.anchors = Anchors()
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        #add arguments alpha, gamma loss_with_no_bboxes, no_bboxes_alpha, no_bboxes_gamma(customized by Yu Han Huang)
        self.focalLoss = losses.FocalLoss(alpha=alpha, gamma=gamma, loss_with_no_bboxes=loss_with_no_bboxes, no_bboxes_alpha=no_bboxes_alpha, no_bboxes_gamma=no_bboxes_gamma)
                
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        prior = 0.01
        
        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0-prior)/prior))
        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)
        self.freeze_bn()
    
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        #add dilate for resnext101 (customized by Yu Han Huang)
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation,))
        return nn.Sequential(*layers)
    
    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs):

        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs
            
        x = self.conv1(img_batch)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        features = self.fpn(x1, x2, x3, x4)

        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)

        classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)

        anchors = self.anchors(img_batch)

        if self.training:
            '''
            l1 = 0
            for p in self.classificationModel.parameters():
                l1 = l1 + p.abs().sum()
            '''
            return self.focalLoss(classification, regression, anchors, annotations)
        else:
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

            scores = torch.max(classification, dim=2, keepdim=True)[0]

            scores_over_thresh = (scores>0.05)[0, :, 0]
            
            if scores_over_thresh.sum() == 0:
                # no boxes to NMS, just return
                return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]

            classification = classification[:, scores_over_thresh, :]
            transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
            scores = scores[:, scores_over_thresh, :]
            

            anchors_nms_idx = nms(torch.cat([transformed_anchors, scores], dim=2)[0, :, :], 0.5)

            nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)

            return [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :]]


#Arguments: dropout1, dropout2, alpha, gamma,loss_with_no_bboxes, no_bboxes_alpha, no_bboxes_gamma are added into all of the retinanet model with different backbone
#(customized by Yu Han Huang)
def resnet18(num_classes, pretrained=False, dropout1=0.25, dropout2=0.25, alpha=0.25, gamma=2.0,
                    loss_with_no_bboxes=False, no_bboxes_alpha=0.5, no_bboxes_gamma=2, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        dropout1: The dropout rate for the 3rd layer of the Classification Model
        dropout2: The dropout rate for the 4th layer of the Classification Model
        alpha: Alpha in focal loss
        gamma: Gamma in focal loss
        loss_with_no_bboxes: If True, picture with no bboxes will be taken into account
        no_bboxes_alpha: Alpha in focal loss for picture with no bboxes
        no_bboxes_gamma: Gamma in focal loss for picture with no bboxes
    """
    model = ResNet(num_classes, BasicBlock, [2, 2, 2, 2], dropout1=0.25, dropout2=0.25, alpha=0.25, gamma=2.0,
                    loss_with_no_bboxes=False, no_bboxes_alpha=0.5, no_bboxes_gamma=2, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='.'), strict=False)
    return model


def resnet34(num_classes, pretrained=False, dropout1=0.25, dropout2=0.25, alpha=0.25, gamma=2.0,
                    loss_with_no_bboxes=False, no_bboxes_alpha=0.5, no_bboxes_gamma=2, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        dropout1: The dropout rate for the 3rd layer of the Classification Model
        dropout2: The dropout rate for the 4th layer of the Classification Model
        alpha: Alpha in focal loss
        gamma: Gamma in focal loss
        loss_with_no_bboxes: If True, picture with no bboxes will be taken into account
        no_bboxes_alpha: Alpha in focal loss for picture with no bboxes
        no_bboxes_gamma: Gamma in focal loss for picture with no bboxes
    """
    model = ResNet(num_classes, BasicBlock, [3, 4, 6, 3], dropout1=0.25, dropout2=0.25, alpha=0.25, gamma=2.0,
                    loss_with_no_bboxes=False, no_bboxes_alpha=0.5, no_bboxes_gamma=2, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir='.'), strict=False)
    return model


def resnet50(num_classes, pretrained=False, dropout1=0.25, dropout2=0.25, alpha=0.25, gamma=2.0,
                    loss_with_no_bboxes=False, no_bboxes_alpha=0.5, no_bboxes_gamma=2, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        dropout1: The dropout rate for the 3rd layer of the Classification Model
        dropout2: The dropout rate for the 4th layer of the Classification Model
        alpha: Alpha in focal loss
        gamma: Gamma in focal loss
        loss_with_no_bboxes: If True, picture with no bboxes will be taken into account
        no_bboxes_alpha: Alpha in focal loss for picture with no bboxes
        no_bboxes_gamma: Gamma in focal loss for picture with no bboxes
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 6, 3], dropout1=0.25, dropout2=0.25, alpha=0.25, gamma=2.0,
                    loss_with_no_bboxes=False, no_bboxes_alpha=0.5, no_bboxes_gamma=2, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='.'), strict=False)
    return model

def resnet101(num_classes, pretrained=False, dropout1=0.25, dropout2=0.25, alpha=0.25, gamma=2.0,
                    loss_with_no_bboxes=False, no_bboxes_alpha=0.5, no_bboxes_gamma=2, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        dropout1: The dropout rate for the 3rd layer of the Classification Model
        dropout2: The dropout rate for the 4th layer of the Classification Model
        alpha: Alpha in focal loss
        gamma: Gamma in focal loss
        loss_with_no_bboxes: If True, picture with no bboxes will be taken into account
        no_bboxes_alpha: Alpha in focal loss for picture with no bboxes
        no_bboxes_gamma: Gamma in focal loss for picture with no bboxes
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 23, 3], dropout1=0.25, dropout2=0.25, alpha=0.25, gamma=2.0,
                    loss_with_no_bboxes=False, no_bboxes_alpha=0.5, no_bboxes_gamma=2, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='.'), strict=False)
    return model


def resnet152(num_classes, pretrained=False, dropout1=0.25, dropout2=0.25, alpha=0.25, gamma=2.0,
                    loss_with_no_bboxes=False, no_bboxes_alpha=0.5, no_bboxes_gamma=2, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        dropout1: The dropout rate for the 3rd layer of the Classification Model
        dropout2: The dropout rate for the 4th layer of the Classification Model
        alpha: Alpha in focal loss
        gamma: Gamma in focal loss
        loss_with_no_bboxes: If True, picture with no bboxes will be taken into account
        no_bboxes_alpha: Alpha in focal loss for picture with no bboxes
        no_bboxes_gamma: Gamma in focal loss for picture with no bboxes
    """
    model = ResNet(num_classes, Bottleneck, [3, 8, 36, 3], dropout1=0.25, dropout2=0.25, alpha=0.25, gamma=2.0, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir='.'), strict=False)
    return model


# add resnext101 backbone (customized by Yu Han Huang)
def resnext101(num_classes, pretrained=False, dropout1=0.25, dropout2=0.25, alpha=0.25, gamma=2.0,
                    loss_with_no_bboxes=False, no_bboxes_alpha=0.5, no_bboxes_gamma=2, **kwargs):
    """Constructs a ResNext-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        dropout1: The dropout rate for the 3rd layer of the Classification Model
        dropout2: The dropout rate for the 4th layer of the Classification Model
        alpha: Alpha in focal loss
        gamma: Gamma in focal loss
        loss_with_no_bboxes: If True, picture with no bboxes will be taken into account
        no_bboxes_alpha: Alpha in focal loss for picture with no bboxes
        no_bboxes_gamma: Gamma in focal loss for picture with no bboxes
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    model = ResNet(num_classes, Bottleneck, [3, 4, 23, 3], dropout1=0.25, dropout2=0.25, alpha=0.25, gamma=2.0,
                    loss_with_no_bboxes=False, no_bboxes_alpha=0.5, no_bboxes_gamma=2, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnext101_32x8d'], model_dir='.'), strict=False)
    return model