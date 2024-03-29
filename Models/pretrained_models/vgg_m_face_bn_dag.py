
import torch
import torch.nn as nn
import numpy as np


class Vgg_m_face_bn_dag(nn.Module):

    def __init__(self, use_cuda):
        super(Vgg_m_face_bn_dag, self).__init__()
        self.meta = {'mean': [131.45376586914062, 103.98748016357422, 91.46234893798828],
                     'std': [1, 1, 1],
                     'imageSize': [224, 224, 3]}
        self.conv1 = nn.Conv2d(3, 96, kernel_size=[7, 7], stride=(2, 2))
        self.bn49 = nn.BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=[5, 5], stride=(2, 2), padding=(1, 1))
        self.bn50 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=(0, 0), dilation=1, ceil_mode=True)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.bn51 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.bn52 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.bn53 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.fc6 = nn.Conv2d(512, 4096, kernel_size=[6, 6], stride=(1, 1))
        self.bn54 = nn.BatchNorm2d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu6 = nn.ReLU(inplace=True)
        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=[1, 1], stride=(1, 1))
        self.bn55 = nn.BatchNorm2d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu7 = nn.ReLU(inplace=True)
        self.fc8 = nn.Linear(in_features=4096, out_features=2622, bias=True)
        # Normalization
        mean = np.array(self.meta['mean'], dtype=np.float32)
        std = np.array(self.meta['std'], dtype=np.float32)
        self.norm_mean = torch.from_numpy(mean).view(1, -1, 1, 1)
        self.norm_std = torch.from_numpy(std).view(1, -1, 1, 1)
        if use_cuda:
            self.norm_mean = self.norm_mean.cuda()
            self.norm_std = self.norm_std.cuda()


    def forward(self, x0, output_layer):
        i = 0
        num = len(output_layer)
        output = []

        if output_layer[i] == 0:
            output.append(x0)
            i += 1
            if i == num:
                return output

        # Pre-processing
        x0 = x0 / 2.0 + 0.5    # from [-1,1] to [0,1]
        x0 = x0 * 255.0   # from [0,1] to [0,255]
        x0 = (x0 - self.norm_mean) / self.norm_std    # normalize

        # Network Forward
        x1 = self.conv1(x0)
        x2 = self.bn49(x1)
        x3 = self.relu1(x2)

        if output_layer[i] == 1:
            output.append(x3)
            i += 1
            if i == num:
                return output

        x4 = self.pool1(x3)
        x5 = self.conv2(x4)
        x6 = self.bn50(x5)
        x7 = self.relu2(x6)

        if output_layer[i] == 2:
            output.append(x7)
            i += 1
            if i == num:
                return output

        x8 = self.pool2(x7)
        x9 = self.conv3(x8)
        x10 = self.bn51(x9)
        x11 = self.relu3(x10)

        if output_layer[i] == 3:
            output.append(x11)
            i += 1
            if i == num:
                return output

        x12 = self.conv4(x11)
        x13 = self.bn52(x12)
        x14 = self.relu4(x13)

        if output_layer[i] == 4:
            output.append(x14)
            i += 1
            if i == num:
                return output

        x15 = self.conv5(x14)
        x16 = self.bn53(x15)
        x17 = self.relu5(x16)

        if output_layer[i] == 5:
            output.append(x17)
            i += 1
            if i == num:
                return output

        x18 = self.pool5(x17)
        x19 = self.fc6(x18)
        x20 = self.bn54(x19)
        x21 = self.relu6(x20)

        if output_layer[i] == 6:
            output.append(x21)
            i += 1
            if i == num:
                return output


        x22 = self.fc7(x21)
        x23 = self.bn55(x22)
        x24_preflatten = self.relu7(x23)

        if output_layer[i] == 7:
            output.append(x24_preflatten)
            i += 1
            if i == num:
                return output

        x24 = x24_preflatten.view(x24_preflatten.size(0), -1)
        x25 = self.fc8(x24)
        return x25
        

def vgg_m_face_bn_dag(use_cuda, weights_path=None, **kwargs):
    """
    load imported model instance

    Args:
        weights_path (str): If set, loads model weights from the given path
    """
    model = Vgg_m_face_bn_dag(use_cuda)
    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)
    return model
