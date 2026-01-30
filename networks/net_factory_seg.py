import torch

from networks.segmentation.Unet2D import UNet
from networks.segmentation.LasagnaUNet import LasagnaUNet

def net_factory(args):

    net_type=args.model
    in_channels=args.in_channels
    n_classes=args.num_classes
    img_dim = args.patch_size[0]

    if net_type == 'LasagnaUNet':
        net = LasagnaUNet(image_size=img_dim, in_channels=in_channels, class_num=n_classes)
    elif net_type == 'UNet2D':
        net = UNet(in_chns=in_channels, class_num=n_classes)
    else:
        raise ValueError(f'{net_type} is not supported')

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            net = torch.nn.DataParallel(net)
        net.cuda()
    return net
