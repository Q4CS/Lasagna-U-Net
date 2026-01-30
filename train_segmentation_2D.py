import argparse
import logging
import os
import random
import shutil
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import time
from tqdm import tqdm

from datasets.BUSI.BUSI_dataset import BUSI_DataSets
from datasets.GlaS.GlaS_dataset import GlaS_Dataset
from datasets.ISIC_2018_Task_1.ISIC_2018_Task_1_dataset import ISIC2018Task1_DataSets
from datasets.KPIs2024.KPIs2024_dataset import KPIs2024_Dataset

from losses.losses2D.CE_Dice_Loss import CEDiceLoss
from networks.net_factory_seg import net_factory
from torchmetrics import MetricCollection, Accuracy, Recall, Specificity, Precision
from torchmetrics.segmentation import DiceScore, MeanIoU

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def parse_option():
    """
    parameter setting
    :return:
    """
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--dataset', type=str, default='KPIs2024', help='dataset name')
    parser.add_argument('--patch_size', type=list,  default=[256, 256], help='patch size of network input')
    
    # model
    parser.add_argument('--model', type=str, default='LasagnaUNet', help='model name')
    parser.add_argument('--need_replace_module', type=int, default=0, help='whether to replace module when loading ckpt')

    # train
    parser.add_argument('--exp_name', type=str, default='seg', help='experiment name')
    parser.add_argument('--epochs', type=int, default=500, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
    parser.add_argument('--seed', type=int,  default=1234, help='random seed')
    args = parser.parse_args()

    return args


def set_loader(args):
    """
    setting data loader
    :param args:
    :return:
    """
    dataset = args.dataset
    patch_size = args.patch_size
    batch_size = args.batch_size

    if dataset == 'BUSI':
        # set dataset info
        args.has_val_set = True
        args.in_channels = 3
        args.images_normalized = False
        args.num_classes = 2

        train_dataset = BUSI_DataSets(split='train', img_size=patch_size)
        val_dataset = BUSI_DataSets(split='val', img_size=patch_size)
        test_dataset = BUSI_DataSets(split='test', img_size=patch_size)
    elif dataset == 'KPIs2024':
        # set dataset info
        args.has_val_set = True
        args.in_channels = 3
        args.num_classes = 2
        args.images_normalized = False

        train_dataset = KPIs2024_Dataset(split='train', img_size=patch_size)
        val_dataset = KPIs2024_Dataset(split='val', img_size=patch_size)
        test_dataset = KPIs2024_Dataset(split='test', img_size=patch_size)
    elif dataset == 'GlaS':
        # set dataset info
        args.has_val_set = True
        args.in_channels = 3
        args.num_classes = 2
        args.images_normalized = False

        train_dataset = GlaS_Dataset(split='train', img_size=patch_size)
        val_dataset = GlaS_Dataset(split='val', img_size=patch_size)
        test_dataset = GlaS_Dataset(split='test', img_size=patch_size)
    elif dataset == 'ISIC_2018_Task_1':
        # set dataset info
        args.has_val_set = True
        args.in_channels = 3
        args.num_classes = 2
        args.images_normalized = False

        train_dataset = ISIC2018Task1_DataSets(split='train', img_size=patch_size)
        val_dataset = ISIC2018Task1_DataSets(split='val', img_size=patch_size)
        test_dataset = ISIC2018Task1_DataSets(split='test', img_size=patch_size)
    else:
        raise ValueError(dataset)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=32, pin_memory=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

    if args.has_val_set:
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    else:
        test_loader = None

    return train_loader, val_loader, test_loader


def set_model(args):
    """
    setting model and criterion
    :param args:
    :return:
    """
    net = net_factory(args)
    criterion = CEDiceLoss(alpha=1.0, beta=1.0, n_classes=args.num_classes,
                           weight=None, softmax=True, include_background=False)
    if torch.cuda.is_available():
        criterion = criterion.cuda()
    return net, criterion


def train(train_loader, model, criterion, optimizer, epoch, writer, args, metric_collection):
    model.train()

    loss_list = []  # loss per iteration
    lr_list = []  # lr per iteration

    iterations_per_epoch = len(train_loader)
    random_show_img_iteration = random.randint(0, (iterations_per_epoch - 1))  # Randomly display the images in iteration
    loop = tqdm(train_loader, total=iterations_per_epoch)
    for i_batch, sampled_batch in enumerate(loop):
        if 'seg_label' in sampled_batch:
            label_name = 'seg_label'
        else:
            label_name = 'label'

        image_batch, label_batch = sampled_batch['image'], sampled_batch[label_name]
        batch_size = image_batch.shape[0]
        image_name_batch = sampled_batch['image_name']

        image_batch, label_batch = image_batch.cuda(), label_batch.long().cuda()

        outputs = model(image_batch)
        # outputs_soft = torch.softmax(outputs, dim=1)
        loss = criterion(outputs, label_batch)  # compute loss
        loss_list.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_ = args.base_lr * (1.0 - ((epoch - 1) * len(train_loader) + i_batch) / args.max_iterations) ** 0.9
        lr_list.append(lr_)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_

        # saving image
        if i_batch == random_show_img_iteration:
            random_int = random.randint(0, (batch_size - 1))
            temp_image = image_batch[random_int, ...]
            if args.images_normalized:
                temp_image = temp_image * 255
            writer.add_image('train/Image', np.array(temp_image.cpu(), dtype='uint8'), epoch, dataformats="CHW")
            temp_output = torch.argmax(outputs, dim=1, keepdim=True)[random_int, ...] * 255
            writer.add_image('train/Prediction',
                             np.array(temp_output.cpu(), dtype='uint8'), epoch)
            temp_label = label_batch[random_int, ...].unsqueeze(0) * 255
            writer.add_image('train/GroundTruth', np.array(temp_label.cpu(), dtype='uint8'), epoch)

        # used for calculating metrics
        preds = torch.argmax(outputs, dim=1, keepdim=True)
        target = torch.unsqueeze(label_batch, dim=1)
        # print(f'preds:{preds.shape}, target:{target.shape}')
        metric_collection.update(preds, target)

        loop.set_description(f'train, epoch [{epoch}/{args.epochs}]')
        loop.set_postfix(loss=loss.item())

    # calculating metrics
    metric_dict_ = metric_collection.compute()
    metric_dict = dict()
    for key, value in metric_dict_.items():
        metric_dict[key] = value.item()
    metric_collection.reset()

    metric_dict['loss_average'] = np.mean(loss_list)
    metric_dict['lr_average'] = np.mean(lr_list)
    return metric_dict


def validate(val_loader, model, criterion, epoch, writer, args, split, metric_collection):
    model.eval()

    # used for calculating metrics
    loss_list = []  # loss per iteration

    if split == 'test':
        other_metric_lsit_dict = {'accuracy':[],
                                  'sensitivity':[],
                                  'specificity':[],
                                  'precision':[]}
        other_metric_collection = get_cls_metric_collection(device='', num_classes=args.num_classes)

    with torch.no_grad():
        iterations_per_epoch = len(val_loader)
        random_show_img_iteration = random.randint(0, (iterations_per_epoch - 1))  # Randomly display the images in iteration
        loop = tqdm(val_loader, total=iterations_per_epoch)
        for i_batch, sampled_batch in enumerate(loop):
            if 'seg_label' in sampled_batch:
                label_name = 'seg_label'
            else:
                label_name = 'label'
            image_batch, label_batch = sampled_batch['image'], sampled_batch[label_name]
            batch_size = image_batch.shape[0]
            image_name_batch = sampled_batch['image_name']

            image_batch, label_batch = image_batch.cuda(), label_batch.long().cuda()

            outputs = model(image_batch)
            # outputs_soft = torch.softmax(outputs, dim=1)
            loss = criterion(outputs, label_batch)  # compute loss
            loss_list.append(loss.item())

            # saving image
            if i_batch == random_show_img_iteration:
                random_int = random.randint(0, (batch_size - 1))
                temp_image = image_batch[random_int, ...]
                if args.images_normalized:
                    temp_image = temp_image * 255
                writer.add_image(f'{split}/Image', np.array(temp_image.cpu(), dtype='uint8'), epoch, dataformats="CHW")
                temp_output = torch.argmax(outputs, dim=1, keepdim=True)[random_int, ...] * 255
                writer.add_image(f'{split}/Prediction',
                                 np.array(temp_output.cpu(), dtype='uint8'), epoch)
                temp_label = label_batch[random_int, ...].unsqueeze(0) * 255
                writer.add_image(f'{split}/GroundTruth', np.array(temp_label.cpu(), dtype='uint8'), epoch)

            # used for calculating metrics
            preds = torch.argmax(outputs, dim=1, keepdim=True)
            target = torch.unsqueeze(label_batch, dim=1)
            metric_collection.update(preds, target)

            if split == 'test':
                other_metric_ = other_metric_collection(torch.flatten(preds), torch.flatten(target))
                for key, value in other_metric_.items():
                    other_metric_lsit_dict[key].append(value.item())

            loop.set_description(f'{split}, epoch [{epoch}/{args.epochs}]')
            loop.set_postfix(loss=loss.item())

        # calculating metrics
        metric_dict_ = metric_collection.compute()
        metric_dict = dict()
        for key, value in metric_dict_.items():
            metric_dict[key] = value.item()
        metric_collection.reset()

        if split == 'test':
            for key, value in other_metric_lsit_dict.items():
                metric_dict[key] = np.mean(value)
            other_metric_collection.reset()

        metric_dict['loss_average'] = np.mean(loss_list)

        return metric_dict


def get_seg_metric_collection(device='', num_classes=2, input_format='index'):
    if device == '' or (device not in ['cuda', 'cpu']):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # DiceScore, MeanIoU
    metric_collection = MetricCollection({
        'dice': DiceScore(num_classes=num_classes,
                          include_background=False,
                          average='macro',  # "micro", "macro", "weighted", "none"
                          input_format=input_format), # "one-hot", "index"
        'miou': MeanIoU(num_classes=num_classes,
                        include_background=False,
                        per_class=False,
                        input_format=input_format),
    }).to(device)
    return metric_collection


def get_cls_metric_collection(device='', num_classes=2):
    if device == '' or (device not in ['cuda', 'cpu']):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Accuracy, Sensitivity/Recall, Specificity, Precision, Dice/F1Score
    if num_classes == 2:
        task = 'binary'
    else:
        task = 'multiclass'
    # task: "binary", "multiclass", "multilabel"
    # average: "micro", "macro", "weighted", "none"
    metric_collection = MetricCollection({
        'accuracy': Accuracy(task=task, average='macro', num_classes=num_classes),
        'sensitivity': Recall(task=task, num_classes=num_classes, average='macro'),
        'specificity': Specificity(task=task, num_classes=num_classes, average='macro'),
        'precision': Precision(task=task, num_classes=num_classes, average='macro'),
    }).to(device)
    return metric_collection


def main():
    # get parameters
    args = parse_option()

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # logs
    base_lr_ = str(args.base_lr).split('.')[-1]
    args.snapshot_path = f'../MedImgSeg_save_model/{args.dataset}/{args.model}/{args.exp_name}_{args.patch_size[0]}x{args.patch_size[1]}_{args.batch_size}bs_{args.epochs}eps_{base_lr_}lr'

    if not os.path.exists(args.snapshot_path):
        os.makedirs(args.snapshot_path)
    if os.path.exists(args.snapshot_path + '/code'):
        shutil.rmtree(args.snapshot_path + '/code')
    shutil.copytree('.', args.snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=args.snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    # build data loader
    train_loader, val_loader, test_loader = set_loader(args)

    args.max_iterations = len(train_loader) * args.epochs
    logging.info(str(args))

    # build model and criterion
    model, criterion = set_model(args)

    # build optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)

    # tensorboardX
    writer = SummaryWriter(args.snapshot_path + '/log')
    logging.info(f'{len(train_loader)} iterations per epoch')

    # training routine
    best_performance = -0.1
    metric_collection = get_seg_metric_collection(device='', num_classes=args.num_classes, input_format='index')
    for epoch in range(1, args.epochs + 1):

        # train for one epoch
        time1 = time.time()
        train_metric_dict = train(train_loader, model, criterion, optimizer, epoch, writer, args, metric_collection)
        time2 = time.time()
        train_epoch_time = (time2 - time1)

        logging.info(f'train: epoch {epoch}, total time {train_epoch_time:.2f}, '
                     f'lr {train_metric_dict["lr_average"]}, loss {train_metric_dict["loss_average"]}, '
                     f'dice {train_metric_dict["dice"]}, miou {train_metric_dict["miou"]}')

        # train tensorboard logger
        writer.add_scalar('info/lr', train_metric_dict['lr_average'], epoch)
        writer.add_scalar('info/train_loss', train_metric_dict['loss_average'], epoch)
        writer.add_scalar('info/train_dice', train_metric_dict['dice'], epoch)
        writer.add_scalar('info/train_miou', train_metric_dict['miou'], epoch)

        # evaluation
        time1 = time.time()
        val_metric_dict = validate(val_loader, model, criterion, epoch, writer, args, 'val', metric_collection)
        time2 = time.time()
        val_epoch_time = (time2 - time1)
        logging.info(f'val: epoch {epoch}, total time {val_epoch_time:.2f}, loss {val_metric_dict["loss_average"]}, '
                     f'dice {val_metric_dict["dice"]}, miou {val_metric_dict["miou"]}')

        # evaluation tensorboard logger
        writer.add_scalar('info/val_loss', val_metric_dict['loss_average'], epoch)
        writer.add_scalar('info/val_dice', val_metric_dict['dice'], epoch)
        writer.add_scalar('info/val_miou', val_metric_dict['miou'], epoch)

        # save the best model of the validation set
        val_dice = val_metric_dict['dice']
        if val_dice > best_performance:
            best_performance = val_dice
            save_mode_path = os.path.join(args.snapshot_path, f'epoch_{epoch}_dice_{round(best_performance, 4)}.pth')
            save_best = os.path.join(args.snapshot_path, f'{args.model}_best_model.pth')
            torch.save(model.state_dict(), save_mode_path)
            torch.save(model.state_dict(), save_best)
            logging.info(f'save model to {save_mode_path}')

        # save the model every 50 epochs
        if epoch % 50 == 0:
            save_mode_path = os.path.join(args.snapshot_path, f'epoch_{epoch}.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info(f'save model to {save_mode_path}')

    logging.info(f'best dice: {best_performance:.4f}')

    # test
    if args.has_val_set:
        model = net_factory(args)
        ckpt = torch.load(os.path.join(args.snapshot_path, f'{args.model}_best_model.pth'))  # loading model

        if (args.need_replace_module == 1) or (args.need_replace_module == '1'):
            new_state_dict = {}
            for k, v in ckpt.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
        else:
            new_state_dict = ckpt

        model.load_state_dict(new_state_dict)

        epoch = 0
        time1 = time.time()
        test_metric_dict = validate(test_loader, model, criterion, epoch, writer, args, 'test', metric_collection)
        time2 = time.time()
        test_time = (time2 - time1)

        logging.info(f'test: total time {test_time:.2f}, '
                     f'loss {test_metric_dict["loss_average"]}, accuracy {test_metric_dict["accuracy"]}, '
                     f'sensitivity {test_metric_dict["sensitivity"]}, specificity {test_metric_dict["specificity"]}, '
                     f'precision {test_metric_dict["precision"]}, dice {test_metric_dict["dice"]}, '
                     f'miou {test_metric_dict["miou"]}')

        # evaluation tensorboard logger
        writer.add_scalar('info/test_loss', test_metric_dict['loss_average'], epoch)
        writer.add_scalar('info/test_accuracy', test_metric_dict['accuracy'], epoch)
        writer.add_scalar('info/test_sensitivity', test_metric_dict['sensitivity'], epoch)
        writer.add_scalar('info/test_specificity', test_metric_dict['specificity'], epoch)
        writer.add_scalar('info/test_precision', test_metric_dict['precision'], epoch)
        writer.add_scalar('info/test_dice', test_metric_dict['dice'], epoch)
        writer.add_scalar('info/test_miou', test_metric_dict['miou'], epoch)

    writer.close()
    return 'Training Finished!'


if __name__ == "__main__":
    main()
