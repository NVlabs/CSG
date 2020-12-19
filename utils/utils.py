# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under a NVIDIA Open Source Non-commercial license.

import glob
import os
import shutil
import numpy as np
import torch
from pdb import set_trace as bp
from PIL import Image
import matplotlib.cm as mpl_color_map
import copy


def get_params(model, layers=["layer4"]):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    if isinstance(layers, str):
        layers = [layers]
    b = []
    for layer in layers:
        b.append(getattr(model, layer))

    for i in range(len(b)):
        for k, v in b[i].named_parameters():
            if v.requires_grad:
                yield v


def adjust_learning_rate_exp(optimizer, power=0.746):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    num_groups = len(optimizer.param_groups)
    for g in range(num_groups):
        optimizer.param_groups[g]['lr'] *= power


def adjust_learning_rate(base_lrs, optimizer, iter_curr, iter_max, power):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    num_groups = len(optimizer.param_groups)
    for g in range(num_groups):
        optimizer.param_groups[g]['lr'] = lr_poly(base_lrs[g], iter_curr, iter_max, power)


def lr_poly(base_lr, iter, max_iter, power):
    return min(0.01+0.99*(float(iter)/100)**2.0, 1.0) * base_lr * ((1-float(iter)/max_iter)**power)  # This is with warm up
    # return min(0.01+0.99*(float(iter)/100)**2.0, 1.0) * base_lr * ((1-min(float(iter)/max_iter, 0.8))**power)  # This is with warm up & no smaller than last 20% LR
    # return base_lr * ((1-float(iter)/max_iter)**power)


def save_checkpoint(name, state, is_best, filename='checkpoint.pth.tar', keep_last=1):
    """Saves checkpoint to disk"""
    directory = name
    if not os.path.exists(directory):
        os.makedirs(directory)
    models_paths = list(filter(os.path.isfile, glob.glob(directory + "/epoch*.pth.tar")))
    models_paths.sort(key=os.path.getmtime, reverse=False)
    if len(models_paths) == keep_last:
        for i in range(len(models_paths) + 1 - keep_last):
            os.remove(models_paths[i])
    torch.save(state, directory + '/epoch_'+str(state['epoch']) + '_' + filename)
    filename = directory + '/latest_' + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '%s/'%(name) + 'model_best.pth.tar')


class IterNums(object):
    def __init__(self, iter_max):
        self.iter_max = iter_max
        self.iter_curr = 0

    def reset(self):
        self.iter_curr = 0

    def update(self):
        self.iter_curr += 1


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vec2sca_avg = 0
        self.vec2sca_val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if torch.is_tensor(self.val) and torch.numel(self.val) != 1:
            self.avg[self.count == 0] = 0
            self.vec2sca_avg = self.avg.sum() / len(self.avg)
            self.vec2sca_val = self.val.sum() / len(self.val)


def accuracy(output, label, num_class, topk=(1,)):
    """Computes the precision@k for the specified values of k, currently only k=1 is supported"""
    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)
    if len(label.size()) == 2:
        # one_hot label
        _, gt = label.topk(maxk, 1, True, True)
    else:
        gt = label
    pred = pred.t()
    pred_class_idx_list = [pred == class_idx for class_idx in range(num_class)]
    gt = gt.t()
    gt_class_number_list = [(gt == class_idx).sum() for class_idx in range(num_class)]
    correct = pred.eq(gt)

    res = []
    gt_num = []
    for k in topk:
        correct_k = correct[:k].float()
        per_class_correct_list = [correct_k[pred_class_idx].sum(0) for pred_class_idx in pred_class_idx_list]
        per_class_correct_array = torch.tensor(per_class_correct_list)
        gt_class_number_tensor = torch.tensor(gt_class_number_list).float()
        gt_class_zeronumber_tensor = gt_class_number_tensor == 0
        gt_class_number_matrix = torch.tensor(gt_class_number_list).float()
        gt_class_acc = per_class_correct_array.mul_(100.0 / gt_class_number_matrix)
        gt_class_acc[gt_class_zeronumber_tensor] = 0
        res.append(gt_class_acc)
        gt_num.append(gt_class_number_matrix)
    return res, gt_num


def apply_colormap_on_image(org_im, activation, colormap_name='hsv'):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.5
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

    # Apply heatmap on iamge
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image


class UnNormalize(object):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

