
"""
通过实现Grad-CAM学习module中的forward_hook和backward_hook函数
"""
import cv2
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.nn import BCELoss, Sigmoid

def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())

def farward_hook(module, input, output):
    fmap_block.append(output)

def show_cam_on_image(img, mask, out_dir):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)

    path_cam_img = os.path.join(out_dir, "cam.jpg")
    path_raw_img = os.path.join(out_dir, "raw.jpg")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    cv2.imwrite(path_cam_img, np.uint8(255 * cam))
    cv2.imwrite(path_raw_img, np.uint8(255 * img))

def comp_class_vec(ouput, label):
    output = Sigmoid(ouput)
    loss = BCELoss(output, label.long())
    return loss

def gen_cam(feature_map, grads):
    """
    依据梯度和特征图，生成cam
    :param feature_map: np.array， in [C, H, W]
    :param grads: np.array， in [C, H, W]
    :return: np.array, [H, W]
    """
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # cam shape (H, W)
    weights = np.mean(grads, axis=(1, 2))  #
    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (32, 32))
    cam -= np.min(cam)
    cam /= np.max(cam)
    return cam

if __name__ == '__main__':

    # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # path_img = os.path.join(BASE_DIR, "..", "..", "Data", "cam_img", "test_img_8.png")
    # path_net = os.path.join(BASE_DIR, "..", "..", "Data", "net_params_72p.pkl")
    output_dir = "heatmap_sample"

    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    fmap_block = list()
    grad_block = list()

    # 图片读取；网络加载
    # img = cv2.imread(path_img, 1)  # H*W*C
    # img_input = img_preprocess(img)
    # net = Net()
    # net.load_state_dict(torch.load(path_net))

    # 注册hook
    net.Conv2d-356.register_forward_hook(farward_hook)
    net.Conv2d-356.register_backward_hook(backward_hook)

    # forward
    output = net(img_input)
    # idx = np.argmax(output.cpu().data.numpy())
    # print("predict: {}".format(classes[idx]))

    # backward
    net.zero_grad()
    class_loss = comp_class_vec(output, label)
    class_loss.backward()

    # 生成cam
    grads_val = grad_block[0].cpu().data.numpy().squeeze()
    fmap = fmap_block[0].cpu().data.numpy().squeeze()
    cam = gen_cam(fmap, grads_val)

    # 保存cam图片
    img_show = np.float32(cv2.resize(img, (32, 32))) / 255
    show_cam_on_image(img_show, cam, output_dir)











