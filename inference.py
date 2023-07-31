import cv2
import torch
import random
import argparse
from glob import glob
from os.path import join
from model.common import freeze_weights
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch.transforms import ToTensorV2
from torchsummary import summary

# fix random seed
seed = 0
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser(description="This code helps you use a trained model to "
                                             "do inference.")
parser.add_argument("--weight", "-w",
                    type=str,
                    default=None,
                    help="Specify the path to the model weight (the state dict file). "
                         "Do not use this argument when '--bin' is set.")
parser.add_argument("--bin", "-b",
                    type=str,
                    default=None,
                    help="Specify the path to the model bin which ends up with '.bin' "
                         "(which is generated by the trainer of this project). "
                         "Do not use this argument when '--weight' is set.")
parser.add_argument("--image", "-i",
                    type=str,
                    default=None,
                    help="Specify the path to the input image. "
                         "Do not use this argument when '--image_folder' is set.")
parser.add_argument("--image_folder", "-f",
                    type=str,
                    default=None,
                    help="Specify the directory to evaluate all the images. "
                         "Do not use this argument when '--image' is set.")
parser.add_argument("--image_file", "-f2",
                    type=str,
                    default=None,
                    help="Specify the directory to evaluate all the images. "
                         "Do not use this argument when '--image' is set.")
parser.add_argument('--device', '-d', type=str,
                    default="cuda:2",
                    help="Specify the device to load the model. Default: 'cpu'.")
parser.add_argument('--image_size', '-s', type=int,
                    default=299,
                    help="Specify the spatial size of the input image(s). Default: 299.")
parser.add_argument('--visualize', '-v', action="store_true",
                    default=False, help='Visualize images.')

# 4476/6067
# 4627/6067

def preprocess(file_path):
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    compose = Compose([Resize(height=args.image_size, width=args.image_size),
                       Normalize(mean=[0.5] * 3, std=[0.5] * 3),
                       ToTensorV2()])
    img = compose(image=img)['image'].unsqueeze(0)
    return img

def prepare_data():
    paths = list()
    images = list()
    # check the console arguments
    if args.image and args.image_folder:
        raise ValueError("Only one of '--image' or '--image_folder' can be set.")
    elif args.image:
        images.append(preprocess(args.image))
        paths.append(args.image)
    elif args.image_folder:
        image_paths = glob(join(args.image_folder, "*.jpg"))
        image_paths.extend(glob(join(args.image_folder, "*.png")))
        for _ in image_paths:
            images.append(preprocess(_))
            paths.append(_)
    elif args.image_file:
        labels = []
        for line in open(args.image_file):
            line = line.strip()
            paths.append(line.split()[0])
            images.append(preprocess(line.split()[0]))
            labels.append(line.split()[2])
    else:
        raise ValueError("Neither of '--image' nor '--image_folder' is set. Please specify either "
                         "one of these two arguments to load input image(s) properly.")
    return paths, images, labels


def prepare_data_aux():
    paths  = list()
    images = list()
    auxs   = list()
    # check the console arguments
    if args.image and args.image_folder:
        raise ValueError("Only one of '--image' or '--image_folder' can be set.")
    elif args.image:
        images.append(preprocess(args.image))
        paths.append(args.image)
    elif args.image_folder:
        image_paths = glob(join(args.image_folder, "*.jpg"))
        image_paths.extend(glob(join(args.image_folder, "*.png")))
        for _ in image_paths:
            images.append(preprocess(_))
            paths.append(_)
    elif args.image_file:
        labels = []
        for line in open(args.image_file):
            line = line.strip()
            paths.append(line.split()[0])
            images.append(preprocess(line.split()[0]))
            auxs.append(preprocess(line.split()[1]))
            labels.append(line.split()[2])
    else:
        raise ValueError("Neither of '--image' nor '--image_folder' is set. Please specify either "
                         "one of these two arguments to load input image(s) properly.")
    return paths, images, auxs, labels


def inference(model, images, paths, labels, device):
    true_num  = 0
    total_num = 0

    for img, pt, label in zip(images, paths, labels):
        total_num += 1
        img = img.to(device)
        prediction = model(img)
        prediction = torch.sigmoid(prediction).cpu()
        fake = True if prediction >= 0.5 else False
        print(f"path: {pt} \t\t| fake probability: {prediction.item():.4f} \t| "
              f"prediction: {'fake' if fake else 'real'}")
        if prediction >= 0.5:
            res = True
        else:
            res = False
        if res == bool(int(label)):
            true_num += 1
        if args.visualize:
            cvimg = cv2.imread(pt)
            cvimg = cv2.putText(cvimg, f'p: {prediction.item():.2f}, ' + f"{'fake' if fake else 'real'}",
                                (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 255) if fake else (255, 0, 0), 2)
            cv2.imshow("image", cvimg)
            cv2.waitKey(0)
            cv2.destroyWindow("image")
    return true_num, total_num

def inference_aux(model, paths, images, auxs, labels, device):
    true_num  = 0
    total_num = 0
    for path, image, aux, label in zip(paths, images, auxs, labels):
        total_num += 1
        image = image.to(device)
        aux = aux.to(device)
        prediction = model(image, aux)
        prediction = torch.sigmoid(prediction).cpu()
        print(prediction)
        fake = True if prediction >= 0.5 else False
        # print(f"path: {path} \t\t| fake probability: {prediction.item():.4f} \t| "
        #       f"prediction: {'fake' if fake else 'real'}")
        if prediction > 0.5:
            res = True
        else:
            res = False
        if res == bool(int(label)):
            true_num = true_num + 1
        # if args.visualize:
        #     cvimg = cv2.imread(pt)
        #     cvimg = cv2.putText(cvimg, f'p: {prediction.item():.2f}, ' + f"{'fake' if fake else 'real'}",
        #                         (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #                         (0, 0, 255) if fake else (255, 0, 0), 2)
        #     cv2.imshow("image", cvimg)
        #     cv2.waitKey(0)
        #     cv2.destroyWindow("image")
    return true_num, total_num


def main():
    from model.network import BRCNet
    print("Arguments:\n", args, end="\n\n")
    # set device
    device = torch.device(args.device)
    # load model
    model_name = "BRCNet"
    model = eval(model_name)(num_classes=1)
    # summary(model, [(32, 3, 299, 299), [32, 3, 299, 299]], device='cpu')
    # print(summary(model, [(32, 3, 299, 299), [32, 3, 299, 299]], device='cpu'))
    # check the console arguments
    if args.weight and args.bin:
        raise ValueError("Only one of '--weight' or '--bin' can be set.")
    elif args.weight:
        weights = torch.load(args.weight, map_location="cpu")
    elif args.bin:
        weights = torch.load(args.bin, map_location="cpu")["model"]
    else:
        raise ValueError("Neither of '--weight' nor '--bin' is set. Please specify either "
                         "one of these two arguments to load model's weight properly.")
    model.load_state_dict(weights)
    model = model.to(device)
    freeze_weights(model)
    model.eval()

    # paths, images, labels = prepare_data()
    paths, images, auxs, labels = prepare_data_aux()
    print("Inference:")
    true_num, total_num = inference_aux(model, paths, images, auxs, labels, device)
    # true_num, total_num = inference(model, images, paths, labels, device)
    print("预测正确的图像数: ", true_num)
    print("总图像数: ", total_num)


if __name__ == '__main__':
    args = parser.parse_args()
    main()
