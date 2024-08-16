import argparse
import torch
import numpy as np
import scipy
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import imageio
import cv2
import os
from pathlib import Path
import tqdm
from utils.factory import create_model_and_transforms
from utils.imagenet_segmentation import ImagenetSegmentation
from utils.segmentation_utils import (batch_pix_accuracy, batch_intersection_union, 
                                      get_ap_scores, Saver)
from sklearn.metrics import precision_recall_curve
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Args
def get_args_parser():
    parser = argparse.ArgumentParser(description='Segmentation scores')
    parser.add_argument('--save_img', action='store_true', default=False, help='')
    parser.add_argument('--train_dataset', type=str, default='imagenet_seg', help='The name of the dataset')
    parser.add_argument('--classifier_dataset', type=str, default='imagenet', help='The name of the classifier dataset')
    parser.add_argument('--image_size', default=224, type=int, help='Image size')
    parser.add_argument('--thr', type=float, default=0., help='threshold')
    parser.add_argument('--data_path', default='imagenet_seg/gtsegs_ijcv.mat', type=str, help='dataset path')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--classifier_dir', default='./output_dir/')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
    parser.add_argument('--model', default='ViT-H-14', type=str, metavar='MODEL', help='Name of model to use')
    parser.add_argument('--pretrained', default='laion2b_s32b_b79k', type=str)
    parser.add_argument('--output_dir', default='./output_dir', help='path where to save')
    parser.add_argument('--device', default='cuda:0', help='device to use for testing')
    return parser

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def eval_batch(model, image, labels, index, args, classifier, saver):
    if args.save_img:
        img = image[0].permute(1, 2, 0).data.cpu().numpy()
        img = 255 * (img - img.min()) / (img.max() - img.min())
        img = img.astype('uint8')
        Image.fromarray(img, 'RGB').save(os.path.join(saver.results_dir, 'input/{}_input.png'.format(index)))
        Image.fromarray((labels.repeat(3, 1, 1).permute(1, 2, 0).data.cpu().numpy() * 255).astype('uint8'), 'RGB').save(
            os.path.join(saver.results_dir, 'input/{}_mask.png'.format(index)))
    
    target_layers = [model.visual.transformer.resblocks[-1].ln_1]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=args.device.startswith('cuda'),
                  reshape_transform=reshape_transform)

    with torch.no_grad():
        representation = model.encode_image(image.to(args.device))
    chosen_class = (representation.detach().cpu().numpy() @ classifier).argmax(axis=1)

    grayscale_cam = cam(input_tensor=image.to(args.device), target_category=chosen_class[0])
    grayscale_cam = grayscale_cam[0, :]

    Res = torch.from_numpy(grayscale_cam).unsqueeze(0).unsqueeze(0).to(args.device)
    Res = torch.nn.functional.interpolate(Res, size=(args.image_size, args.image_size), mode='bilinear')
    Res = (Res - Res.min()) / (Res.max() - Res.min())

    ret = Res.mean()

    Res_1 = Res.gt(ret).type(Res.type())
    Res_0 = Res.le(ret).type(Res.type())

    Res_1_AP = Res
    Res_0_AP = 1-Res

    Res_1[Res_1 != Res_1] = 0
    Res_0[Res_0 != Res_0] = 0
    Res_1_AP[Res_1_AP != Res_1_AP] = 0
    Res_0_AP[Res_0_AP != Res_0_AP] = 0

    pred = Res.clamp(min=args.thr) / Res.max()
    pred = pred.view(-1).data.cpu().numpy()
    target = labels.view(-1).data.cpu().numpy()

    output = torch.cat((Res_0, Res_1), 1)
    output_AP = torch.cat((Res_0_AP, Res_1_AP), 1)

    if args.save_img:
        mask = Res_1[0].squeeze().data.cpu().numpy()
        mask = 255 * mask
        mask = mask.astype('uint8')
        imageio.imsave(os.path.join(args.exp_img_path, 'mask_' + str(index) + '.jpg'), mask)

        heatmap = show_cam_on_image(img / 255., grayscale_cam, use_rgb=True)
        imageio.imsave(os.path.join(args.exp_img_path, 'heatmap_' + str(index) + '.jpg'), heatmap)

    correct, labeled = batch_pix_accuracy(output[0].data.cpu(), labels[0])
    inter, union = batch_intersection_union(output[0].data.cpu(), labels[0], 2)
    ap = np.nan_to_num(get_ap_scores(output_AP, labels))
    
    return correct, labeled, inter, union, ap, pred, target

def _create_saver_and_folders(args):
    saver = Saver(args)
    saver.results_dir = os.path.join(saver.experiment_dir, 'results')
    if not os.path.exists(saver.results_dir):
        os.makedirs(saver.results_dir)
    if not os.path.exists(os.path.join(saver.results_dir, 'input')):
        os.makedirs(os.path.join(saver.results_dir, 'input'))
    if not os.path.exists(os.path.join(saver.results_dir, 'explain')):
        os.makedirs(os.path.join(saver.results_dir, 'explain'))

    args.exp_img_path = os.path.join(saver.results_dir, 'explain/img')
    if not os.path.exists(args.exp_img_path):
        os.makedirs(args.exp_img_path)
    return saver

def main(args):
    model, _, preprocess = create_model_and_transforms(args.model, pretrained=args.pretrained)
    model.to(args.device)
    model.eval()

    target_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size), Image.NEAREST),
    ])

    ds = ImagenetSegmentation(args.data_path, transform=preprocess, target_transform=target_transform)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    iterator = tqdm.tqdm(dl)
    
    saver = _create_saver_and_folders(args) 
    
    with open(os.path.join(args.classifier_dir, f'{args.classifier_dataset}_classifier_{args.model}.npy'), 'rb') as f:
        classifier = np.load(f)
    
    total_inter, total_union, total_correct, total_label = np.int64(0), np.int64(0), np.int64(0), np.int64(0)
    total_ap = []

    predictions, targets = [], []
    for batch_idx, (image, labels) in enumerate(iterator):
        correct, labeled, inter, union, ap, pred, target = eval_batch(model, image, labels, batch_idx, args, classifier, saver)

        predictions.append(pred)
        targets.append(target)

        total_correct += correct.astype('int64')
        total_label += labeled.astype('int64')
        total_inter += inter.astype('int64')
        total_union += union.astype('int64')
        total_ap += [ap]
        pixAcc = np.float64(1.0) * total_correct / (np.spacing(1, dtype=np.float64) + total_label)
        IoU = np.float64(1.0) * total_inter / (np.spacing(1, dtype=np.float64) + total_union)
        mIoU = IoU.mean()
        mAp = np.mean(total_ap)
        iterator.set_description('pixAcc: %.4f, mIoU: %.4f, mAP: %.4f' % (pixAcc, mIoU, mAp))

    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    pr, rc, thr = precision_recall_curve(targets, predictions)
    np.save(os.path.join(saver.experiment_dir, 'precision.npy'), pr)
    np.save(os.path.join(saver.experiment_dir, 'recall.npy'), rc)

    txtfile = os.path.join(saver.experiment_dir, 'result_mIoU_%.4f.txt' % mIoU)
    with open(txtfile, 'w') as fh:
        fh.write("Mean IoU over %d classes: %.4f\n" % (2, mIoU))
        fh.write("Pixel-wise Accuracy: %2.2f%%\n" % (pixAcc * 100))
        fh.write("Mean AP over %d classes: %.4f\n" % (2, mAp))

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)