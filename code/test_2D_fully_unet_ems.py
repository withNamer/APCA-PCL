import argparse
import os
import shutil

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
from networks.vision_transformer import SwinUnet as ViT_seg

# from networks.efficientunet import UNet
from networks.net_factory import net_factory
from config import get_config

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='ACDC/Fully_Supervised', help='experiment_name')
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4, help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=3, help='labeled data')

# parser = argparse.ArgumentParser()
# parser.add_argument('--root_path', type=str,
#                     default='../data/ACDC', help='Name of Experiment')
# parser.add_argument('--exp', type=str,
#                     default='ACDC/Cross_teaching_min_max', help='experiment_name')
# parser.add_argument('--model', type=str,
#                     default='unet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum iteration number to train')
parser.add_argument('--batch_size', type=int, default=2,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[224, 224],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
# parser.add_argument('--num_classes', type=int,  default=4,
#                     help='output channel of network')

parser.add_argument("--load", default=True, action="store_true", help="restore previous checkpoint")
parser.add_argument(
    "--conf_thresh",
    type=float,
    default=0.95,
    help="confidence threshold for using pseudo-labels",
)
# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=1,
                    help='labeled_batch_size per epoch')
# parser.add_argument('--labeled_num', type=int, default=35,
#                     help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.999, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency1', type=float,
                    default=1, help='consistency')
parser.add_argument('--consistency2', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')

parser.add_argument(
    '--cfg', type=str, default="../code/configs/swin_tiny_patch4_window7_224_lite.yaml", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                    'full: cache all data, '
                    'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')

args = parser.parse_args()
config = get_config(args)

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    # asd = metric.binary.asd(pred, gt)
    # hd95 = metric.binary.hd95(pred, gt)
    return dice
    # , hd95
    # , asd

def normalize(tensor):
    min_val = tensor.min(1, keepdim=True)[0]
    max_val = tensor.max(1, keepdim=True)[0]
    result = tensor - min_val
    result = result / max_val
    return result

def test_single_volume_ems(case, net1, net2, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (224 / x, 224 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net1.eval()
        net2.eval()
        with torch.no_grad():
            if FLAGS.model == "unet_urpc": # unet_urds
                out_main1, _, _ = net1(input)
                out_main2, _, _ = net2(input)
            else:
                out_main1 = net1(input)
                out_main2 = net2(input)

            out_main1_soft = torch.softmax(out_main1, dim=1)
            out_main2_soft = torch.softmax(out_main2, dim=1)

            # normalize_1 =  normalize(out_main1_soft)
            # normalize_2 =  normalize(out_main2_soft)

            # mask_h1 = (normalize_1 > 0.95)
            # mask_h2 = (normalize_2 > 0.95)
            # mask_m1 = (normalize_1 < 0.95) & (normalize_1 > 0.95)
            # mask_m2 = (normalize_2 < 0.95) & (normalize_2 > 0.95)
            # mask_l1 = (normalize_1 < 0.95)
            # mask_l2 = (normalize_2 < 0.95)
            
            # mask_hh = mask_h1 & mask_h2
            # pseudo_outputs_hh = (mask_hh.float() * normalize_1 + mask_hh.float() * normalize_2) / 2

            # mask_hm = mask_h1 & mask_m2
            # pseudo_outputs_hm = (mask_hm.float() * normalize_1 + mask_hm.float() * normalize_2 * 0.5) / 2
            # mask_mh = mask_m1 & mask_h2
            # pseudo_outputs_mh = (mask_mh.float() * 0.5 * normalize_1 + mask_mh.float() * normalize_2) / 2

            # mask_hl = mask_h1 & mask_l2
            # pseudo_outputs_hl = (mask_hl.float() * normalize_1 + mask_hl.float() * normalize_2 * 0.0) / 2
            # mask_lh = mask_l1 & mask_h2
            # pseudo_outputs_lh = (mask_lh.float() * normalize_1 * 0.0 + mask_lh.float() * normalize_2) / 2
            
            # mask_mm_ml_lm = (~mask_h1) & (~mask_h2)
            # pseudo_outputs_mm_ml_lm = (mask_mm_ml_lm.float() * normalize_1 * 0.0 + mask_mm_ml_lm.float() * normalize_2 * 0.0) / 2 

            # out = torch.argmax((pseudo_outputs_hh + pseudo_outputs_hm + pseudo_outputs_mh + pseudo_outputs_hl + pseudo_outputs_lh).detach(), dim=1, keepdim=False).squeeze(0) #  + pseudo_outputs_mm_ml_lm

            # out_main1_sum = torch.sum((0.7 < out_main1_soft) * (out_main1_soft < 0.95))
            # out_main1_sum_ = torch.sum((0.1 < out_main1_soft) * (out_main1_soft < 0.7))
            
            # out_main2_sum = torch.sum((0.7 < out_main2_soft) * (out_main2_soft < 0.95))
            # out_main2_sum_ = torch.sum((0.1 < out_main2_soft) * (out_main2_soft < 0.7))

            # out = torch.argmax(torch.softmax((out_main1 + out_main2) / 2, dim=1), dim=1).squeeze(0)
            out = torch.argmax((torch.softmax(out_main1, dim=1) + torch.softmax(out_main2, dim=1)) / 2, dim=1).squeeze(0)

            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 224, y / 224), order=0)
            prediction[ind] = pred

    first_metric = calculate_metric_percase(prediction == 1, label == 1)
    second_metric = calculate_metric_percase(prediction == 2, label == 2)
    third_metric = calculate_metric_percase(prediction == 3, label == 3)

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return first_metric, second_metric, third_metric


def Inference(FLAGS):
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0] for item in image_list])
    # snapshot_path = "../model/{}_{}_labeled/{}".format(
    # snapshot_path = "../model/{}_{}/{}".format(FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    snapshot_path = "/ldap_shared/home/s_fsw/mySSL_CL/model/ACDC/Cross_teaching_min_max_3_labeled/unet/2024-11-24 00:51:14"
    # test_save_path = "../model/{}_{}_labeled/{}_predictions/".format(
    test_save_path = "../model/{}_{}/{}_predictions/".format(FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    # net = net_factory(net_type=FLAGS.model, in_chns=1, class_num=FLAGS.num_classes)

    # net1 = ViT_seg(config, img_size=[224, 224], num_classes=4).cuda()
    net1 = net_factory(net_type='unet', in_chns=1, class_num=4)
    net1 = torch.nn.DataParallel(net1)
    save_mode_path = '/ldap_shared/home/s_fsw/TraCoCo/Code/UnetACDC/saved/traCoCo(3-label, unsup_weight=1.0)[ACDC]/model10.8987240828807729_6850.pth'
    net1.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net1.eval()

    # net2 = ViT_seg(config, img_size=[224, 224], num_classes=4).cuda()
    net2 = net_factory(net_type='unet', in_chns=1, class_num=4)
    net2 = torch.nn.DataParallel(net2)
    save_mode_path = '/ldap_shared/home/s_fsw/TraCoCo/Code/UnetACDC/saved/traCoCo(3-label, unsup_weight=1.0)[ACDC]/model20.8947960926210732_5575.pth'
    net2.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net2.eval()

    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    for case in tqdm(image_list):
        first_metric, second_metric, third_metric = test_single_volume_ems(case, net1, net2, test_save_path, FLAGS)
        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)
    avg_metric = [first_total / len(image_list), second_total / len(image_list), third_total / len(image_list)]
    return avg_metric

if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric = Inference(FLAGS)
    print(metric)
    print((metric[0]+metric[1]+metric[2])/3)
