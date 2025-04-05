import argparse
import logging
import os
import random
import shutil
import sys
from datetime import datetime
from info_nce import *
from collections import Counter
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.losses import contrastive_loss_sup_seq, NegEntropy, semi_crc_loss, semi_mycrc_loss
from config import get_config
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataloaders.dataset import (
    BaseDataSets,
    CTATransform,
    TwoStreamBatchSampler,
)
from networks.vision_transformer import SwinUnet as ViT_seg
from networks.net_factory import net_factory
from utils import losses, ramps, util
from val_2D import test_single_volume
from utils.warm_up import GradualWarmupScheduler
import augmentations
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='ACDC/Cross_teaching_min_max', help='experiment_name')
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum iteration number to train')
parser.add_argument('--batch_size', type=int, default=32, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[224, 224], help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4, help='output channel of network')
parser.add_argument("--load", default=True, action="store_true", help="restore previous checkpoint")
parser.add_argument("--conf_thresh", type=float, default=0.95, help="confidence threshold for using pseudo-labels",)
parser.add_argument('--labeled_bs', type=int, default=16, help='labeled_batch_size per epoch')
parser.add_argument('--labeled_num', type=int, default=3, help='labeled data')
parser.add_argument('--ema_decay', type=float,  default=0.999, help='ema_decay')
parser.add_argument('--consistency_type', type=str, default="mse", help='consistency_type')
parser.add_argument('--consistency1', type=float, default=1, help='consistency')
parser.add_argument('--consistency2', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--cfg', type=str, default="../code/configs/swin_tiny_patch4_window7_224_lite.yaml", help='path to config file', )
parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs. ", default=None, nargs='+',)
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, ' 'full: cache all data, '
                    'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true', help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'], help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')
parser.add_argument('--beta', default=1.0, type=float, help='hyperparameter beta')
parser.add_argument('--cutmix_prob', default=1.0, type=float, help='cutmix probability')

args = parser.parse_args()
config = get_config(args)

os.environ["CUDA_VISIBLE_DEVICES"] = '2,3' # ,2,3

def patients_to_slices(dataset, patiens_num): 
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136, "14": 256, "21": 396, "28": 512, "35": 664, "130":1132,"126":1058, "140": 1312}
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120, "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]

def get_current_consistency_weight(consistency,epoch):  
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)  

def check_models_equal(model1, model2):  
    for p1, p2 in zip(model1.parameters(), model2.parameters()):  
        if not torch.allclose(p1.data, p2.data):  
            return False  
    return True  

def copy_weights(model, model_ema):
    for ema_param, param in zip(model_ema.parameters(), model.parameters()):
        ema_param.data.copy_(param.data)

class KLD(nn.Module):
    def forward(self, inputs, targets):
        inputs = F.log_softmax(inputs, dim=1)
        targets = F.softmax(targets, dim=1)
        return F.kl_div(inputs, targets, reduction='none').mean()  # inputs和targets应该是相同形状

def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    def create_model(net_type,ema=False):
        model = net_factory(net_type=net_type, in_chns=1, class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model1 = ViT_seg(config, img_size=args.patch_size, num_classes=args.num_classes).cuda()
    model1.load_from(config)
    model2 = ViT_seg(config, img_size=args.patch_size, num_classes=args.num_classes).cuda()
    model2.load_from(config)

    projector_1 = create_model('projectors_feature')
    projector_2 = create_model('projectors_feature')
    projector_3 = create_model('projectors_linear')
    projector_4 = create_model('projectors_linear')

    kld_loss = KLD().cuda()

    model1 = torch.nn.DataParallel(model1)
    model2 = torch.nn.DataParallel(model2)
    projector_1 = torch.nn.DataParallel(projector_1)
    projector_2 = torch.nn.DataParallel(projector_2)
    projector_3 = torch.nn.DataParallel(projector_3)
    projector_4 = torch.nn.DataParallel(projector_4)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    
    def normalize(tensor):
        min_val = tensor.min(1, keepdim=True)[0]
        max_val = tensor.max(1, keepdim=True)[0]
        result = tensor - min_val
        result = result / max_val
        return result

    def rand_bbox(size, lam=None):
        W = size[2]
        H = size[3]
        B = size[0]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        cx = np.random.randint(size=[B, ], low=int(W / 8), high=W) # int(W / 8)
        cy = np.random.randint(size=[B, ], low=int(W / 8), high=H)
        # cx = np.random.randint(W)
        # cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2
    
    def cut_mix(volume=None, pseudo_outputs=None, mask=None):
        mix_volume = volume.clone()
        mix_target = mask.clone()
        mix_pseudo = pseudo_outputs.clone()

        u_rand_index = torch.randperm(volume.size()[0])[:volume.size()[0]].cuda()
        u_bbx1, u_bby1, u_bbx2, u_bby2 = rand_bbox(volume.size(), lam=np.random.beta(4, 4))

        for i in range(0, mix_volume.shape[0]):
            mix_volume[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = volume[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
            if len(mix_target.shape) > 3:
                mix_target[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = mask[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
            else:
                mix_target[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = mask[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
            mix_pseudo[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = pseudo_outputs[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        return mix_volume, mix_pseudo, mix_target

    def refresh_policies(db_train, cta,random_depth_weak, random_depth_strong):
        db_train.ops_weak = cta.policy(probe=False, weak=True)
        db_train.ops_strong = cta.policy(probe=False, weak=False)
        cta.random_depth_weak = random_depth_weak
        cta.random_depth_strong = random_depth_strong
        if max(Counter([a.f for a in db_train.ops_weak]).values()) >=3 or max(Counter([a.f for a in db_train.ops_strong]).values()) >= 3:
            print('too deep with one transform, refresh again')
            refresh_policies(db_train, cta,random_depth_weak, random_depth_strong)
        logging.info(f"CTA depth weak: {cta.random_depth_weak}")
        logging.info(f"CTA depth strong: {cta.random_depth_strong}")
        logging.info(f"\nWeak Policy: {db_train.ops_weak}")
        logging.info(f"Strong Policy: {db_train.ops_strong}")

    def A(outputs, mask_bool_all, sample_num):
        prob_all_all_i = outputs[mask_bool_all]
        if len(prob_all_all_i) > sample_num:
            index = np.random.choice(np.arange(0, len(prob_all_all_i)), sample_num, replace=False)
            prob_all_all_i = prob_all_all_i[index]
        return prob_all_all_i
    
    def B(prob_mean_i_cutout):
        with torch.no_grad():
            dot_product_matrix = torch.zeros((args.num_classes, args.num_classes))
            prob_similarity = torch.zeros(args.num_classes)
            for i in range(args.num_classes):
                for j in range(args.num_classes):
                    dot_product_matrix[i, j] = torch.dot(prob_mean_i_cutout[i], prob_mean_i_cutout[j])
                prob_similarity[i] = (torch.sum(dot_product_matrix[i, :i].sum() + dot_product_matrix[i, i+1:].sum()) / (args.num_classes - 1)).item()
            prob_similarity = torch.softmax(prob_similarity / 0.7, dim=0)
        return prob_similarity
    
    def C(prob_all, prob_similarity, num_hard):
        with torch.no_grad():
            len_ = torch.tensor([len(prob_all[idx]) for idx in range(args.num_classes)])
            min_value, min_index = torch.min(torch.floor(prob_similarity * len_), dim=0)
            min_value_ = torch.floor(num_hard * prob_similarity[min_index])
            if min_value > min_value_:
                min_value = min_value_
            len_real = torch.zeros(args.num_classes)
            for i in range(args.num_classes):
                if i == min_index:
                    len_real[i] = min_value
                    continue
                len_real[i] = torch.floor(min_value / prob_similarity[min_index] * prob_similarity[i])
            if (len_real == 0).sum() != 0:
                for i in range(args.num_classes):
                    if len_real[i] == 0:
                        len_real[i] = 1
        # print("len_real: ", len_real)
        return len_, len_real
            
    cta = augmentations.CTAugment()
    transform = CTATransform(args.patch_size, cta)

    ops_weak = cta.policy(probe=False, weak=True)
    ops_strong = cta.policy(probe=False, weak=False)

    db_train = BaseDataSets(
        base_dir=args.root_path,
        split="train",
        num=None,
        transform=transform,
        ops_weak=ops_weak,
        ops_strong=ops_strong,
    )

    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    print("Total silices is: {}, labeled slices is: {}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    trainloader = DataLoader(
        db_train,
        batch_sampler=batch_sampler,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    # print(len(trainloader))
    
    model1.train()
    model2.train()  

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
 
    parameters_to_optimize = list(model1.parameters()) + list(model2.parameters()) + list(projector_3.parameters()) + list(projector_4.parameters())
    optimizer =optim.AdamW(parameters_to_optimize, lr=0.0002, weight_decay=0.02)
    warm_up_iter_num = 10
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=args.max_iterations - warm_up_iter_num, eta_min=1e-8) # args.max_iterations // len(trainloader) + 1
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warm_up_iter_num, after_scheduler=scheduler_cosine)

    iter_num = 0
    start_epoch = 0

    ce_loss = CrossEntropyLoss(ignore_index=-1)
    dice_loss = losses.DiceLoss(num_classes)

    contrastive_loss_sup_criter_seq = contrastive_loss_sup_seq()
    NegEntropy_loss = NegEntropy()

    writer = SummaryWriter(snapshot_path + '/log' + data_flag )
    logging.info("{} iterations per epoch".format(len(trainloader)))

    max_epoch = max_iterations // len(trainloader) + 1
    best_performance1 = 0.0
    best_performance2 = 0.0
    performance1 = 0.0
    performance2 = 0.0
    lr_ = base_lr
    iterator = tqdm(range(start_epoch, max_epoch), ncols=70)

    low_thresh_iter = iter(np.linspace(0.90, 0.95, 30))
    # low_thresh = 0.9

    for epoch_num in iterator:
        epoch_errors = []
        if iter_num <= 10000:
            random_depth_weak = np.random.randint(3, high=5)
            random_depth_strong = np.random.randint(2, high=5)
        elif (iter_num >= 20000):
            random_depth_weak = 2
            random_depth_strong = 2
        elif (iter_num > 10000) and (iter_num < 20000):
            random_depth_weak = np.random.randint(2, high=5)
            random_depth_strong = np.random.randint(2, high=5)

        refresh_policies(db_train, cta,random_depth_weak, random_depth_strong)

        running_loss = 0.0
        running_sup_loss = 0.0
        for i_batch, sampled_batch in enumerate(zip(trainloader)):
            
            raw_batch, weak_batch, strong_batch, label_batch_aug, label_batch = (
                sampled_batch[0]["image"],
                sampled_batch[0]["image_weak"],
                sampled_batch[0]["image_strong"],
                sampled_batch[0]["label_aug"],
                sampled_batch[0]["label"],
            )
            label_batch_aug[label_batch_aug>=args.num_classes] = 0
            label_batch_aug[label_batch_aug<0] = 0

            r = np.random.rand(1)
            if args.beta > 0 and r < args.cutmix_prob:
                lam = np.random.beta(args.beta, args.beta)
       
                rand_index_l = torch.randperm(weak_batch[:args.labeled_bs].shape[0])
                rand_index_u = torch.randperm(weak_batch[:args.labeled_bs].shape[0])

                bbx1_l, bby1_l, bbx2_l, bby2_l = rand_bbox(weak_batch[:args.labeled_bs].shape, lam)
                bbx1_u, bby1_u, bbx2_u, bby2_u = rand_bbox(weak_batch[args.labeled_bs:].shape, lam)
                for i in range(args.labeled_bs):
                    strong_batch[:args.labeled_bs][i, :, bbx1_l[i]:bbx2_l[i], bby1_l[i]:bby2_l[i]] = strong_batch[:args.labeled_bs][rand_index_l[i], :, bbx1_l[i]:bbx2_l[i], bby1_l[i]:bby2_l[i]]
                    weak_batch[:args.labeled_bs][i, :, bbx1_l[i]:bbx2_l[i], bby1_l[i]:bby2_l[i]] = weak_batch[:args.labeled_bs][rand_index_l[i], :, bbx1_l[i]:bbx2_l[i], bby1_l[i]:bby2_l[i]]
                    label_batch_aug[:args.labeled_bs][i, bbx1_l[i]:bbx2_l[i], bby1_l[i]:bby2_l[i]] = label_batch_aug[:args.labeled_bs][rand_index_l[i], bbx1_l[i]:bbx2_l[i], bby1_l[i]:bby2_l[i]]

                    strong_batch[args.labeled_bs:][i, :, bbx1_u[i]:bbx2_u[i], bby1_u[i]:bby2_u[i]] = strong_batch[args.labeled_bs:][rand_index_u[i], :, bbx1_u[i]:bbx2_u[i], bby1_u[i]:bby2_u[i]]
                    weak_batch[args.labeled_bs:][i, :, bbx1_u[i]:bbx2_u[i], bby1_u[i]:bby2_u[i]] = weak_batch[args.labeled_bs:][rand_index_u[i], :, bbx1_u[i]:bbx2_u[i], bby1_u[i]:bby2_u[i]]
                    label_batch_aug[args.labeled_bs:][i, bbx1_u[i]:bbx2_u[i], bby1_u[i]:bby2_u[i]] = label_batch_aug[args.labeled_bs:][rand_index_u[i], bbx1_u[i]:bbx2_u[i], bby1_u[i]:bby2_u[i]]
            
            # r = np.random.rand(1)
            # if args.beta > 0 and r < args.cutmix_prob:
            #     lam = np.random.beta(args.beta, args.beta)
       
            #     rand_index_l = torch.randperm(weak_batch[:args.labeled_bs].shape[0])
            #     rand_index_u = torch.randperm(weak_batch[:args.labeled_bs].shape[0])

            #     # target_a = strong_batch
            #     # target_b = weak_batch[rand_index] 

            #     bbx1_l, bby1_l, bbx2_l, bby2_l = rand_bbox(weak_batch[:args.labeled_bs].shape, lam)
            #     strong_batch[:args.labeled_bs][:, :, bbx1_l:bbx2_l, bby1_l:bby2_l] = strong_batch[:args.labeled_bs][rand_index_l, :, bbx1_l:bbx2_l, bby1_l:bby2_l]
            #     weak_batch[:args.labeled_bs][:, :, bbx1_l:bbx2_l, bby1_l:bby2_l] = weak_batch[:args.labeled_bs][rand_index_l, :, bbx1_l:bbx2_l, bby1_l:bby2_l]
            #     label_batch_aug[:args.labeled_bs][:, bbx1_l:bbx2_l, bby1_l:bby2_l] = label_batch_aug[:args.labeled_bs][rand_index_l, bbx1_l:bbx2_l, bby1_l:bby2_l]

            #     bbx1_u, bby1_u, bbx2_u, bby2_u = rand_bbox(weak_batch[args.labeled_bs:].shape, lam)
            #     strong_batch[args.labeled_bs:][:, :, bbx1_u:bbx2_u, bby1_u:bby2_u] = strong_batch[args.labeled_bs:][rand_index_u, :, bbx1_u:bbx2_u, bby1_u:bby2_u]
            #     weak_batch[args.labeled_bs:][:, :, bbx1_u:bbx2_u, bby1_u:bby2_u] = weak_batch[args.labeled_bs:][rand_index_u, :, bbx1_u:bbx2_u, bby1_u:bby2_u]
            #     label_batch_aug[args.labeled_bs:][:, bbx1_u:bbx2_u, bby1_u:bby2_u] = label_batch_aug[args.labeled_bs:][rand_index_u, bbx1_u:bbx2_u, bby1_u:bby2_u]

            r = np.random.rand(1)
            cutout_flag = False
            if args.beta > 0 and r < 1.0: 
                cutout_flag = True
                lam = np.random.beta(args.beta, args.beta)
                bbx1, bby1, bbx2, bby2 = rand_bbox(strong_batch.shape, lam)
                label_batch_aug_cutout = label_batch_aug.clone().detach()
                for i in range(0, args.batch_size):
                    strong_batch[i, :, bbx1[i]:bbx2[i], bby1[i]:bby2[i]] = 0
                    label_batch_aug_cutout[i, bbx1[i]:bbx2[i], bby1[i]:bby2[i]] = -1
                # strong_batch[:, :, bbx1:bbx2, bby1:bby2] = 0
                # label_batch_aug_cutout[i, bbx1:bbx2, bby1:bby2] = -1
            else:
                label_batch_aug_cutout = label_batch_aug.clone().detach()
            label_batch_aug_cutout = label_batch_aug_cutout.long()
                
            weak_batch, strong_batch, label_batch_aug, label_batch_aug_cutout = (
                weak_batch.cuda(),
                strong_batch.cuda(),
                label_batch_aug.cuda(),
                label_batch_aug_cutout.cuda(),
            )
            
            # handle unfavorable cropping
            non_zero_ratio = torch.count_nonzero(label_batch) / (args.batch_size * 224 * 224)
            non_zero_ratio_aug = torch.count_nonzero(label_batch_aug) / (args.batch_size * 224 * 224)
            if non_zero_ratio > 0 and non_zero_ratio_aug < 0.005:   # try 0.01
                logging.info("Refreshing policy...")
                refresh_policies(db_train, cta,random_depth_weak, random_depth_strong)
#################################################################################################################################
            # noise = torch.zeros_like(weak_batch).uniform_(-.02, .02)
            outputs_weak1,_,_,_,_ = model1(weak_batch)  # outputs_weak1_
            outputs_weak_soft1 = torch.softmax(outputs_weak1, dim=1)  

            # noise = torch.zeros_like(strong_batch).uniform_(-.05, .05)
            outputs_strong1,_,_,_,_ = model1(strong_batch)
            outputs_strong_soft1 = torch.softmax(outputs_strong1, dim=1)
            
            # noise = torch.zeros_like(weak_batch).uniform_(-.02, .02)
            outputs_weak2,_,_,_,_ = model2(weak_batch)
            outputs_weak_soft2 = torch.softmax(outputs_weak2, dim=1)

            # noise = torch.zeros_like(strong_batch).uniform_(-.05, .05)
            outputs_strong2,_,_,_,_ = model2(strong_batch)
            outputs_strong_soft2 = torch.softmax(outputs_strong2, dim=1)
#################################################################################################################################           
            with torch.no_grad():             
                pseudo_mask1 = (normalize(outputs_weak_soft1) > args.conf_thresh).float()
                outputs_weak_soft_masked1 = (normalize(outputs_weak_soft1)) * pseudo_mask1
                pseudo_outputs1 = torch.argmax(outputs_weak_soft_masked1.detach(), dim=1, keepdim=False)
                
                pseudo_mask2 = (normalize(outputs_weak_soft2) > args.conf_thresh).float()
                outputs_weak_soft_masked2 = (normalize(outputs_weak_soft2)) * pseudo_mask2
                pseudo_outputs2 = torch.argmax(outputs_weak_soft_masked2.detach(), dim=1, keepdim=False)
                
                # outputs_weak_soft_masked = (outputs_weak_soft_masked1 + outputs_weak_soft_masked2) / 2
                # pseudo_outputs = torch.argmax(outputs_weak_soft_masked.detach(), dim=1, keepdim=False)

                normalize_1 =  normalize(outputs_weak_soft1)
                normalize_2 =  normalize(outputs_weak_soft2)

                if (iter_num < 24000) and (iter_num % 800 == 0):
                    low_thresh = next(low_thresh_iter)
                
                mask_h1 = (normalize_1 > args.conf_thresh)
                mask_h2 = (normalize_2 > args.conf_thresh)
                mask_m1 = (normalize_1 < args.conf_thresh) & (normalize_1 > low_thresh)
                mask_m2 = (normalize_2 < args.conf_thresh) & (normalize_2 > low_thresh)
                mask_l1 = (normalize_1 < low_thresh)
                mask_l2 = (normalize_2 < low_thresh)
                
                mask_hh = mask_h1 & mask_h2
                pseudo_outputs_hh = (mask_hh.float() * normalize_1 + mask_hh.float() * normalize_2) / 2

                mask_hm = mask_h1 & mask_m2
                pseudo_outputs_hm = (mask_hm.float() * normalize_1 + mask_hm.float() * normalize_2 * 0.5) / 2  
                mask_mh = mask_m1 & mask_h2
                pseudo_outputs_mh = (mask_mh.float() * 0.5 * normalize_1 + mask_mh.float() * normalize_2) / 2

                mask_hl = mask_h1 & mask_l2
                pseudo_outputs_hl = (mask_hl.float() * normalize_1 + mask_hl.float() * normalize_2 * 0.0) / 2
                mask_lh = mask_l1 & mask_h2
                pseudo_outputs_lh = (mask_lh.float() * normalize_1 * 0.0 + mask_lh.float() * normalize_2) / 2
                
                # mask_mm_ml_lm = (~mask_h1) & (~mask_h2)
                # pseudo_outputs_mm_ml_lm = (mask_mm_ml_lm.float() * normalize_1 * 0.0 + mask_mm_ml_lm.float() * normalize_2 * 0.0) / 2 

                pseudo_outputs = torch.argmax((pseudo_outputs_hh + pseudo_outputs_hm + pseudo_outputs_mh + pseudo_outputs_hl + pseudo_outputs_lh).detach(), dim=1, keepdim=False) #  + pseudo_outputs_mm_ml_lm
                if cutout_flag:
                    pseudo_outputs_cutout = pseudo_outputs.clone().detach()
                    for i in range(args.batch_size):
                        pseudo_outputs_cutout[i, bbx1[i]:bbx2[i], bby1[i]:bby2[i]] = -1
                else:
                    pseudo_outputs_cutout = pseudo_outputs.clone().detach() 
                pseudo_outputs_cutout = pseudo_outputs_cutout.long()

            consistency_weight1 = get_current_consistency_weight(args.consistency1, iter_num // 150)
            if iter_num < 0:
                consistency_weight2 = 0
            else:
                consistency_weight2 = get_current_consistency_weight(args.consistency2, iter_num // 150)
#############################################################################################################################
            sup_loss1 = ce_loss(outputs_weak1[: args.labeled_bs], label_batch_aug[:][: args.labeled_bs].long(),) + dice_loss(
                outputs_weak_soft1[: args.labeled_bs],
                label_batch_aug[: args.labeled_bs],
            )
            
            sup_loss2 = ce_loss(outputs_weak2[: args.labeled_bs], label_batch_aug[:][: args.labeled_bs].long(),) + dice_loss(
                outputs_weak_soft2[: args.labeled_bs],
                label_batch_aug[: args.labeled_bs],
            )
            sup_loss = sup_loss1 + sup_loss2
#############################################################################################################################
            unsup_loss1 = (
                # ce_loss(outputs_strong1[args.labeled_bs :], pseudo_outputs_cutout[args.labeled_bs :]) + \
                dice_loss(outputs_strong_soft1[args.labeled_bs :], pseudo_outputs_cutout[args.labeled_bs :], ignore = pseudo_outputs_cutout[args.labeled_bs :])
            )
            unsup_loss2 = (
                # ce_loss(outputs_strong2[args.labeled_bs :], pseudo_outputs_cutout[args.labeled_bs :]) + \
                dice_loss(outputs_strong_soft2[args.labeled_bs :], pseudo_outputs_cutout[args.labeled_bs :], ignore = pseudo_outputs_cutout[args.labeled_bs :])
            )
  
            unsup_loss = unsup_loss1 + unsup_loss2
###########################################################################################################               
            # cut_strong_batch1, cut_pseudo_outputs_cutout, cut_outputs_weak2 = cut_mix(strong_batch, pseudo_outputs_cutout, outputs_weak2)
            # cut_outputs_strong1,_,_,_,_ = model1(cut_strong_batch1)  

            loss_crc1, _ = semi_mycrc_loss(inputs=outputs_strong1[args.labeled_bs:],
                                           pseudo_outputs=pseudo_outputs_cutout[args.labeled_bs:].clone().detach(),
                                           targets=outputs_weak2[args.labeled_bs:],
                                           threshold=0.65,
                                           neg_threshold=0.1,
                                           conf_mask=True)
            
            # cut_strong_batch2, cut_pseudo_outputs_cutout, cut_outputs_weak1 = cut_mix(strong_batch, pseudo_outputs_cutout, outputs_weak1)
            # cut_outputs_strong2,_,_,_,_ = model2(cut_strong_batch2)

            loss_crc2, _ = semi_mycrc_loss(inputs=outputs_strong2[args.labeled_bs:],
                                           pseudo_outputs=pseudo_outputs_cutout[args.labeled_bs:].clone().detach(),  
                                           targets=outputs_weak1[args.labeled_bs:],
                                           threshold=0.65,
                                           neg_threshold=0.1,
                                           conf_mask=True)
            loss_crc = loss_crc1 + loss_crc2
###########################################################################################################         
            with torch.no_grad():
                mask_weak1_hard = (normalize_1 < low_thresh) * (normalize_1 > 0.7)
                mask_weak2_hard = (normalize_2 < low_thresh) * (normalize_2 > 0.7)
                mask_weak_hard = mask_weak1_hard * mask_weak2_hard
                mask_weak_hard_cutout = mask_weak_hard.clone()
                mask_weak_hard_cutout[(label_batch_aug_cutout == -1).unsqueeze(1).repeat(1, args.num_classes, 1, 1)] = 0
            
            prob_mean_l_i = []
            prob_mean_l_i_all = []
            prob_mean_u_i_cutout = []
            prob_mean_u_i_all_cutout = []

            prob_all_l_weak1 = []
            prob_all_l_weak1_all = []
            prob_all_l_weak2 = []
            prob_all_l_weak2_all = []

            prob_all_u_weak1_cutout = []
            prob_all_u_weak1_all_cutout = []
            prob_all_u_weak2_cutout = []
            prob_all_u_weak2_all_cutout = []

            prob_all_u_strong1_cutout = []
            prob_all_u_strong1_all_cutout = []
            prob_all_u_strong2_cutout = []
            prob_all_u_strong2_all_cutout = []

            outputs_weak1 = outputs_weak1.permute(0, 2, 3, 1)
            outputs_weak2 = outputs_weak2.permute(0, 2, 3, 1)
            outputs_strong1 = outputs_strong1.permute(0, 2, 3, 1)
            outputs_strong2 = outputs_strong2.permute(0, 2, 3, 1)

            exsit_nan_cutout = False
            exsit_nan = False
            sample_num = 4096
            num_hard = 4096
            num_pixel = 4096
            for i in range(args.num_classes):
                with torch.no_grad():
                    mask_weak_l_bool = (mask_weak_hard[:,i,:,:])[:args.labeled_bs]  
                    mask_weak_l_bool_all = (pseudo_outputs == i)[:args.labeled_bs]
                    mask_weak_bool_l_batch = torch.sum(mask_weak_l_bool.view(args.labeled_bs, -1), dim=1)
                    mask_weak_bool_l_batch_all = torch.sum(mask_weak_l_bool_all.view(args.labeled_bs, -1), dim=1)

                if (mask_weak_bool_l_batch != 0).sum() != 0 and (mask_weak_bool_l_batch_all != 0).sum() != 0:
                    prob_all_l_weak1.append(outputs_weak1[:args.labeled_bs][mask_weak_l_bool])  
                    prob_all_l_weak2.append(outputs_weak2[:args.labeled_bs][mask_weak_l_bool])

                    prob_all_l_weak1_all.append(A(outputs_weak1[:args.labeled_bs], mask_weak_l_bool_all, sample_num))
                    prob_all_l_weak2_all.append(A(outputs_weak2[:args.labeled_bs], mask_weak_l_bool_all, sample_num))

                    prob_mean_l_i.append(torch.mean((nn.functional.normalize(outputs_weak1[args.labeled_bs:][mask_weak_l_bool], p=1, dim=-1) + nn.functional.normalize(outputs_weak2[args.labeled_bs:][mask_weak_l_bool], p=1, dim=-1) / 2), dim=0))  
                    prob_mean_l_i_all.append(torch.mean((nn.functional.normalize(outputs_weak1[args.labeled_bs:][mask_weak_l_bool_all], p=1, dim=-1) + nn.functional.normalize(outputs_weak2[args.labeled_bs:][mask_weak_l_bool_all], p=1, dim=-1) / 2), dim=0))
                else:
                    exsit_nan = True

                with torch.no_grad():
                    mask_weak_u_bool_cutout = (mask_weak_hard_cutout[:,i,:,:])[args.labeled_bs:]
                    mask_weak_u_bool_all_cutout = (pseudo_outputs_cutout == i)[args.labeled_bs:]
                    mask_weak_bool_u_batch_cutout = torch.sum(mask_weak_u_bool_cutout.view(args.batch_size - args.labeled_bs, -1), dim=1)
                    mask_weak_bool_u_batch_all_cutout = torch.sum(mask_weak_u_bool_all_cutout.view(args.batch_size - args.labeled_bs, -1), dim=1)

                if (mask_weak_bool_u_batch_cutout != 0).sum() != 0 and (mask_weak_bool_u_batch_all_cutout != 0).sum() != 0:
                    prob_all_u_weak1_cutout.append(outputs_weak1[args.labeled_bs:][mask_weak_u_bool_cutout])                 
                    prob_all_u_weak2_cutout.append(outputs_weak2[args.labeled_bs:][mask_weak_u_bool_cutout])               
                    prob_all_u_strong1_cutout.append(outputs_strong1[args.labeled_bs:][mask_weak_u_bool_cutout])
                    prob_all_u_strong2_cutout.append(outputs_strong2[args.labeled_bs:][mask_weak_u_bool_cutout])

                    prob_all_u_weak1_all_cutout.append(A(outputs_weak1[args.labeled_bs:], mask_weak_u_bool_all_cutout, sample_num))
                    prob_all_u_weak2_all_cutout.append(A(outputs_weak2[args.labeled_bs:], mask_weak_u_bool_all_cutout, sample_num))
                    prob_all_u_strong1_all_cutout.append(A(outputs_strong1[args.labeled_bs:], mask_weak_u_bool_all_cutout, sample_num))
                    prob_all_u_strong2_all_cutout.append(A(outputs_strong2[args.labeled_bs:], mask_weak_u_bool_all_cutout, sample_num))

                    prob_mean_u_i_cutout.append(torch.mean((nn.functional.normalize(outputs_weak1[args.labeled_bs:][mask_weak_u_bool_cutout], p=1, dim=-1) + nn.functional.normalize(outputs_weak2[args.labeled_bs:][mask_weak_u_bool_cutout], p=1, dim=-1) / 2), dim=0))  
                    prob_mean_u_i_all_cutout.append(torch.mean((nn.functional.normalize(outputs_weak1[args.labeled_bs:][mask_weak_u_bool_all_cutout], p=1, dim=-1) + nn.functional.normalize(outputs_weak2[args.labeled_bs:][mask_weak_u_bool_all_cutout], p=1, dim=-1) / 2), dim=0))
                else:
                    exsit_nan_cutout = True

            if not exsit_nan:
                prob_l_similarity = B(prob_mean_l_i)
                prob_l_similarity_all = B(prob_mean_l_i_all)

                contrast_hard_l_weak1 = []
                contrast_hard_l_weak2 = []

                len_l, len_l_real = C(prob_all_l_weak1, prob_l_similarity, num_hard)

                for idx, value in enumerate(len_l_real):
                    contrast_hard_l_index = np.random.choice(np.arange(0, len_l[idx].long().item()), size=value.long().item(), replace=False)
                    contrast_hard_l_weak1.append(prob_all_l_weak1[idx][contrast_hard_l_index])
                    contrast_hard_l_weak2.append(prob_all_l_weak2[idx][contrast_hard_l_index])

                with torch.no_grad():
                    negative_dist = torch.distributions.categorical.Categorical(probs=prob_l_similarity_all)
                    samp_class = negative_dist.sample(sample_shape=(num_pixel, ))
                    samp_class_counter = [(samp_class == c).sum().item() for c in range(args.num_classes)]

                for idx, value in enumerate(samp_class_counter):
                    contrast_hard_l_index_all = np.random.choice(np.arange(0, len(prob_all_l_weak1_all[idx])), size=value, replace=True)
                    contrast_hard_l_weak1.append(prob_all_l_weak1_all[idx][contrast_hard_l_index_all])
                    contrast_hard_l_weak2.append(prob_all_l_weak2_all[idx][contrast_hard_l_index_all])

                contrast_hard_l_weak1 = torch.cat(contrast_hard_l_weak1, dim=0)
                contrast_hard_l_weak2 = torch.cat(contrast_hard_l_weak2, dim=0)
                shuffle_idx = torch.randperm(len(contrast_hard_l_weak1))
                contrast_hard_l_weak1 = contrast_hard_l_weak1[shuffle_idx]
                contrast_hard_l_weak2 = contrast_hard_l_weak2[shuffle_idx]

                feat_l_q = projector_3(contrast_hard_l_weak1)  
                feat_l_k = projector_4(contrast_hard_l_weak2)
                Loss_contrast_l = contrastive_loss_sup_criter_seq(feat_l_q, feat_l_k) + contrastive_loss_sup_criter_seq(feat_l_k, feat_l_q)

                reg_loss_weak_l = -NegEntropy_loss(contrast_hard_l_weak1.clamp(min=1e-6, max=1.)) - NegEntropy_loss(contrast_hard_l_weak2.clamp(min=1e-6, max=1.))
            else:
                Loss_contrast_l = torch.tensor(0.0).to(sup_loss.device)
                reg_loss_weak_l = -torch.tensor(0.0).to(sup_loss.device)  

            if not exsit_nan_cutout:
                prob_u_similarity = B(prob_mean_u_i_cutout)
                prob_u_similarity_all = B(prob_mean_u_i_all_cutout)

                contrast_hard_u_weak1_cutout = []
                contrast_hard_u_weak2_cutout = []
                contrast_hard_u_strong1_cutout = []
                contrast_hard_u_strong2_cutout = []

                len_u, len_u_real = C(prob_all_u_weak1_cutout, prob_u_similarity, num_hard)
                
                for idx, value in enumerate(len_u_real):
                    contrast_hard_u_index = np.random.choice(np.arange(0, len_u[idx].long().item()), size=value.long().item(), replace=False)
                    contrast_hard_u_weak1_cutout.append(prob_all_u_weak1_cutout[idx][contrast_hard_u_index])
                    contrast_hard_u_weak2_cutout.append(prob_all_u_weak2_cutout[idx][contrast_hard_u_index])
                    contrast_hard_u_strong1_cutout.append(prob_all_u_strong1_cutout[idx][contrast_hard_u_index])
                    contrast_hard_u_strong2_cutout.append(prob_all_u_strong2_cutout[idx][contrast_hard_u_index])

                with torch.no_grad():
                    negative_dist = torch.distributions.categorical.Categorical(probs=prob_u_similarity_all)
                    samp_class = negative_dist.sample(sample_shape=(num_pixel, ))
                    samp_class_counter = [(samp_class == c).sum().item() for c in range(args.num_classes)]

                for idx, value in enumerate(samp_class_counter):
                    contrast_hard_u_index_all = np.random.choice(np.arange(0, len(prob_all_u_weak1_all_cutout[idx])), size=value, replace=True)
                    contrast_hard_u_weak1_cutout.append(prob_all_u_weak1_all_cutout[idx][contrast_hard_u_index_all])
                    contrast_hard_u_weak2_cutout.append(prob_all_u_weak2_all_cutout[idx][contrast_hard_u_index_all])
                    contrast_hard_u_strong1_cutout.append(prob_all_u_strong1_all_cutout[idx][contrast_hard_u_index_all])
                    contrast_hard_u_strong2_cutout.append(prob_all_u_strong2_all_cutout[idx][contrast_hard_u_index_all])

                contrast_hard_u_weak1_cutout = torch.cat(contrast_hard_u_weak1_cutout, dim=0)
                contrast_hard_u_weak2_cutout = torch.cat(contrast_hard_u_weak2_cutout, dim=0)
                contrast_hard_u_strong1_cutout = torch.cat(contrast_hard_u_strong1_cutout, dim=0)
                contrast_hard_u_strong2_cutout = torch.cat(contrast_hard_u_strong2_cutout, dim=0)
                shuffle_idx = torch.randperm(len(contrast_hard_u_weak1_cutout))
                contrast_hard_u_weak1_cutout = contrast_hard_u_weak1_cutout[shuffle_idx]
                contrast_hard_u_weak2_cutout = contrast_hard_u_weak2_cutout[shuffle_idx]
                contrast_hard_u_strong1_cutout = contrast_hard_u_strong1_cutout[shuffle_idx]
                contrast_hard_u_strong2_cutout = contrast_hard_u_strong2_cutout[shuffle_idx]
            
                feat_q = projector_3(contrast_hard_u_weak1_cutout)
                feat_k = projector_4(contrast_hard_u_strong2_cutout)
                Loss_contrast_u_1 = contrastive_loss_sup_criter_seq(feat_k,feat_q)
                
                feat_q = projector_4(contrast_hard_u_weak2_cutout)
                feat_k = projector_3(contrast_hard_u_strong1_cutout)
                Loss_contrast_u_2 = contrastive_loss_sup_criter_seq(feat_k,feat_q)
                
                Loss_contrast_u = Loss_contrast_u_1 + Loss_contrast_u_2
                
                kl_loss = kld_loss(contrast_hard_u_weak1_cutout.clamp(min=1e-6, max=1.), contrast_hard_u_strong1_cutout.clamp(min=1e-6, max=1.)) + kld_loss(contrast_hard_u_weak2_cutout.clamp(min=1e-6, max=1.), contrast_hard_u_strong2_cutout.clamp(min=1e-6, max=1.))
                reg_loss_weak_u = -NegEntropy_loss(contrast_hard_u_weak1_cutout.clamp(min=1e-6, max=1.)) - NegEntropy_loss(contrast_hard_u_weak2_cutout.clamp(min=1e-6, max=1.))
                reg_loss_strong_u = -NegEntropy_loss(contrast_hard_u_strong1_cutout.clamp(min=1e-6, max=1.)) - NegEntropy_loss(contrast_hard_u_strong2_cutout.clamp(min=1e-6, max=1.))
            else:
                kl_loss = torch.tensor(0.0).to(sup_loss.device)
                Loss_contrast_u = torch.tensor(0.0).to(sup_loss.device)
                reg_loss_weak_u = -torch.tensor(0.0).to(sup_loss.device)
                reg_loss_strong_u = -torch.tensor(0.0).to(sup_loss.device)

            hyp = 0.1
            kl_weight = 0.2
            #  + loss_crc
            loss = sup_loss + consistency_weight1 * (unsup_loss + loss_crc) + consistency_weight1 * (Loss_contrast_l) + consistency_weight2 * (Loss_contrast_u) + consistency_weight1 * kl_weight * kl_loss  # + consistency_weight1 * hyp * (reg_loss_weak_l + reg_loss_weak_u + reg_loss_strong_u)
            
            running_loss += loss
            running_sup_loss += sup_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if iter_num <=10000:  
                coefficient = 0.5
            elif 10000 <iter_num <= 20000:
                coefficient = 0.4
            else:
                coefficient = 0.3

            with  torch.no_grad():
                epoch_errors.append(coefficient * loss.item())
                iter_num = iter_num + 1

                writer.add_scalar("lr", lr_, iter_num)
                writer.add_scalar("consistency_weight/consistency_weight1", consistency_weight1, iter_num)
                writer.add_scalar("consistency_weight/consistency_weight2", consistency_weight2, iter_num)
                writer.add_scalar("loss/model_loss", loss, iter_num)
                logging.info("iteration %d : model loss : %f" % (iter_num, loss.item()))

                mean_epoch_error = np.mean(epoch_errors)
                cta.update_rates(db_train.ops_weak, 1.0 - coefficient * mean_epoch_error)
                cta.update_rates(db_train.ops_strong, 1.0 - coefficient * mean_epoch_error)

                if iter_num < 18000:
                    continue

                if iter_num % 50 == 0:
                    idx  = args.labeled_bs
                
                    image = raw_batch[idx, 0:1, :, :]
                    writer.add_image("train/RawImage", image, iter_num)

                    image = weak_batch[idx, 0:1, :, :]
                    writer.add_image("train/WeakImage", image, iter_num)
         
                    image_strong = strong_batch[idx, 0:1, :, :]
                    writer.add_image("train/StrongImage", image_strong, iter_num)                
       
                    outputs_strong1 = torch.argmax(outputs_strong_soft1, dim=1, keepdim=True)
                    writer.add_image("train/model_Prediction1", outputs_strong1[idx, ...] * 50, iter_num)
                    outputs_strong2 = torch.argmax(outputs_strong_soft2, dim=1, keepdim=True)
                    writer.add_image("train/model_Prediction2", outputs_strong2[idx, ...] * 50, iter_num)
    
                    labs = label_batch_aug[idx, ...].unsqueeze(0) * 50
                    writer.add_image("train/GroundTruth", labs, iter_num)

                    pseudo_labs1 = pseudo_outputs1[idx, ...].unsqueeze(0) * 50
                    writer.add_image("train/PseudoLabel1", pseudo_labs1, iter_num)
                    pseudo_labs2 = pseudo_outputs2[idx, ...].unsqueeze(0) * 50
                    writer.add_image("train/PseudoLabel2", pseudo_labs2, iter_num)
                    pseudo_labs = pseudo_outputs[idx, ...].unsqueeze(0) * 50
                    writer.add_image("train/PseudoLabel", pseudo_labs, iter_num)
                    
                if (iter_num > 0 and iter_num % 10 == 0) or performance1 > 0.89 or performance2 > 0.89:
                    model1.eval()
                    metric_list = 0.0
                    for _, sampled_batch in enumerate(valloader):
                        metric_i = test_single_volume(
                            sampled_batch["image"], sampled_batch["label"], model1, classes=num_classes, patch_size=args.patch_size)
                        metric_list += np.array(metric_i)
                    metric_list = metric_list / len(db_val)
                    for class_i in range(num_classes-1):
                        writer.add_scalar('info/model1_val_{}_dice'.format(class_i+1), metric_list[class_i, 0], iter_num)
                        writer.add_scalar('info/model1_val_{}_hd95'.format(class_i+1), metric_list[class_i, 1], iter_num)

                    performance1 = np.mean(metric_list, axis=0)[0]

                    mean_hd951 = np.mean(metric_list, axis=0)[1]
                    writer.add_scalar('eval/model1_val_mean_dice', performance1, iter_num)
                    writer.add_scalar('eval/model1_val_mean_hd95', mean_hd951, iter_num)

                    if performance1 > best_performance1:
                        best_performance1 = performance1
                        if performance1 > 0:
                            save_mode_path = os.path.join(snapshot_path, 'model1_iter_{}_dice_{}.pth'.format(iter_num, round(best_performance1, 4)))
                            save_best = os.path.join(snapshot_path,'{}_best_model1.pth'.format(args.model))
                            util.save_checkpoint(epoch_num, model1, optimizer1, loss, save_mode_path)
                            util.save_checkpoint(epoch_num, model1, optimizer1, loss, save_best)

                    logging.info('iteration %d : model1_mean_dice : %f model1_mean_hd95 : %f' % (iter_num, performance1, mean_hd951))
                    model1.train()

                    model2.eval()
                    metric_list = 0.0
                    for _, sampled_batch in enumerate(valloader):
                        metric_i = test_single_volume(sampled_batch["image"], sampled_batch["label"], model2, classes=num_classes, patch_size=args.patch_size)
                        metric_list += np.array(metric_i)
                    metric_list = metric_list / len(db_val)
                    for class_i in range(num_classes-1):
                        writer.add_scalar('info/model2_val_{}_dice'.format(class_i+1), metric_list[class_i, 0], iter_num)
                        writer.add_scalar('info/model2_val_{}_hd95'.format(class_i+1), metric_list[class_i, 1], iter_num)

                    performance2 = np.mean(metric_list, axis=0)[0]

                    mean_hd952 = np.mean(metric_list, axis=0)[1]
                    writer.add_scalar('eval/model2_val_mean_dice', performance2, iter_num)
                    writer.add_scalar('eval/model2_val_mean_hd95', mean_hd952, iter_num)

                    if performance2 > best_performance2:
                        best_performance2 = performance2
                        if performance2 > 0:
                            save_mode_path = os.path.join(snapshot_path, 'model2_iter_{}_dice_{}.pth'.format(iter_num, round(best_performance2, 4)))
                            save_best = os.path.join(snapshot_path,'{}_best_model2.pth'.format(args.model))
                            util.save_checkpoint(epoch_num, model2, optimizer2, loss, save_mode_path)
                            util.save_checkpoint(epoch_num, model2, optimizer2, loss, save_best)

                    logging.info('iteration %d : model2_mean_dice : %f model2_mean_hd95 : %f' % (iter_num, performance2, mean_hd952))
                    model2.train()

                    logging.info('current best dice coef model 1 {}, model 2 {}'.format(best_performance1, best_performance2))

                if iter_num % 3000 == 0:
                    save_mode_path = os.path.join(snapshot_path, 'model1_iter_' + str(iter_num) + '.pth')
                    util.save_checkpoint(epoch_num, model1, optimizer1, loss, save_mode_path)
                    logging.info("save model1 to {}".format(save_mode_path))

                    save_mode_path = os.path.join(snapshot_path, 'model2_iter_' + str(iter_num) + '.pth')
                    util.save_checkpoint(epoch_num, model2, optimizer2, loss, save_mode_path)
                    logging.info("save model2 to {}".format(save_mode_path))

                if iter_num >= max_iterations:
                    break
                    
            if iter_num >= max_iterations:
                iterator.close()
                break
                
            epoch_loss = running_loss / len(trainloader)
            epoch_sup_loss = running_sup_loss / len(trainloader)
            
            logging.info('{} Epoch [{:03d}/{:03d}]'.format(datetime.now(), epoch_num, max_epoch))
            logging.info('Train loss: {}'.format(epoch_loss))
            writer.add_scalar('Train/Loss', epoch_loss, epoch_num)        
            logging.info('Train sup loss: {}'.format(epoch_sup_loss))
            writer.add_scalar('Train/sup_loss', epoch_sup_loss, epoch_num)

    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed) 
    os.environ['PYTHONHASHSEED'] = str(0)

    snapshot_path = "../model/{}_{}_labeled/{}".format(args.exp, args.labeled_num, args.model)
    data_flag = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    snapshot_path = snapshot_path + '/' + data_flag
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
    train(args, snapshot_path)
