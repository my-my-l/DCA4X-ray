import random
import time
import math
import argparse
import shutil
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import SGD
import torch.utils.data
from torch.utils.data import DataLoader

from dca_utils import mixing_criterion, DataGenerator, auc, get_target_samples, SDLoss, get_mask
from utils.model import ImageClassifierHead, get_model
from utils.meter import AverageMeter, ProgressMeter
from utils.logger import CompleteLogger
from utils.data import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_map = {0:"ATL", 1:"PM", 2:"CM", 3:"CS", 4:"PE", 5:"EDMA"}
                    



def main(args:argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase, args.resume, args.resume_path)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
     
    cudnn.benchmark = True



    train_transform = get_train_transform(args.train_resizing, scale=args.scale, ratio=args.ratio,
                                                random_horizontal_flip=not args.no_hflip,
                                                random_color_jitter=False, resize_size=args.resize_size,
                                                norm_mean=args.norm_mean, norm_std=args.norm_std)
    val_transform = get_val_transform(args.val_resizing, resize_size=args.resize_size,
                                            norm_mean=args.norm_mean, norm_std=args.norm_std)

    print("train_transform: ", train_transform)
    print("val_transform: ", val_transform)


    train_source_dataset, train_target_dataset, target_val_dataset, target_test_dataset, source_val_dataset, source_test_dataset, num_classes, args.class_names = \
        get_dataset(args.data, args.root, args.source, args.target, train_transform, val_transform, sl=True, class_index=args.current_class)
    print(num_classes)
    print(args.class_names)
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)

    target_val_loader = DataLoader(target_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    target_test_loader = DataLoader(target_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
  

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)
    

    G = get_model(pretrain=not args.scratch).to(device)
    pool_layer = nn.Identity() if args.no_pool else None
    F = ImageClassifierHead(G.out_features, args.bottleneck_dim, pool_layer, num_classes=1).to(device)
    sd_loss_function = SDLoss().to(device)
    cls_loss_function = nn.BCEWithLogitsLoss()

    best_auc = 0.
    print('train {}'.format(class_map[args.current_class]))
    optimizer = SGD([{'params': G.parameters(), 'lr':0.1}, {'params': F.head.parameters(), 'lr':1.0}], lr=1,
            momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    mixup_source_target = DataGenerator(alpha=args.lam_alpha, sup=args.sup)
    for epoch in range(args.epochs):
        print("lr:", lr_scheduler.get_last_lr())
        train(train_source_iter, train_target_iter, 
              G, F, mixup_source_target, cls_loss_function,sd_loss_function,
              optimizer, lr_scheduler, epoch, args)
        valid_auc = validate(target_val_loader, G, F, args)

        if valid_auc > best_auc:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_auc = max(valid_auc, best_auc)
      
      
       
    G.load_state_dict(torch.load(logger.get_checkpoint_path('best'))['g'])
    F.load_state_dict(torch.load(logger.get_checkpoint_path('best'))['f'])
    test_auc = validate(target_test_loader, G, F, args)


    print("Evaluate")
    print('auc   :{:3.1f}'.format(test_auc))
 
    logger.close()

   


def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator, 
          G: nn.Module, F: nn.Module, mixup_source_target: DataGenerator,
          cls_loss_function: nn.BCEWithLogitsLoss, sd_loss_function: SDLoss, optimizer: SGD,
          lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    loss_s_meter = AverageMeter('Loss Source', ':6.2f')
    loss_st_meter = AverageMeter('Loss ST', ':6.2f')
    loss_sd_meter = AverageMeter('Loss BSP', ':6.2f')
    auc_t_meter = AverageMeter('target '+class_map[args.current_class], ':3.1f')
   

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, loss_s_meter, loss_st_meter, loss_sd_meter, auc_t_meter], 
        prefix="Epoch: [{}]".format(epoch))

    G.train()
    F.train()
   

    end = time.time()
  
 
    print_freq = math.ceil(args.iters_per_epoch / 10.)
    for i in range(args.iters_per_epoch):

        x_s, labels_s = next(train_source_iter)[:2]
        x_s = x_s.to(device)
        labels_s = labels_s.to(device).float()
        x_t, labels_t = next(train_target_iter)[:2]
        x_t = x_t.to(device)
        labels_t = labels_t.to(device).float()


        data_time.update(time.time() - end)


        g_s = G(x_s) 
        y_s, g_s = F(g_s, get_f=True) 
        cls_loss_s = cls_loss_function(y_s, labels_s)
        loss = cls_loss_s
        loss_s_meter.update(cls_loss_s.item(), x_s.size(0))


        g_t = G(x_t)
        y_t, g_t = F(g_t, get_f=True)
        y_t = y_t.detach()
        mask_f_p, mask_f_n = get_mask(args.hi_threshold, args.lo_threshold, y_t)
        sd_loss = sd_loss_function(g_s, torch.cat((g_t[mask_f_n==1], g_s[labels_s==0]),dim=0),
                                    torch.cat((g_t[mask_f_p==1], g_s[labels_s==1]),dim=0),
                                  )
     
        
        loss += args.trade_off_sd * sd_loss
        loss_sd_meter.update(sd_loss.item(), x_s.size(0))

        cls_auc_t = auc(y_t, labels_t)
        auc_t_meter.update(cls_auc_t[-1], x_t.size(0))
        
        
        x_t, pseudo_labels_t = get_target_samples(args.hi_threshold,args.lo_threshold,x_t,y_t,device,x_t.size(0),class_balance=False)
        mixed_x_st, labels_st_a, labels_st_b, lam_st = mixup_source_target(x_s, x_t, labels_s, pseudo_labels_t, device)
        g_st = G(mixed_x_st)
        y_st = F(g_st)
        cls_loss_st = mixing_criterion(cls_loss_function, y_st, labels_st_a, labels_st_b, lam_st)
        loss += args.trade_off_st * cls_loss_st
        loss_st_meter.update(cls_loss_st.item(), x_s.size(0))

        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        batch_time.update(time.time() - end)
        end = time.time()


        if i % print_freq == 0:
            progress.display(i)
    return 


def validate(val_loader: DataLoader, G: nn.Module, F: ImageClassifierHead, args: argparse.Namespace):

    batch_num = len(val_loader)
    print_freq = math.ceil(batch_num / 5)
    batch_time = AverageMeter('Time', ':6.3f')
    auc_meter = AverageMeter(class_map[args.current_class], ':3.1f')


    progress = ProgressMeter(
        batch_num,
        [batch_time, auc_meter],
        prefix="{} Test: ".format(type))
      
    G.eval()
    F.eval()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[:2]
            images = images.to(device)
            target = target.to(device).float()

            g = G(images)
            y = F(g)
           
            cls_auc = auc(y, target)
            auc_meter.update(cls_auc[-1],images.size(0))

          
            batch_time.update(time.time() - end)
            end = time.time()
            if i % print_freq == 0:
                progress.display(i)
    
    return auc_meter.avg



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unsupervised_Adversarial_Domain_Adaptation_for_Multi-Label_Classification_of_Chest_X-Ray based on multi_label DANN ')


    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-s', '--source', default="NIH_CXR14", help='source domain(s)', nargs='+')
    parser.add_argument('-t', '--target', default="Open-i", help='target domain(s)', nargs='+')
    parser.add_argument('--preresized', action='store_true', help = 'ues pre-resized 256*256 images')
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    parser.add_argument('--resize-size', type=int, default=224,
                        help='the image size after resizing')
    parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                        help='Random resize scale (default: 0.08 1.0)')
    parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                        help='Random resize aspect ratio (default: 0.75 1.33)')
    parser.add_argument('--no-hflip', action='store_true',
                        help='no random horizontal flipping during training')
    parser.add_argument('--norm-mean', type=float, nargs='+',
                        default=(0.485, 0.456, 0.406), help='normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+',
                        default=(0.229, 0.224, 0.225), help='normalization std')  
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.001, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=500, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument("--log", type=str, default='multi_label_dann',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test'],
                        help="When phase is 'test', only test the model.")
    parser.add_argument('--current-class', type=int, default=0)
    parser.add_argument('--hi-threshold', default=0.6, type=float)
    parser.add_argument('--lo-threshold', default=0.4, type=float)
    parser.add_argument('--lam-alpha', default=1., type=float)
    parser.add_argument('--trade-off-st', default=1., type=float)
    parser.add_argument('--trade-off-sd', default=2e-4, type=float)
    parser.add_argument('--sup', default=10, type=float)
    args = parser.parse_args()
    main(args)
