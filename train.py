from __future__ import print_function

import argparse
import os
import sys
import random
import math
import shutil
import torch
import torch.nn.parallel
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
from tensorboardX import SummaryWriter # https://github.com/lanpa/tensorboard-pytorch
import multiprocessing as mp
from dataset import PointcloudPatchDataset, RandomPointcloudPatchSampler, SequentialShapeRandomPointcloudPatchSampler

from dsac import DSAC
import torchsnooper
import dist_chamfer as ext
def parse_arguments():
    parser = argparse.ArgumentParser()

    # naming / file handling
    parser.add_argument('--name', type=str, default='my_single_scale_normal', help='training run name')
    parser.add_argument('--desc', type=str, default='My training run for single-scale normal2 estimation.', help='description')
    parser.add_argument('--indir', type=str, default='/data/pclouds', help='input folder (point clouds)')
    parser.add_argument('--outdir', type=str, default='../../data/dsacmodels', help='output folder (trained models)')
    parser.add_argument('--logdir', type=str, default='../../data/dsaclogs', help='training log folder')
    parser.add_argument('--trainset', type=str, default='trainingset_whitenoise.txt', help='training set file name')
    parser.add_argument('--testset', type=str, default='validationset_whitenoise.txt', help='test set file name')
    parser.add_argument('--saveinterval', type=int, default='10', help='save model each n epochs')
    #parser.add_argument('--refine', type=str, default='', help='refine model at this patch')
    #parser.add_argument('--gpu_idx', type=str, default='1,2,3', help='set < 0 to use CPU')
    parser.add_argument('--gpu_idx', type=int, default=0, help='set < 0 to use CPU')
    # training parameters
    parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--patch_radius', type=float, default=[0.03], nargs='+', help='patch radius in multiples of the shape\'s bounding box diagonal, multiple values for multi-scale.')
    parser.add_argument('--patch_center', type=str, default='point', help='center patch at...\n'
                        'point: center point\n'
                        'mean: patch mean')
    #parser.add_argument('--patch_point_count_std', type=float, default=0, help='standard deviation of the number of points in a patch')
    parser.add_argument('--patches_per_shape', type=int, default=1000, help='number of patches sampled from each shape in an epoch')
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers - 0 means same thread as main execution')
    parser.add_argument('--cache_capacity', type=int, default=100, help='Max. number of dataset elements (usually shapes) to hold in the cache at the same time.')
    parser.add_argument('--seed', type=int, default=3627474, help='manual seed')
    parser.add_argument('--training_order', type=str, default='random', help='order in which the training patches are presented:\n'
                        'random: fully random over the entire dataset (the set of all patches is permuted)\n'
                        'random_shape_consecutive: random over the entire dataset, but patches of a shape remain consecutive (shapes and patches inside a shape are permuted)')
    parser.add_argument('--identical_epochs', type=int, default=False, help='use same patches in each epoch, mainly for debugging')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='gradient descent momentum')
    #parser.add_argument('--use_pca', type=int, default=False, help='Give both inputs and ground truth in local PCA coordinate frame')
    parser.add_argument('--normal_loss', type=str, default='ms_euclidean', help='Normal loss type:\n'
                        'ms_euclidean: mean square euclidean distance\n'
                        'ms_oneminuscos: mean square 1-cos(angle error)')

    # model hyperparameters
    parser.add_argument('--outputs', type=str, nargs='+', default=['unoriented_normals'], help='outputs of the network, a list with elements of:\n'
                        'unoriented_normals: unoriented (flip-invariant) point normals\n'
                        'oriented_normals: oriented point normals\n'
                        'max_curvature: maximum curvature\n'
                        'min_curvature: mininum curvature')
    parser.add_argument('--use_point_stn', type=int, default=True, help='use point spatial transformer')
    
    parser.add_argument('--use_feat_stn', type=int, default=True, help='use feature spatial transformer')
    parser.add_argument('--use_mask', type=int, default=True, help='use point mask')
    #parser.add_argument('--sym_op', type=str, default='max', help='symmetry operation')
    parser.add_argument('--point_tuple', type=int, default=1, help='use n-tuples of points as input instead of single points')
    parser.add_argument('--points_per_patch', type=int, default=512, help='max. number of points per patch')
    parser.add_argument('--hypotheses', '-hyps', type=int, default=32, 
        help='number of line hypotheses sampled for each image')

    parser.add_argument('--inlierthreshold', '-it', type=float, default=0.1, 
        help='threshold used in the soft inlier count. Its measured in relative image size (1 = image width)')

    parser.add_argument('--inlieralpha', '-ia', type=float, default=0.5, 
        help='scaling factor for the soft inlier scores (controls the peakiness of the hypothesis distribution)')

    parser.add_argument('--inlierbeta', '-ib', type=float, default=100.0, 
        help='scaling factor within the sigmoid of the soft inlier count')
    return parser.parse_args()

def train_dsacpnet(opt):


    # gpu init
    # multi_gpus = False
    # if ',' in opt.gpu_idx:
    #     gpu_ids = [int(id) for id in opt.gpu_idx.split(',')]
    #     multi_gpus = True
    # else:
    #     gpu_ids = [int(opt.gpu_idx)]

    # device = torch.device('cuda:{}'.format(gpu_ids[0]) if torch.cuda.is_available() else 'cpu')
    # 此处如果使用 下面一行代码，则会报错，RuntimeError: all tensors must be on devices[0]
    # 默认情况下 device 为0，因此需要指定 device id.
    device = torch.device("cpu" if opt.gpu_idx < 0 else "cuda:%d" % opt.gpu_idx)

    #if multi_gpus:
       # net = DataParallel(net, device_ids=gpu_ids).to(device)
       # margin = DataParallel(margin, device_ids=gpu_ids).to(device)
    #else:
     #   net = net.to(device)
      #  margin = margin.to(device)
    #device = torch.device("cpu" if opt.gpu_idx < 0 else "cuda:%d" % opt.gpu_idx)

    # colored console output
    green = lambda x: '\033[92m' + x + '\033[0m'
    blue = lambda x: '\033[94m' + x + '\033[0m'

    log_dirname = os.path.join(opt.logdir, opt.name)
    params_filename = os.path.join(opt.outdir, '%s_params.pth' % (opt.name))
    model_filename = os.path.join(opt.outdir, '%s_model.pth' % (opt.name))
    desc_filename = os.path.join(opt.outdir, '%s_description.txt' % (opt.name))

    if os.path.exists(log_dirname) or os.path.exists(model_filename):
        response = input('A training run named "%s" already exists, overwrite? (y/n) ' % (opt.name))
        if response == 'y':
            if os.path.exists(log_dirname):
                shutil.rmtree(os.path.join(opt.logdir, opt.name))
        else:
            sys.exit()

    #output_loss_weight =1.0
    #dsacpnet = DSACPNet(
     #   num_points=opt.points_per_patch,

      #  )
    criterion = nn.BCEWithLogitsLoss()
    dsac = DSAC(
        opt.hypotheses, 
        opt.inlierthreshold, 
        opt.inlierbeta, 
        opt.inlieralpha, 
        opt.normal_loss,
        opt.seed,device,
        use_point_stn=opt.use_point_stn,
        use_feat_stn=opt.use_feat_stn,
        use_mask=opt.use_mask
        )
    #distChamfer =  ext.chamferDist()
    #dsacpnet.to(device)
    if opt.seed < 0:
        opt.seed = random.randint(1, 10000)

    print("Random Seed: %d" % (opt.seed))
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    # create train and test dataset loaders
    train_dataset = PointcloudPatchDataset(
        root=opt.indir,
        shape_list_filename=opt.trainset,
        patch_radius=opt.patch_radius,
        points_per_patch=opt.points_per_patch,
        seed=opt.seed,
        identical_epochs=opt.identical_epochs,
        center=opt.patch_center,
        cache_capacity=opt.cache_capacity)
    if opt.training_order == 'random':
        train_datasampler = RandomPointcloudPatchSampler(
            train_dataset,
            patches_per_shape=opt.patches_per_shape,
            seed=opt.seed,
            identical_epochs=opt.identical_epochs)
    elif opt.training_order == 'random_shape_consecutive':
        train_datasampler = SequentialShapeRandomPointcloudPatchSampler(
            train_dataset,
            patches_per_shape=opt.patches_per_shape,
            seed=opt.seed,
            identical_epochs=opt.identical_epochs)
    else:
        raise ValueError('Unknown training order: %s' % (opt.training_order))

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_datasampler,
        batch_size=opt.batchSize,
        num_workers=int(opt.workers))

    test_dataset = PointcloudPatchDataset(
        root=opt.indir,
        shape_list_filename=opt.testset,
        patch_radius=opt.patch_radius,
        points_per_patch=opt.points_per_patch,
        seed=opt.seed,
        identical_epochs=opt.identical_epochs,
        center=opt.patch_center,
        cache_capacity=opt.cache_capacity)
    if opt.training_order == 'random':
        test_datasampler = RandomPointcloudPatchSampler(
            test_dataset,
            patches_per_shape=opt.patches_per_shape,
            seed=opt.seed,
            identical_epochs=opt.identical_epochs)
    elif opt.training_order == 'random_shape_consecutive':
        test_datasampler = SequentialShapeRandomPointcloudPatchSampler(
            test_dataset,
            patches_per_shape=opt.patches_per_shape,
            seed=opt.seed,
            identical_epochs=opt.identical_epochs)
    else:
        raise ValueError('Unknown training order: %s' % (opt.training_order))

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        sampler=test_datasampler,
        batch_size=opt.batchSize,
        num_workers=int(opt.workers))

    # keep the exact training shape names for later reference
    opt.train_shapes = train_dataset.shape_names
    opt.test_shapes = test_dataset.shape_names

    print('training set: %d patches (in %d batches) - test set: %d patches (in %d batches)' %
          (len(train_datasampler), len(train_dataloader), len(test_datasampler), len(test_dataloader)))

    try:
        os.makedirs(opt.outdir)
    except OSError:
        pass


    train_writer = SummaryWriter(os.path.join(log_dirname, 'train'))
    test_writer = SummaryWriter(os.path.join(log_dirname, 'test'))

    optimizer = optim.SGD(dsac.parameters(), lr=opt.lr, momentum=opt.momentum)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=0.1) # milestones in number of optimizer iterations
    #dsacpnet= torch.nn.DataParallel(dsacpnet, device_ids=gpu_ids).to(device)
    #dsac= torch.nn.DataParallel(dsac, device_ids=gpu_ids).to(device)
    dsac.to(device)
    train_num_batch = len(train_dataloader)
    test_num_batch = len(test_dataloader)

    # save parameters
    torch.save(opt, params_filename)

    # save description
    with open(desc_filename, 'w+') as text_file:
        print(opt.desc, file=text_file)

    for epoch in range(opt.nepoch):

        train_batchind = -1
        train_fraction_done = 0.0
        train_enum = enumerate(train_dataloader, 0)

        test_batchind = -1
        test_fraction_done = 0.0
        test_enum = enumerate(test_dataloader, 0)

        for train_batchind, data in train_enum:

            # update learning rate
            scheduler.step(epoch * train_num_batch + train_batchind)

            # set to training mode
            dsac.train()
            #index =[]
            # get trainingset batch and upload to GPU
            points = data[0]#这时的point是64*512*3的类型
            target = data[1]
            mask = data[2]
            #valid = data[3]
            
            # for i in range(len(valid)):
            #     if valid[i]:
            #         index.append(i)
            

            points=points
            # for point in points:
            
            points = points.transpose(2, 1)
            points = points.to(device)
            
            # print("input",points[0])
            #print("input",points.size())
            #print("the points before input")
            #print(points)
            
            target = target.to(device)
            mask =mask.to(device)
            
            
            # zero gradients
            optimizer.zero_grad()

            # forward pass
            #points, _ = dsacpnet(points)
            #print("after trans")
            #print(points[0])
            #with torchsnooper.snoop():
            exp_loss, top_loss,_,pts,mask_p = dsac(
                points,
                target
                #output_loss_weight = output_loss_weight,
                
                
            )
            #dist1, _ = distChamfer(pts, points.transpose(2, 1).contiguous())
            # print(dist1)
            # print(pts)
            # print(points.transpose(2, 1))
            #chamfer_loss=torch.mean(dist1)
            if opt.use_mask:
                mask_p=mask_p.view(-1,opt.points_per_patch)
            #mask_loss=(mask-mask_p).pow(2).sum(1).mean()
            #mask_loss=(mask-mask_p).pow(2).sum(1).mean()*0.01
                mask_loss=criterion(mask_p, mask)#
            else:mask_loss=0
            exp_loss=exp_loss.mean()

            loss=exp_loss+mask_loss #+chamfer_loss
            # backpropagate through entire network to compute gradients of loss w.r.t. parameters
            loss.backward()

            # parameter optimization step
            optimizer.step()

            train_fraction_done = (train_batchind+1) / train_num_batch

            # print info and update log file
            print('[%s %d: %d/%d] %s tloss: %f loss: %f Top Loss:%f mask Loss:%f' % (opt.name, epoch, train_batchind, train_num_batch-1, green('train'),loss, exp_loss.mean().item(),top_loss.mean(),mask_loss))
            train_writer.add_scalar('loss', exp_loss.item(), (epoch + train_fraction_done) * train_num_batch * opt.batchSize)

            while test_fraction_done <= train_fraction_done and test_batchind+1 < test_num_batch:

                # set to evaluation mode
                dsac.eval()

                test_batchind, data = next(test_enum)

                #index =[]
            # get trainingset batch and upload to GPU
                points = data[0]#这时的point是64*512*3的类型
                target = data[1]
                mask = data[2]
                #valid = data[3]
                
                # for i in range(len(valid)):
                #     if valid[i]:
                #         index.append(i)
            

                points=points
            # for point in points:
            #     print(point)
                points = points.transpose(2, 1)
                points = points.to(device)

            # print("input",points[0])
            #print("input",points.size())
            #print("the points before input")
            #print(points)
                target = target.to(device)
                mask =mask.to(device)

                # forward pass
                with torch.no_grad():
                    exp_loss, top_loss,_,pts,mask_p = dsac(points,target)
                #dist1, _ = distChamfer(pts, points.transpose(2, 1).contiguous())

                #chamfer_loss=torch.mean(dist1)
                
                if opt.use_mask:
                    mask_p=mask_p.view(-1,opt.points_per_patch)
            #mask_loss=(mask-mask_p).pow(2).sum(1).mean()
            #mask_loss=(mask-mask_p).pow(2).sum(1).mean()*0.01
                    mask_loss=criterion(mask_p, mask)#
                else:mask_loss=0
                #mask_loss=criterion(mask_p, mask)
                loss=exp_loss+mask_loss
                test_fraction_done = (test_batchind+1) / test_num_batch

                # print info and update log file
                print('[%s %d: %d/%d] %s tloss: %f loss: %f Top Loss:%f mask Loss:%f ' % (opt.name, epoch, train_batchind, train_num_batch-1, blue('test'),loss, exp_loss.mean().item(),top_loss.mean(),mask_loss))
                test_writer.add_scalar('loss', exp_loss.mean().item(), (epoch + test_fraction_done) * train_num_batch * opt.batchSize)

        # save model, overwriting the old model
        if epoch % opt.saveinterval == 0 or epoch == opt.nepoch-1:
            torch.save(dsac.state_dict(), model_filename)

        # save model in a separate file in epochs 0,5,10,50,100,500,1000, ...
        if epoch % (5 * 10**math.floor(math.log10(max(2, epoch-1)))) == 0 or epoch % 100 == 0 or epoch == opt.nepoch-1:
            torch.save(dsac.state_dict(), os.path.join(opt.outdir, '%s_model_%d.pth' % (opt.name, epoch)))


if __name__ == '__main__':
    mp.set_start_method('spawn')
    train_opt = parse_arguments()
    train_dsacpnet(train_opt)
