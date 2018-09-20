import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# torch.manual_seed(456)
# torch.cuda.manual_seed(789)  # reproducible
import torch.nn as nn
import matplotlib.patches as patches
import scipy.io
from collections import OrderedDict
# torch can only train on Variable, so convert them to Variable
import torch.optim as optim
import os
from options import *
actor_list = ['conv1', 'conv2', 'conv3','conv4', 'fc1', 'fc2', 'fc3']
from data_prov import *
from sample_generator import *
# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()
def append_params(params, module, prefix):
    for child in module.children():
        for k, p in child._parameters.iteritems():
            if p is None: continue

            if isinstance(child, nn.BatchNorm2d):
                name = prefix + '_bn_' + k
            else:
                name = prefix + '_' + k

            if name not in params:
                params[name] = p
            else:
                raise RuntimeError("Duplicated param name: %s" % (name))

class LRN(nn.Module):
    def __init__(self):
        super(LRN, self).__init__()

    def forward(self, x):
        #
        # x: N x C x H x W
        pad = Variable(x.data.new(x.size(0), 1, 1, x.size(2), x.size(3)).zero_())
        x_sq = (x**2).unsqueeze(dim=1)
        x_tile = torch.cat((torch.cat((x_sq,pad,pad,pad,pad),2),
                            torch.cat((pad,x_sq,pad,pad,pad),2),
                            torch.cat((pad,pad,x_sq,pad,pad),2),
                            torch.cat((pad,pad,pad,x_sq,pad),2),
                            torch.cat((pad,pad,pad,pad,x_sq),2)),1)
        x_sumsq = x_tile.sum(dim=1).squeeze(dim=1)[:,2:-2,:,:]
        x = x / ((2.+0.0001*x_sumsq)**0.75)
        return x

class DDPG(nn.Module):
    def __init__(self, model_path=None, K=1):
        super(DDPG, self).__init__()
        self.K = K
        self.layers = nn.Sequential(OrderedDict([
            ('conv1', nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2),
                                    nn.ReLU(),
                                    LRN(),
                                    nn.MaxPool2d(kernel_size=3, stride=2))),
            ('conv2', nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=2),
                                    nn.ReLU(),
                                    LRN(),
                                    nn.MaxPool2d(kernel_size=3, stride=2))),
            ('conv3', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1),
                                    nn.ReLU())),
            ('conv4', nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1),
                                    nn.ReLU())),
            ('fc1', nn.Sequential(nn.Linear(512, 512),
                                  nn.ReLU())),
            ('fc3', nn.Sequential(nn.Linear(512, 3),
                                  nn.Tanh()))]))



        if model_path is not None:
            if os.path.splitext(model_path)[1] == '.pth':
                self.load_model(model_path)
            elif os.path.splitext(model_path)[1] == '.npy':
                self.load_npy_model(model_path)
            else:
                raise RuntimeError("Unkown model format: %s" % (model_path))
        self.build_param_dict()

    def build_param_dict(self):
        self.params = OrderedDict()
        for name, module in self.layers.named_children():
            append_params(self.params, module, name)
        # for k, module in enumerate(self.branches):
        #     append_params(self.params, module, 'fc3_%d' % (k))

    def set_learnable_params(self, layers):
        for k, p in self.params.iteritems():
            if any([k.startswith(l) for l in layers]):
                p.requires_grad = True
            else:
                p.requires_grad = False

    def get_learnable_params(self):
        params = OrderedDict()
        for k, p in self.params.iteritems():
            if p.requires_grad:
                params[k] = p
        return params

    def forward(self, x, k=0, in_layer='conv1', out_layer='fc3'):
        #
        # forward model from in_layer to out_layer
        run = False
        for name, module in self.layers.named_children():
            if name == in_layer:
                run = True
            if run:
                x = module(x)
                if name == 'conv4':
                    x = x.view(x.size(0), -1)
                if name == out_layer:
                    return x

    def load_model(self, model_path):
        states = torch.load(model_path)
        shared_layers = states['shared_layers']
        self.layers.load_state_dict(shared_layers)

    def load_npy_model(self, matfile):
        mat = np.load(matfile).item()

        # copy conv weights
        for i in range(4):
            weight, bias = mat[actor_list[i]][0],mat[actor_list[i]][1]
            self.layers[i][0].weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))
            self.layers[i][0].bias.data = torch.from_numpy(bias)
        for i in range(4, 6):
            weight, bias = mat[actor_list[i]][0],mat[actor_list[i]][1]
            self.layers[i][0].weight.data = torch.from_numpy(np.transpose(weight,(1,0)))
            self.layers[i][0].bias.data = torch.from_numpy(bias)

def set_optimizer(model, lr_base, lr_mult=opts['lr_mult'], momentum=opts['momentum'], w_decay=opts['w_decay']):
    params = model.get_learnable_params()
    param_list = []
    for k, p in params.iteritems():
        lr = lr_base
        for l, m in lr_mult.iteritems():
            if k.startswith(l):
                lr = lr_base * m
        param_list.append({'params': [p], 'lr':lr})
    optimizer = optim.SGD(param_list, lr = lr, momentum=momentum, weight_decay=w_decay)
    return optimizer

def cal_distance(samples, ground_th):
    distance = samples[:, 0:2] + samples[:, 2:4] / 2 - ground_th[:, 0:2] - ground_th[:, 2:4] / 2
    distance = distance / samples[:, 2:4]
    rate = samples[:, 3] / ground_th[:, 3]
    rate = np.array(rate).reshape(rate.shape[0], 1)
    rate = rate - 1.0
    distance = np.hstack([distance, rate])
    return  distance


def init_actor(image, gt):
    np.random.seed(123)
    torch.manual_seed(456)
    torch.cuda.manual_seed(789)
    model = DDPG('model/Actor170000.npy')
    # model = DDPG('model/vggm1-4.npy')
    # model = DDPG('model/actor_50000.npy')
    model = model.cuda()
    model.train()
    # model.set_learnable_params(['fc'])

    # init_optimizer = set_optimizer(model, 0.01)
    init_optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_func = torch.nn.MSELoss()

    actor_samples = gen_samples(SampleGenerator('uniform', image.size, 0.3, 1.5, 1.1),
                                gt, opts['n_bbreg'], opts['overlap_distance'], opts['scale_distance'] )

    for t in range(30):
        actor_num = np.random.choice(actor_samples.shape[0], 256)
        actor_sample = np.round(actor_samples[actor_num, :])
        extractor = RegionExtractor(image, actor_sample, opts['img_size'], 0, 256)
        for i, regions in enumerate(extractor):
            regions = Variable(regions)
            regions = regions.cuda()
            feat = model(regions)
        # feat.data.clone().cpu().numpy()
        distance = cal_distance(actor_sample, np.tile(gt,[256,1]))


        loss = loss_func(feat, Variable(torch.FloatTensor(distance).cuda()))     # must be (1. nn output, 2. target)

        model.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        init_optimizer.step()        # apply gradients
        if opts['show_train']:
            print "Iter %d, Loss %.4f" % (t, loss.data[0])

    return model

def show_pf_once(frame, bbox, fig_n):
    if bbox.shape.__len__() == 1:
        bbox = np.array(bbox).reshape([1,4])
    fig = plt.figure(fig_n)
    ax = fig.add_subplot(111)
    r = np.zeros(bbox.shape[0])
    ax.imshow(np.uint8(frame))
    for i in range(bbox.shape[0]):
        ax.add_patch(patches.Rectangle((bbox[i,0],bbox[i,1]), bbox[i,2], bbox[i,3], linewidth=1, edgecolor='r', fill=False))
    plt.ion()
    plt.show()
    plt.pause(0.001)

def move_crop(pos_, deta_pos, img_size, rate):
    flag = 0
    if pos_.shape.__len__() == 1:
        pos_ = np.array(pos_).reshape([1,4])
        deta_pos = np.array(deta_pos).reshape([1, 3])
        flag = 1
    try:
        pos_deta = deta_pos[:, 0:2] * pos_[:, 2:]
    except:
        a = 1
    pos = np.copy(pos_)
    center = pos[:, 0:2] + pos[:, 2:4] / 2
    center_ = center - pos_deta
    pos[:, 2] = pos[:, 2] / (1 + deta_pos[:, 2])
    pos[:, 3] = pos[:, 3] / (1 + deta_pos[:, 2])

    if np.max((pos[:, 2] > img_size[1])*1.0) == 1.0:
        num = pos[:, 2] > img_size[1]
        pos[num, 2] = img_size[1]/2
        pos[num, 3] = pos[num, 2]/rate

    if np.max((pos[:, 3] > img_size[0])*1.0) == 1.0:
        num = pos[:, 3] > img_size[0]
        pos[num, 3] = img_size[0]/2
        pos[num, 2] = pos[num, 3] * rate
    if np.max((pos[:, 3] > (pos[:, 2] / rate) *1.1)) == 1.0  or  np.max((pos[:, 3] < (pos[:, 2] / rate) / 1.1)) == 1.0:
        pos[:, 3] = pos[:, 2] / rate


    pos[:, 0:2] = center_ - pos[:, 2:4] / 2

    pos[pos[:, 0] + pos[:, 2] > img_size[1], 0] = \
        img_size[1] - pos[pos[:, 0] + pos[:, 2] > img_size[1], 2] - 1
    pos[pos[:, 1] + pos[:, 3] > img_size[0], 1] = \
        img_size[0] - pos[pos[:, 1] + pos[:, 3] > img_size[0], 3] - 1
    pos[pos[:, 0] < 0, 0] = 0
    pos[pos[:, 1] < 0, 1] = 0
    pos[pos[:, 2] < 1, 2] = 1
    pos[pos[:, 3] < 1, 3] = 1
    pos[pos[:, 2] > img_size[1], 2] = img_size[1]
    pos[pos[:, 3] > img_size[0], 3] = img_size[0]
    if flag == 1:
        pos = pos[0]

    return np.round(pos)