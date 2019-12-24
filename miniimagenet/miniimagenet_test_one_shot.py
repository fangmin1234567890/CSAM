
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import task_generator_test as tg
import os
import math
import argparse
import scipy as sp
import scipy.stats

parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 64)
parser.add_argument("-r","--relation_dim",type = int, default = 8)
parser.add_argument("-w","--class_num",type = int, default = 5)
parser.add_argument("-s","--sample_num_per_class",type = int, default = 1)
parser.add_argument("-b","--batch_num_per_class",type = int, default = 15)
parser.add_argument("-e","--episode",type = int, default= 10)
parser.add_argument("-t","--test_episode", type = int, default = 300)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-u","--hidden_unit",type=int,default=10)
args = parser.parse_args()


# Hyper Parameters
FEATURE_DIM = args.feature_dim
RELATION_DIM = args.relation_dim
CLASS_NUM = args.class_num
SAMPLE_NUM_PER_CLASS = 1
BATCH_NUM_PER_CLASS = args.batch_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit


class cha_att(nn.Module):
    def __init__(self):
        super(cha_att, self).__init__()
        #
        self.layer1 = nn.Linear(64, 16)
        self.layer1_1 = nn.ReLU()
        self.layer1_2 = nn.Linear(16, 64)
        self.layer1_3 = nn.Sigmoid()

    def forward(self, fm):
        fm1 = F.avg_pool2d(fm, fm.size()[2])
        fm1 = fm1.view(-1, 64)
        fm1 = self.layer1(fm1)
        fm2 = self.layer1_1(fm1)
        fm3 = self.layer1_2(fm2)
        fm3 = self.layer1_3(fm3)
        fm3 = fm3.view(-1, 64, 1, 1)
        return fm3


class spatial_att1(nn.Module):
    def __init__(self):
        super(spatial_att1, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1, padding=0))
        self.layer2 = nn.Sigmoid()

    def forward(self, x):
        fm = self.layer1(x)
        n_grid = fm.size()[2] * fm.size()[3]
        fm_reshape = fm.view(-1, n_grid)
        prob = self.layer2(fm_reshape)
        prob = prob.view(-1, 1, fm.size()[2], fm.size()[3])
        return prob


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1 + confidence) / 2., n - 1)
    return m, h


class CNNEncoder(nn.Module):
    """docstring for ClassName"""

    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU()

        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU()
        )
        self.layer2_1 = nn.MaxPool2d(2)
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU()

        )
        self.layer4_1 = nn.MaxPool2d(2)
        self.chan_t = cha_att()
        self.spat_t = spatial_att1()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        att1 = self.chan_t(out)
        out1_1 = att1 * out
        att1_1 = self.spat_t(out)
        out1_2 = att1_1 * out
        out3 = torch.add(out1_1, out1_2)
        out4 = torch.add(out3, out)
        out4 = self.layer2_1(out4)
        out5 = self.layer3(out4)
        out5 = self.layer4(out5)
        att = self.chan_t(out5)
        out2 = att * out5
        att2_1 = self.spat_t(out5)
        out2_2 = att2_1 * out5
        out3 = torch.add(out2_2, out2)
        out6 = torch.add(out3, out5)
        out6 = self.layer4_1(out6)
        # out = out.view(out.size(0),-1)
        return out6  # 64


class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""

    def __init__(self, input_size, hidden_size):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU()
        )

        self.fc1 = nn.Linear(input_size * 15 * 15, 64)
        self.fc1_1 = nn.Linear(64, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.layer3 = nn.ReLU()
        self.chan_t = cha_att()
        self.spat_t = spatial_att1()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        att = self.chan_t(out)
        out1 = att * out
        att1 = self.spat_t(out)
        out2 = att1 * out
        out3 = torch.add(out1, out2)
        out4 = torch.add(out3, out)
        out4 = out4.view(out4.size(0), -1)
        out4 = F.relu(self.fc1(out4))
        out4 = F.relu(self.fc1_1(out4))
        out4 = F.sigmoid(self.fc2(out4))
        return out4

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

def main():
    # Step 1: init data folders
    print("init data folders")
    # init character folders for dataset construction
    metatrain_folders,metatest_folders = tg.mini_imagenet_folders()

    # Step 2: init neural networks
    print("init neural networks")

    feature_encoder = CNNEncoder()
    relation_network = RelationNetwork(FEATURE_DIM,RELATION_DIM)


    feature_encoder.cuda(GPU)
    relation_network.cuda(GPU)


    if os.path.exists(str("./miniimagenet_feature_encoder1_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")):
        feature_encoder.load_state_dict(torch.load(str("./miniimagenet_feature_encoder1_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
        print("load feature encoder success")
    if os.path.exists(str("./miniimagenet_relation_network1_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")):
        relation_network.load_state_dict(torch.load(str("./miniimagenet_relation_network1_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
        print("load relation network success")

    # Step 3: build graph

    total_accuracy = 0.0
    total_h=0.0
    for episode in range(EPISODE):

            # test
            print("Testing...")

            accuracies = []
            for i in range(TEST_EPISODE):
                total_rewards = 0
                counter = 0
                task = tg.MiniImagenetTask(metatest_folders,CLASS_NUM,1,15)
                sample_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=1,split="train",shuffle=False)

                num_per_class = 3
                test_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=num_per_class,split="test",shuffle=True)
                sample_images,sample_labels = sample_dataloader.__iter__().next()
                for test_images,test_labels in test_dataloader:
                    batch_size = test_labels.shape[0]
                    # calculate features
                    sample_features = feature_encoder(Variable(sample_images).cuda(GPU)) # 5x64
                    test_features = feature_encoder(Variable(test_images).cuda(GPU)) # 20x64

                    # calculate relations
                    # each batch sample link to every samples to calculate relations
                    # to form a 100x128 matrix for relation network
                    sample_features_ext = sample_features.unsqueeze(0).repeat(batch_size,1,1,1,1)
                    test_features_ext = test_features.unsqueeze(0).repeat(1*CLASS_NUM,1,1,1,1)
                    test_features_ext = torch.transpose(test_features_ext,0,1)
                    relation_pairs = torch.cat((sample_features_ext,test_features_ext),2).view(-1,FEATURE_DIM*2,19,19)
                    relations = relation_network(relation_pairs).view(-1,CLASS_NUM)

                    _,predict_labels = torch.max(relations.data,1)
                    predict_labels=predict_labels.cuda(GPU)
                    test_labels=test_labels.cuda(GPU)
                    rewards = [1 if predict_labels[j]==test_labels[j] else 0 for j in range(batch_size)]

                    total_rewards += np.sum(rewards)
                    counter += batch_size
                accuracy = total_rewards/1.0/counter
                accuracies.append(accuracy)

            test_accuracy,h = mean_confidence_interval(accuracies)

            print("test accuracy:",test_accuracy,"h:",h)

            total_accuracy += test_accuracy
            total_h+=h
    print("aver_accuracy:",total_accuracy/EPISODE)
    print("total_h:",total_h/EPISODE)




if __name__ == '__main__':
    main()