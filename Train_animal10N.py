from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.models as models
from models.CNN import CNN
import random
import os
import argparse
import numpy as np
import dataloader_animal10N as animal_dataloader
from sklearn.mixture import GaussianMixture
import copy

parser = argparse.ArgumentParser(description='PyTorch Clothing1M Training')
parser.add_argument('--batch_size', default=128, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=0, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--id', default='animal10N-dmix')
# parser.add_argument('--data_path', default='E:/Dataset_All/clothing1M/images', type=str, help='path to dataset')
parser.add_argument('--data_path', default='C:/Users/Administrator/Desktop/All_dataset/Animal-10N', type=str, help='path to dataset')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=10, type=int)
# parser.add_argument('--num_batches', default=1000, type=int)
args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


# Training
def train(epoch, net, net2, optimizer, labeled_trainloader, unlabeled_trainloader):
    net.train()
    net2.eval()  # fix one network and train the other

    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = (len(labeled_trainloader.dataset) // args.batch_size) + 1
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):
        try:
            inputs_u, inputs_u2 = unlabeled_train_iter.__next__()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = unlabeled_train_iter.__next__()
        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1, 1), 1)
        w_x = w_x.view(-1, 1).type(torch.FloatTensor)

        inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11 = net(inputs_u)
            outputs_u12 = net(inputs_u2)
            outputs_u21 = net2(inputs_u)
            outputs_u22 = net2(inputs_u2)

            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) +
                  torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4
            ptu = pu ** (1 / args.T)  # temparature sharpening

            targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # normalize
            targets_u = targets_u.detach()

            # label refinement of labeled samples
            outputs_x = net(inputs_x)
            outputs_x2 = net(inputs_x2)

            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x * labels_x + (1 - w_x) * px
            ptx = px ** (1 / args.T)  # temparature sharpening

            targets_x = ptx / ptx.sum(dim=1, keepdim=True)  # normalize
            targets_x = targets_x.detach()

            # mixmatch
        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1 - l)

        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a[:batch_size * 2] + (1 - l) * input_b[:batch_size * 2]
        mixed_target = l * target_a[:batch_size * 2] + (1 - l) * target_b[:batch_size * 2]

        logits = net(mixed_input)

        Lx = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * mixed_target, dim=1))

        # regularization
        prior = torch.ones(args.num_class) / args.num_class
        prior = prior.cuda()
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior * torch.log(prior / pred_mean))

        loss = Lx + penalty

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write('Animal10N | Epoch [%3d/%3d] Iter[%3d/%3d]\t  Labeled loss: %.4f '
                         % (epoch, args.num_epochs, batch_idx + 1, num_iter, Lx.item()))
        sys.stdout.flush()

def mixup_criterion(pred, y_a, y_b, lam):
    c = F.log_softmax(pred, 1)
    return lam * F.cross_entropy(c, y_a) + (1 - lam) * F.cross_entropy(c, y_b)

def warmup(net, optimizer, dataloader):
    net.train()
    num_batches = 50000/args.batch_size
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):

        l = np.random.beta(args.alpha, args.alpha)
        # 促使X'更加靠近labeled sample而不是无监督样本
        l = max(l, 1 - l)
        idx = torch.randperm(inputs.size(0))
        targets = torch.zeros(inputs.size(0), args.num_class).scatter_(1, labels.view(-1, 1), 1).cuda()
        targets = torch.clamp(targets, 1e-4, 1.)
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        # outputs = net(inputs)
        # loss = CEloss(outputs, labels)
        #
        # penalty = conf_penalty(outputs)
        # L = loss + penalty
        input_a, input_b = inputs, inputs[idx]
        target_a, target_b = targets, targets[idx]
        labels_a, labels_b = labels, labels[idx]

        # 利用mix但是促使模型更偏向于label而不是UNlabel
        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        outputs = net(mixed_input)
        loss = mixup_criterion(outputs, labels_a, labels_b, l)
        L = loss
        L.backward()
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write('|Warm-up: Iter[%3d/%3d]\t CE-loss: %.4f  Conf-Penalty: %.4f'
                         % (2*(batch_idx + 1), num_batches, loss.item(), 0.))
        sys.stdout.flush()


def warmup2(net, optimizer, dataloader):
    net.train()
    num_batches = 50000 / args.batch_size
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = CEloss(outputs, labels)

        penalty = conf_penalty(outputs)
        L = loss #+ penalty
        L.backward()
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write('|Warm-up: Iter[%3d/%3d]\t CE-loss: %.4f  Conf-Penalty: %.4f'
                         % (2 * (batch_idx + 1), num_batches, loss.item(), penalty.item()))
        sys.stdout.flush()

def val(net, val_loader, best_acc, w_glob=None):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()
    acc = 100. * correct / total
    print("\n| Validation\t Net%d  Acc: %.2f%%" % (k, acc))
    if acc > best_acc[k - 1]:
        best_acc[k - 1] = acc
        print('| Saving Best Net%d ...' % k)
        save_point = './checkpoint/%s_net%d.pth.tar' % (args.id, k)
        torch.save(net.state_dict(), save_point)
    return acc


def test(epoch, net1, net2, test_loader, best_acc, w_glob=None):
    if w_glob is None:
        net1.eval()
        net2.eval()
        correct = 0
        correct2 = 0
        correct1 = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs1 = net1(inputs)
                outputs2 = net2(inputs)
                outputs = outputs1 + outputs2
                _, predicted = torch.max(outputs, 1)
                _, predicted1 = torch.max(outputs1, 1)
                _, predicted2 = torch.max(outputs2, 1)

                total += targets.size(0)
                correct += predicted.eq(targets).cpu().sum().item()
                correct1 += predicted1.eq(targets).cpu().sum().item()
                correct2 += predicted2.eq(targets).cpu().sum().item()
        acc = 100. * correct / total
        acc1 = 100. * correct / total
        acc2 = 100. * correct / total
        if best_acc < acc:
            best_acc = acc
        print(
            "\n| Ensemble network Test Epoch #%d\t Accuracy: %.2f, Accuracy1: %.2f, Accuracy2: %.2f, best_acc: %.2f%%\n" % (
            epoch, acc, acc1, acc2, best_acc))
        log.write('ensemble_Epoch:%d   Accuracy:%.2f, Accuracy1: %.2f, Accuracy2: %.2f, best_acc: %.2f\n' % (
        epoch, acc, acc1, acc2, best_acc))
        log.flush()
    else:
        net1_w_bak = net1.state_dict()
        net1.load_state_dict(w_glob)
        net1.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs1 = net1(inputs)
                _, predicted = torch.max(outputs1, 1)
                total += targets.size(0)
                correct += predicted.eq(targets).cpu().sum().item()
        acc = 100. * correct / total
        if best_acc < acc:
            best_acc = acc
        print("\n| Global network Test Epoch #%d\t Accuracy: %.2f, best_acc: %.2f%%\n" % (epoch, acc, best_acc))
        log.write('global_Epoch:%d   Accuracy:%.2f, best_acc: %.2f\n' % (epoch, acc, best_acc))
        log.flush()
        #   恢复权重
        net1.load_state_dict(net1_w_bak)
    return best_acc


def eval_train(epoch, model):
    model.eval()
    num_samples = eval_loader.dataset.__len__()
    losses = torch.zeros(num_samples)
    paths = []
    n = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, path) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = CE(outputs, targets)
            for b in range(inputs.size(0)):
                losses[n] = loss[b]
                paths.append(path[b])
                n += 1
            sys.stdout.write('\r')
            sys.stdout.write('| Evaluating loss Iter %3d\t' % (batch_idx))
            sys.stdout.flush()

    losses = (losses - losses.min()) / (losses.max() - losses.min())
    losses = losses.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, max_iter=10, reg_covar=5e-4, tol=1e-2)
    gmm.fit(losses)
    prob = gmm.predict_proba(losses)
    prob = prob[:, gmm.means_.argmin()]
    return prob, paths


class NegEntropy(object):
    def __call__(self, outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))


def create_model():
    use_cnn = False
    if use_cnn:
        model = CNN()
        model = model.cuda()
    else:
        model = models.vgg19_bn(pretrained=False)
        model.classifier._modules['6'] = nn.Linear(4096, 10)
        model = model.cuda()
    return model


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
            # 只考虑iid noise的话，每个client训练样本数一样，所以不用做nk/n
        w_avg[k] = torch.div(w_avg[k], len(w))

    return w_avg


log = open('./checkpoint/%s.txt' % args.id, 'w')
log.flush()

loader = animal_dataloader.animal_dataloader(root=args.data_path, batch_size=args.batch_size, num_workers=0)

print('| Building net')
net1 = create_model()
net2 = create_model()
cudnn.benchmark = True

optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
conf_penalty = NegEntropy()

local_round = 5
balance_crit = 'median'  # 'median'
exp_path = './checkpoint/c2mt_animal10N'

boot_loader = None
w_glob = None
best_en_acc = 0.
best_gl_acc = 0.
resume_epoch = 0
warm_up = 30
if resume_epoch > 0:
    snapLast = exp_path + str(resume_epoch - 1) + "_model1.pth"
    global_state = torch.load(snapLast)
    net1.load_state_dict(global_state)
    snapLast = exp_path + str(resume_epoch - 1) + "_model2.pth"
    global_state = torch.load(snapLast)
    net2.load_state_dict(global_state)
mid_wp = 15

for epoch in range(resume_epoch, args.num_epochs + 1):
    lr = args.lr
    if 140 <= epoch < 80:
        lr /= 10
    elif epoch >= 140:
        lr /= 100
    # if 15 <= epoch:
    #     lr /= 2
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr

    local_weights = []
    if epoch < warm_up:  # warm up
        train_loader = loader.run('warmup')
        print('Warmup Net1')
        warmup(net1, optimizer1, train_loader)
        train_loader = loader.run('warmup')
        print('\nWarmup Net2')
        warmup(net2, optimizer2, train_loader)
    elif (epoch+1)%mid_wp == 0 and epoch > warm_up:
        warmup_trainloader = loader.run('warmup')
        print('Warmup Net1')
        warmup2(net1, optimizer1, warmup_trainloader)
        print('\nWarmup Net2')
        warmup2(net2, optimizer2, warmup_trainloader)
    else:

        print('\n==== net 1 evaluate next epoch training data loss ====')
        eval_loader = loader.run('eval_train')  # evaluate training data loss for next epoch
        prob1, paths1 = eval_train(epoch, net1)
        print('\n==== net 2 evaluate next epoch training data loss ====')
        eval_loader = loader.run('eval_train')
        prob2, paths2 = eval_train(epoch, net2)

        pred1 = (prob1 > args.p_threshold)  # divide dataset
        pred2 = (prob2 > args.p_threshold)


        print(f'dmix selection, Train Net1')
        labeled_trainloader, unlabeled_trainloader = loader.run('train', pred2, prob2, paths=paths2)  # co-divide
        train(epoch, net1, net2, optimizer1, labeled_trainloader, unlabeled_trainloader)  # train net1

        print(f'\ndmix selection, Train Net2')
        labeled_trainloader, unlabeled_trainloader = loader.run('train', pred1, prob1, paths=paths1)  # co-divide
        train(epoch, net2, net1, optimizer2, labeled_trainloader, unlabeled_trainloader)  # train net2

        if epoch % 5 == 0:
            snapLast = exp_path + str(epoch) + "_model1.pth"
            torch.save(net1.state_dict(), snapLast)
            snapLast = exp_path + str(epoch) + "_model2.pth"
            torch.save(net2.state_dict(), snapLast)

    test_loader = loader.run('test')
    best_en_acc = test(epoch, net1, net2, test_loader, best_en_acc)
    # best_gl_acc = test(epoch, net1, net2, test_loader, best_gl_acc, w_glob=w_glob)

