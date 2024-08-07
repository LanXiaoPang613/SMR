from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
from PreResNet import *
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import dataloader_cifar as dataloader
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=128, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--noise_mode', default='sym')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--r', default=0.8, type=float, help='noise ratio')
parser.add_argument('--id', default='')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--data_path', default='./data/cifar-10-batches-py', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
# parser.add_argument('--data_path', default='./data/cifar-100-python', type=str, help='path to dataset')
# parser.add_argument('--dataset', default='cifar100', type=str)
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

        # Transform label to one-hot，转为0-1矩阵
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

            # 取average of 所有网络的输出，作者利用了所谓的augmentation
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1)
                  + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4
            ptu = pu ** (1 / args.T)  # temparature sharpening

            # Algorithm 1 中的shapen(qb,T)
            targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # normalize
            targets_u = targets_u.detach()

            # label refinement of labeled samples
            outputs_x = net(inputs_x)
            outputs_x2 = net(inputs_x2)

            # 取labeled的输出平均值
            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2

            # 公式(3)(4)退火
            px = w_x * labels_x + (1 - w_x) * px
            ptx = px ** (1 / args.T)  # temparature sharpening

            targets_x = ptx / ptx.sum(dim=1, keepdim=True)  # normalize
            targets_x = targets_x.detach()

            # mixmatch
        l = np.random.beta(args.alpha, args.alpha)
        # 促使X'更加靠近labeled sample而不是无监督样本
        l = max(l, 1 - l)

        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        # 随机输出mini batch的序号，来mixup
        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        # 利用mix但是促使模型更偏向于label而不是UNlabel
        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        logits = net(mixed_input)
        # 输出被排列成两部分，input_x、Input_u
        logits_x = logits[:batch_size * 2]
        logits_u = logits[batch_size * 2:]

        # 利用公式(9)-(10)计算损失函数，其中lamb是所谓的warm up
        Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size * 2],
                                 logits_u, mixed_target[batch_size * 2:],
                                 epoch + batch_idx / num_iter, warm_up)

        # regularization
        prior = torch.ones(args.num_class) / args.num_class
        prior = prior.cuda()
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        # 一般来说会省略固定的prior部分，只取last term
        # lambR=1
        penalty = torch.sum(prior * torch.log(prior / pred_mean))

        # lamb是通过warm和current epoch比较得出的百分数，意味着随着epoch进行，Lu所占比重会逐渐增加
        # 前期需要保持标准CE损失，但是实际还有penalty
        loss = Lx + lamb * Lu + penalty
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % 200 ==0:
            sys.stdout.write('\r')
            sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.8f  Unlabeled loss: %.8f'
                             % (args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx + 1, num_iter,
                                Lx.item(), Lu.item()))
            sys.stdout.flush()

def mixup_criterion(pred, y_a, y_b, lam):
    c = F.log_softmax(pred, 1)
    return lam * F.cross_entropy(c, y_a) + (1 - lam) * F.cross_entropy(c, y_b)

soft_mix_warm = True

def warmup(epoch, net, optimizer, dataloader):
    net.train()
    num_iter = (len(dataloader.dataset) // dataloader.batch_size) + 1
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):
        optimizer.zero_grad()
        l = np.random.beta(args.alpha, args.alpha)
        # 促使X'更加靠近labeled sample而不是无监督样本
        l = max(l, 1 - l)
        idx = torch.randperm(inputs.size(0))
        targets = torch.zeros(inputs.size(0), args.num_class).scatter_(1, labels.view(-1, 1), 1).cuda()
        targets = torch.clamp(targets, 1e-4, 1.)
        inputs, labels = inputs.cuda(), labels.cuda()
        if soft_mix_warm:
            input_a, input_b = inputs, inputs[idx]
            target_a, target_b = targets, targets[idx]
            labels_a, labels_b = labels, labels[idx]

            # 利用mix但是促使模型更偏向于label而不是UNlabel
            mixed_input = l * input_a + (1 - l) * input_b
            mixed_target = l * target_a + (1 - l) * target_b

            outputs = net(mixed_input)
            loss = mixup_criterion(outputs, labels_a, labels_b, l)
            L = loss
        else:
            outputs = net(inputs)
            loss = CEloss(outputs, labels)
            if args.noise_mode == 'asym':  # penalize confident prediction for asymmetric noise
                penalty = conf_penalty(outputs)
                L = loss + penalty
            elif args.noise_mode == 'sym':
                L = loss

        L.backward()
        optimizer.step()
        if batch_idx % 200 == 0:
            sys.stdout.write('\r')
            sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                             % (args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx + 1, num_iter,
                                loss.item()))
            sys.stdout.flush()

def warmup2(epoch, net, optimizer, dataloader):
    net.train()
    num_iter = (len(dataloader.dataset) // dataloader.batch_size) + 1
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):
        optimizer.zero_grad()
        l = np.random.beta(args.alpha, args.alpha)
        # 促使X'更加靠近labeled sample而不是无监督样本
        l = max(l, 1 - l)
        idx = torch.randperm(inputs.size(0))
        targets = torch.zeros(inputs.size(0), args.num_class).scatter_(1, labels.view(-1, 1), 1).cuda()
        targets = torch.clamp(targets, 1e-4, 1.)
        inputs, labels = inputs.cuda(), labels.cuda()

        # CE校正

        outputs = net(inputs)
        loss = CEloss(outputs, labels)
        L = loss

        # mixup校正
        # input_a, input_b = inputs, inputs[idx]
        # target_a, target_b = targets, targets[idx]
        # labels_a, labels_b = labels, labels[idx]

        # 利用mix但是促使模型更偏向于label而不是UNlabel
        # mixed_input = l * input_a + (1 - l) * input_b
        # mixed_target = l * target_a + (1 - l) * target_b
        #
        # outputs = net(mixed_input)
        # loss = mixup_criterion(outputs, labels_a, labels_b, l)
        # L = loss

        L.backward()
        optimizer.step()
        if batch_idx % 200 == 0:
            sys.stdout.write('\r')
            sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                             % (args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx + 1, num_iter,
                                loss.item()))
            sys.stdout.flush()

best_aa=0.
def test(epoch, net1, net2):
    global best_aa
    net1.eval()
    net2.eval()
    correct = 0
    total = 0

    precision = 0.
    f1 = 0.
    recall = 0.
    predicted_label = np.zeros(len(test_loader.dataset), dtype=int)


    with torch.no_grad():
        for batch_idx, (inputs, targets, idx) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)
            outputs = outputs1 + outputs2
            _, predicted = torch.max(outputs, 1)

            predicted_label[idx] = predicted.cpu().data.numpy()

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()
    acc = 100. * correct / total
    if best_aa < acc:
        best_aa = acc
    confusion = confusion_matrix(np.array(test_loader.dataset.test_label), predicted_label)
    precision = 100. * precision_score(np.array(test_loader.dataset.test_label), predicted_label, average='macro')
    recall = 100. * recall_score(np.array(test_loader.dataset.test_label), predicted_label, average='macro')
    f1 = 100. * f1_score(np.array(test_loader.dataset.test_label), predicted_label, average='macro')

    print('\n | Epoch:%d   confusion matrix', confusion)
    print('\n | Epoch:%d   Accuracy:%.2f, best Accuracy:%.2f,    precision:%.2f,    recall:%.2f,    f1:%.2f \n' % (epoch, acc, best_aa, precision, recall, f1))
    test_log.write('Epoch:%d   Accuracy:%.2f, best Accuracy:%.2f, precision:%.2f, recall:%.2f, f1:%.2f \n' % (epoch, acc, best_aa, precision, recall, f1))
    test_log.flush()


def eval_train(model, all_loss):
    model.eval()
    losses = torch.zeros(50000)
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = CE(outputs, targets)
            for b in range(inputs.size(0)):
                losses[index[b]] = loss[b]
    losses = (losses - losses.min()) / (losses.max() - losses.min())
    all_loss.append(losses)

    if args.r == 0.9:
        # average loss over last 5 epochs to improve convergence stability
        history = torch.stack(all_loss)
        input_loss = history[-5:].mean(0)
        input_loss = input_loss.reshape(-1, 1)
    else:
        input_loss = losses.reshape(-1, 1)

    # fit a two-component GMM to the loss
    # 参数如下：
    # n_components 聚类数量，max_iter 最大迭代次数，tol 阈值低于停止，reg_covar 协方差矩阵对角线上非负正则化参数，接近0即可
    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss)
    prob = prob[:, gmm.means_.argmin()]
    return prob, all_loss, losses.numpy()


def linear_rampup(current, warm_up, rampup_length=16):
    # 线性warm_up，对sym噪声使用标准CE训练一段时间
    # 实际warm up epoch是warm_up+rampup_length
    current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
    re_val = args.lambda_u * float(current)
    # print("   current warm up parameters:", current)
    # print("return parameters:", re_val)
    return re_val


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        # 利用mixup后的交叉熵，px输出*log(px_model)
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        # 而UNlabel则是均方误差，p_u输出-pu_model
        Lu = torch.mean((probs_u - targets_u) ** 2)

        return Lx, Lu, linear_rampup(epoch, warm_up)


class NegEntropy(object):
    def __call__(self, outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))


def create_model():
    # 其实是pre-resnet18，使用的是pre-resnet block
    model = ResNet18(num_classes=args.num_class)
    model = model.cuda()
    return model

def plotHistogram(model_1_loss, model_2_loss, noise_index, clean_index, epoch, noise_rate):
    fig = plt.figure()
    plt.subplot(121)
    plt.hist(np.array(model_1_loss[noise_index]), bins=300, alpha=0.5, color='red', label='Noisy subset')
    plt.hist(np.array(model_1_loss[clean_index]), bins=300, alpha=0.5, color='blue', label='Clean subset')
    plt.legend(loc='upper right')
    plt.title('Model_1')
    plt.subplot(122)
    plt.hist(np.array(model_2_loss[noise_index]), bins=300, alpha=0.5, color='red', label='Noisy subset')
    plt.hist(np.array(model_2_loss[clean_index]), bins=300, alpha=0.5, color='blue', label='Clean subset')
    plt.legend(loc='upper right')
    plt.title('Model_2')
    print('\nlogging histogram...')
    title = 'cifar10_Sym_double_'+ str(noise_rate)
    plt.savefig(os.path.join('./figure_his/', '{}_{}.{}'.format(epoch, title, ".tif")), dpi=300)
    # plt.show()
    plt.close()

if os.path.exists('checkpoint') == False:
    os.mkdir('checkpoint')
    print("新建日志文件夹")
stats_log = open('./checkpoint/%s_%.1f_%s' % (args.dataset, args.r, args.noise_mode) + '_stats.txt', 'w')
test_log = open('./checkpoint/%s_%.1f_%s' % (args.dataset, args.r, args.noise_mode) + '_acc.txt', 'w')

if args.dataset == 'cifar10':
    warm_up = 10
elif args.dataset == 'cifar100':
    warm_up = 30

loader = dataloader.cifar_dataloader(args.dataset, r=args.r, noise_mode=args.noise_mode,
                                     batch_size=args.batch_size, num_workers=0,
                                     root_dir=args.data_path, log=stats_log,
                                     noise_file='%s/%.1f_%s.json' % (args.data_path, args.r, args.noise_mode))

print('| Building net')
net1 = create_model()
net2 = create_model()
cudnn.benchmark = True

criterion = SemiLoss()
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
if args.noise_mode == 'asym':
    # 本文第一个问题，对于非对称和对称需要不同措施，这很不适用
    # 其次本文在不同步骤中噪声数据处理措施很凌乱
    conf_penalty = NegEntropy()

all_loss = [[], []]  # save the history of losses from two networks
mid_wp = 30
for epoch in range(args.num_epochs + 1):
    lr = args.lr
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr
    test_loader = loader.run('test')
    eval_loader = loader.run('eval_train')

    noise_ind, clean_ind = eval_loader.dataset.if_noise()

    if epoch < warm_up:
        warmup_trainloader = loader.run('warmup')
        print('Warmup Net1')
        warmup(epoch, net1, optimizer1, warmup_trainloader)
        print('\nWarmup Net2')
        warmup(epoch, net2, optimizer2, warmup_trainloader)
    elif (epoch+1)%mid_wp == 0 and epoch > warm_up:
        warmup_trainloader = loader.run('warmup')
        print('Warmup Net1')
        warmup2(epoch, net1, optimizer1, warmup_trainloader)
        print('\nWarmup Net2')
        warmup2(epoch, net2, optimizer2, warmup_trainloader)
    else:
        prob1, all_loss[0], loss1 = eval_train(net1, all_loss[0])
        prob2, all_loss[1], loss2 = eval_train(net2, all_loss[1])

        pred1 = (prob1 > args.p_threshold)
        pred2 = (prob2 > args.p_threshold)

        temp_pred1 = (prob1[clean_ind] > args.p_threshold)
        temp_pred2 = (prob2[clean_ind] > args.p_threshold)

        if epoch % 10 ==0:
            plotHistogram(np.array(loss1), np.array(loss2), noise_ind, clean_ind, epoch, args.r)
        eval_loader.dataset.if_noise(pred1)
        eval_loader.dataset.if_noise(pred2)

        print('Train Net1')
        # prob2就是先验概率wi,通过GMM拟合出来的，大于阈值就认为是clean，否则noisy
        labeled_trainloader, unlabeled_trainloader = loader.run('train', pred2, prob2)  # co-divide
        train(epoch, net1, net2, optimizer1, labeled_trainloader, unlabeled_trainloader)  # train net1

        print('\nTrain Net2')
        labeled_trainloader, unlabeled_trainloader = loader.run('train', pred1, prob1)  # co-divide
        train(epoch, net2, net1, optimizer2, labeled_trainloader, unlabeled_trainloader)  # train net2

    test(epoch, net1, net2)


