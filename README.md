# A Noisy Sample Selection Framework Based on a Mixup Loss and Recalibration Strategy

<h5 align="center">

*Qian Zhang, De Yu, Xinru Zhou, Hanmeng Gong, Zheng Li, Yiming Liu, Ruirui Shao* 

[Mathematics](https://doi.org/10.3390/math12152389)
[License: Apache License 2.0](https://github.com/LanXiaoPang613/SMR/blob/main/LICENSE)

</h5>

The PyTorch implementation code of the paper, [A Noisy Sample Selection Framework Based on a Mixup Loss and Recalibration Strategy](https://doi.org/10.3390/math12152389).

**Abstract**
Deep neural networks (DNNs) have achieved breakthrough progress in various fields, largely owing to the support of large-scale datasets with manually annotated labels. However, obtaining such datasets is costly and time-consuming, making high-quality annotation a challenging task. In this work, we propose an improved noisy sample selection method, termed “sample selection framework”, based on a mixup loss and recalibration strategy (SMR). This framework enhances the robustness and generalization abilities of models. First, we introduce a robust mixup loss function to pre-train two models with identical structures separately. This approach avoids additional hyperparameter adjustments and reduces the need for prior knowledge of noise types. Additionally, we use a Gaussian Mixture Model (GMM) to divide the entire training set into labeled and unlabeled subsets, followed by robust training using semi-supervised learning (SSL) techniques. Furthermore, we propose a recalibration strategy based on cross-entropy (CE) loss to prevent the models from converging to local optima during the SSL process, thus further improving performance. Ablation experiments on CIFAR-10 with 50% symmetric noise and 40% asymmetric noise demonstrate that the two modules introduced in this paper improve the accuracy of the baseline (i.e., DivideMix) by 1.5% and 0.5%, respectively. Moreover, the experimental results on multiple benchmark datasets demonstrate that our proposed method effectively mitigates the impact of noisy labels and significantly enhances the performance of DNNs on noisy datasets. For instance, on the WebVision dataset, our method improves the top-1 accuracy by 0.7% and 2.4% compared to the baseline method.

![SMR Framework](./framework.tiff)

[//]: # (<img src="./framework.tiff" alt="SMR Framework" style="margin-left: 10px; margin-right: 50px;"/>)

## Installation

```shell
# Please install PyTorch using the official installation instructions (https://pytorch.org/get-started/locally/).
pip install -r requirements.txt
```

## Training

To train on the CIFAR dataset(https://www.cs.toronto.edu/~kriz/cifar.html), run the following command:

```shell
python train_cifar.py --r 0.4 --noise_mode 'asym' --lambda_u 0 --data_path './data/cifar-10-batches-py' --dataset 'cifar10' --num_class 10
python train_cifar.py --r 0.5 --noise_mode 'sym' --lambda_u 25 --data_path './data/cifar-10-batches-py' --dataset 'cifar10' --num_class 10
python train_cifar.py --r 0.2 --noise_mode 'sym' --lambda_u 25 --data_path './data/cifar-100-python' --dataset 'cifar100' --num_class 100
```

To train on the Animal-10N dataset(https://dm.kaist.ac.kr/datasets/animal-10n/), run the following command:

```shell
python train_animal10N.py --num_epochs 200 --lambda_u 0 --data_path './data/Animal-10N' --dataset 'animal10N' --num_class 10
```


## Citation

If you have any questions, do not hesitate to contact zhangqian@jsou.edu.cn

Also, if you find our work useful please consider citing our work:

```bibtex
Zhang, Q.; Yu, D.; Zhou, X.; Gong, H.; Li, Z.; Liu, Y.; Shao, R.
A Noisy Sample Selection Framework Based on a Mixup Loss and Recalibration Strategy.
Mathematics 2024, 12, 2389.
https://doi.org/10.3390/math12152389
```

## Acknowledgement

* [DivideMix](https://github.com/LiJunnan1992/DivideMix): The algorithm that our framework is based on.
* [UNICON](https://github.com/nazmul-karim170/UNICON-Noisy-Label): Inspiration for our framework.
