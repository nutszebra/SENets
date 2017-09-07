# What's this
Implementation of Squeeze and Excitation Networks (SENets) [[2]][Paper2] by chainer  

# Dependencies

    git clone https://github.com/nutszebra/SENets.git
    cd SENets
    git submodule init
    git submodule update
    # Note: chainer==1.24.0

# How to run
     # for SENets
     python main_se_residual_net.py -g 0 -p ./result_senet -e 200 -b 128 -lr 0.1 -k 1 -n 18 -multiplier 4 -r 4
     # for resnet
     python main_residual_net.py -g 0 -p ./result_resnet -e 200 -b 128 -lr 0.1 -k 1 -n 18 -multiplier 4


# Details about my implementation

* Data Augmentation

Train: Pictures are randomly resized in the range of [32, 36], then 32x32 patches are extracted randomly and are normalized locally. Horizontal flipping is applied with 0.5 probability. Cutout [[3]][Paper3] is applied with 0.5 probability (16x16 window). 

Test: Pictures are resized to 32x32, then they are normalized locally. Single image test is used to calculate total accuracy.  
* Optimization: Momentum SGD (momentum is 0.9)

* Scheduling: 0.1 is multiplied to learning rate at [100, 150] epochs.

* Initial learning rate: 0.1, but warm-up learnig rate, 0.01, is only used at first epoch.

* Weight decay    
0.0001  



# Cifar10 result

| network                                                             | depth | total accuracy (%) |
|:-----------------------------------------------------|--------------|-------------------:|
| SEResNet(my implementation) [[2]][Paper2]            | 164 + 108    | soon               |
| ResNet [[1]][Paper]                                  | 164          | 94.54              |
| my implementation[[1]][Paper]                        | 164          | soon               |


<img src="https://github.com/nutszebra/SENets/blob/master/loss.jpg" alt="loss" title="loss">
<img src="https://github.com/nutszebra/SENets/blob/master/accuracy.jpg" alt="total accuracy" title="total accuracy">


# References
Identity Mappings in Deep Residual Networks [[1]][Paper]

Squeeze-and-Excitation Networks [[2]][Paper2]

Improved Regularization of Convolutional Neural Networks with Cutout [[3]][Paper3]


[paper]: https://arxiv.org/abs/1603.05027 "Paper"
[paper2]: https://arxiv.org/abs/1709.01507 "Paper2"
[paper3]: https://arxiv.org/abs/1708.04552 "Paper3"
