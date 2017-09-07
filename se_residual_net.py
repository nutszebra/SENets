import six
import chainer
import numpy as np
import chainer.links as L
import chainer.functions as F
import nutszebra_chainer
import functools
from collections import defaultdict


class BN_ReLU_Conv(nutszebra_chainer.Model):

    def __init__(self, in_channel, out_channel, filter_size=(3, 3), stride=(1, 1), pad=(1, 1)):
        super(BN_ReLU_Conv, self).__init__(
            conv=L.Convolution2D(in_channel, out_channel, filter_size, stride, pad),
            bn=L.BatchNormalization(in_channel),
        )

    def weight_initialization(self):
        self.conv.W.data = self.weight_relu_initialization(self.conv)
        self.conv.b.data = self.bias_initialization(self.conv, constant=0)

    def count_parameters(self):
        return functools.reduce(lambda a, b: a * b, self.conv.W.data.shape)

    def __call__(self, x, train=False):
        return self.conv(F.relu(self.bn(x, test=not train)))


class SEBlock(nutszebra_chainer.Model):

    def __init__(self, in_channel, r=4, non_linear1=F.relu, non_linear2=F.sigmoid):
        super(SEBlock, self).__init__(
            f1=L.Linear(in_channel, int(in_channel / r)),
            f2=L.Linear(int(in_channel / r), in_channel),
        )
        self.non_linear1 = non_linear1
        self.non_linear2 = non_linear2

    def weight_initialization(self):
        self.f1.W.data = self.weight_relu_initialization(self.f1)
        self.f1.b.data = self.bias_initialization(self.f1, constant=0)
        self.f2.W.data = self.weight_relu_initialization(self.f2)
        self.f2.b.data = self.bias_initialization(self.f2, constant=0)

    def count_parameters(self):
        return functools.reduce(lambda a, b: a * b, self.f1.W.data.shape) + functools.reduce(lambda a, b: a * b, self.f1.W.data.shape)

    def __call__(self, x, train=False):
        # global average pooling
        num, categories, height, width = x.data.shape
        h = F.reshape(F.average_pooling_2d(x, (height, width)), (num, categories))
        h = self.non_linear1(self.f1(h))
        h = self.non_linear2(self.f2(h))
        return x * F.transpose(F.broadcast_to(h, (height, width, num, categories)), (2, 3, 0, 1))


class SEResBlock(nutszebra_chainer.Model):

    def __init__(self, in_channel, out_channel, n=18, stride_at_first_layer=2, multiplier=4, r=4):
        super(SEResBlock, self).__init__()
        modules = []
        modules.append(('skip_bn', L.BatchNormalization(in_channel)))
        modules.append(('bn_relu_conv1_1', BN_ReLU_Conv(in_channel, out_channel, 1, stride_at_first_layer, 0)))
        modules.append(('bn_relu_conv2_1', BN_ReLU_Conv(out_channel, out_channel)))
        modules.append(('bn_relu_conv3_1', BN_ReLU_Conv(out_channel, int(multiplier * out_channel), 1, 1, 0)))
        modules.append(('se_block_1', SEBlock(int(multiplier * out_channel), r=r)))
        for i in six.moves.range(2, n + 1):
            modules.append(('bn_relu_conv1_{}'.format(i), BN_ReLU_Conv(int(multiplier * out_channel), out_channel, 1, 1, 0)))
            modules.append(('bn_relu_conv2_{}'.format(i), BN_ReLU_Conv(out_channel, out_channel)))
            modules.append(('bn_relu_conv3_{}'.format(i), BN_ReLU_Conv(out_channel, int(multiplier * out_channel), 1, 1, 0)))
            modules.append(('se_block_{}'.format(i), SEBlock(int(multiplier * out_channel), r=r)))
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.n = n
        self.stride_at_first_layer = stride_at_first_layer
        self.multiplier = multiplier
        self.r = r

    def weight_initialization(self):
        for i in six.moves.range(1, self.n + 1):
            self['bn_relu_conv1_{}'.format(i)].weight_initialization()
            self['bn_relu_conv2_{}'.format(i)].weight_initialization()
            self['bn_relu_conv3_{}'.format(i)].weight_initialization()
            self['se_block_{}'.format(i)].weight_initialization()

    def count_parameters(self):
        count = 0
        for i in six.moves.range(1, self.n + 1):
            count += self['bn_relu_conv1_{}'.format(i)].count_parameters()
            count += self['bn_relu_conv2_{}'.format(i)].count_parameters()
            count += self['bn_relu_conv3_{}'.format(i)].count_parameters()
            count += self['se_block_{}'.format(i)].count_parameters()
        return count

    @staticmethod
    def concatenate_zero_pad(x, h_shape, volatile, h_type):
        _, x_channel, _, _ = x.data.shape
        batch, h_channel, h_y, h_x = h_shape
        if x_channel == h_channel:
            return x
        pad = chainer.Variable(np.zeros((batch, h_channel - x_channel, h_y, h_x), dtype=np.float32), volatile=volatile)
        if h_type is not np.ndarray:
            pad.to_gpu()
        return F.concat((x, pad))

    def maybe_pooling(self, x):
        if self.stride_at_first_layer == 2:
            return F.average_pooling_2d(x, 1, 2, 0)
        return x

    def __call__(self, x, train=False):
        h = self['bn_relu_conv1_1'](x, train=train)
        h = self['bn_relu_conv2_1'](h, train=train)
        h = self['bn_relu_conv3_1'](h, train=train)
        h = self['se_block_1'](h, train=train)
        x = h + SEResBlock.concatenate_zero_pad(self.maybe_pooling(F.relu(self.skip_bn(x, test=not train))), h.data.shape, h.volatile, type(h.data))
        for i in six.moves.range(2, self.n + 1):
            h = self['bn_relu_conv1_{}'.format(i)](x, train=train)
            h = self['bn_relu_conv2_{}'.format(i)](h, train=train)
            h = self['bn_relu_conv3_{}'.format(i)](h, train=train)
            x = self['se_block_{}'.format(i)](h, train=train) + x
        return x


class SEResidualNetwork(nutszebra_chainer.Model):

    def __init__(self, category_num, block_num=3, out_channels=(16, 32, 64), N=(18, 18, 18), multiplier=4, r=4):
        super(SEResidualNetwork, self).__init__()
        # conv
        modules = [('conv1', L.Convolution2D(3, out_channels[0], 3, 1, 1))]
        in_channel = out_channels[0]
        strides = [1] + [2] * (block_num - 1)
        for i, out_channel, n, stride in six.moves.zip(six.moves.range(1, block_num + 1), out_channels, N, strides):
            modules.append(('res_block{}'.format(i), SEResBlock(in_channel, out_channel, n=n, stride_at_first_layer=stride, multiplier=multiplier, r=r)))
            in_channel = int(out_channel * multiplier)
        modules.append(('bn_relu_conv', BN_ReLU_Conv(in_channel, category_num, filter_size=(1, 1), stride=(1, 1), pad=(0, 0))))
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules
        self.category_num = category_num
        self.block_num = block_num
        self.out_channels = out_channels
        self.N = N
        self.multiplier = multiplier
        self.r = r
        self.name = 'se_residual_network_{}_{}_{}_{}_{}_{}'.format(category_num, block_num, out_channels, N, multiplier, r)

    def weight_initialization(self):
        self.conv1.W.data = self.weight_relu_initialization(self.conv1)
        self.conv1.b.data = self.bias_initialization(self.conv1, constant=0)
        for i in six.moves.range(1, self.block_num + 1):
            self['res_block{}'.format(i)].weight_initialization()
        self.bn_relu_conv.weight_initialization()

    def __call__(self, x, train=False):
        h = F.relu(self.conv1(x))
        for i in six.moves.range(1, self.block_num + 1):
            h = self['res_block{}'.format(i)](h, train=train)
        h = self.bn_relu_conv(F.relu(h), train=train)
        num, categories, height, width = h.data.shape
        h = F.reshape(F.average_pooling_2d(h, (height, width)), (num, categories))
        return h

    def count_parameters(self):
        count = 0
        count += functools.reduce(lambda a, b: a * b, self.conv1.W.data.shape)
        for i in six.moves.range(1, self.block_num + 1):
            count += self['res_block{}'.format(i)].count_parameters()
        count += self.bn_relu_conv.count_parameters()
        return count

    def calc_loss(self, y, t):
        loss = F.softmax_cross_entropy(y, t)
        return loss

    def accuracy(self, y, t, xp=np):
        y.to_cpu()
        t.to_cpu()
        indices = np.where((t.data == np.argmax(y.data, axis=1)) == True)[0]
        accuracy = defaultdict(int)
        for i in indices:
            accuracy[t.data[i]] += 1
        indices = np.where((t.data == np.argmax(y.data, axis=1)) == False)[0]
        false_accuracy = defaultdict(int)
        false_y = np.argmax(y.data, axis=1)
        for i in indices:
            false_accuracy[(t.data[i], false_y[i])] += 1
        return accuracy, false_accuracy
