import chainer
from chainer import optimizers
import nutszebra_basic_print


class Optimizer(object):

    def __init__(self, model=None):
        self.model = model
        self.optimizer = None

    def __call__(self, i):
        pass

    def update(self):
        self.optimizer.update()


class OptimizerResnet(Optimizer):

    def __init__(self, model=None, schedule=(100, 150), lr=0.1, momentum=0.9, weight_decay=1.0e-4, warm_up_lr=0.01):
        super(OptimizerResnet, self).__init__(model)
        optimizer = optimizers.MomentumSGD(warm_up_lr, momentum)
        weight_decay = chainer.optimizer.WeightDecay(weight_decay)
        optimizer.setup(self.model)
        optimizer.add_hook(weight_decay)
        self.optimizer = optimizer
        self.schedule = schedule
        self.lr = lr
        self.warmup_lr = warm_up_lr
        self.momentum = momentum
        self.weight_decay = weight_decay

    def __call__(self, i):
        if i == 1:
            lr = self.lr
            print('finishded warming up')
            print('lr is changed: {} -> {}'.format(self.optimizer.lr, lr))
            self.optimizer.lr = lr
        if i in self.schedule:
            lr = self.optimizer.lr / 10
            print('lr is changed: {} -> {}'.format(self.optimizer.lr, lr))
            self.optimizer.lr = lr
