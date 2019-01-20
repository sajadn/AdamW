from keras import backend as K
from keras.legacy import interfaces
from keras.optimizers import Adam
from keras.callbacks import Callback


class DecoupleWeightDecay:
    """Adam optimizer.
    Default parameters follow those provided in the original paper.
    # Arguments
        weight_decay: weight decay value that will be mutltiplied to the parameter
    # References
        - [AdamW - DECOUPLED WEIGHT DECAY REGULARIZATION](
           https://arxiv.org/pdf/1711.05101.pdf)

    """

    def __init__(self, weight_decay, **kwargs):
        with K.name_scope(self.__class__.__name__):
            self.weight_decay = K.variable(weight_decay, name='weight_decay')
        super(DecoupleWeightDecay, self).__init__(**kwargs)

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        updates = super(DecoupleWeightDecay, self).get_updates(loss, params)
        #TODO change loop to vectorized
        for p in params:
            updates.append(K.update_sub(p, self.weight_decay*p))
        return updates


def create_decouple_optimizer(optimizer):
    class OptimizerW(DecoupleWeightDecay, optimizer):
        def __init__(self, weight_decay, **kwargs):
            super(OptimizerW, self).__init__(weight_decay, **kwargs)


class WeightDecayScheduler(Callback):
    def __init__(self, init_lr):
        super(WeightDecayScheduler, self).__init__()
        self.previous_lr = init_lr


    def on_epoch_begin(self, epoch, logs=None):
        current_lr = float(K.get_value(self.model.optimizer.lr))
        coeff = current_lr / self.previous_lr
        new_weight_decay = float(K.get_value(self.model.optimizer.weight_decay)) * coeff
        K.set_value(self.model.optimizer.weight_decay, new_weight_decay)
        self.previous_lr = current_lr
        if coeff!=1:
            print(epoch, coeff)

    def on_epoch_end(self, epoch, logs=None):
        return


class AdamW(DecoupleWeightDecay, Adam):
    def __init__(self, weight_decay, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., amsgrad=False, **kwargs):
        super(AdamW, self).__init__(weight_decay=weight_decay, lr=lr, beta_1=beta_1, beta_2=beta_2,
                 epsilon=epsilon, decay=decay, amsgrad=amsgrad, **kwargs)