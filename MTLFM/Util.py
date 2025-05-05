import torch
import numpy as np
import random
import os
from model.FM import FM, MTLNFM, STLNFM, MTLNFM_2task
from model.LR import LR


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # for prohibiting hash randomization，and make the experiments reproducible
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_model(name, embedding_dim=30, device=''):
    """
    the embedding_dim is the maximal interger, which is 138 + 1,(0,139)
    """
    if name == 'lr':
        return LR(139, embedding_dim)
    elif name == 'fm':
        return FM(139, embed_dim=embedding_dim)
    elif name == 'stlnfm':
        return STLNFM(139, embed_dim=embedding_dim, hidden_layers=(64, ), dropout=(0.4, 0.4))
    elif name == 'mtlnfm':
        return MTLNFM(139, embed_dim=embedding_dim, hidden_layers=(64, ), dropout=(0.4, 0.4))
    elif name == 'mtlnfm_for2':
        return MTLNFM_2task(139, embed_dim=embedding_dim, hidden_layers=(64, ), dropout=(0.4, 0.4))
    else:
        raise ValueError('unknown model name: ' + name)


def prob_to_label(prob):
    result = []
    for i in prob:
        if i >= 0.5:
            result.append(1)
        else:
            result.append(0)
    return result


class EarlyStopper(object):

    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.best_logloss = 10
        self.save_path = save_path

    def is_continuable(self, model, accuracy, logloss):
        # if accuracy > self.best_accuracy and logloss < self.best_logloss:
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_logloss = logloss
            self.trial_counter = 0
            torch.save(model, self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False
