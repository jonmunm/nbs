import math
import matplotlib.pyplot as plt

class CLR():
    def __init__(self, train_dataloader, base_lr=1e-5, max_lr=100):
        self.base_lr = base_lr # The lower boundary for learning rate (initial lr)
        self.max_lr = max_lr # The upper boundary for learning rate
        self.bn = len(train_dataloader) - 1 # Total number of iterations used for this test run (lr is calculated based on this)
        ratio = self.max_lr/self.base_lr # n
        self.mult = ratio ** (1/self.bn) # q = (max_lr/init_lr)^(1/n)
        self.best_loss = 1e9 # our assumed best loss
        self.iteration = 0 # current iteration, initialized to 0
        self.lrs = []
        self.losses = []
        
    def calc_lr(self, loss):
        self.iteration +=1
        if math.isnan(loss) or loss > 4 * self.best_loss: # stopping criteria (if current loss > 4*best loss) 
            return -1
        if loss < self.best_loss and self.iteration > 1: # if current_loss < best_loss, replace best_loss with current_loss
            self.best_loss = loss
        mult = self.mult ** self.iteration # q = q^i
        lr = self.base_lr * mult # lr_i = init_lr * q
        self.lrs.append(lr) # append the learing rate to lrs
        self.losses.append(loss) # append the loss to losses
        return lr
        
    def plot(self, start=10, end=-5): # plot lrs vs losses
        plt.xlabel("Learning Rate")
        plt.ylabel("Losses")
        plt.plot(self.lrs[start:end], self.losses[start:end])
        plt.xscale('log') # learning rates are in log scale