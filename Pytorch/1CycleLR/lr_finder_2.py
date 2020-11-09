import torch
from torch import nn

loss_func = nn.CrossEntropyLoss()  # loss function
opt = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.95) # optimizer
clr = CLR(train_dataloader) # CLR instance

def find_lr(clr):
    running_loss = 0. 
    avg_beta = 0.98 # useful in calculating smoothed loss
    model.train() # set the model in training mode
    for i, (input, target) in enumerate(train_dataloader):
        input, target = input.to(device), target.to(device) # move the inputs and labels to gpu if available
        output = model(var_ip) # predict output
        loss = loss_func(output, var_tg) # calculate loss 
        
        # calculate the smoothed loss 
        running_loss = avg_beta * running_loss + (1-avg_beta) *loss # the running loss
        smoothed_loss = running_loss / (1 - avg_beta**(i+1)) # smoothening effect of the loss 
        
        lr = clr.calc_lr(smoothed_loss) # calculate learning rate using CLR
        if lr == -1 : # the stopping criteria
            break
        for pg in opt.param_groups: # update learning rate
            pg['lr'] = lr   

        # compute gradient and do parameter updates
        opt.zero_grad()
        loss.backward()
        opt.step()