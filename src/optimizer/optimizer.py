import torch.optim.lr_scheduler
from torch.optim import SGD, Adam

def get_optimizer(model, config):
    lr, weight_decay = config["lr"], config["weight_decay"]
    if type(config['lr']) == str:
        lr = eval(config['lr'])
    if type(config['weight_decay']) == str:
        weight_decay = eval(config['weight_decay'])

    if config["optimizer_name"] == "Adam":
        optimizer = Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    elif config["optimizer_name"] == "SGD":
        optimizer = SGD(params=model.parameters(), lr=lr, weight_decay=weight_decay)

    return optimizer