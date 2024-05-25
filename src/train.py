from datasets import get_data, get_dataloader
from nets import get_model
from loss import get_loss
from optimizer import get_optimizer
from criterions import get_evaluator
import torch
import os
from tqdm import tqdm
from utils import Logger


def save_model(model, epoch, config, log):
    save_dir = os.path.join(config["save_dir"], config["exp_name"])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_name = f"{config['exp_name']}-{epoch}.pth"
    torch.save(model.state_dict(), os.path.join(save_dir, save_name))
    log.info(f"save model at {os.path.join(save_dir, save_name)} at the end of epoch {epoch + 1}")

def test_in_train(model, dataloader, config, epoch, log):
    
    evaluator = get_evaluator()
    metrics = config["metrics"] 
    use_gpu = config["settings"]["use_gpu"]
    device = 'cuda' if use_gpu else 'cpu'

    test_model = get_model(config["model"], is_train=False, log=log)
    test_model.load_state_dict(model.state_dict())
    test_model.to(device)
    test_model.eval()


    with torch.no_grad():
        for (i, item) in enumerate(tqdm(dataloader)):
            image, mask, weight, name = item
            image, weight = image.to(device), weight.to(device)

            output = test_model(image)
            img_out = output.detach().cpu()
            weight = weight.detach().cpu()
            evaluator.add_batch(img_out, mask, metrics, weight, True)

        mae, si_mae, avg_auc, si_auc, mean_F, max_F, si_mean_F, si_max_F, Em = evaluator.get_result(metrics)

        # print(f"mae:{mae}, si_mae:{si_mae}, auc:{avg_auc}, si_auc:{si_auc}, mean_F:{mean_F}, si_mean_f:{si_mean_F}, max_F:{max_F}, si_max_f:{si_max_F}, Em:{Em}")
        log.info("=======================================================")
        log.info(f"         Evaluation Result at epoch: {epoch}")
        log.info("=======================================================")
        log.info(f"        mae:{mae:.4f}")
        log.info(f"        si-mae:{si_mae:.4f}")
        log.info(f"        auc:{avg_auc:.4f}")
        log.info(f"        si-auc:{si_auc:.4f}")
        log.info(f"        mean_F:{mean_F:.4f}")
        log.info(f"        si-mean_F:{si_mean_F:.4f}")
        log.info(f"        max_F:{max_F:.4f}")
        log.info(f"        si-max_F:{si_max_F:.4f}")
        log.info(f"        Em:{Em:.4f}")
        log.info("=======================================================")


def train(model, dataloader, config, log):

    model.train()
    use_gpu = config["settings"]["use_gpu"]
    device = 'cuda' if use_gpu else 'cpu'

    model = model.to(device)

    loss_config = config["loss"]
    loss_function = get_loss(loss_config)
    log.info(f"loss adopted: {loss_config['losses']}")
    log.info(f"SI: {loss_config['SI']}")
    log.info(f"warm up: {loss_config['warm_up']}")

    optimizer_config = config["optimizer"]
    optimizer = get_optimizer(model, optimizer_config)
    log.info(f"optimizer adopted: {optimizer_config['optimizer_name']}")
    log.info(f"Learning rate: {optimizer_config['lr']}")

    training_epoch = config["settings"]["epoch"]
    iter_size = config["settings"]["iter_size"]
    save_epoch = config["settings"]["save_epoch"]
    eval_epoch = config["settings"]["eval_epoch"]
    total_iter = 0
    train_dataloader, test_dataloader = dataloader
    for epoch in range(training_epoch):

        # possible warm_up losses
        loss_config = config["loss"]
        loss_function = get_loss(loss_config, epoch)

        print(f"********** EPOCH: {epoch + 1} **********")
        bar = tqdm(train_dataloader)
        for (i, item) in enumerate(bar):
            image, mask, weight, name = item
            image, mask = image.to(device), mask.to(device)

            output = model(image)
            loss = loss_function(output, mask, weight)
            bar.set_description(f"epoch: {epoch + 1}, loss: {loss:.5f}")
            loss.backward()
            total_iter += 1
            if total_iter % iter_size == 0:
                optimizer.step()

            optimizer.zero_grad()

        if (epoch + 1) % save_epoch == 0:
            save_model(model, epoch + 1, config["save_log"], log)

        if eval_epoch > 0 and (epoch + 1) % eval_epoch == 0:
            test_in_train(model, test_dataloader, config, epoch + 1, log)


