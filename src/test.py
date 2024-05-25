from datasets import get_data, get_dataloader
from nets import get_model
from criterions import get_evaluator
from utils import visualize
import torch
from tqdm import tqdm
import os

    


def test(model, dataloader, config, log):

    model.eval()

    metrics = config["metrics"]
    use_gpu = config["settings"]["use_gpu"]
    device = 'cuda' if use_gpu else 'cpu'
    save_dir = os.path.join(config["save_log"]["save_dir"], config["save_log"]["exp_name"])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = model.to(device)

    evaluator = get_evaluator()
 
    with torch.no_grad():
        for (i, item) in enumerate(tqdm(dataloader)):
            
            image, mask, weight, name = item
            image, weight = image.to(device), weight.to(device)

            output = model(image)
            img_out = output.detach().cpu()
            weight = weight.detach().cpu()
            evaluator.add_batch(img_out, mask, metrics, weight, True)
            if config["save_log"]["save_predict"]:
                visualize(img_out, name, save_dir)

        mae, si_mae, avg_auc, si_auc, mean_F, max_F, si_mean_F, si_max_F, Em = evaluator.get_result(metrics)

        # print(f"mae:{mae}, si_mae:{si_mae}, auc:{avg_auc}, si_auc:{si_auc}, mean_F:{mean_F}, si_mean_f:{si_mean_F}, max_F:{max_F}, si_max_f:{si_max_F}, Em:{Em}")
        log.info("=======================================================")
        log.info("         Evaluation Result")
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
        if config["save_log"]["save_predict"]:
            log.info("------------------------------------------------------")
            log.info(f"         predictions are saved at {save_dir}")
        log.info("=======================================================")