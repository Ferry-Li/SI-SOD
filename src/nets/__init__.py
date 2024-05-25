import torch
import yaml
from .EDN import EDN
from .ICON import ICON
from .GateNet import GateNet
from .PoolNet import build_PoolNet as PoolNet


def get_model(config, is_train=True, log=None):
    model_dict = ["EDN", "ICON", "GateNet", "ICON"]
    model_name = config["model_name"]
    if model_name not in model_dict:
        assert "Model not implemented!"

    with open(config["model_config"], 'r') as f:
        model_config = yaml.load(f, Loader=yaml.Loader)    
    if model_name == "EDN":
        EDN_config = model_config["EDN"]
        model = EDN(EDN_config, is_train=is_train, log=log)
        log.info("load model EDN")
        if EDN_config["pretrained"]:
            state_dict = torch.load(EDN_config["ckpt_path"])
            model.load_state_dict(state_dict)
            log.info(f"load ckpt from {EDN_config['ckpt_path']}")

    elif model_name == "ICON":
        ICON_config = model_config["ICON"]
        model  = ICON(ICON_config, is_train=is_train, log=log)
        log.info("load model ICON")
        if ICON_config["pretrained"]:
            state_dict = torch.load(ICON_config["ckpt_path"])
            model.load_state_dict(state_dict)
            log.info(f"load ckpt from {ICON_config['ckpt_path']}")

    elif model_name == "GateNet":
        GateNet_config = model_config["GateNet"]
        model = GateNet(GateNet_config, is_train=is_train, log=log)
        log.info("load model GateNet")
        if GateNet_config["pretrained"]:
            state_dict = torch.load(GateNet_config["ckpt_path"])
            model.load_state_dict(state_dict)
            log.info(f"load ckpt from {GateNet_config['ckpt_path']}")

    elif model_name == "PoolNet":
        PoolNet_config = model_config["PoolNet"]
        model = PoolNet(PoolNet_config, is_train=is_train, log=log)
        log.info("load model PoolNet")

        if PoolNet_config["backbone"] == 'resnet':
            model.base.load_pretrained_model(torch.load(PoolNet_config["resnet_path"]))
            log.info(f"load pretrained backbone from {PoolNet_config['resnet_path']}")
        else:
            model.base.load_pretrained_model(torch.load(PoolNet_config["vgg_path"]))
            log.info(f"load pretrained backbone from {PoolNet_config['vgg_path']}")

        if PoolNet_config["pretrained"]:
            state_dict = torch.load(PoolNet_config["ckpt_path"])
            model.load_state_dict(state_dict)
            log.info(f"load ckpt from {PoolNet_config['ckpt_path']}")




        
    # model = model.to()
    return model