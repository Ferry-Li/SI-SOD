import yaml
import argparse
from datasets import get_data, get_dataloader
from nets import get_model
from train import train as train_model
from test import test as test_model
from utils import Logger, set_seed
import os


class Runner:
    def __init__(self, config):
        self.config = config
        self.log = Logger(os.path.join(config["save_log"]["log_dir"], config["save_log"]["exp_name"]))
        self.log.info("=======================================================")
        self.log.info("         SI-SOD Runner Init!")
        self.log.info("=======================================================")
        self.log.info(f"exp: {config['save_log']['exp_name']}")


    def train(self):
        train_dataset, test_dataset = get_data(self.config['dataset'])
        train_dataloader = get_dataloader(train_dataset, self.config['dataset'])
        test_dataloader = get_dataloader(test_dataset, self.config['dataset'])
        dataloader = (train_dataloader, test_dataloader)
        self.log.info(f"load training dataset {self.config['dataset']}")
        model = get_model(config["model"], log=self.log)
        # self.log.info(f"model arch: {self.config['model']['arch']}")

        train_model(model, dataloader, self.config, self.log)

    def test(self):
        _, test_dataset = get_data(self.config["dataset"])
        test_dataloader = get_dataloader(test_dataset, self.config['dataset'])
        self.log.info(f"load test dataset {self.config['dataset']}")
        model = get_model(config["model"], is_train=False, log=self.log)
        # self.log.info(f"model arch: {self.config['model']['arch']}")
        # self.log.info(f"ckpt path: {self.config['model']['ckpt_path']}")

        test_model(model, test_dataloader, self.config, self.log)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start!")
    parser.add_argument('--config',      type=str,       default=None,       help="path to config file")
    parser.add_argument('--test',       action='store_true',       help="set when testing")


    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)


    set_seed(config["settings"]["seed"])
    runner = Runner(config)

    if not args.test:
        runner.train()
    else:
        runner.test()


