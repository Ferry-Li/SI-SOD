from generate_weight import generate_connection, generate_weight, generate_single_connection
import yaml
import os
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Start!")
    parser.add_argument('--config',      type=str,       default=None,       help="path to config file")
    parser.add_argument('--dataset',     type=str,       default=None,       help="choose among keys in configs/datasets.yaml")
    parser.add_argument('--visualize',   action="store_true",                help="visualize the generated weight")
    parser.add_argument('--size',       type=int,         default=384,       help="resize image size")
    parser.add_argument('--epsilon',       type=int,         default=25,       help="denoise parameter")


    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    dataset = args.dataset

    generate_connection(root_dir=config[dataset]["root_dir"], size=args.size, epsilon=args.epsilon)
    generate_weight(root_dir=config[dataset]["root_dir"], size=args.size, visualize=True)
    # generate_single_connection(root_dir=config[dataset]["root_dir"], size=args.size, epsilon=args.epsilon)
    