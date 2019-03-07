import argparse
import datetime
import json
import os
import shutil

from evaluate import evaluate
from train import train
from visualize import visualize

parser = argparse.ArgumentParser(description='Train a RL agent.')
parser.add_argument('config', type=str, help="Path to a configuration file")

if __name__ == "__main__":
    args = parser.parse_args()

    # Parses the config file
    with open(args.config) as f:
        config = json.load(f)

    # Creates the result directory
    path = "experiments/" + datetime.datetime.today().strftime("%Y_%m_%d_%H_%M_%S") + "/"
    os.makedirs(path)

    # Moves the config file in the result directory
    shutil.copyfile(args.config, path + "config.json")

    train(config, path)
    evaluate(config, path)
    visualize(config, path)
