import argparse
import numpy as np
import pandas as pd
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from gp.model_tuning import split_data, build_models, evaluate_error, validate_model


parser = argparse.ArgumentParser()
parser.add_argument("--data-location", type=str, required=True, help='location of the data file')


def main():
    args = parser.parse_args()
    data = pd.read_csv(args.data_location)
    train, test, validate = split_data(data)
    kernel = ConstantKernel() + Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1)
    naive_probs = {"Pass": (train.event_type == 'Pass').mean(),
                   "Shot": (train.event_type == 'Shot').mean(),
                   "Take on": (train.event_type == "Take on").mean()}
    models = build_models(train, kernel)
    print validate_model(models, validate, naive_probs)


if __name__ == "__main__":
    main()
