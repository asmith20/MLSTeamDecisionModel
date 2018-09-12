import argparse
import numpy as np
import pandas as pd
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel, WhiteKernel,RationalQuadratic
from gp.model_tuning import split_data, build_models, evaluate_error

parser = argparse.ArgumentParser()
parser.add_argument("--data-location", type=str, required=True, help='location of the data file')


def main():
    args = parser.parse_args()
    data = pd.read_csv(args.data_location)
    train, test, validate = split_data(data)
    # kernels = [1.0 * RBF(length_scale=1.0),
    #            Matern(length_scale=1.0, nu=1.5),
    #            ConstantKernel() + Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1),
    #            ConstantKernel() + Matern(length_scale=2.0, nu=1.5) + WhiteKernel(noise_level=1)]
    kernels = [ConstantKernel() + RationalQuadratic(length_scale=1,alpha=1.5),
               ConstantKernel() + Matern(length_scale=1,nu=1.5)]

    error = []
    for k in kernels:
        print(k)
        models = build_models(train, k)
        error.append(evaluate_error(models, test))

    print(error)
    # then need to tune the model and pick hyperparams
    # it takes so long to run we probably just want to evaluate a handful of kernels
    # and get a baseline calc somewwhere against the holdout
    # then we need somewhere to output the models that are built or score them and output those results maybe


if __name__ == "__main__":
        main()
