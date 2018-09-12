import argparse
import numpy as np
import pandas as pd
from gp.output_models import build_model, plot_model


parser = argparse.ArgumentParser()
parser.add_argument("--data-location", type=str, required=True, help="location of the data file")
parser.add_argument("--output-location", type=str, required=True, help="location to write the plots to")


def main():
    args = parser.parse_args()
    data = pd.read_csv(args.data_location)
    for team_id in data.team_id.unique():
        team_data = data[data.team_id == team_id]
        model1 = build_model(team_data[team_data.score == 1])
        model2 = build_model(team_data[team_data.score == 0])
        model3 = build_model(team_data[team_data.score == -1])
        team = str(team_data.team.iloc[0])
        plot_model(model1,model2,model3, team_id, team, args.output_location)


if __name__ == "__main__":
    main()
