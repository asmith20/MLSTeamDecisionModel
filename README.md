# MLSTeamDecisionModel
GPC model code for team decisions (article link: www.americansocceranalysis.com/home/2018/8/10/why-is-atlantas-attack-so-dangerous-its-consistent)


To run the code (you will need opta data or something similar) and produce heatmaps

1. unpack the /bin and /gp directories, cd to /bin
2. run “python3 read_data.py input_data_file.csv output_data_file.csv”
3. run “python3 build_plot_models.py --data-location input_data_file.csv --output-location output_plot_directory”
