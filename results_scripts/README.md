# How to retrieve the data

To retrieve the logs, you need to download the data from [here](TBD).
The experiments are in the three zip files.

Unzip them, typically in `results_scripts/logs`.

Then, you can run the notebooks in the following order:

You will be able to create all figures with the different notebook.
Beware that the first cell may be updated to find the logs if you chose something else than `results_scripts/logs`.

The name of the notebooks are self-explanatory:

- [boxplots](boxplots.ipynb): create the firgure 2 of the paper with the relative boxplots.
- [find_best_network](find_best_network.ipynb): find the best network for each run.
- [generate_table](generate_table.ipynb): generate the tables of best results the paper.
- [plot_variance](plot_variance.ipynb): create the figure X of the paper with the variance of the results.
- [plot_testing_time](plot_testing_time.ipynb): create the figures X and Y of the paper with the testing time.
- [plot_training_time](plot_training_time.ipynb): create the figures X and Y of the paper with the training time.
- [plot_learning_curves](plot_learning_curves.ipynb): create the figure X and Y of the paper with the learning curves.