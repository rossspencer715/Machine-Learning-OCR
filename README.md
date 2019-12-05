# project01-naive-baes

## Running the Code

Use command `python test.py <input_data_path> <output_data_path>` to get an output numpy file at the location specified
 by the output data path.
 
For the class-wide challenge, run the command `python test.py <input_data_path> <output_data_path> --hard` and the 
output will be the labels file for all 9 classes of letters.

### Flags in the files (only applies to training)
If you would like to train a neural net using this repo, go into the script called `pytorch_training.py`. This file 
has methods to train both neural nets and a small script at the bottom that will run the methods and save the models.

In the final `if` statement, change the zare_data boolean. If `True`, then the model will be trained using the 
TrainImages.npy file. If it is `False`, then the model will be trained using class's data.

In that same `if` statement, change the `AB` boolean to `True` to train only three classes 'A', 'B', and None. If AB is 
`False`, the model will train all nine handwritten classes.

These same flags need to be changed at the top of the file called `net_classes.py` to the correct values.

### Other files

knn_implementation* : (multiple files) Show how we implemented our KNN classifier that was not ultimately successful.

compare_models.py : usage - `python compare_models.py <data_file_path> <label_file_path> <prefix_to_output_data_files>` will load up all of the trained models in this repo and compare
them against the data set of your choosing

features.py : Contains the methods used for feature extraction in the KNN method.

net_classes.py : Contains the class definitions for the neural nets and various evaluation/data loading methods

preprocess.py : contains the methods used to preprocess any data that is run through the neural net

./data/plot_data.npy : Plots the metrics created by training the models

./models/ : folder containing trained models

./data/ : folder containing training metrics

./plots/ : folder containing output plots