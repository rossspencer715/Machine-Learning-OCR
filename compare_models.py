import torch
import numpy as np
from net_classes import *
from sklearn.metrics import precision_score, accuracy_score
import sys
import os

def make_labels(indata_fp, outdata_fp):
    # load the models
    model0 = torch.load('./models/CNN_AB_zare_classifier_30_epochs.pt')
    model1 = torch.load('./models/NN_AB_zare_classifier_30_epochs.pt')
    model2 = torch.load('./models/CNN_class_all_letters_classifier_30_epochs.pt')
    model3 = torch.load('./models/NN_class_all_letters_classifier_30_epochs.pt')

    # save into dictionary
    models = {
        'CNN_Zare': model0,
        'NN_Zare': model1,
        'CNN_Class': model2,
        'NN_Class': model3
    }
    print("Models loaded into list")

    # evaluate each model
    for (k, v) in sorted(models.items()):
        print(f'Testing model {k}...')
        # load the data and turn into dataloader
        full_dataloader = load_data_from_file(indata_fp)
        # run the data loader through the model
        if isinstance(v, SimpleCNN):
            pred_vals = eval_model_no_labels(v, full_dataloader, CNN=True)
        else:
            pred_vals = eval_model_no_labels(v, full_dataloader, CNN=False)
        # create a labels array in the form that is demanded
        pred_vals[pred_vals == 0] = -1
        # save the labels array in the path declared initially
        filepath = f'./data/final_model_output/{outdata_fp}_{k}.npy'
        np.save(filepath, pred_vals)
        print(f'saved to file name {filepath}')


if __name__ == '__main__':

    if not os.path.exists('./data/final_model_output'):
        os.mkdir('./data/final_model_output')

    # To not call from command line, comment the following code block and use example below
    if len(sys.argv) != 4:
        print(f'usage: {sys.argv[0]} <data_file_path> <label_file_path> <prefix_to_output_data_files>')
        sys.exit(0)
    in_fname = sys.argv[1]
    label_file_path = sys.argv[2]
    out_fname = sys.argv[3]

    # create the AB labels
    true_labels = np.load(label_file_path).astype(int)
    true_labels[true_labels > 2] = -1

    # create the full class labels
    true_labels_class = np.load(label_file_path).astype(int)
    true_labels_class[true_labels_class == 0] = -1

    make_labels(in_fname, out_fname)

    for filename in os.listdir(f'./data/final_model_output/'):
        if filename.__contains__('Class'):
            predicted = np.load(f'./data/final_model_output/{filename}')
            print(f'{filename} evaluated at accuracy {accuracy_score(true_labels_class, predicted)*100}')
            print(f'{filename} evaluated at macro-precision '
                  f'{precision_score(true_labels_class, predicted,average="macro")*100}')
            print(f'{filename} evaluated at weighted-precision '
                  f'{precision_score(true_labels_class, predicted,average="weighted")*100}')
        else:
            predicted = np.load(f'./data/final_model_output/{filename}')
            print(f'{filename} evaluated at accuracy {accuracy_score(true_labels, predicted)*100}')
            print(f'{filename} evaluated at macro-precision '
                  f'{precision_score(true_labels, predicted,average="macro")*100}')
            print(f'{filename} evaluated at weighted-precision '
                  f'{precision_score(true_labels, predicted,average="weighted")*100}')
