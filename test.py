import torch
import numpy as np
from net_classes import *
import sys

def make_labels(indata_fp, outdata_fp, hard=False):
    # load the model
    if not hard:
        model = torch.load('./models/CNN_AB_zare_classifier_30_epochs.pt')
    else:
        model = torch.load('./models/CNN_class_all_letters_classifier_30_epochs.pt')

    # load the data and turn into dataloader
    full_dataloader = load_data_from_file(indata_fp)

    # run the data loader through the model - this will be a CNN in both cases
    pred_vals = eval_model_no_labels(model, full_dataloader, CNN=True)

    # create a labels array in the form that is demanded
    pred_vals[pred_vals == 0] = -1

    # save the labels array in the path declared initially
    filepath = f'{outdata_fp}.npy'
    np.save(filepath, pred_vals)

    print(f'saved to file name: {filepath}')


if __name__ == '__main__':

    # To not call from command line, comment the following code block and use example below
    if len(sys.argv) != 3 and len(sys.argv) != 4:
        print(f'usage: {sys.argv[0]} <data_file_path> <label_file_path> --hard (last argument optional)')
        sys.exit(0)
    in_fname = sys.argv[1]
    out_fname = sys.argv[2]

    # determine whether or not to classify the hard challenge
    hard = False
    if len(sys.argv) > 3:
        if sys.argv[3] == '--hard':
            hard = True

    make_labels(in_fname, out_fname, hard)
