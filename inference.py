import os
import pickle
import argparse
import numpy as np

import torch


# model architectures, specific to each submission
import auxiliary_scripts.models as models
import auxiliary_scripts.backbones as backbones


# this is the path that contains files specific to your submission
auxiliary_files_path = 'auxiliary_files'


# read the model (customized depending on the submission)
def load_model(model_name, device):

    # configuration parameters of the model

    with open(os.path.join(auxiliary_files_path, f'{model_name}.cfg'), 'rb') as fp:
        model_cfg = pickle.load(fp)

    
    # load the model

    model = models.Classifier(backbones.get_resnet_backbone(model_cfg))
    model.load_state_dict(torch.load(os.path.join(auxiliary_files_path, f'{model_name}.dat')))

    # tranfer the model to the device
    model.eval().to(device)

    return model


def write_results(model, dataset_path, features_filepath, results_filepath):

    # load the feature vectors file written by the process_features script
    with open(features_filepath, 'rb') as fp:
        feature_vectors = np.load(fp)

    # the order of features in feature_vectors was based on the order of files in filenames
    filenames = os.listdir(dataset_path)

    # make inference to get prediction scores from the model, we care about the malwareness score 
    # ([:,0] gives the benignness score and [:,1 ] gives the malwareness score in our model implementation)
    probs = model.predict_proba(feature_vectors)[:,1]

    # create the results dict to map each filename to the corresponding probability
    results_dict = {fn:p for fn,p in zip(filenames, probs)}

    # write the results to the expected file
    with open(results_filepath, 'wb') as fp:
        pickle.dump(results_dict, fp)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Malware Detection In the Wild Leaderboard (https://malwaredetectioninthewild.github.io) Example Submission Format for Inference.')

    parser.add_argument('--dataset_path', type=str, help='Path to the folder input traces for inference are located', default='./dataset')

    parser.add_argument('--device', type=str, help='Device for performing model inference on feature vectors (GPU or CPU)', default='cuda')

    parser.add_argument('--features_filepath', type=str, help='Path to load the processed features in a format that is expected by the model (must be written by process_features.py script).', default='./features.pkl')


    parser.add_argument('--results_filepath', type=str, help='File path to the file where the predicted malware probabilities for each trace in the dataset will be written as a dictionary (filenames are the keys and the scores are the values)', default='./results.pkl')

    args = parser.parse_args()

    print(args)

    # load your model based on your submission files (customized according to the submission)
    model_name = 'invariance_model'

    model = load_model(model_name, args.device)


    write_results(model, args.dataset_path, args.features_filepath, args.results_filepath)