import os
import pickle
import argparse
import numpy as np


# feature processing routines, specific to each submission
import auxiliary_scripts.ngrams_to_feature_vector as feats
import auxiliary_scripts.trace_to_ngrams as ngrams

# this is the path that contains files specific to your submission
auxiliary_files_path = 'auxiliary_files'

# read the raw reports in standardize format and convert them to the feature vectors the model expects
def write_feature_vectors(dataset_path, features_filepath):

    # PCA learned on the training set for dimensionality reduction
    with open(os.path.join(auxiliary_files_path, 'pca.pkl'), 'rb') as fp:
        pca = pickle.load(fp)

    # the token dictionary that contains the top-10k most frequent tokens (to eliminate the rare tokens)
    with open(os.path.join(auxiliary_files_path, 'kept_tokens_dict.pkl'), 'rb') as fp:
        kept_tokens_dict = pickle.load(fp)


    filenames = os.listdir(dataset_path)

    reports_file_paths = [os.path.join(dataset_path, path) for path in filenames]

    # convert the reports to n-gram sequences
    hashed_ngrams = ngrams.extract_ngrams(reports_file_paths, kept_tokens_dict, n=2)

    # from ngram sequences to bag-of-ngrams (count each n-gram to create a feature vector)
    # counts will be turned to log scale for stability and better behaved features
    feat_counts = feats.ngram_sequence_to_feature_counts(hashed_ngrams, log_scaler=True)

    # use the pca to reduce the dimensionality of the n-grams feature vector 

    # this saves one feature vector corresponding to each report
    feature_vectors = pca.transform(feat_counts)

    # write the feature vectors to a file as an np array
    with open(features_filepath, 'wb') as fp:
        np.save(fp, feature_vectors)

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Malware Detection In the Wild Leaderboard (https://malwaredetectioninthewild.github.io) Example Submission for Feature Processing.')

    parser.add_argument('--dataset_path', type=str, help='Path to the folder input traces for inference are located', default='./dataset')

    parser.add_argument('--features_filepath', type=str, help='Path to write the processed features in a format that is expected by the model.', default='./features.pkl')

    args = parser.parse_args()

    print(args)

    write_feature_vectors(args.dataset_path, args.features_filepath)