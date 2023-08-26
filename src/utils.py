import pickle
import os

def save_file(output_path,data):
    """
    saving a pickle file to local disk

    args:
        output_path: (str) path_to_save_the_file/file.pkl
    returns: nothing
    """
    print(f"saving to {output_path}\n")
    with open(output_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_data_from_file(file_path):
    """
    reads pickle file
    
    args:
        file_path: (str) 'path_to_file/file.pkl'
    
    returns:
        data: (any format) data in the pickle file
    """
    print(f'\nreading data from {file_path}\n')

    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    return data

def make_dir(path):
    """
    make directory if directory not exists

    """
    if not os.path.exists(path):
        os.mkdir(path)