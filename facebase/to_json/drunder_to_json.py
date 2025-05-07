import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

# Define the root path to the dataset and the output file path
root = '/media/vision/FastStorage-1/alireza-sm/datasets/diabetic-retinopathy-detection'
output_file = 'facebase/benchmarks/databases/DRUnder.json'

def convert_csv_to_json(root, output_file, n_folds=5, random_state=123):
    """
    Convert Adience dataset's cleaned fold txt files into a single JSON file.

    Parameters
    ----------
    root : str
        Root path of the Adience dataset.
    fold_folder : str
        Path to the folder containing cleaned fold txt files.
    output_file : str
        Path to save the output JSON file.
    """
    # Initialize a list to hold all JSON objects
    all_data = []

    labels_trn = pd.read_csv(os.path.join(root, "trainLabels.csv"))
    labels_tst = pd.read_csv(os.path.join(root, "retinopathy_solution.csv"))

    level_0_sample = labels_trn[labels_trn['level'] == 0].sample(n=5810, random_state=42)

    # Keep all rows where level != 0
    non_level_0 = labels_trn[labels_trn['level'] != 0]

    # Combine the sampled level 0 rows with the rest
    labels_trn = pd.concat([level_0_sample, non_level_0], ignore_index=True)
    
    labels_trn['fold'] = -1
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(labels_trn, labels_trn['level'])):
        labels_trn.loc[val_idx, 'folder'] = fold
    
    id_num = 0
    for _, row in tqdm(labels_trn.iterrows()):
        # Format the JSON object
        data_instance = {
            "img_path": os.path.join('diabetic-retinopathy-detection', 'train', row['image']+'.jpeg'),
            "id_num": id_num,
            "level": row['level'],
            "database": "DRUnder",
            "folder": int(row['folder'])
        }

        # Append the instance to the list
        all_data.append(data_instance)
        id_num += 1

    for _, row in tqdm(labels_tst.iterrows()):
        # Format the JSON object
        data_instance = {
            "img_path": os.path.join('diabetic-retinopathy-detection', 'test', row['image']+'.jpeg'),
            "id_num": id_num,
            "level": row['level'],
            "database": "DRUnder",
            "folder": 5
        }

        # Append the instance to the list
        all_data.append(data_instance)
        id_num += 1


    # Write the list of JSON objects to a file
    with open(output_file, 'w') as f:
        json.dump(all_data, f, indent=4)

    print(f"DRUnder data saved to {output_file}")

# Example usage
if __name__ == "__main__":
    convert_csv_to_json(root, output_file)
