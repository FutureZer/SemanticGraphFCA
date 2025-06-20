import os

from classification.main import all_raw_graphs_actions, make_gsofia_dir, create_gsofia_input
from read_dataset import load_graphs

if __name__ == "__main__":
    # Define the path to your dataset and the notation you want to load
    dataset_path = "data\\bbcsport"  # Replace with the actual path

    # Load the data
    train_ptg, test_ptg = load_graphs(dataset_path, "ptg", 0.12, 42)
    train_ucca, test_ucca = load_graphs(dataset_path, "ucca", 0.12, 42)

    train_processed_ptg = all_raw_graphs_actions(train_ptg)
    test_processed_ptg = all_raw_graphs_actions(test_ptg)

    all_preprocessed = {}
    for class_name in train_processed_ptg:
        all_preprocessed[class_name] = []
        all_preprocessed[class_name].extend(train_processed_ptg[class_name])
        all_preprocessed[class_name].extend(test_processed_ptg[class_name])

    # Here is example for PTG graphs
    gsofia_dir = make_gsofia_dir(dataset_path)
    labels_enc_dir = os.path.join(gsofia_dir, "ptg-train")
    labels_enc_dir = os.path.join(labels_enc_dir, "labels_enc.pkl")
    create_gsofia_input(train_processed_ptg, gsofia_dir, 'ptg-train')


