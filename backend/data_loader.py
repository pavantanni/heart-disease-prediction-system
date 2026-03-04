import pandas as pd

def load_dataset(path):
    """
    Load the heart disease dataset
    """
    try:
        data = pd.read_csv(path)
        print("Dataset loaded successfully")
        return data
    except Exception as e:
        print("Error loading dataset:", e)
        return None


if __name__ == "__main__":
    dataset = load_dataset("../datasets/heart.csv")
    if dataset is not None:
        print(dataset.head())