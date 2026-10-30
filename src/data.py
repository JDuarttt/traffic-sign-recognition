from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np

def build_index(train_dir):
    files = sorted(train_dir.rglob("*.png"))

    rows = []

    #Create a pandas DataFrame with the path, labels and the group of our images. The group would be for splitting the training set later
    for f in files:
        parts = f.stem.split('_')
        group = parts[0]+parts[1]
        fileDict = {"path": str(f), "label": int(f.parent.name), "group": group}
        rows.append(fileDict)

    df = pd.DataFrame(rows)

    #Check the DataFrame
    print(f"Number of paths: {len(df)}")
    print(f"Number of Tracks: {df["group"].nunique()}")
    print(f"Number of classes: {df["label"].nunique()}")
    return df
    
def make_split(df, n_splits, random_state):
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    _, val_idx = next(skf.split(df, df["label"], groups=df["group"]))

    col = np.full(len(df), "train")
    col[val_idx] = "val"
    df["split"] = col
    return df


def check_split(df):
    N_CLASSES = len(df['label'].unique())

    #Which classes are present in each split
    for name in ["train", "val"]:
        present = set(df.loc[df["split"] == name, "label"])
        missing = set(range(N_CLASSES)) - present
        if(missing):
            print(f"{name}: {len(present)}/{N_CLASSES} classes, missing: {missing}")
        else:
            print(f"{name}: {len(present)}/{N_CLASSES} classes")

    #Tracks per class and split
    print("")
    tracks = df.groupby(["label", "split"])["group"].nunique().unstack()
    print("Leak check:")
    if(df.groupby('group')['split'].nunique().max() > 1):
        print("ERROR: SOME TRACKS IN BOTH TRAIN AND VALIDATION SET")
    else:
        print("No leaked tracks.")
    print("")
    print("5 least common classes in Validation set:")
    print(tracks.sort_values("val").head(5))