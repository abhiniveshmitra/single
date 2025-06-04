import os
import shutil
import kagglehub

def download_dataset(dataset_id="denniswang07/datasets-for-rag"):
    """
    Downloads the KaggleHub dataset and moves it to the current working directory.
    """
    print(f"Downloading dataset: {dataset_id}")
    dataset_path = kagglehub.dataset_download(dataset_id)
    print(f"Dataset downloaded to: {dataset_path}")

    # Move all files from the downloaded directory to the current directory
    current_dir = os.getcwd()
    versioned_dir = os.path.join(dataset_path, "versions", "1")
    
    if os.path.exists(versioned_dir):
        for filename in os.listdir(versioned_dir):
            src = os.path.join(versioned_dir, filename)
            dst = os.path.join(current_dir, filename)
            print(f"Moving {src} -> {dst}")
            shutil.move(src, dst)
    else:
        print("Could not find versioned directory structure. Please check manually.")

    print("Dataset download and relocation complete.")

if __name__ == "__main__":
    download_dataset()
