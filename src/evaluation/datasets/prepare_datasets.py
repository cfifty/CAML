import os
import shutil
from torchvision import datasets, transforms
import torch
from typing import Literal
import argparse
from multiprocessing import Pool

DATASETS = Literal["mnist", "fashionmnist", "cifar10", "svhn", "usps", "stl", "kmnist"]

dataset_map = {
    "mnist": datasets.MNIST,
    "fashionmnist": datasets.FashionMNIST,
    "cifar10": datasets.CIFAR10,
    "svhn": datasets.SVHN,
    "usps": datasets.USPS,
    "stl": datasets.STL10,
    "kmnist": datasets.KMNIST,
}


def process_class_images(args):
    """Process and save images for a specific class."""
    images, label, class_dir = args
    for i, image in enumerate(images):
        image_path = os.path.join(class_dir, f"{i}.jpg")
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)  # Convert tensor to PIL image
        image.save(image_path)


def download_and_save_datasets(
    output_dir: str,
    datasets_to_download: list[DATASETS],
    num_workers: int = os.cpu_count(),
):
    """
    Download specified datasets and save them in ImageFolder-compatible format.

    Args:
        output_dir (str): Directory to save the processed datasets.
        datasets_to_download (list): List of dataset names to download (e.g., ["MNIST", "CIFAR10"]).
        num_workers (int): Number of workers to use for parallel processing.
    """

    for dataset_name in datasets_to_download:
        if dataset_name not in dataset_map:
            print(f"Dataset {dataset_name} is not supported.")
            continue

        # Create output directory for the dataset
        dataset_dir = os.path.join(output_dir, dataset_name)
        if os.path.exists(dataset_dir):
            print(
                f"Dataset {dataset_name} already exists in {dataset_dir}. Skipping download."
            )
            continue
        os.makedirs(dataset_dir, exist_ok=True)

        # Download the dataset
        print(f"Downloading {dataset_name}...")
        if dataset_name == "svhn" or dataset_name == "stl":
            dataset = dataset_map[dataset_name](
                root="./temp_data", split="train", download=True
            )
        else:
            dataset = dataset_map[dataset_name](
                root="./temp_data", train=True, download=True
            )

        # Prepare arguments for multiprocessing
        class_images = {}
        for i, (image, label) in enumerate(dataset):
            if label not in class_images:
                class_images[label] = []
            class_images[label].append(image)

        args = []
        for label, images in class_images.items():
            class_dir = os.path.join(dataset_dir, str(label))
            os.makedirs(class_dir, exist_ok=True)
            args.append((images, label, class_dir))

        # Process images in parallel
        print(f"Processing {dataset_name} with {num_workers} workers...")
        with Pool(num_workers) as pool:
            pool.map(process_class_images, args)

        print(f"Saved {dataset_name} to {dataset_dir}.")

    # Clean up temporary files
    shutil.rmtree("./temp_data", ignore_errors=True)
    print("Temporary files cleaned up.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and save datasets.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/ubuntu/flowtolearn-D2NWG/data/datasets",
        help="Directory to save the processed datasets.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        type=str,
        default=["MNIST", "CIFAR10"],
        help="List of dataset names to download (e.g., MNIST CIFAR10).",
    )
    args = parser.parse_args()

    download_and_save_datasets(args.output_dir, args.datasets)
