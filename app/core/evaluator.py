import os
import zipfile
import shutil
from werkzeug.utils import secure_filename

def extract_dataset(zip_path, extract_to):
    """
    Extracts a zip file containing the dataset.
    Returns the path to the extracted images (the folder containing the class subfolders).
    """
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Zip file not found: {zip_path}")
        
    # Clean the extraction directory if it already exists to avoid garbage
    if os.path.exists(extract_to):
        shutil.rmtree(extract_to)
    os.makedirs(extract_to, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        
    # Find the actual root of the dataset (the directory containing class folders)
    # Sometimes users ZIP a folder instead of the contents directly
    dataset_root = extract_to
    
    # Check if there is exactly one directory inside the extracted root
    items = os.listdir(extract_to)
    if len(items) == 1 and os.path.isdir(os.path.join(extract_to, items[0])):
        dataset_root = os.path.join(extract_to, items[0])
        
    return dataset_root

import torch
import time
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def evaluate_model(model, input_shape, dataset_path):
    """
    Evaluates the model against an ImageFolder dataset.
    Args:
        model: Built PyTorch nn.Module
        input_shape: e.g., [1, 3, 224, 224] representing [Batch, Channels, Height, Width]
        dataset_path: Extracted folder containing class subdirectories
    """
    if len(input_shape) != 4:
        raise ValueError("Evaluation currently requires a 4D input shape: [Batch, Channels, Height, Width]")

    # Extract required dimensions for transformations
    channels = input_shape[1]
    height = input_shape[2]
    width = input_shape[3]

    # Dynamically build transforms to match the model's required input size
    transform_list = [
        transforms.Resize((height, width)),
        transforms.ToTensor(),
    ]

    # Note: If channels == 1, convert RGB images to Grayscale
    if channels == 1:
        transform_list.insert(1, transforms.Grayscale(num_output_channels=1))
        
    data_transforms = transforms.Compose(transform_list)

    # Load Dataset
    try:
        dataset = datasets.ImageFolder(dataset_path, data_transforms)
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset as ImageFolder. Ensure it contains subdirectories for classes. {str(e)}")

    if len(dataset) == 0:
        raise ValueError("Dataset contains no valid images.")

    # Create DataLoader
    # Use batch_size=1 since our dummy input is usually batch_size=1, but allow standard eval batches
    batch_size = max(1, input_shape[0])
    # Don't use too many workers to avoid crashing small VMs or causing pickling errors dynamically
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Prepare for evaluation
    device = next(model.parameters()).device if list(model.parameters()) else torch.device('cpu')
    model = model.to(device)
    model.eval()

    correct = 0
    total = 0
    
    start_time = time.time()
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Predict
            outputs = model(inputs)
            
            # Assuming standard classification output where dim 1 is logits
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    end_time = time.time()
    
    total_time_seconds = end_time - start_time
    fps = total / total_time_seconds if total_time_seconds > 0 else 0
    accuracy = (correct / total) * 100 if total > 0 else 0

    return {
        "accuracy": round(accuracy, 2),
        "total_images": total,
        "correct_predictions": correct,
        "total_time_seconds": round(total_time_seconds, 2),
        "fps": round(fps, 2),
        "classes": dataset.classes
    }
