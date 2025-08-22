import numpy as np
import torch
from collections import Counter

def get_entropy_of_dataset(data) -> float:
    """
    Calculate the entropy of the entire dataset using the target variable (last column).
    PyTorch optimized version.
    
    Args:
        data: Dataset where the last column is the target variable (torch.Tensor or np.ndarray)
    
    Returns:
        float: Entropy value calculated using the formula: 
               Entropy = -Σ(p_i * log2(p_i)) where p_i is the probability of class i
    """
    # Convert to PyTorch tensor if needed
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()
    elif not isinstance(data, torch.Tensor):
        data = torch.tensor(data).float()
    
    # Extract target column (last column)
    target_column = data[:, -1]
    
    # Get unique classes and their counts using PyTorch
    unique_classes, counts = torch.unique(target_column, return_counts=True)
    
    # Calculate total number of samples
    total_samples = target_column.shape[0]
    
    # Calculate probabilities
    probabilities = counts.float() / total_samples
    
    # Calculate entropy using PyTorch operations
    # Use torch.where to handle log2(0) case
    log_probs = torch.where(probabilities > 0, torch.log2(probabilities), torch.zeros_like(probabilities))
    entropy = -torch.sum(probabilities * log_probs)
    
    return float(entropy.item())


def get_avg_info_of_attribute(data, attribute: int) -> float:
    """
    Calculate the average information (weighted entropy) of a specific attribute.
    PyTorch optimized version.
    
    Args:
        data: Dataset where the last column is the target variable (torch.Tensor or np.ndarray)
        attribute (int): Index of the attribute column to calculate average information for
    
    Returns:
        float: Average information calculated using the formula:
               Avg_Info = Σ((|S_v|/|S|) * Entropy(S_v)) 
               where S_v is subset of data with attribute value v
    """
    # Convert to PyTorch tensor if needed
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()
    elif not isinstance(data, torch.Tensor):
        data = torch.tensor(data).float()
    
    # Extract attribute column
    attr_column = data[:, attribute]
    
    # Get unique values in the attribute column
    unique_values = torch.unique(attr_column)
    
    # Calculate total number of samples
    total_samples = data.shape[0]
    
    # Calculate weighted average information
    avg_info = 0.0
    for value in unique_values:
        # Create boolean mask for this attribute value
        mask = attr_column == value
        subset = data[mask]
        
        if subset.shape[0] > 0:
            # Calculate weight (proportion of samples with this value)
            weight = subset.shape[0] / total_samples
            
            # Calculate entropy of this subset
            subset_entropy = get_entropy_of_dataset(subset)
            avg_info += weight * subset_entropy
    
    return float(avg_info)


def get_information_gain(data, attribute: int) -> float:
    """
    Calculate the Information Gain for a specific attribute.
    PyTorch optimized version.
    
    Args:
        data: Dataset where the last column is the target variable (torch.Tensor or np.ndarray)
        attribute (int): Index of the attribute column to calculate information gain for
    
    Returns:
        float: Information gain calculated using the formula:
               Information_Gain = Entropy(S) - Avg_Info(attribute)
               Rounded to 4 decimal places
    """
    # Calculate dataset entropy
    dataset_entropy = get_entropy_of_dataset(data)
    
    # Calculate average information of the attribute
    avg_info = get_avg_info_of_attribute(data, attribute)
    
    # Calculate information gain
    information_gain = dataset_entropy - avg_info
    
    # Round to 4 decimal places
    return round(float(information_gain), 4)


def get_selected_attribute(data) -> tuple:
    """
    Select the best attribute based on highest information gain.
    PyTorch optimized version.
    
    Args:
        data: Dataset where the last column is the target variable (torch.Tensor or np.ndarray)
    
    Returns:
        tuple: A tuple containing:
            - dict: Dictionary mapping attribute indices to their information gains
            - int: Index of the attribute with the highest information gain
    """
    # Convert to PyTorch tensor if needed
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()
    elif not isinstance(data, torch.Tensor):
        data = torch.tensor(data).float()
    
    # Number of attributes (excluding target variable which is the last column)
    num_attributes = data.shape[1] - 1
    
    # Dictionary to store information gains
    information_gains = {}
    
    # Calculate information gain for each attribute
    for attribute_idx in range(num_attributes):
        gain = get_information_gain(data, attribute_idx)
        information_gains[attribute_idx] = gain
    
    # Find attribute with maximum information gain using PyTorch operations
    gains_tensor = torch.tensor(list(information_gains.values()))
    max_idx = torch.argmax(gains_tensor).item()
    selected_attribute = list(information_gains.keys())[max_idx]
    
    return (information_gains, selected_attribute)