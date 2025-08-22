import numpy as np
from collections import Counter

def get_entropy_of_dataset(data: np.ndarray) -> float:
    """
    Calculate the entropy of the entire dataset using the target variable (last column).
    NumPy optimized version.
    
    Args:
        data (np.ndarray): Dataset where the last column is the target variable
    
    Returns:
        float: Entropy value calculated using the formula: 
               Entropy = -Σ(p_i * log2(p_i)) where p_i is the probability of class i
    """
    # Extract target column (last column)
    target_column = data[:, -1]
    
    # Get unique classes and their counts using NumPy
    unique_classes, counts = np.unique(target_column, return_counts=True)
    
    # Calculate total number of samples
    total_samples = len(target_column)
    
    # Calculate probabilities
    probabilities = counts / total_samples
    
    # Calculate entropy using vectorized operations
    # Use np.where to handle log2(0) case
    entropy = -np.sum(probabilities * np.log2(probabilities, out=np.zeros_like(probabilities), where=(probabilities!=0)))
    
    return float(entropy)


def get_avg_info_of_attribute(data: np.ndarray, attribute: int) -> float:
    """
    Calculate the average information (weighted entropy) of a specific attribute.
    NumPy optimized version.
    
    Args:
        data (np.ndarray): Dataset where the last column is the target variable
        attribute (int): Index of the attribute column to calculate average information for
    
    Returns:
        float: Average information calculated using the formula:
               Avg_Info = Σ((|S_v|/|S|) * Entropy(S_v)) 
               where S_v is subset of data with attribute value v
    """
    # Extract attribute column
    attr_column = data[:, attribute]
    
    # Get unique values in the attribute column
    unique_values = np.unique(attr_column)
    
    # Calculate total number of samples
    total_samples = len(data)
    
    # Calculate weighted average information
    avg_info = 0.0
    for value in unique_values:
        # Create boolean mask for this attribute value
        mask = attr_column == value
        subset = data[mask]
        
        if len(subset) > 0:
            # Calculate weight (proportion of samples with this value)
            weight = len(subset) / total_samples
            
            # Calculate entropy of this subset
            subset_entropy = get_entropy_of_dataset(subset)
            avg_info += weight * subset_entropy
    
    return float(avg_info)


def get_information_gain(data: np.ndarray, attribute: int) -> float:
    """
    Calculate the Information Gain for a specific attribute.
    NumPy optimized version.
    
    Args:
        data (np.ndarray): Dataset where the last column is the target variable
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


def get_selected_attribute(data: np.ndarray) -> tuple:
    """
    Select the best attribute based on highest information gain.
    NumPy optimized version.
    
    Args:
        data (np.ndarray): Dataset where the last column is the target variable
    
    Returns:
        tuple: A tuple containing:
            - dict: Dictionary mapping attribute indices to their information gains
            - int: Index of the attribute with the highest information gain
    """
    # Number of attributes (excluding target variable which is the last column)
    num_attributes = data.shape[1] - 1
    
    # Dictionary to store information gains
    information_gains = {}
    
    # Calculate information gain for each attribute using vectorized operations
    for attribute_idx in range(num_attributes):
        gain = get_information_gain(data, attribute_idx)
        information_gains[attribute_idx] = gain
    
    # Find attribute with maximum information gain
    selected_attribute = max(information_gains, key=information_gains.get)
    
    return (information_gains, selected_attribute)