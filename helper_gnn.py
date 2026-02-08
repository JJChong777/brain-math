import numpy as np

def average_gnn_matrices_by_subject(X_raw, y_labels, subject_ids):
    """
    Averages 3D correlation matrices for each unique subject.
    
    Parameters:
    X_raw: 3D array of shape (N_scans, 400, 400)
    y_labels: 1D array of labels (MLD vs TD)
    subject_ids: 1D array of subject identifiers
    """
    X_avg, y_avg = [], []
    unique_subs = np.unique(subject_ids)
    
    for sub in unique_subs:
        # Find all indices (e.g., Mult and Sub scans) for this specific child
        sub_indices = np.where(subject_ids == sub)[0]
        
        # Mean across the first axis (the scans) to get one 400x400 matrix
        # This is the same logic as your helper.py, but works for 3D arrays
        avg_matrix = np.mean(X_raw[sub_indices], axis=0)
        
        X_avg.append(avg_matrix)
        y_avg.append(y_labels[sub_indices[0]])
        
    return np.array(X_avg), np.array(y_avg)

def get_task_specific_gnn_data(X_raw, y_labels, y_tasks, subject_ids, task_id):
    """
    Filters the raw 3D matrices for a specific task (1 for Mult, 0 for Sub).
    """
    # Create a mask for the specific task
    mask = (y_tasks == task_id)
    
    # Apply the mask to all arrays
    X_task = X_raw[mask]
    y_task_labels = y_labels[mask]
    s_task_ids = subject_ids[mask]
    
    return X_task, y_task_labels, s_task_ids

# Example usage with your raw saved files:
X_raw = np.load('X_raw_gnn.npy')
y_labels = np.load('y_labels.npy')
subject_ids = np.load('subject_ids.npy')

X_final, y_final = average_gnn_matrices_by_subject(X_raw, y_labels, subject_ids)
# X_final will now have the shape (30, 400, 400)