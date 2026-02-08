import numpy as np
import os
from nilearn import datasets

def filter_task_data(X, y_group, y_task, groups, task_id):
    """Returns only the data and labels associated with a specific task_id."""
    mask = (y_task == task_id)
    return X[mask], y_group[mask], groups[mask]

def average_by_subject(X_subset, y_subset, groups_subset):
    """Averages multiple runs/trials for each unique subject."""
    X_avg, y_avg = [], []
    unique_subs = np.unique(groups_subset)
    
    for sub in unique_subs:
        sub_indices = np.where(groups_subset == sub)[0]
        # Mean across features for this participant
        X_avg.append(np.mean(X_subset[sub_indices], axis=0))
        # Keep the label (TD or MLD)
        y_avg.append(y_subset[sub_indices[0]])
        
    return np.array(X_avg), np.array(y_avg)

X = np.load('data/X_features.npy')
y_groups = np.load('data/y_labels.npy')
y_task = np.load('data/y_tasks.npy')
groups = np.load('data/subject_groups.npy')

def get_all_data():
    X_avg, y_avg = average_by_subject(X, y_groups, groups)
    return X_avg, y_avg

def get_mult_data():
    X_m, y_m, g_m = filter_task_data(X, y_groups, y_task, groups, task_id=1)
    X_avg_m, y_avg_m = average_by_subject(X_m, y_m, g_m)
    return X_avg_m, y_avg_m

def get_sub_data():
    X_s, y_s, g_s = filter_task_data(X, y_groups, y_task, groups, task_id=0)
    X_avg_s, y_avg_s = average_by_subject(X_s, y_s, g_s)
    return X_avg_s, y_avg_s

def get_labels():
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7)
    labels = atlas.labels
    return labels