"""
Utility functions for model evaluation and result processing.
Moved from engine_for_finetuning.py to improve code organization.
"""

import os
import pickle
import numpy as np
from scipy.special import softmax
from typing import Dict, List, Tuple, Any, Union

from src.utils.config import get_cfg


def compute_video(lst: List[Any]) -> List[Union[int, float]]:
    """
    Compute video-level prediction from multiple clips.
    
    Args:
        lst: List containing [video_index, video_id, feature_data, label]
        
    Returns:
        List containing [prediction, top1_accuracy, top5_accuracy, label]
    """
    i, video_id, data, label = lst
    feat = [x for x in data]
    feat = np.mean(feat, axis=0)
    pred = np.argmax(feat)
    top1 = (int(pred) == int(label)) * 1.0
    top5 = (int(label) in np.argsort(-feat)[:5]) * 1.0
    return [pred, top1, top5, int(label)]


def merge_distributed_results(eval_path: str, num_tasks: int, best: bool = False) -> Tuple[float, float, Dict[str, List]]:
    """
    Merge results from distributed evaluation.
    
    Args:
        eval_path: Path to evaluation results
        num_tasks: Number of distributed tasks
        args: Configuration arguments (for backward compatibility)
        best: Whether to use best checkpoint results
        
    Returns:
        Tuple of (final_top1, final_top5, prediction_dict)
    """
    # Use global config if args is not provided
    cfg = get_cfg()
    
    dict_feats = {}
    dict_label = {}
    dict_pos = {}
    print("Reading individual output files")

    # For saving feature in the last layer
    overall_saved_features = {}

    # Read results from each distributed process
    for x in range(num_tasks):
        file = os.path.join(eval_path, str(x) + '.txt') if not best else os.path.join(eval_path, str(x) + '_best.txt')
        lines = open(file, 'r').readlines()[1:]
        
        for line in lines:
            line = line.strip()
            name = line.split('[')[0]
            
            if cfg.DATA.DATASET_NAME == 'Gaze360':
                # Regression task parsing
                parts = line.split(']')
                label_str = parts[1].split(' ')[1]
                
                # Parse label (might be in list format)
                if label_str.startswith('[') and label_str.endswith(']'):
                    label = eval(label_str)  # Convert string list to actual list
                else:
                    label = float(label_str)
                    
                chunk_nb = parts[1].split(' ')[2]
                split_nb = parts[1].split(' ')[3]
                data = np.fromstring(parts[0].split('[')[1], dtype=np.float32, sep=',')
                # Don't apply softmax for regression tasks
            else:
                # Classification task parsing
                label = line.split(']')[1].split(' ')[1]
                chunk_nb = line.split(']')[1].split(' ')[2]
                split_nb = line.split(']')[1].split(' ')[3]
                data = np.fromstring(line.split('[')[1].split(']')[0], dtype=np.float, sep=',')
                data = softmax(data)
                
            if name not in dict_feats:
                dict_feats[name] = []
                dict_label[name] = 0
                dict_pos[name] = []
                
            if chunk_nb + split_nb in dict_pos[name]:
                continue
                
            dict_feats[name].append(data)
            dict_pos[name].append(chunk_nb + split_nb)
            dict_label[name] = label

        # Load saved features if available
        if cfg.SYSTEM.SAVE_FEATURE:
            feature_file = file.replace(file[-4:], '_feature.pkl')
            if os.path.exists(feature_file):
                saved_features = pickle.load(open(feature_file, 'rb'))
                
                for sample_id in saved_features.keys():
                    if sample_id not in overall_saved_features:
                        overall_saved_features[sample_id] = {
                            'feature': [],
                            'prob': [],
                            'chunk_id': [],
                            'split_id': []
                        }
                    
                    chunk_ids = saved_features[sample_id]['chunk_id']
                    split_ids = saved_features[sample_id]['split_id']
                    
                    for idx, (chunk_id, split_id) in enumerate(zip(chunk_ids, split_ids)):
                        if chunk_id + split_id not in overall_saved_features[sample_id]['chunk_id']:
                            overall_saved_features[sample_id]['feature'].append(
                                saved_features[sample_id]['feature'][idx]
                            )
                            overall_saved_features[sample_id]['prob'].append(
                                saved_features[sample_id]['prob'][idx]
                            )
                            overall_saved_features[sample_id]['chunk_id'].append(chunk_id + split_id)

    print("Computing final results")

    # Compute final predictions
    input_lst = []
    print(f"Total videos: {len(dict_feats)}")
    
    # Prepare prediction dictionary
    pred_dict = {'id': [], 'label': [], 'pred': []}
    
    for i, item in enumerate(dict_feats):
        input_lst.append([i, item, dict_feats[item], dict_label[item]])
        
        if cfg.DATA.DATASET_NAME == 'Gaze360':
            # For regression tasks, use mean prediction
            pred = np.mean(dict_feats[item], axis=0)
            pred_dict['pred'].append(pred.tolist() if isinstance(pred, np.ndarray) else pred)
        else:
            # For classification tasks, use argmax
            pred = int(np.argmax(np.mean(dict_feats[item], axis=0)))
            pred_dict['pred'].append(pred)
            
        pred_dict['label'].append(dict_label[item])
        pred_dict['id'].append(item.strip())

    # Compute accuracy (only for classification tasks)
    if cfg.DATA.DATASET_NAME == 'Gaze360':
        # For regression tasks, we don't compute top1/top5 accuracy
        final_top1 = 0.0
        final_top5 = 0.0
    else:
        # Use single-threaded computation to avoid hanging
        ans = [compute_video(lst) for lst in input_lst]
        top1 = [x[1] for x in ans]
        top5 = [x[2] for x in ans]
        final_top1, final_top5 = np.mean(top1), np.mean(top5)

    # Save aggregated features if requested
    if cfg.SYSTEM.SAVE_FEATURE and overall_saved_features:
        # Get average feature and prediction
        for sample_id in overall_saved_features.keys():
            overall_saved_features[sample_id]['feature'] = np.mean(
                overall_saved_features[sample_id]['feature'], axis=0
            )
            if cfg.DATA.DATASET_NAME != 'Gaze360':
                overall_saved_features[sample_id]['pred'] = int(
                    np.argmax(np.mean(overall_saved_features[sample_id]['prob'], axis=0))
                )
            else:
                overall_saved_features[sample_id]['pred'] = np.mean(
                    overall_saved_features[sample_id]['prob'], axis=0
                )
        
        feature_file = os.path.join(eval_path, 'overall_feature.pkl') if not best else os.path.join(eval_path, 'overall_feature_best.pkl')
        pickle.dump(overall_saved_features, open(feature_file, 'wb'))

    return final_top1 * 100, final_top5 * 100, pred_dict


def save_evaluation_results(file_path: str, results: Dict[str, List], args=None):
    """
    Save evaluation results to file.
    
    Args:
        file_path: Path to save results
        results: Dictionary containing evaluation results
        args: Configuration arguments (for backward compatibility)
    """
    cfg = get_cfg()
    
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
    with open(file_path, 'w') as f:
        if cfg.DATA.DATASET_NAME == 'Gaze360':
            # For regression tasks, save mean angular error
            f.write("Mean Angular Error\n")
            for i, (vid_id, label, pred) in enumerate(zip(results['id'], results['label'], results['pred'])):
                f.write(f"{vid_id}: label={label}, pred={pred}\n")
        else:
            # For classification tasks, save accuracy
            acc1 = sum([1 for p, l in zip(results['pred'], results['label']) if p == l]) / len(results['label'])
            f.write(f"Accuracy: {acc1:.4f}\n")
            for i, (vid_id, label, pred) in enumerate(zip(results['id'], results['label'], results['pred'])):
                f.write(f"{vid_id}: label={label}, pred={pred}\n")


def compute_angular_error(pred_angles: np.ndarray, target_angles: np.ndarray) -> float:
    """
    Compute angular error between predicted and target gaze angles.
    
    Args:
        pred_angles: Predicted angles [N, 2] (pitch, yaw)
        target_angles: Target angles [N, 2] (pitch, yaw)
        
    Returns:
        Mean angular error in degrees
    """
    # Convert to radians
    pred_rad = np.radians(pred_angles)
    target_rad = np.radians(target_angles)
    
    # Compute angular error using dot product
    cos_error = np.sum(pred_rad * target_rad, axis=1)
    cos_error = np.clip(cos_error, -1.0, 1.0)  # Avoid numerical errors
    
    angular_error = np.arccos(cos_error)
    angular_error_degrees = np.degrees(angular_error)
    
    return np.mean(angular_error_degrees)
