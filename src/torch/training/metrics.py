import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.metrics as skm


class HeteroEdgeMetricWrapper:
    """
    Wrapper to compute metrics on edges of heterogeneous graphs.

    Args:
        metric_fn (callable): Metric function to compute (e.g., accuracy, AUC)
    """

    def __init__(self, metric_fn):
        self.metric_fn = metric_fn

    def __call__(self, outputs, targets):
        """
        Compute the metric for a specific edge type.

        Args:
            outputs (dict): Dictionary of model outputs for each edge type
            targets (dict): Dictionary of true labels for each edge type
            edge_type (tuple): Edge type to compute the metric for (e.g., ('mppc', 'to', 'pixel'))
        Returns:
            dict: Dictionary with metric results for each edge type
        """
        results = {}
        for edge_type in targets:
            if edge_type in outputs:
                preds = outputs[edge_type].detach().cpu()
                labels = targets[edge_type].detach().cpu()
                if preds.ndim == 2 and preds.shape[1] == 1:
                    preds = preds.squeeze(1)
                metric_value = self.metric_fn(labels, preds)
                results[edge_type] = metric_value
        return results

class HeteroEdgeNodeWrapper:
    """
    Wrapper to compute metrics on nodes of heterogeneous graphs.

    Args:
        metric_fn (callable): Metric function to compute (e.g., accuracy, AUC)
    """

    def __init__(self, metric_fn):
        self.metric_fn = metric_fn

    def __call__(self, outputs, targets):
        """
        Compute the metric for a specific node type.

        Args:
            outputs (dict): Dictionary of model outputs for each node type
            targets (dict): Dictionary of true labels for each node type
        Returns:
            dict: Dictionary with metric results for each node type
        """
        results = {}
        for node_type in targets:
            if node_type in outputs:
                preds = outputs[node_type].detach().cpu()
                labels = targets[node_type].detach().cpu()
                if preds.ndim == 2 and preds.shape[1] == 1:
                    preds = preds.squeeze(1)
                metric_value = self.metric_fn(labels, preds)
                results[node_type] = metric_value
        return results