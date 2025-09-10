"""
Graph batching utilities for detector data processing.

This module provides PyTorch Geometric compatible classes for building different types of graphs from detector events:
- PixelGraphBuilder: Homogeneous graphs from pixel data only
- HeteroGraphBuilder: Heterogeneous graphs with pixel and MPPC detectors  
- LayerSeparatedHeteroGraphBuilder: Layer-separated heterogeneous graphs
- CombinedGraphBuilder: Combined homogeneous graphs with both detector types
"""

import torch
import numpy as np
from torch_geometric.data import Data, Dataset, Batch, HeteroData, InMemoryDataset
from torch_geometric.loader import DataLoader
from typing import List, Tuple, Optional, Union, Dict, Any, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import os
import os.path as osp
import pickle


@dataclass
class DetectorData:
    """Container for detector data components."""
    
    positions: torch.Tensor
    layers: torch.Tensor
    tracks: torch.Tensor
    times: torch.Tensor

    def __len__(self) -> int:
        return self.positions.size(0)

    def filter_by_time(self, time_value: torch.Tensor) -> "DetectorData":
        """Filter data by time value."""
        mask = self.times == time_value
        return DetectorData(
            positions=self.positions[mask],
            layers=self.layers[mask],
            tracks=self.tracks[mask],
            times=self.times[mask],
        )

    def has_sufficient_data(self, min_nodes: int = 2) -> bool:
        """Check if data has sufficient nodes."""
        return len(self) >= min_nodes


@dataclass
class EdgeConfig:
    """Configuration for edge creation between detector types."""
    
    src_type: str
    dst_type: str
    edge_type: Tuple[str, str, str]
    same_type: bool
    use_timing: bool = False
    use_layers: bool = True


def generate_all_pairs(n_src: int, n_dst: int, include_self: bool = True, same_type: bool = False):
    """
    Generate all possible (i, j) index pairs between two node sets.

    Args:
        n_src: Number of source nodes
        n_dst: Number of destination nodes  
        include_self: Keep diagonal if same_type=True
        same_type: If True, treat source and destination as the same set

    Returns:
        Tuple of (row, col) tensors with source and destination indices
    """
    row = torch.arange(n_src).repeat_interleave(n_dst)
    col = torch.arange(n_dst).repeat(n_src)

    if same_type and not include_self:
        mask = row != col
        row, col = row[mask], col[mask]

    return row, col


class GraphBuilderBase(ABC):
    """Abstract base class for graph builders."""

    def __init__(self, connect_layers: bool = True):
        self.connect_layers = connect_layers

    @staticmethod
    def _validate_tensor(tensor: Union[torch.Tensor, any], name: str) -> torch.Tensor:
        """Validate and prepare input tensor."""
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor, dtype=torch.float32)

        if tensor.dim() == 3:
            tensor = tensor.squeeze(0)
        elif tensor.dim() != 2:
            raise ValueError(f"{name} must be 2D or 3D tensor. Got {tensor.dim()}D.")

        return tensor

    @staticmethod
    def _extract_detector_data(tensor: torch.Tensor) -> DetectorData:
        """Extract valid detector data from tensor."""
        mask = tensor[:, -1] != -1  # Valid entries have time != -1
        return DetectorData(
            positions=tensor[:, :3][mask],
            layers=tensor[:, 3][mask],
            tracks=tensor[:, 4][mask],
            times=tensor[:, -1][mask],
        )

    def _create_edge_labels(self, src_tracks: torch.Tensor, dst_tracks: torch.Tensor, 
                           row: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
        """Create edge labels based on track matching."""
        return ((src_tracks[row] > 0) & (src_tracks[row] == dst_tracks[col])).float()

    @abstractmethod
    def build_graphs_from_event(self, **kwargs) -> List[Union[Data, HeteroData]]:
        """Build graphs from event data."""
        pass


class PixelGraphBuilder(GraphBuilderBase):
    """Builder for pixel-only homogeneous graphs."""

    def _create_pixel_edges(self, data: DetectorData) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Create edges between pixel nodes."""
        if len(data) < 2:
            return None, None

        row, col = generate_all_pairs(len(data), len(data), include_self=False, same_type=True)

        if self.connect_layers and row.numel() > 0:
            edge_mask = (data.layers[row] - data.layers[col]).abs() <= 1
            row, col = row[edge_mask], col[edge_mask]

        if row.numel() == 0:
            return None, None

        edge_labels = self._create_edge_labels(data.tracks, data.tracks, row, col)
        edge_index = torch.stack([row, col], dim=0)

        return edge_index, edge_labels

    def build_graphs_from_event(self, pixel: Union[torch.Tensor, any], **kwargs) -> List[Data]:
        """Build homogeneous graphs from pixel event data."""
        pixel = self._validate_tensor(pixel, "pixel")
        data = self._extract_detector_data(pixel)

        graphs = []
        for time_slice in torch.unique(data.times):
            data_t = data.filter_by_time(time_slice)

            if not data_t.has_sufficient_data():
                continue

            edge_index, edge_labels = self._create_pixel_edges(data_t)

            if edge_index is not None:
                graph = Data(
                    x=data_t.positions, 
                    edge_index=edge_index, 
                    edge_labels=edge_labels
                )
                graphs.append(graph)

        return graphs


class HeteroGraphBuilder(GraphBuilderBase):
    """Builder for heterogeneous graphs with multiple detector types."""

    def __init__(self, connect_layers: bool = True, mppc_timing_cutoff: float = 0.2):
        super().__init__(connect_layers)
        self.mppc_timing_cutoff = mppc_timing_cutoff
        self.edge_configs = [
            EdgeConfig("pixel", "pixel", ("pixel", "to", "pixel"), True, False, True),
            EdgeConfig("mppc", "mppc", ("mppc", "to", "mppc"), True, True, False),
            EdgeConfig("pixel", "mppc", ("pixel", "to", "mppc"), False, False, True),
            EdgeConfig("mppc", "pixel", ("mppc", "to", "pixel"), False, False, True),
        ]

    def _create_edges(self, src_data: DetectorData, dst_data: DetectorData, 
                     config: EdgeConfig) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Create edges between source and destination nodes."""
        if len(src_data) == 0 or len(dst_data) == 0:
            return None, None

        row, col = generate_all_pairs(len(src_data), len(dst_data), 
                                     include_self=False, same_type=config.same_type)

        if row.numel() == 0:
            return None, None

        # Apply layer constraints
        if self.connect_layers and config.use_layers:
            layer_diff = (src_data.layers[row] - dst_data.layers[col]).abs()
            layer_mask = (layer_diff <= 1) & ~(
                ((src_data.layers[row] == 2) & (dst_data.layers[col] == 3)) |
                ((src_data.layers[row] == 3) & (dst_data.layers[col] == 2))
            )
            row, col = row[layer_mask], col[layer_mask]

        # Apply timing constraints for MPPC
        if config.use_timing and row.numel() > 0:
            time_diff = (src_data.times[row] - dst_data.times[col]).abs()
            timing_mask = time_diff <= self.mppc_timing_cutoff
            row, col = row[timing_mask], col[timing_mask]

        if row.numel() == 0:
            return None, None

        edge_labels = self._create_edge_labels(src_data.tracks, dst_data.tracks, row, col)
        return torch.stack([row, col], dim=0), edge_labels

    def _create_hetero_graph(self, pixel_data: DetectorData, mppc_data: DetectorData,
                           time_slice: torch.Tensor) -> Optional[HeteroData]:
        """Create heterogeneous graph for a specific time slice."""
        # Filter data for current time slice
        pixel_t = pixel_data.filter_by_time(time_slice)
        mppc_cut_times = (mppc_data.times // 8) * 8
        mppc_t = DetectorData(
            positions=mppc_data.positions[mppc_cut_times == time_slice],
            layers=mppc_data.layers[mppc_cut_times == time_slice],
            tracks=mppc_data.tracks[mppc_cut_times == time_slice],
            times=mppc_data.times[mppc_cut_times == time_slice],
        )

        # Skip if insufficient data
        if not pixel_t.has_sufficient_data() or not mppc_t.has_sufficient_data():
            return None

        # Create graph with node features
        graph = HeteroData()
        graph["pixel"].x = pixel_t.positions
        graph["mppc"].x = torch.cat([mppc_t.positions, mppc_t.times.unsqueeze(1)], dim=1)

        # Create edges for all configured types
        data_map = {"pixel": pixel_t, "mppc": mppc_t}
        edges_created = 0

        for config in self.edge_configs:
            src_data = data_map[config.src_type]
            dst_data = data_map[config.dst_type]

            edge_index, edge_labels = self._create_edges(src_data, dst_data, config)
            
            if edge_index is not None and edge_labels is not None:
                graph[config.edge_type].edge_index = edge_index
                graph[config.edge_type].edge_labels = edge_labels
                edges_created += 1

        return graph if edges_created == 4 else None

    def build_graphs_from_event(self, pixel: Union[torch.Tensor, any], 
                              mppc: Union[torch.Tensor, any], **kwargs) -> List[HeteroData]:
        """Build heterogeneous graphs from a single event."""
        mppc = self._validate_tensor(mppc, "mppc")
        pixel = self._validate_tensor(pixel, "pixel")

        pixel_data = self._extract_detector_data(pixel)
        mppc_data = self._extract_detector_data(mppc)

        graphs = []
        for time_slice in torch.unique(pixel_data.times):
            graph = self._create_hetero_graph(pixel_data, mppc_data, time_slice)
            if graph is not None:
                graphs.append(graph)

        return graphs


class LayerSeparatedHeteroGraphBuilder(GraphBuilderBase):
    """
    Builder for layer-separated heterogeneous graphs with specific connectivity:
    layer_1 <-> layer_2 <-> mppc <-> mppc <-> layer_3 <-> layer_4
    """

    def __init__(self, connect_layers: bool = True, mppc_timing_cutoff: float = 0.2):
        super().__init__(connect_layers)
        self.mppc_timing_cutoff = mppc_timing_cutoff
        
        # Define the connectivity pattern
        self.edge_configs = [
            # Layer connections
            EdgeConfig("layer_1", "layer_2", ("layer_1", "to", "layer_2"), False, False, False),
            EdgeConfig("layer_2", "layer_1", ("layer_2", "to", "layer_1"), False, False, False),
            
            # Layer 2 to MPPC connections
            EdgeConfig("layer_2", "mppc", ("layer_2", "to", "mppc"), False, False, False),
            EdgeConfig("mppc", "layer_2", ("mppc", "to", "layer_2"), False, False, False),
            
            # MPPC internal connections
            EdgeConfig("mppc", "mppc", ("mppc", "to", "mppc"), True, True, False),
            
            # MPPC to Layer 3 connections
            EdgeConfig("mppc", "layer_3", ("mppc", "to", "layer_3"), False, False, False),
            EdgeConfig("layer_3", "mppc", ("layer_3", "to", "mppc"), False, False, False),
            
            # Layer 3 to Layer 4 connections
            EdgeConfig("layer_3", "layer_4", ("layer_3", "to", "layer_4"), False, False, False),
            EdgeConfig("layer_4", "layer_3", ("layer_4", "to", "layer_3"), False, False, False),
        ]

    def _separate_pixel_layers(self, pixel_data: DetectorData) -> Dict[str, DetectorData]:
        """Separate pixel data into individual layer datasets."""
        layer_data = {}
        
        for layer_id in [1, 2, 3, 4]:
            layer_mask = pixel_data.layers == layer_id
            if layer_mask.sum() > 0:
                layer_data[f"layer_{layer_id}"] = DetectorData(
                    positions=pixel_data.positions[layer_mask],
                    layers=pixel_data.layers[layer_mask],
                    tracks=pixel_data.tracks[layer_mask],
                    times=pixel_data.times[layer_mask],
                )
            else:
                # Create empty DetectorData for missing layers
                empty_tensor = torch.empty((0, 3), dtype=pixel_data.positions.dtype, 
                                         device=pixel_data.positions.device)
                empty_1d = torch.empty(0, dtype=pixel_data.layers.dtype,
                                     device=pixel_data.layers.device)
                layer_data[f"layer_{layer_id}"] = DetectorData(
                    positions=empty_tensor,
                    layers=empty_1d,
                    tracks=empty_1d,
                    times=empty_1d,
                )
        
        return layer_data

    def _create_edges(self, src_data: DetectorData, dst_data: DetectorData, 
                     config: EdgeConfig) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Create edges between source and destination nodes."""
        if len(src_data) == 0 or len(dst_data) == 0:
            return None, None

        # Generate all possible connections
        src_indices = torch.arange(len(src_data))
        dst_indices = torch.arange(len(dst_data))
        
        if config.same_type:
            # For same type connections (mppc-mppc), create all pairs excluding self-connections
            row = src_indices.repeat_interleave(len(dst_data))
            col = dst_indices.repeat(len(src_data))
            mask = row != col
            row, col = row[mask], col[mask]
        else:
            # For different type connections, create all possible pairs
            row = src_indices.repeat_interleave(len(dst_data))
            col = dst_indices.repeat(len(src_data))

        if row.numel() == 0:
            return None, None

        # Apply timing constraints for MPPC-MPPC connections
        if config.use_timing and row.numel() > 0:
            time_diff = (src_data.times[row] - dst_data.times[col]).abs()
            timing_mask = time_diff <= self.mppc_timing_cutoff
            row, col = row[timing_mask], col[timing_mask]

        if row.numel() == 0:
            return None, None

        edge_labels = self._create_edge_labels(src_data.tracks, dst_data.tracks, row, col)
        return torch.stack([row, col], dim=0), edge_labels

    def _create_layer_separated_graph(self, pixel_data: DetectorData, mppc_data: DetectorData,
                                    time_slice: torch.Tensor) -> Optional[HeteroData]:
        """Create layer-separated heterogeneous graph for a specific time slice."""
        # Filter pixel data for current time slice
        pixel_t = pixel_data.filter_by_time(time_slice)
        
        # Filter MPPC data (using 8ns time bins)
        mppc_cut_times = (mppc_data.times // 8) * 8
        mppc_t = DetectorData(
            positions=mppc_data.positions[mppc_cut_times == time_slice],
            layers=mppc_data.layers[mppc_cut_times == time_slice],
            tracks=mppc_data.tracks[mppc_cut_times == time_slice],
            times=mppc_data.times[mppc_cut_times == time_slice],
        )

        # Separate pixel data into layers
        layer_data = self._separate_pixel_layers(pixel_t)
        
        # Check if we have sufficient data - need connected path through the graph
        if not (len(layer_data["layer_2"]) > 0 and len(mppc_t) > 0 and len(layer_data["layer_3"]) > 0):
            return None

        # Create graph with node features
        graph = HeteroData()
        
        # Add layer nodes (only if they exist)
        for layer_name, data in layer_data.items():
            if len(data) > 0:
                graph[layer_name].x = data.positions

        # Add MPPC nodes (include timing information)
        if len(mppc_t) > 0:
            graph["mppc"].x = torch.cat([mppc_t.positions, mppc_t.times.unsqueeze(1)], dim=1)

        # Create data mapping for edge creation
        data_map = {**layer_data, "mppc": mppc_t}

        # Create edges for all configured types
        edges_added = 0
        for config in self.edge_configs:
            src_data = data_map[config.src_type]
            dst_data = data_map[config.dst_type]

            edge_index, edge_labels = self._create_edges(src_data, dst_data, config)
            if edge_index is not None:
                graph[config.edge_type].edge_index = edge_index
                graph[config.edge_type].edge_labels = edge_labels
                edges_added += 1

        return graph if edges_added > 0 else None

    def build_graphs_from_event(self, pixel: Union[torch.Tensor, any], 
                              mppc: Union[torch.Tensor, any], **kwargs) -> List[HeteroData]:
        """Build layer-separated heterogeneous graphs from a single event."""
        mppc = self._validate_tensor(mppc, "mppc")
        pixel = self._validate_tensor(pixel, "pixel")

        pixel_data = self._extract_detector_data(pixel)
        mppc_data = self._extract_detector_data(mppc)

        graphs = []
        for time_slice in torch.unique(pixel_data.times):
            graph = self._create_layer_separated_graph(pixel_data, mppc_data, time_slice)
            if graph is not None:
                graphs.append(graph)

        return graphs

    def get_node_types(self) -> List[str]:
        """Return list of node types in this graph."""
        return ["layer_1", "layer_2", "layer_3", "layer_4", "mppc"]

    def get_edge_types(self) -> List[Tuple[str, str, str]]:
        """Return list of edge types in this graph."""
        return [config.edge_type for config in self.edge_configs]


class CombinedGraphBuilder(GraphBuilderBase):
    """Builder for combined homogeneous graphs with both detector types."""

    def __init__(self, connect_layers: bool = True, mppc_timing_cutoff: float = 0.1):
        super().__init__(connect_layers)
        self.mppc_timing_cutoff = mppc_timing_cutoff

    def _create_combined_edges(self, combined_data: DetectorData, 
                             num_pixel: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Create edges for combined pixel+MPPC nodes."""
        if len(combined_data) < 2:
            return None, None

        row, col = generate_all_pairs(len(combined_data), len(combined_data), 
                                     include_self=False, same_type=True)

        if self.connect_layers and row.numel() > 0:
            # Complex layer logic from original code
            layer_diff_mask = (combined_data.layers[row] - combined_data.layers[col]).abs() <= 1

            # Exclude connections between layers 2 and 3
            layer_23_mask = ~(
                ((combined_data.layers[row] == 2) & (combined_data.layers[col] == 3)) |
                ((combined_data.layers[row] == 3) & (combined_data.layers[col] == 2))
            )

            # Exclude MPPC-MPPC connections with large time differences
            mppc_timing_mask = ~(
                (combined_data.layers[row] == 2.5) & 
                (combined_data.layers[col] == 2.5) & 
                (torch.abs(combined_data.times[row] - combined_data.times[col]) > self.mppc_timing_cutoff)
            )

            edge_mask = layer_diff_mask & layer_23_mask & mppc_timing_mask
            row, col = row[edge_mask], col[edge_mask]

        if row.numel() == 0:
            return None, None

        edge_labels = self._create_edge_labels(combined_data.tracks, combined_data.tracks, row, col)
        edge_index = torch.stack([row, col], dim=0)

        return edge_index, edge_labels

    def build_graphs_from_event(self, pixel: Union[torch.Tensor, any], 
                              mppc: Union[torch.Tensor, any], **kwargs) -> List[Data]:
        """Build combined homogeneous graphs from event data."""
        mppc = self._validate_tensor(mppc, "mppc")
        pixel = self._validate_tensor(pixel, "pixel")

        pixel_data = self._extract_detector_data(pixel)
        mppc_data = self._extract_detector_data(mppc)

        mppc_cut_times = (mppc_data.times // 8) * 8
        graphs = []

        for time_slice in torch.unique(pixel_data.times):
            pixel_t = pixel_data.filter_by_time(time_slice)
            mppc_mask = mppc_cut_times == time_slice

            if not pixel_t.has_sufficient_data() or mppc_mask.sum() < 2:
                continue

            # Create combined node features
            pixel_nodes = torch.cat([
                pixel_t.positions,
                torch.full((len(pixel_t), 1), 1, device=pixel_t.positions.device),  # Type flag
                torch.full((len(pixel_t), 1), time_slice, device=pixel_t.positions.device),
            ], dim=1)

            mppc_nodes = torch.cat([
                mppc_data.positions[mppc_mask],
                torch.full((mppc_mask.sum(), 1), 0, device=mppc_data.positions.device),  # Type flag
                mppc_data.times[mppc_mask].unsqueeze(1),
            ], dim=1)

            # Combine all data
            combined_nodes = torch.cat([pixel_nodes, mppc_nodes], dim=0)
            combined_data = DetectorData(
                positions=combined_nodes,
                tracks=torch.cat([pixel_t.tracks, mppc_data.tracks[mppc_mask]], dim=0),
                layers=torch.cat([pixel_t.layers, mppc_data.layers[mppc_mask]], dim=0),
                times=torch.cat([pixel_t.times, mppc_data.times[mppc_mask]], dim=0),
            )

            edge_index, edge_labels = self._create_combined_edges(combined_data, len(pixel_t))

            if edge_index is not None:
                graph = Data(
                    x=combined_nodes,
                    edge_index=edge_index, 
                    edge_labels=edge_labels
                )
                graphs.append(graph)

        return graphs