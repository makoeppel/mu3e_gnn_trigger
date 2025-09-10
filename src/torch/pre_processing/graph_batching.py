"""
Graph batching utilities for detector data processing from .npy files.

This module provides PyTorch Geometric compatible classes for building different types of graphs from detector events:
- HeteroGraphBuilder: Heterogeneous graphs with pixel and MPPC detectors
- CombinedGraphBuilder: Homogeneous graphs with type features (0=MPPC, 1=pixel)
- LayerSeparatedHeteroGraphBuilder: Layer-separated heterogeneous graphs (layer_1, layer_2, layer_3, layer_4, mppc)

Supports two processing modes:
1. Single slice mode: Create graphs of 8ns time slices for individual classification
2. Sequence mode: Create sequences of graphs with varying lengths for per-sequence classification
"""

import torch
import numpy as np
from torch_geometric.data import Data, Dataset, Batch, HeteroData, InMemoryDataset
from torch_geometric.loader import DataLoader
from typing import List, Tuple, Optional, Union, Dict, Any, Callable, Sequence
from dataclasses import dataclass
from abc import ABC, abstractmethod
import os
import os.path as osp
import pickle
from pathlib import Path


@dataclass
class DetectorData:
    """Container for detector data components."""

    positions: torch.Tensor
    layers: torch.Tensor
    tracks: torch.Tensor
    times: torch.Tensor
    track_truth: Optional[torch.Tensor] = None  # [px, py, pz, e, pdg]

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
            track_truth=(
                self.track_truth[mask] if self.track_truth is not None else None
            ),
        )

    def has_sufficient_data(self, min_nodes: int = 2) -> bool:
        """Check if data has sufficient nodes."""
        return len(self) >= min_nodes


@dataclass
class EventData:
    """Container for a complete event with spacetime and track label data."""

    spacetime: (
        torch.Tensor
    )  # [n_hits, n_features] where features are [x,y,z,(layer),time]
    track_labels: (
        torch.Tensor
    )  # [n_hits, 6] where features are [track_id, px, py, pz, e, pdg]
    has_layer_feature: bool = False

    def to_detector_data(self) -> DetectorData:
        """Convert EventData to DetectorData format."""
        # Extract positions (first 3 features)
        positions = self.spacetime[:, :3]

        # Extract layer information
        if self.has_layer_feature:
            layers = self.spacetime[:, 3]
            time_idx = 4
        else:
            # If no layer feature, infer from z-coordinate ranges or use default
            z_coords = positions[:, 2]
            # Simple layer assignment based on z-coordinate (adjust based on your detector geometry)
            layers = torch.ones(positions.size(0))  # Default to layer 1
            time_idx = 3

        # Extract time (last feature of spacetime)
        times = self.spacetime[:, time_idx]

        # Extract track information
        tracks = self.track_labels[:, 0]  # First column is track_id
        track_truth = self.track_labels[
            :, 1:
        ]  # Remaining columns are [px, py, pz, e, pdg]

        return DetectorData(
            positions=positions,
            layers=layers,
            tracks=tracks,
            times=times,
            track_truth=track_truth,
        )


@dataclass
class EdgeConfig:
    """Configuration for edge creation between detector types."""

    src_type: str
    dst_type: str
    edge_type: Tuple[str, str, str]
    same_type: bool
    use_timing: bool = False
    use_layers: bool = True


def generate_all_pairs(
    n_src: int, n_dst: int, include_self: bool = True, same_type: bool = False
):
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


def load_npy_data(
    spacetime_path: str, track_labels_path: str, has_layer_feature: bool = False
) -> List[EventData]:
    """
    Load data from .npy files and convert to EventData format.

    Args:
        spacetime_path: Path to spacetime .npy file [n_events, n_hits, n_features]
        track_labels_path: Path to track labels .npy file [n_events, n_hits, 6]
        has_layer_feature: Whether the spacetime data includes layer information

    Returns:
        List of EventData objects, one per event
    """
    spacetime_data = np.load(spacetime_path)
    track_labels_data = np.load(track_labels_path)

    if spacetime_data.shape[:2] != track_labels_data.shape[:2]:
        raise ValueError(
            "Spacetime and track labels must have matching event and hit dimensions"
        )

    events = []
    padding_value = -999  # Assuming same padding value as in root_to_numpy.py

    for i in range(spacetime_data.shape[0]):
        # Filter out padded entries (assuming padding value is -999)
        spacetime_event = spacetime_data[i]
        track_labels_event = track_labels_data[i]

        # Valid entries have time != padding_value (time is last feature)
        if has_layer_feature:
            valid_mask = spacetime_event[:, 4] != padding_value  # time at index 4
        else:
            valid_mask = spacetime_event[:, 3] != padding_value  # time at index 3

        if valid_mask.sum() > 0:  # Only include events with valid hits
            event_data = EventData(
                spacetime=torch.tensor(
                    spacetime_event[valid_mask], dtype=torch.float32
                ),
                track_labels=torch.tensor(
                    track_labels_event[valid_mask], dtype=torch.float32
                ),
                has_layer_feature=has_layer_feature,
            )
            events.append(event_data)

    return events


def group_time_slices_into_sequences(
    times: torch.Tensor, time_step: float = 8.0
) -> List[List[torch.Tensor]]:
    """
    Group time slices into sequences with varying lengths.

    Args:
        times: Unique time values in the event
        time_step: Time step between slices (8ns)

    Returns:
        List of sequences, where each sequence is a list of consecutive time slice values
    """
    sorted_times = torch.sort(times)[0]
    sequences = []

    if len(sorted_times) == 0:
        return sequences

    current_sequence = [sorted_times[0]]

    for i in range(1, len(sorted_times)):
        current_time = sorted_times[i]
        last_time = current_sequence[-1]

        # Check if this time slice is consecutive (within tolerance)
        if abs(current_time - last_time - time_step) < time_step * 0.1:
            current_sequence.append(current_time)
        else:
            # Start new sequence
            if len(current_sequence) >= 2:  # Only keep sequences with multiple slices
                sequences.append(current_sequence)
            current_sequence = [current_time]

    # Add the last sequence if it has sufficient length
    if len(current_sequence) >= 2:
        sequences.append(current_sequence)

    return sequences


class GraphBuilderBase(ABC):
    """Abstract base class for graph builders."""

    def __init__(self, connect_layers: bool = True, sequence_mode: bool = False):
        self.connect_layers = connect_layers
        self.sequence_mode = sequence_mode

    def _create_edge_labels(
        self,
        src_tracks: torch.Tensor,
        dst_tracks: torch.Tensor,
        row: torch.Tensor,
        col: torch.Tensor,
    ) -> torch.Tensor:
        """Create edge labels based on track matching."""
        return ((src_tracks[row] > 0) & (src_tracks[row] == dst_tracks[col])).float()

    @abstractmethod
    def build_graphs_from_event(
        self, pixel_event: EventData, mppc_event: EventData
    ) -> Union[List[Union[Data, HeteroData]], List[List[Union[Data, HeteroData]]]]:
        """Build graphs from event data. Returns single graphs or sequences based on mode."""
        pass

    def _build_sequence_graphs(
        self, pixel_data: DetectorData, mppc_data: DetectorData
    ) -> List[List[Union[Data, HeteroData]]]:
        """Build sequences of graphs with varying lengths."""
        unique_times = torch.unique(pixel_data.times)
        sequences = group_time_slices_into_sequences(unique_times)

        sequence_graphs = []
        for time_sequence in sequences:
            graphs_in_sequence = []
            for time_slice in time_sequence:
                time_graphs = self._create_graphs_for_time_slice(
                    pixel_data, mppc_data, time_slice
                )
                graphs_in_sequence.extend(time_graphs)

            if len(graphs_in_sequence) >= 1:  # Include sequences with at least 1 graph
                sequence_graphs.append(graphs_in_sequence)

        return sequence_graphs

    @abstractmethod
    def _create_graphs_for_time_slice(
        self,
        pixel_data: DetectorData,
        mppc_data: DetectorData,
        time_slice: torch.Tensor,
    ) -> List[Union[Data, HeteroData]]:
        """Create graphs for a specific time slice."""
        pass

    @abstractmethod
    def get_edge_types(self) -> List[Tuple[str, str, str]]:
        """Return list of edge types in the graph."""
        pass

    @abstractmethod
    def get_node_dims(self) -> Dict[str, int]:
        """Return dictionary of node feature dimensions per node type."""
        pass


class HeteroGraphBuilder(GraphBuilderBase):
    """Builder for heterogeneous graphs with pixel and MPPC node types."""

    def __init__(
        self,
        connect_layers: bool = True,
        mppc_timing_cutoff: float = 0.2,
        sequence_mode: bool = False,
    ):
        super().__init__(connect_layers, sequence_mode)
        self.mppc_timing_cutoff = mppc_timing_cutoff
        self.edge_configs = [
            EdgeConfig("pixel", "pixel", ("pixel", "to", "pixel"), True, False, True),
            EdgeConfig("mppc", "mppc", ("mppc", "to", "mppc"), True, True, False),
            EdgeConfig("pixel", "mppc", ("pixel", "to", "mppc"), False, False, True),
            EdgeConfig("mppc", "pixel", ("mppc", "to", "pixel"), False, False, True),
        ]

    def _create_edges(
        self, src_data: DetectorData, dst_data: DetectorData, config: EdgeConfig
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Create edges between source and destination nodes."""
        if len(src_data) == 0 or len(dst_data) == 0:
            return None, None

        row, col = generate_all_pairs(
            len(src_data), len(dst_data), include_self=False, same_type=config.same_type
        )

        if row.numel() == 0:
            return None, None

        # Apply layer constraints
        if self.connect_layers and config.use_layers:
            layer_diff = (src_data.layers[row] - dst_data.layers[col]).abs()
            layer_mask = (layer_diff <= 1) & ~(
                ((src_data.layers[row] == 2) & (dst_data.layers[col] == 3))
                | ((src_data.layers[row] == 3) & (dst_data.layers[col] == 2))
            )
            row, col = row[layer_mask], col[layer_mask]

        # Apply timing constraints for MPPC
        if config.use_timing and row.numel() > 0:
            time_diff = (src_data.times[row] - dst_data.times[col]).abs()
            timing_mask = time_diff <= self.mppc_timing_cutoff
            row, col = row[timing_mask], col[timing_mask]

        if row.numel() == 0:
            return None, None

        edge_labels = self._create_edge_labels(
            src_data.tracks, dst_data.tracks, row, col
        )
        return torch.stack([row, col], dim=0), edge_labels

    def _create_graphs_for_time_slice(
        self,
        pixel_data: DetectorData,
        mppc_data: DetectorData,
        time_slice: torch.Tensor,
    ) -> List[HeteroData]:
        """Create heterogeneous graphs for a specific time slice."""
        # Filter data for current time slice
        pixel_t = pixel_data.filter_by_time(time_slice)
        mppc_cut_times = (mppc_data.times // 8) * 8
        mppc_t = DetectorData(
            positions=mppc_data.positions[mppc_cut_times == time_slice],
            layers=mppc_data.layers[mppc_cut_times == time_slice],
            tracks=mppc_data.tracks[mppc_cut_times == time_slice],
            times=mppc_data.times[mppc_cut_times == time_slice],
            track_truth=(
                mppc_data.track_truth[mppc_cut_times == time_slice]
                if mppc_data.track_truth is not None
                else None
            ),
        )

        # Skip if insufficient data
        if not pixel_t.has_sufficient_data() or not mppc_t.has_sufficient_data():
            return []

        # Create graph with node features
        graph = HeteroData()
        graph["pixel"].x = pixel_t.positions
        graph["mppc"].x = torch.cat(
            [mppc_t.positions, mppc_t.times.unsqueeze(1)], dim=1
        )

        # Add track truth information if available
        if pixel_t.track_truth is not None:
            graph["pixel"].track_truth = pixel_t.track_truth
        if mppc_t.track_truth is not None:
            graph["mppc"].track_truth = mppc_t.track_truth

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

        return [graph] if edges_created >= 1 else []

    def build_graphs_from_event(
        self, pixel_event: EventData, mppc_event: EventData
    ) -> Union[List[HeteroData], List[List[HeteroData]]]:
        """Build heterogeneous graphs from pixel and MPPC event data."""
        pixel_data = pixel_event.to_detector_data()
        mppc_data = mppc_event.to_detector_data()

        if self.sequence_mode:
            return self._build_sequence_graphs(pixel_data, mppc_data)
        else:
            graphs = []
            for time_slice in torch.unique(pixel_data.times):
                time_graphs = self._create_graphs_for_time_slice(
                    pixel_data, mppc_data, time_slice
                )
                graphs.extend(time_graphs)
            return graphs

    def get_edge_types(self) -> List[Tuple[str, str, str]]:
        """Return list of edge types in the graph."""
        return [config.edge_type for config in self.edge_configs]

    def get_node_dims(self) -> Dict[str, int]:
        """Return dictionary of node feature dimensions per node type."""
        return {
            "pixel": 3,  # x, y, z
            "mppc": 4,  # x, y, z, time
        }

    def get_viable_event_indices(
        self, pixel_events: List[EventData], mppc_events: List[EventData]
    ) -> List[int]:
        """Identify indices of events with sufficient data for graph construction."""
        viable_indices = []
        for idx, (pixel_event, mppc_event) in enumerate(zip(pixel_events, mppc_events)):
            pixel_data = pixel_event.to_detector_data()
            mppc_data = mppc_event.to_detector_data()
            if pixel_data.has_sufficient_data() and mppc_data.has_sufficient_data():
                viable_indices.append(idx)
        return viable_indices


class CombinedGraphBuilder(GraphBuilderBase):
    """Builder for combined homogeneous graphs with type features (0=MPPC, 1=pixel)."""

    def __init__(
        self,
        connect_layers: bool = True,
        mppc_timing_cutoff: float = 0.1,
        sequence_mode: bool = False,
    ):
        super().__init__(connect_layers, sequence_mode)
        self.mppc_timing_cutoff = mppc_timing_cutoff

    def _create_combined_edges(
        self, combined_data: DetectorData, num_pixel: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Create edges for combined pixel+MPPC nodes."""
        if len(combined_data) < 2:
            return None, None

        row, col = generate_all_pairs(
            len(combined_data), len(combined_data), include_self=False, same_type=True
        )

        if self.connect_layers and row.numel() > 0:
            # Complex layer logic from original code
            layer_diff_mask = (
                combined_data.layers[row] - combined_data.layers[col]
            ).abs() <= 1

            # Exclude connections between layers 2 and 3
            layer_23_mask = ~(
                ((combined_data.layers[row] == 2) & (combined_data.layers[col] == 3))
                | ((combined_data.layers[row] == 3) & (combined_data.layers[col] == 2))
            )

            # Exclude MPPC-MPPC connections with large time differences
            mppc_timing_mask = ~(
                (combined_data.layers[row] == 2.5)
                & (combined_data.layers[col] == 2.5)
                & (
                    torch.abs(combined_data.times[row] - combined_data.times[col])
                    > self.mppc_timing_cutoff
                )
            )

            edge_mask = layer_diff_mask & layer_23_mask & mppc_timing_mask
            row, col = row[edge_mask], col[edge_mask]

        if row.numel() == 0:
            return None, None

        edge_labels = self._create_edge_labels(
            combined_data.tracks, combined_data.tracks, row, col
        )
        edge_index = torch.stack([row, col], dim=0)

        return edge_index, edge_labels

    def _create_graphs_for_time_slice(
        self,
        pixel_data: DetectorData,
        mppc_data: DetectorData,
        time_slice: torch.Tensor,
    ) -> List[Data]:
        """Create combined homogeneous graphs for a specific time slice."""
        pixel_t = pixel_data.filter_by_time(time_slice)
        mppc_cut_times = (mppc_data.times // 8) * 8
        mppc_mask = mppc_cut_times == time_slice

        if not pixel_t.has_sufficient_data() or mppc_mask.sum() < 2:
            return []

        # Create combined node features with type indicator
        pixel_nodes = torch.cat(
            [
                pixel_t.positions,  # x, y, z
                pixel_t.layers.unsqueeze(1),  # layer
                torch.ones(
                    (len(pixel_t), 1), device=pixel_t.positions.device
                ),  # Type: 1=pixel
                torch.full(
                    (len(pixel_t), 1), time_slice, device=pixel_t.positions.device
                ),  # time
            ],
            dim=1,
        )

        mppc_nodes = torch.cat(
            [
                mppc_data.positions[mppc_mask],  # x, y, z
                mppc_data.layers[mppc_mask].unsqueeze(1),  # layer (2.5 for MPPC)
                torch.zeros(
                    (mppc_mask.sum(), 1), device=mppc_data.positions.device
                ),  # Type: 0=MPPC
                mppc_data.times[mppc_mask].unsqueeze(1),  # actual time
            ],
            dim=1,
        )

        # Combine all data
        combined_nodes = torch.cat([pixel_nodes, mppc_nodes], dim=0)
        combined_data = DetectorData(
            positions=combined_nodes,  # Now includes [x, y, z, layer, type, time]
            tracks=torch.cat([pixel_t.tracks, mppc_data.tracks[mppc_mask]], dim=0),
            layers=torch.cat([pixel_t.layers, mppc_data.layers[mppc_mask]], dim=0),
            times=torch.cat([pixel_t.times, mppc_data.times[mppc_mask]], dim=0),
        )

        edge_index, edge_labels = self._create_combined_edges(
            combined_data, len(pixel_t)
        )

        graphs = []
        if edge_index is not None:
            graph = Data(
                x=combined_nodes, edge_index=edge_index, edge_labels=edge_labels
            )

            # Add track truth if available
            if pixel_t.track_truth is not None and mppc_data.track_truth is not None:
                combined_track_truth = torch.cat(
                    [pixel_t.track_truth, mppc_data.track_truth[mppc_mask]], dim=0
                )
                graph.track_truth = combined_track_truth

            graphs.append(graph)

        return graphs

    def build_graphs_from_event(
        self, pixel_event: EventData, mppc_event: EventData
    ) -> Union[List[Data], List[List[Data]]]:
        """Build combined homogeneous graphs from event data."""
        pixel_data = pixel_event.to_detector_data()
        mppc_data = mppc_event.to_detector_data()

        if self.sequence_mode:
            return self._build_sequence_graphs(pixel_data, mppc_data)
        else:
            graphs = []
            for time_slice in torch.unique(pixel_data.times):
                time_graphs = self._create_graphs_for_time_slice(
                    pixel_data, mppc_data, time_slice
                )
                graphs.extend(time_graphs)
            return graphs


class LayerSeparatedHeteroGraphBuilder(GraphBuilderBase):
    """
    Builder for layer-separated heterogeneous graphs with connectivity:
    layer_1 <-> layer_2 <-> mppc <-> mppc <-> layer_3 <-> layer_4
    """

    def __init__(
        self,
        connect_layers: bool = True,
        mppc_timing_cutoff: float = 0.2,
        sequence_mode: bool = False,
    ):
        super().__init__(connect_layers, sequence_mode)
        self.mppc_timing_cutoff = mppc_timing_cutoff

        # Define the connectivity pattern
        self.edge_configs = [
            # Layer connections
            EdgeConfig(
                "layer_1", "layer_2", ("layer_1", "to", "layer_2"), False, False, False
            ),
            EdgeConfig(
                "layer_2", "layer_1", ("layer_2", "to", "layer_1"), False, False, False
            ),
            # Layer 2 to MPPC connections
            EdgeConfig(
                "layer_2", "mppc", ("layer_2", "to", "mppc"), False, False, False
            ),
            EdgeConfig(
                "mppc", "layer_2", ("mppc", "to", "layer_2"), False, False, False
            ),
            # MPPC internal connections
            EdgeConfig("mppc", "mppc", ("mppc", "to", "mppc"), True, True, False),
            # MPPC to Layer 3 connections
            EdgeConfig(
                "mppc", "layer_3", ("mppc", "to", "layer_3"), False, False, False
            ),
            EdgeConfig(
                "layer_3", "mppc", ("layer_3", "to", "mppc"), False, False, False
            ),
            # Layer 3 to Layer 4 connections
            EdgeConfig(
                "layer_3", "layer_4", ("layer_3", "to", "layer_4"), False, False, False
            ),
            EdgeConfig(
                "layer_4", "layer_3", ("layer_4", "to", "layer_3"), False, False, False
            ),
            EdgeConfig(
                "layer_4", "layer_4", ("layer_4", "to", "layer_4"), True, False, False
            ),
        ]

    def _separate_pixel_layers(
        self, pixel_data: DetectorData
    ) -> Dict[str, DetectorData]:
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
                    track_truth=(
                        pixel_data.track_truth[layer_mask]
                        if pixel_data.track_truth is not None
                        else None
                    ),
                )
            else:
                # Create empty DetectorData for missing layers
                empty_tensor = torch.empty(
                    (0, 3),
                    dtype=pixel_data.positions.dtype,
                    device=pixel_data.positions.device,
                )
                empty_1d = torch.empty(
                    0, dtype=pixel_data.layers.dtype, device=pixel_data.layers.device
                )
                layer_data[f"layer_{layer_id}"] = DetectorData(
                    positions=empty_tensor,
                    layers=empty_1d,
                    tracks=empty_1d,
                    times=empty_1d,
                    track_truth=None,
                )

        return layer_data

    def _create_edges(
        self, src_data: DetectorData, dst_data: DetectorData, config: EdgeConfig
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
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

        edge_labels = self._create_edge_labels(
            src_data.tracks, dst_data.tracks, row, col
        )
        return torch.stack([row, col], dim=0), edge_labels

    def _create_graphs_for_time_slice(
        self,
        pixel_data: DetectorData,
        mppc_data: DetectorData,
        time_slice: torch.Tensor,
    ) -> List[HeteroData]:
        """Create layer-separated heterogeneous graphs for a specific time slice."""
        # Filter pixel data for current time slice
        pixel_t = pixel_data.filter_by_time(time_slice)

        # Filter MPPC data (using 8ns time bins)
        mppc_cut_times = (mppc_data.times // 8) * 8
        mppc_t = DetectorData(
            positions=mppc_data.positions[mppc_cut_times == time_slice],
            layers=mppc_data.layers[mppc_cut_times == time_slice],
            tracks=mppc_data.tracks[mppc_cut_times == time_slice],
            times=mppc_data.times[mppc_cut_times == time_slice],
            track_truth=(
                mppc_data.track_truth[mppc_cut_times == time_slice]
                if mppc_data.track_truth is not None
                else None
            ),
        )

        # Separate pixel data into layers
        layer_data = self._separate_pixel_layers(pixel_t)

        # Check if we have sufficient data - need connected path through the graph
        if not (
            len(layer_data["layer_2"]) > 0
            and len(mppc_t) > 0
            and len(layer_data["layer_3"]) > 0
        ):
            return []

        # Create graph with node features
        graph = HeteroData()

        # Add layer nodes (only if they exist)
        for layer_name, data in layer_data.items():
            if len(data) > 0:
                graph[layer_name].x = data.positions
                if data.track_truth is not None:
                    graph[layer_name].track_truth = data.track_truth

        # Add MPPC nodes (include timing information)
        if len(mppc_t) > 0:
            graph["mppc"].x = torch.cat(
                [mppc_t.positions, mppc_t.times.unsqueeze(1)], dim=1
            )
            if mppc_t.track_truth is not None:
                graph["mppc"].track_truth = mppc_t.track_truth

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

        return [graph] if edges_added > 0 else []

    def build_graphs_from_event(
        self, pixel_event: EventData, mppc_event: EventData
    ) -> Union[List[HeteroData], List[List[HeteroData]]]:
        """Build layer-separated heterogeneous graphs from event data."""
        pixel_data = pixel_event.to_detector_data()
        mppc_data = mppc_event.to_detector_data()

        if self.sequence_mode:
            return self._build_sequence_graphs(pixel_data, mppc_data)
        else:
            graphs = []
            for time_slice in torch.unique(pixel_data.times):
                time_graphs = self._create_graphs_for_time_slice(
                    pixel_data, mppc_data, time_slice
                )
                graphs.extend(time_graphs)
            return graphs

    def get_viable_event_indices(
        self,
        pixel_events: List[EventData],
        mppc_events: List[EventData],
        min_nodes: int = 2,
    ) -> List[int]:
        """Get indices of events that have sufficient data to form graphs."""
        viable_indices = []
        for idx, (pixel_event, mppc_event) in enumerate(zip(pixel_events, mppc_events)):
            pixel_data = pixel_event.to_detector_data()
            mppc_data = mppc_event.to_detector_data()
            layer_data = self._separate_pixel_layers(pixel_data)

            # Check if we have sufficient data - need connected path through the graph
            if (
                len(layer_data["layer_2"]) >= min_nodes
                and len(mppc_data) >= min_nodes
                and len(layer_data["layer_3"]) >= min_nodes
            ):
                viable_indices.append(idx)
        return viable_indices

    def get_edge_types(self) -> List[Tuple[str, str, str]]:
        """Return list of edge types in the graph."""
        return [config.edge_type for config in self.edge_configs]

    def get_node_dims(self) -> Dict[str, int]:
        """Return dictionary of node feature dimensions per node type."""
        return {
            "layer_1": 3,  # x, y, z
            "layer_2": 3,  # x, y, z
            "mppc": 4,  # x, y, z, time
            "layer_3": 3,  # x, y, z
            "layer_4": 3,  # x, y, z
        }


class DetectorDataset(Dataset):
    """PyTorch Geometric Dataset for detector data from .npy files."""

    def __init__(
        self,
        pixel_spacetime_path: str,
        pixel_track_labels_path: str,
        mppc_spacetime_path: str,
        mppc_track_labels_path: str,
        graph_builder: GraphBuilderBase,
        has_layer_feature: bool = False,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        cache_dir: Optional[str] = None,
    ):

        self.pixel_spacetime_path = pixel_spacetime_path
        self.pixel_track_labels_path = pixel_track_labels_path
        self.mppc_spacetime_path = mppc_spacetime_path
        self.mppc_track_labels_path = mppc_track_labels_path
        self.has_layer_feature = has_layer_feature
        self.graph_builder = graph_builder

        # Load event data
        self.pixel_events = load_npy_data(
            pixel_spacetime_path, pixel_track_labels_path, has_layer_feature
        )
        self.mppc_events = load_npy_data(
            mppc_spacetime_path, mppc_track_labels_path, has_layer_feature
        )

        if len(self.pixel_events) != len(self.mppc_events):
            raise ValueError("Number of pixel and MPPC events must match")

        super().__init__(
            transform=transform, pre_transform=pre_transform, pre_filter=pre_filter
        )
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, "detector_dataset_cache.pt")

    def precompute(self, cache_path: str):
        """Precompute and cache the dataset."""
        data_list = []
        for idx in range(len(self)):
            graphs = self[idx]
            if isinstance(graphs, list):
                data_list.extend(graphs)
            else:
                data_list.append(graphs)

        torch.save(data_list, cache_path)
        print(f"Dataset cached at {cache_path}")

    def len(self) -> int:
        return len(self.pixel_events)

    def get(
        self, idx: int
    ) -> Union[List[Union[Data, HeteroData]], List[List[Union[Data, HeteroData]]]]:
        """Get graphs for a specific event."""
        pixel_event = self.pixel_events[idx]
        mppc_event = self.mppc_events[idx]
        return self.graph_builder.build_graphs_from_event(pixel_event, mppc_event)

    def get_node_dims(self) -> Optional[Dict[str, int]]:
        """Get node feature dimensions if available."""
        if hasattr(self.graph_builder, "get_node_dims"):
            return self.graph_builder.get_node_dims()
        return None

    def get_edge_types(self) -> Optional[List[Tuple[str, str, str]]]:
        """Get edge types if available."""
        if hasattr(self.graph_builder, "get_edge_types"):
            return self.graph_builder.get_edge_types()
        return None


class SequenceDataset(Dataset):
    """Dataset that returns sequences of graphs with varying lengths."""

    def __init__(self, base_dataset: DetectorDataset):
        self.base_dataset = base_dataset
        self.sequences = []

        # Pre-compute all sequences
        for i in range(len(base_dataset)):
            graphs = base_dataset[i]
            if len(graphs) > 0 and isinstance(graphs[0], list):  # Sequence mode
                self.sequences.extend(graphs)

        super().__init__()

    def len(self) -> int:
        return len(self.sequences)

    def get(self, idx: int) -> List[Union[Data, HeteroData]]:
        return self.sequences[idx]


# Factory functions for creating datasets
def create_hetero_dataset(
    prefix: str,
    has_layer_feature: bool = False,
    sequence_mode: bool = False,
    mppc_timing_cutoff: float = 0.2,
) -> DetectorDataset:
    """Create a dataset for heterogeneous graphs."""
    builder = HeteroGraphBuilder(
        sequence_mode=sequence_mode, mppc_timing_cutoff=mppc_timing_cutoff
    )
    pixel_spacetime_path = f"{prefix}_pixel_spacetime.npy"
    pixel_track_labels_path = f"{prefix}_pixel_track_labels.npy"
    mppc_spacetime_path = f"{prefix}_mppc_spacetime.npy"
    mppc_track_labels_path = f"{prefix}_mppc_track_labels.npy"
    return DetectorDataset(
        pixel_spacetime_path=pixel_spacetime_path,
        pixel_track_labels_path=pixel_track_labels_path,
        mppc_spacetime_path=mppc_spacetime_path,
        mppc_track_labels_path=mppc_track_labels_path,
        has_layer_feature=has_layer_feature,
        graph_builder=builder,
    )


def create_combined_dataset(
    prefix: str,
    has_layer_feature: bool = False,
    sequence_mode: bool = False,
    mppc_timing_cutoff: float = 0.1,
) -> DetectorDataset:
    """Create a dataset for combined homogeneous graphs with type features."""
    builder = CombinedGraphBuilder(
        sequence_mode=sequence_mode, mppc_timing_cutoff=mppc_timing_cutoff
    )
    pixel_spacetime_path = f"{prefix}_pixel_spacetime.npy"
    pixel_track_labels_path = f"{prefix}_pixel_track_labels.npy"
    mppc_spacetime_path = f"{prefix}_mppc_spacetime.npy"
    mppc_track_labels_path = f"{prefix}_mppc_track_labels.npy"

    return DetectorDataset(
        pixel_spacetime_path=pixel_spacetime_path,
        pixel_track_labels_path=pixel_track_labels_path,
        mppc_spacetime_path=mppc_spacetime_path,
        mppc_track_labels_path=mppc_track_labels_path,
        has_layer_feature=has_layer_feature,
        graph_builder=builder,
    )


def create_layer_separated_dataset(
    prefix: str,
    has_layer_feature: bool = False,
    sequence_mode: bool = False,
    mppc_timing_cutoff: float = 0.2,
) -> DetectorDataset:
    """Create a dataset for layer-separated heterogeneous graphs."""
    builder = LayerSeparatedHeteroGraphBuilder(
        sequence_mode=sequence_mode, mppc_timing_cutoff=mppc_timing_cutoff
    )
    pixel_spacetime_path = f"{prefix}_pixel_spacetime.npy"
    pixel_track_labels_path = f"{prefix}_pixel_track_labels.npy"
    mppc_spacetime_path = f"{prefix}_mppc_spacetime.npy"
    mppc_track_labels_path = f"{prefix}_mppc_track_labels.npy"

    return DetectorDataset(
        pixel_spacetime_path=pixel_spacetime_path,
        pixel_track_labels_path=pixel_track_labels_path,
        mppc_spacetime_path=mppc_spacetime_path,
        mppc_track_labels_path=mppc_track_labels_path,
        has_layer_feature=has_layer_feature,
        graph_builder=builder,
    )
