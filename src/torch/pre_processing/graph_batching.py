"""
Graph batching utilities for detector data processing from .npy files.

This module provides PyTorch Geometric compatible classes for building different types of graphs from detector events:
- HeteroGraphBuilder: Heterogeneous graphs with pixel and MPPC detectors
- CombinedGraphBuilder: Homogeneous graphs with type features (0=MPPC, 1=pixel)
- LayerSeparatedHeteroGraphBuilder: Layer-separated heterogeneous graphs (layer_1, layer_2, layer_3, layer_4, mppc)

Supports three processing modes:
1. Single slice mode: Create graphs of 8ns time slices for individual classification
2. Sequence mode: Create sequences of graphs with varying lengths for per-sequence classification
3. Whole event mode: Create single graphs from entire events with 8ns timing constraints on edges
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
    pixel_spacetime_path: str,
    pixel_track_labels_path: str,
    mppc_spacetime_path: str,
    mppc_track_labels_path: str,
    has_layer_feature: bool = False,
    max_events: Optional[int] = None,
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
    pixel_spacetime_data = np.load(pixel_spacetime_path)
    pixel_track_labels_data = np.load(pixel_track_labels_path)
    mppc_spacetime_data = np.load(mppc_spacetime_path)
    mppc_track_labels_data = np.load(mppc_track_labels_path)

    if len(pixel_spacetime_data) != len(mppc_spacetime_data):
        raise ValueError("Mismatch in number of events between pixel and mppc data")
    if len(pixel_track_labels_data) != len(mppc_track_labels_data):
        raise ValueError("Mismatch in number of events between pixel and mppc data")
    if len(pixel_spacetime_data) != len(pixel_track_labels_data):
        raise ValueError(
            "Mismatch in number of events between pixel spacetime and track labels"
        )

    padding_value = -999  # Assuming same padding value as in root_to_numpy.py

    pixel_data = []
    mppc_data = []

    for i in range(len(pixel_spacetime_data)):
        # Filter out padded entries (assuming padding value is -999)
        pixel_spacetime_event = pixel_spacetime_data[i]
        pixel_track_labels_event = pixel_track_labels_data[i]
        mppc_spacetime_event = mppc_spacetime_data[i]
        mppc_track_labels_event = mppc_track_labels_data[i]

        # Valid entries have time != padding_value (time is last feature)
        valid_mask_pixel = pixel_spacetime_event[:, -1] != padding_value
        valid_mask_mppc = mppc_spacetime_event[:, -1] != padding_value

        if (valid_mask_mppc.sum() > 1) & (
            valid_mask_pixel.sum() > 1
        ):  # Only include events with valid hits
            pixel_event = EventData(
                spacetime=torch.tensor(
                    pixel_spacetime_event[valid_mask_pixel], dtype=torch.float
                ),
                track_labels=torch.tensor(
                    pixel_track_labels_event[valid_mask_pixel], dtype=torch.float
                ),
                has_layer_feature=has_layer_feature,
            )
            mppc_event = EventData(
                spacetime=torch.tensor(
                    mppc_spacetime_event[valid_mask_mppc], dtype=torch.float
                ),
                track_labels=torch.tensor(
                    mppc_track_labels_event[valid_mask_mppc], dtype=torch.float
                ),
                has_layer_feature=has_layer_feature,
            )
            pixel_data.append(pixel_event)
            mppc_data.append(mppc_event)
            if max_events is not None and len(pixel_data) >= max_events:
                break
    return pixel_data, mppc_data


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

    def __init__(
        self,
        connect_layers: bool = True,
        sequence_mode: bool = False,
        whole_event_mode: bool = False,
        timing_cutoff: float = 8.0,
    ):
        self.connect_layers = connect_layers
        self.sequence_mode = sequence_mode
        self.whole_event_mode = whole_event_mode
        self.timing_cutoff = timing_cutoff

        if sequence_mode and whole_event_mode:
            raise ValueError("Cannot enable both sequence_mode and whole_event_mode")

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
    def _create_whole_event_graph(
        self,
        pixel_data: DetectorData,
        mppc_data: DetectorData,
    ) -> List[Union[Data, HeteroData]]:
        """Create a single graph from the entire event with timing constraints."""
        pass

    @abstractmethod
    def get_edge_types(self) -> List[Tuple[str, str, str]]:
        """Return list of edge types in the graph."""
        pass

    @abstractmethod
    def get_node_dims(self) -> Dict[str, int]:
        """Return dictionary of node feature dimensions per node type."""
        pass

    def build_graphs_from_event(
        self, pixel_event: EventData, mppc_event: EventData
    ) -> Union[List[HeteroData], List[List[HeteroData]]]:
        pixel_data = pixel_event.to_detector_data()
        mppc_data = mppc_event.to_detector_data()

        if self.whole_event_mode:
            return self._create_whole_event_graph(pixel_data, mppc_data)
        elif self.sequence_mode:
            return self._build_sequence_graphs(pixel_data, mppc_data)
        else:
            graphs = []
            for time_slice in torch.unique(pixel_data.times):
                time_graphs = self._create_graphs_for_time_slice(
                    pixel_data, mppc_data, time_slice
                )
                graphs.extend(time_graphs)
            return graphs


class HeteroGraphBuilder(GraphBuilderBase):
    """Builder for heterogeneous graphs with pixel and MPPC node types."""

    def __init__(
        self,
        connect_layers: bool = True,
        mppc_timing_cutoff: float = 0.2,
        sequence_mode: bool = False,
        whole_event_mode: bool = False,
        timing_cutoff: float = 8.0,
    ):
        super().__init__(connect_layers, sequence_mode, whole_event_mode, timing_cutoff)
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

        # Apply timing constraints
        if row.numel() > 0:
            time_diff = (src_data.times[row] - dst_data.times[col]).abs()

            if self.whole_event_mode and not config.use_timing:
                # In whole event mode, apply global timing cutoff
                timing_mask = time_diff <= self.timing_cutoff
            elif config.use_timing:
                # In slice mode, apply MPPC-specific timing cutoff
                timing_mask = time_diff <= self.mppc_timing_cutoff
            else:
                # No timing constraint for this edge type in slice mode
                timing_mask = torch.ones_like(time_diff, dtype=torch.bool)

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
        graph["pixel"].x = torch.cat(
            [pixel_t.positions, pixel_t.layers.unsqueeze(1)], dim=1
        )
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

    def _create_whole_event_graph(
        self,
        pixel_data: DetectorData,
        mppc_data: DetectorData,
    ) -> List[HeteroData]:
        """Create a single graph from the entire event with timing constraints."""
        # Skip if insufficient data
        if not pixel_data.has_sufficient_data() or not mppc_data.has_sufficient_data():
            return []

        # Create graph with node features
        graph = HeteroData()
        graph["pixel"].x = torch.cat(
            [pixel_data.positions, pixel_data.times.unsqueeze(1), pixel_data.layers.unsqueeze(1)],
            dim=1,
        )
        graph["mppc"].x = torch.cat(
            [mppc_data.positions, mppc_data.times.unsqueeze(1)], dim=1
        )

        # Add track truth information if available
        if pixel_data.track_truth is not None:
            graph["pixel"].track_truth = pixel_data.track_truth
        if mppc_data.track_truth is not None:
            graph["mppc"].track_truth = mppc_data.track_truth

        # Create edges for all configured types
        data_map = {"pixel": pixel_data, "mppc": mppc_data}
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

    def get_edge_types(self) -> List[Tuple[str, str, str]]:
        """Return list of edge types in the graph."""
        return [config.edge_type for config in self.edge_configs]

    def get_node_dims(self) -> Dict[str, int]:
        """Return dictionary of node feature dimensions per node type."""
        if self.whole_event_mode:
            return {"pixel": 5, "mppc": 4}  # [x,y,z,time,layer] and [x,y,z,time]
        else:
            return {"pixel": 4, "mppc": 4}  # [x,y,z,layer] and [x,y,z,time]

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
        whole_event_mode: bool = False,
        timing_cutoff: float = 8.0,
    ):
        super().__init__(connect_layers, sequence_mode, whole_event_mode, timing_cutoff)
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

        # Apply timing constraints
        if row.numel() > 0:
            time_diff = (src_data.times[row] - dst_data.times[col]).abs()

            if self.whole_event_mode:
                # In whole event mode, apply global timing cutoff to all edges
                timing_mask = time_diff <= self.timing_cutoff
            elif config.use_timing:
                # In slice mode, apply MPPC-specific timing cutoff
                timing_mask = time_diff <= self.mppc_timing_cutoff
            else:
                # No timing constraint for this edge type in slice mode
                timing_mask = torch.ones_like(time_diff, dtype=torch.bool)

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

    def _create_whole_event_graph(
        self,
        pixel_data: DetectorData,
        mppc_data: DetectorData,
    ) -> List[HeteroData]:
        """Create layer-separated heterogeneous graphs for the entire event."""
        # Separate pixel data into layers
        layer_data = self._separate_pixel_layers(pixel_data)

        # Check if we have sufficient data - need connected path through the graph
        if not (
            len(layer_data["layer_2"]) > 0
            and len(mppc_data) > 0
            and len(layer_data["layer_3"]) > 0
        ):
            return []

        # Create graph with node features
        graph = HeteroData()

        # Add layer nodes (only if they exist)
        for layer_name, data in layer_data.items():
            if len(data) > 0:
                graph[layer_name].x = torch.cat([data.positions, data.times.unsqueeze(1)], dim=1
                )
                if data.track_truth is not None:
                    graph[layer_name].track_truth = data.track_truth

        # Add MPPC nodes (include timing information)
        if len(mppc_data) > 0:
            graph["mppc"].x = torch.cat(
                [mppc_data.positions, mppc_data.times.unsqueeze(1)], dim=1
            )
            if mppc_data.track_truth is not None:
                graph["mppc"].track_truth = mppc_data.track_truth

        # Create data mapping for edge creation
        data_map = {**layer_data, "mppc": mppc_data}

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
        if self.whole_event_mode:
            return {
                "layer_1": 3,
                "layer_2": 3,
                "mppc": 4,  # [x,y,z,time]
                "layer_3": 3,
                "layer_4": 3,
            }
        else:
            return {
                "layer_1": 3,
                "layer_2": 3,
                "mppc": 4,  # [x,y,z,time]
                "layer_3": 3,
                "layer_4": 3,
            }


class DetectorDataset(Dataset):
    """Dataset for detector graphs from .npy files using a specified graph builder."""

    def __init__(
        self,
        pixel_events,
        mppc_events,
        graph_builder: Optional[GraphBuilderBase] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        super().__init__(None, transform, pre_transform, pre_filter)
        self.length = 0
        self.graphs = []
        if len(pixel_events) != len(mppc_events):
            raise ValueError("Pixel and MPPC data must have the same number of events")

        if graph_builder is None:
            self.graph_builder = HeteroGraphBuilder(sequence_mode=False)
        else:
            self.graph_builder = graph_builder

        # Precompute all graphs
        self._prepare_graphs(pixel_events, mppc_events)

    def _prepare_graphs(self, pixel_events, mppc_events):
        """Prepare graphs for all events."""
        graphs = []
        for pixel_event, mppc_event in zip(pixel_events, mppc_events):
            event_graphs = self.graph_builder.build_graphs_from_event(
                pixel_event, mppc_event
            )
            if self.graph_builder.sequence_mode:
                graphs.extend(event_graphs)  # List of sequences
            else:
                graphs.extend(event_graphs)  # List of single graphs

        # Apply pre_filter if provided
        if self.pre_filter is not None:
            graphs = [g for g in self.graphs if self.pre_filter(g)]
        # Apply pre_transform if provided
        if self.pre_transform is not None:
            graphs = [self.pre_transform(g) for g in self.graphs]
        self.length = len(graphs)
        self.graphs = graphs

    def get_edge_types(self) -> List[Tuple[str, str, str]]:
        return self.graph_builder.get_edge_types()

    def get_node_dims(self) -> Dict[str, int]:
        return self.graph_builder.get_node_dims()

    def len(self) -> int:
        return self.length

    def get(self, idx: int) -> Union[Data, HeteroData, List[Union[Data, HeteroData]]]:
        return self.graphs[idx]


class SequenceDataset(Dataset):
    """Dataset that returns sequences of graphs and labels for each sequence with varying lengths.

    Args:
        prefices: List of file path prefixes for the .npy data files.
        labels: List of integer labels corresponding to each prefix.
        has_layer_feature: Whether the pixel spacetime data includes layer information.
        graph_builder: Instance of GraphBuilderBase to use for graph construction.
        transform: Optional transform to apply to each graph.
        pre_transform: Optional pre-transform to apply to each graph before saving.
        pre_filter: Optional filter to apply to each graph before saving.
    Returns:
        List of tuples (graph_sequence, label) where graph_sequence is a list of graphs for the sequence.
    """

    def __init__(
        self,
        mppc_events: List[EventData],
        pixel_events: List[EventData],
        labels: List[int],
        has_layer_feature: bool = False,
        graph_builder: Optional[GraphBuilderBase] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        if len(pixel_events) != len(mppc_events):
            raise ValueError("Pixel and MPPC data must have the same number of events")
        if len(pixel_events) != len(labels):
            raise ValueError("Number of events must match number of labels")
        super().__init__(None, transform, pre_transform, pre_filter)
        self.length = 0
        self.sequences = []
        self.labels = labels

        if graph_builder is None:
            self.graph_builder = HeteroGraphBuilder(sequence_mode=True)
        else:
            self.graph_builder = graph_builder
        if not self.graph_builder.sequence_mode:
            raise ValueError(
                "Graph builder must be in sequence mode for SequenceDataset"
            )
        # Precompute all sequences
        self._prepare_sequences(pixel_events, mppc_events)

    def _prepare_sequences(self, pixel_events, mppc_events):
        """Prepare sequences for all events."""
        sequences = []
        for pixel_event, mppc_event in zip(pixel_events, mppc_events):
            event_sequences = self.graph_builder.build_graphs_from_event(
                pixel_event, mppc_event
            )
            sequences.extend(event_sequences)  # List of sequences

        # Apply pre_filter if provided
        if self.pre_filter is not None:
            sequences = [s for s in self.sequences if self.pre_filter(s)]
        # Apply pre_transform if provided
        if self.pre_transform is not None:
            sequences = [self.pre_transform(s) for s in self.sequences]
        self.length = len(sequences)
        self.sequences = sequences

    def get_edge_types(self) -> List[Tuple[str, str, str]]:
        return self.graph_builder.get_edge_types()

    def get_node_dims(self) -> Dict[str, int]:
        return self.graph_builder.get_node_dims()

    def len(self) -> int:
        return self.length

    def get(self, idx: int) -> Tuple[List[Union[Data, HeteroData]], int]:
        return self.sequences[idx], self.labels[idx]


# Factory functions for creating datasets
def create_dataset(
    prefix: Union[str, List[str]],
    split=(1,),
    labels: Optional[List[int]] = None,
    n_events: Optional[int] = None,
    has_layer_feature: bool = False,
    mppc_timing_cutoff: float = 0.2,
    type: str = "hetero",
    sequence_mode: bool = False,
    whole_event_mode: bool = False,
    timing_cutoff: float = 8.0,
) -> Union[
    DetectorDataset,
    Tuple[DetectorDataset, ...],
    SequenceDataset,
    Tuple[SequenceDataset, ...],
]:
    """Create a DetectorDataset from .npy files.
    Args:
        prefix: File path prefix or list of prefixes for the .npy data files.
        split: Tuple of fractions for (train, val, test) splits. Must sum to 1.
        labels: Optional list of integer labels corresponding to each prefix.
        n_events: Optional maximum number of events to load from each file.
        has_layer_feature: Whether the pixel spacetime data includes layer information.
        mppc_timing_cutoff: Timing cutoff in ns for MPPC-MPPC edges in slice mode.
        type: Type of graph builder to use ('hetero' or 'layer_separated').
        sequence_mode: Whether to build sequences of graphs (True) or single graphs (False).
        whole_event_mode: Whether to build single graphs from entire events (True) or time slices (False).
        timing_cutoff: Global timing cutoff in ns for edges in whole event mode (default 8.0ns).

    Returns:
        Instance(s) of DetectorDataset with the specified configuration.
    """

    if sequence_mode and whole_event_mode:
        raise ValueError("Cannot enable both sequence_mode and whole_event_mode")

    if isinstance(prefix, str):
        prefix = [prefix]
    if labels is not None and len(prefix) != len(labels):
        raise ValueError("Number of prefixes must match number of labels")
    all_pixel_events = []
    all_mppc_events = []
    all_labels = []
    for i, p in enumerate(prefix):
        pixel_spacetime_path = f"{p}_pixel_spacetime.npy"
        pixel_track_labels_path = f"{p}_pixel_track_labels.npy"
        mppc_spacetime_path = f"{p}_mppc_spacetime.npy"
        mppc_track_labels_path = f"{p}_mppc_track_labels.npy"
        pixel_events, mppc_events = load_npy_data(
            pixel_spacetime_path,
            pixel_track_labels_path,
            mppc_spacetime_path,
            mppc_track_labels_path,
            has_layer_feature,
            max_events=n_events,
        )
        all_pixel_events.extend(pixel_events)
        all_mppc_events.extend(mppc_events)
        if labels is not None:
            all_labels.extend([labels[i]] * len(pixel_events))
        else:
            all_labels.extend([0] * len(pixel_events))  # Dummy labels if none provided

    if (
        len(split) < 0
        or not all(0 < s <= 1 for s in split)
        or not abs(sum(split) - 1.0) < 1e-6
    ):
        raise ValueError("Split must be a tuple of three fractions summing to 1")
    n_events = len(pixel_events)
    if n_events != len(mppc_events):
        raise ValueError("Pixel and MPPC data must have the same number of events")
    n_splits = len(split)
    split_indices = [int(sum(split[:i]) * n_events) for i in range(n_splits)]
    split_indices.append(n_events)  # Ensure we include all events
    datasets = []
    builder = None
    if type == "hetero":
        builder = HeteroGraphBuilder(
            connect_layers=True,
            mppc_timing_cutoff=mppc_timing_cutoff,
            sequence_mode=sequence_mode,
            whole_event_mode=whole_event_mode,
            timing_cutoff=timing_cutoff,
        )
    elif type == "layer_separated":
        builder = LayerSeparatedHeteroGraphBuilder(
            connect_layers=True,
            mppc_timing_cutoff=mppc_timing_cutoff,
            sequence_mode=sequence_mode,
            whole_event_mode=whole_event_mode,
            timing_cutoff=timing_cutoff,
        )
    else:
        raise ValueError(f"Unknown graph builder type: {type}")

    for i in range(n_splits):
        start_idx = split_indices[i]
        end_idx = split_indices[i + 1]
        pixel_subset = pixel_events[start_idx:end_idx]
        mppc_subset = mppc_events[start_idx:end_idx]
        dataset = None
        if sequence_mode:
            # For sequence mode, we need dummy labels (e.g., all zeros)
            labels = [0] * len(pixel_subset)
            dataset = SequenceDataset(
                mppc_subset,
                pixel_subset,
                labels,
                graph_builder=builder,
            )
        else:
            dataset = DetectorDataset(
                pixel_subset,
                mppc_subset,
                graph_builder=builder,
            )
        datasets.append(dataset)
    return tuple(datasets) if n_splits > 1 else datasets[0]
