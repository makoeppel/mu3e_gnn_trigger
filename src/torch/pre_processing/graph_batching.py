import torch
import numpy as np
from torch_geometric.data import Data, Dataset, Batch, HeteroData
from torch_geometric.loader import DataLoader
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod


def all_indices_hetero(
    N_src: int, N_dst: int, include_self: bool = True, same_type: bool = False
):
    """
    Generate all possible (i, j) index pairs between two node sets.

    Args:
        N_src: Number of source nodes
        N_dst: Number of destination nodes
        include_self: Keep diagonal if same_type=True
        same_type: If True, treat source and destination as the same set

    Returns:
        Tuple of (row, col) tensors with source and destination indices
    """
    row = torch.arange(N_src).repeat_interleave(N_dst)
    col = torch.arange(N_dst).repeat(N_src)

    if same_type and not include_self:
        mask = row != col
        row, col = row[mask], col[mask]

    return row, col


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


class PixelGraphBuilder(GraphBuilderBase):
    """Builder for pixel-only homogeneous graphs."""

    def _create_pixel_edges(
        self, data: DetectorData
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Create edges between pixel nodes."""
        if len(data) < 2:
            return None, None

        row, col = all_indices_hetero(
            len(data), len(data), include_self=True, same_type=True
        )

        if self.connect_layers and row.numel() > 0:
            edge_mask = (data.layers[row] - data.layers[col]).abs() <= 1
            row, col = row[edge_mask], col[edge_mask]

        if row.numel() == 0:
            return None, None

        # Create edge labels (1.0 if both nodes belong to same positive track)
        edge_labels = (
            (data.tracks[row] > 0) & (data.tracks[row] == data.tracks[col])
        ).float()
        edge_index = torch.stack([row, col], dim=0)

        return edge_index, edge_labels

    def build_graphs_from_event(self, event: Union[torch.Tensor, any]) -> List[Data]:
        """Build homogeneous graphs from pixel event data."""
        event = self._validate_tensor(event, "event")
        data = self._extract_detector_data(event)

        graphs = []
        for time_slice in torch.unique(data.times):
            data_t = data.filter_by_time(time_slice)

            if not data_t.has_sufficient_data():
                continue

            edge_index, edge_labels = self._create_pixel_edges(data_t)

            if edge_index is not None:
                graph = Data(
                    x=data_t.positions, edge_index=edge_index, edge_labels=edge_labels
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

    def _create_edges(
        self, src_data: DetectorData, dst_data: DetectorData, config: EdgeConfig
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Create edges between source and destination nodes."""
        if len(src_data) == 0 or len(dst_data) == 0:
            return None, None

        row, col = all_indices_hetero(
            len(src_data), len(dst_data), include_self=True, same_type=config.same_type
        )

        if row.numel() == 0:
            return None, None

        # Apply layer constraints
        if self.connect_layers and config.use_layers:
            layer_diff = (src_data.layers[row] - dst_data.layers[col]).abs()
            layer_mask = layer_diff <= 1
            row, col = row[layer_mask], col[layer_mask]

        # Apply timing constraints for MPPC
        if config.use_timing and row.numel() > 0:
            time_diff = (src_data.times[row] - dst_data.times[col]).abs()
            timing_mask = time_diff <= self.mppc_timing_cutoff
            row, col = row[timing_mask], col[timing_mask]

        if row.numel() == 0:
            return None, None

        # Create edge labels
        edge_labels = (
            (src_data.tracks[row] > 0) & (src_data.tracks[row] == dst_data.tracks[col])
        ).float()

        return torch.stack([row, col], dim=0), edge_labels

    def _add_edges_to_graph(
        self,
        graph: HeteroData,
        config: EdgeConfig,
        edge_index: Optional[torch.Tensor],
        edge_labels: Optional[torch.Tensor],
    ):
        """Add edges to graph if they exist."""
        if edge_index is not None and edge_labels is not None:
            graph[config.edge_type].edge_index = edge_index
            graph[config.edge_type].edge_labels = edge_labels

    def _create_hetero_graph(
        self,
        pixel_data: DetectorData,
        mppc_data: DetectorData,
        time_slice: torch.Tensor,
    ) -> Optional[HeteroData]:
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
        graph["mppc"].x = torch.cat(
            [mppc_t.positions, mppc_t.times.unsqueeze(1)], dim=1
        )

        # Create edges for all configured types
        data_map = {"pixel": pixel_t, "mppc": mppc_t}

        for config in self.edge_configs:
            src_data = data_map[config.src_type]
            dst_data = data_map[config.dst_type]

            edge_index, edge_labels = self._create_edges(src_data, dst_data, config)
            self._add_edges_to_graph(graph, config, edge_index, edge_labels)

        return graph if len(graph.edge_types) > 0 else None

    def build_graphs_from_event(
        self, mppc: Union[torch.Tensor, any], pixel: Union[torch.Tensor, any]
    ) -> List[HeteroData]:
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


class CombinedGraphBuilder(GraphBuilderBase):
    """Builder for combined homogeneous graphs with both detector types."""

    def __init__(self, connect_layers: bool = True, mppc_timing_cutoff: float = 0.1):
        super().__init__(connect_layers)
        self.mppc_timing_cutoff = mppc_timing_cutoff

    def _create_combined_edges(
        self, combined_data: DetectorData, num_pixel: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Create edges for combined pixel+MPPC nodes."""
        if len(combined_data) < 2:
            return None, None

        row, col = all_indices_hetero(
            len(combined_data), len(combined_data), include_self=True, same_type=True
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

        edge_labels = (
            (combined_data.tracks[row] > 0)
            & (combined_data.tracks[row] == combined_data.tracks[col])
        ).float()
        edge_index = torch.stack([row, col], dim=0)

        return edge_index, edge_labels

    def build_graphs_from_event(
        self, mppc: Union[torch.Tensor, any], pixel: Union[torch.Tensor, any]
    ) -> List[Data]:
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
            pixel_nodes = torch.cat(
                [
                    pixel_t.positions,
                    torch.full(
                        (len(pixel_t), 1), 1, device=pixel_t.positions.device
                    ),  # Type flag
                    torch.full(
                        (len(pixel_t), 1), time_slice, device=pixel_t.positions.device
                    ),
                ],
                dim=1,
            )

            mppc_nodes = torch.cat(
                [
                    mppc_data.positions[mppc_mask],
                    torch.full(
                        (mppc_mask.sum(), 1), 0, device=mppc_data.positions.device
                    ),  # Type flag
                    mppc_data.times[mppc_mask].unsqueeze(1),
                ],
                dim=1,
            )

            # Combine all data
            combined_nodes = torch.cat([pixel_nodes, mppc_nodes], dim=0)
            combined_data = DetectorData(
                positions=combined_nodes,
                tracks=torch.cat([pixel_t.tracks, mppc_data.tracks[mppc_mask]], dim=0),
                layers=torch.cat([pixel_t.layers, mppc_data.layers[mppc_mask]], dim=0),
                times=torch.cat([pixel_t.times, mppc_data.times[mppc_mask]], dim=0),
            )

            edge_index, edge_labels = self._create_combined_edges(
                combined_data, len(pixel_t)
            )

            if edge_index is not None:
                graph = Data(
                    x=combined_nodes, edge_index=edge_index, edge_labels=edge_labels
                )
                graphs.append(graph)

        return graphs


class EventProcessor:
    """Main processor for converting events to graphs."""

    def __init__(self, graph_builder: GraphBuilderBase):
        self.graph_builder = graph_builder

    def process_single_event(self,**kwargs) -> List[Union[Data, HeteroData]]:
        """Process single event to graphs."""
        if isinstance(self.graph_builder, PixelGraphBuilder):
            if "X_pixel" not in kwargs:
                raise ValueError("X_pixel is required for PixelGraphBuilder")
            return self.graph_builder.build_graphs_from_event(kwargs["X_pixel"])
        elif isinstance(self.graph_builder, HeteroGraphBuilder):
            if "X_pixel" not in kwargs or "X_mppc" not in kwargs:
                raise ValueError("X_pixel and X_mppc are required for HeteroGraphBuilder")
            return self.graph_builder.build_graphs_from_event(
                kwargs["X_mppc"], kwargs["X_pixel"]
            )
        elif isinstance(self.graph_builder, CombinedGraphBuilder):
            if "X_pixel" not in kwargs or "X_mppc" not in kwargs:
                raise ValueError("X_pixel and X_mppc are required for CombinedGraphBuilder")
            return self.graph_builder.build_graphs_from_event(
                kwargs["X_mppc"], kwargs["X_pixel"]
            )
        else:
            raise ValueError("Unsupported graph builder type")

    def _validate_input_tensors(
        self, **kwargs
    ) -> Tuple[dict, Optional[torch.Tensor]]:
        """Validate and prepare input tensors."""
        input_tensors = {}
        labels = None

        for key, value in kwargs.items():
            if key == "labels":
                labels = (
                    value
                    if isinstance(value, torch.Tensor)
                    else torch.tensor(value, dtype=torch.float32)
                )
            else:
                tensor = (
                    value
                    if isinstance(value, torch.Tensor)
                    else torch.tensor(value, dtype=torch.float32)
                )
                if tensor.dim() == 3:
                    tensor = tensor.squeeze(0)
                elif tensor.dim() != 2:
                    raise ValueError(f"{key} must be 2D or 3D tensor")
                input_tensors[key] = tensor

        return input_tensors, labels

    def process_graphs(
        self, **kwargs
    ) -> List[Union[Data, HeteroData]]:
        """Process batch of events to list of graphs with labels."""
        input_tensors, labels = self._validate_input_tensors(**kwargs)

        all_graphs = []
        event_indices = []
        valid_labels = []
        
        num_events = next(iter(input_tensors.values())).size(0)

        for event_idx in range(num_events):
            event_data = {
                key: tensor[event_idx] for key, tensor in input_tensors.items()
            }
            graphs = self.process_single_event(**event_data)

            if graphs:
                all_graphs.extend(graphs)
                event_indices.extend([event_idx] * len(graphs))
                if labels is not None:
                    for graph in graphs:
                        graph.y = labels[event_idx]
                    valid_labels.append(labels[event_idx])
                

        return all_graphs, torch.tensor(event_indices, dtype=torch.long), torch.stack(valid_labels) if valid_labels else None
    
    def process_to_graphs(
        self, **kwargs
    ) -> List[Union[Data, HeteroData]]:
        """Process batch of events to list of graphs."""
        all_graphs, _, _ = self.process_graphs(**kwargs)
        return all_graphs

    def process_to_graph_set(
        self, **kwargs
    ) -> Optional[Batch]:
        """Process batch of events to PyTorch Geometric batch."""
        all_graphs, event_indices, valid_labels = self.process_graphs(**kwargs)

        if not all_graphs:
            return None

        batch = Batch.from_data_list(all_graphs)
        batch.event_idx = event_indices
        batch.y = valid_labels if valid_labels is not None else None

        return batch


class EventFilter:
    """Utility for filtering viable events."""

    @staticmethod
    def get_viable_events(
        events: torch.Tensor,
        track_ids: torch.Tensor,
        layers: Optional[torch.Tensor] = None,
        min_tracks: int = 2,
    ) -> List[int]:
        """Return indices of events that produce at least one valid graph."""
        viable = []

        for idx in range(events.size(0)):
            event = events[idx]
            tracks = track_ids[idx] if track_ids.dim() > 1 else track_ids

            mask = event[:, -1] != -1
            if mask.sum() < 2:
                continue

            hits = event[mask]
            tracks_valid = tracks[mask]
            times = hits[:, -1]

            for t in torch.unique(times):
                tm = times == t
                tracks_t = tracks_valid[tm]

                if tm.sum() >= 2 and torch.unique(tracks_t).numel() >= min_tracks:
                    viable.append(idx)
                    break

        return viable


class GraphDataset(Dataset):
    """Enhanced dataset for graph processing with caching support."""

    def __init__(
        self,
        events: torch.Tensor,
        labels: torch.Tensor,
        track_ids: torch.Tensor,
        processor: EventProcessor,
        layer_ids: Optional[torch.Tensor] = None,
        cache_graphs: bool = False,
    ):
        super().__init__()

        # Validate and convert inputs
        self.events = (
            events
            if isinstance(events, torch.Tensor)
            else torch.tensor(events, dtype=torch.float32)
        )
        self.labels = (
            labels
            if isinstance(labels, torch.Tensor)
            else torch.tensor(labels, dtype=torch.float32)
        )
        self.track_ids = (
            track_ids
            if isinstance(track_ids, torch.Tensor)
            else torch.tensor(track_ids, dtype=torch.long)
        )
        self.layer_ids = (
            layer_ids
            if layer_ids is None or isinstance(layer_ids, torch.Tensor)
            else torch.tensor(layer_ids, dtype=torch.long)
        )

        self.processor = processor
        self.cache_graphs = cache_graphs

        # Filter to viable events only
        self.valid_indices = EventFilter.get_viable_events(
            self.events, self.track_ids, self.layer_ids
        )

        # Cache graphs if requested
        if self.cache_graphs:
            self._cache_all_graphs()

    def _cache_all_graphs(self):
        """Pre-compute and cache all graphs."""
        self.cached_graphs = {}
        for idx in self.valid_indices:
            graphs = self.processor.process_single_event(self.events[idx])
            self.cached_graphs[idx] = graphs

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(
        self, idx: int
    ) -> Tuple[List[Union[Data, HeteroData]], torch.Tensor]:
        actual_idx = self.valid_indices[idx]

        if self.cache_graphs:
            graphs = self.cached_graphs[actual_idx]
        else:
            graphs = self.processor.process_single_event(self.events[actual_idx])

        return graphs, self.labels[actual_idx]


class GraphDataLoader:
    """Enhanced data loader for batching graphs."""

    def __init__(
        self, dataset: GraphDataset, batch_size: int = 32, shuffle: bool = True
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = torch.arange(len(self.dataset))
        if self.shuffle:
            indices = indices[torch.randperm(len(indices))]

        for start in range(0, len(self.dataset), self.batch_size):
            batch_indices = indices[start : start + self.batch_size]
            batch_graphs = []
            event_indices = []
            labels = []

            for event_idx, dataset_idx in enumerate(batch_indices):
                graphs, label = self.dataset[dataset_idx.item()]

                # Attach labels to individual graphs
                for graph in graphs:
                    graph.y = label

                batch_graphs.extend(graphs)
                event_indices.extend([event_idx] * len(graphs))
                labels.append(label)

            if batch_graphs:  # Only yield if we have graphs
                batch = Batch.from_data_list(batch_graphs)
                batch.event_idx = torch.tensor(event_indices, dtype=torch.long)
                batch.y = torch.tensor(labels, dtype=torch.float32)
                yield batch

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


# Legacy function wrappers for backward compatibility
def pixel_hits_to_graphs(event, connect_layers=True):
    """Legacy wrapper - use PixelGraphBuilder instead."""
    builder = PixelGraphBuilder(connect_layers)
    return builder.build_graphs_from_event(event)


def full_event_to_hetero_graphs(
    mppc, pixel, connect_layers=True, mppc_timing_cutoff=0.2
):
    """Legacy wrapper - use HeteroGraphBuilder instead."""
    builder = HeteroGraphBuilder(connect_layers, mppc_timing_cutoff)
    return builder.build_graphs_from_event(mppc, pixel)


def full_event_to_graphs(mppc, pixel, connect_layers=True, mppc_timing_cutoff=0.1):
    """Legacy wrapper - use CombinedGraphBuilder instead."""
    builder = CombinedGraphBuilder(connect_layers, mppc_timing_cutoff)
    return builder.build_graphs_from_event(mppc, pixel)


def batch_pixel_hits_to_graph_sets(pixel, labels, connect_layers=True):
    """Legacy wrapper for pixel batch processing."""
    builder = PixelGraphBuilder(connect_layers)
    processor = EventProcessor(builder)

    events_data = [pixel[i] for i in range(pixel.size(0))]
    batch = processor.process_batch_to_batch(events_data, labels)
    return batch


def batch_full_events_to_hetero_graph_set(
    mppc, pixel, labels, connect_layers=True, mppc_timing_cutoff=0.1
):
    """Legacy wrapper for hetero batch processing."""
    builder = HeteroGraphBuilder(connect_layers, mppc_timing_cutoff)
    processor = EventProcessor(builder)

    events_data = [(mppc[i], pixel[i]) for i in range(mppc.size(0))]
    batch = processor.process_batch_to_batch(events_data, labels)
    return batch


def get_viable_events(events, track_ids, layers=None):
    """Legacy wrapper - use EventFilter.get_viable_events() instead."""
    return EventFilter.get_viable_events(events, track_ids, layers)


# Factory functions for easy instantiation
def create_pixel_processor(connect_layers: bool = True) -> EventProcessor:
    """Create processor for pixel-only graphs."""
    builder = PixelGraphBuilder(connect_layers)
    return EventProcessor(builder)


def create_hetero_processor(
    connect_layers: bool = True, mppc_timing_cutoff: float = 0.2
) -> EventProcessor:
    """Create processor for heterogeneous graphs."""
    builder = HeteroGraphBuilder(connect_layers, mppc_timing_cutoff)
    return EventProcessor(builder)


def create_combined_processor(
    connect_layers: bool = True, mppc_timing_cutoff: float = 0.1
) -> EventProcessor:
    """Create processor for combined homogeneous graphs."""
    builder = CombinedGraphBuilder(connect_layers, mppc_timing_cutoff)
    return EventProcessor(builder)
