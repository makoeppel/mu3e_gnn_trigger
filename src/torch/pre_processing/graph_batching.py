import torch
import numpy
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.loader import DataLoader


def batch_pixel_hits_to_graph_set(
    pixel_hits: torch.Tensor,
    track_ids: torch.Tensor,
    labels: torch.Tensor,
    layer_ids: torch.Tensor = None,
):
    """
    Convert a batch of events into a list of PyG graphs.
    
    Args:
        events (torch.Tensor): [num_events, num_hits, feature_dim] with -1 padding.
                               Last column must be time.
        track_ids (torch.Tensor): [num_events, num_hits] with track IDs (0 = noise).
        labels (torch.Tensor): [num_events] events-level labels.
        layer_ids (torch.Tensor, optional): [num_events, num_hits] detector layer IDs.

    Returns:
        List[Data]: PyG graphs for all time slices of all events.
    """
    all_graphs = []
    event_indices = []

    for event_idx, event in enumerate(pixel_hits):
        # Mask valid hits
        event_mask = event[:, -1] != -1
        hits = event[event_mask]
        tracks = track_ids[event_idx][event_mask]
        layers = layer_ids[event_idx][event_mask] if layer_ids is not None else None

        times = hits[:, -1]
        positions = hits[:, :3]

        # Use torch.unique instead of np.unique
        unique_times = torch.unique(times)

        for t in unique_times:
            # Hits at this time slice
            time_mask = times == t
            nodes = positions[time_mask]
            tracks_t = tracks[time_mask]
            layers_t = layers[time_mask] if layers is not None else None

            num_nodes = nodes.size(0)
            if num_nodes < 2:
                continue
            if torch.unique(tracks_t).numel() < 2:
                continue

            # Build edges (upper triangular, no self-loops)
            row, col = torch.triu_indices(num_nodes, num_nodes, offset=1, device=nodes.device)

            # Optional layer constraint
            if layers_t is not None:
                edge_mask = (layers_t[row] - layers_t[col]).abs() <= 1
                row, col = row[edge_mask], col[edge_mask]

            if row.numel() == 0:
                continue

            # Edge labels: same nonzero track
            edge_labels = ((tracks_t[row] > 0) & (tracks_t[row] == tracks_t[col])).float()

            # Construct graph
            all_graphs.append(
                Data(
                    x=nodes,
                    edge_index=torch.stack([row, col], dim=0),
                    edge_labels=edge_labels,
                )
            )
            event_indices.append(event_idx)
        batch = Batch.from_data_list(all_graphs)
        batch.event_idx = torch.tensor(event_indices, dtype=torch.long)
        batch.y = labels.float() if isinstance(labels, torch.Tensor) else torch.tensor(labels, dtype=torch.float)
    return batch


# ------------------------
# Helper: Pre-filter viable events
# ------------------------
def get_viable_events(events, track_ids, layers=None):
    """
    Return indices of events that produce at least one valid graph.
    """
    viable = []
    for idx, event in enumerate(events):
        mask = event[:, -1] != -1
        hits = event[mask]
        tracks = track_ids[idx][mask]

        times = hits[:, -1]
        positions = hits[:, :3]

        for t in torch.unique(times):
            tm = times == t
            nodes = positions[tm]
            tracks_t = tracks[tm]

            if nodes.size(0) >= 2 and torch.unique(tracks_t).numel() >= 2:
                viable.append(idx)
                break  # first valid slice is enough
    return viable

def pixel_hits_to_graphs(event, connect_layers = False):
    graphs = []
    mask = event[:, -1] != -1
    positions = event[:, :3][mask]
    tracks = event[:, 4][mask]
    layers = event[:, 3][mask]
    times = event[:, -1][mask]

    for t in torch.unique(times):
        tm = times == t
        nodes = positions[tm]
        tracks_t = tracks[tm]
        layers_t = layers[tm]

        num_nodes = nodes.size(0)
        if num_nodes < 2 or torch.unique(tracks_t).numel() < 2:
            continue

        row, col = torch.triu_indices(num_nodes, num_nodes, offset=1, device=nodes.device)
        if connect_layers:
            edge_mask = (layers_t[row] - layers_t[col]).abs() <= 1
            row, col = row[edge_mask], col[edge_mask]
        if row.numel() == 0:
            continue

        edge_labels = ((tracks_t[row] > 0) & (tracks_t[row] == tracks_t[col])).float()

        graph = Data(
            x=nodes,
            edge_index=torch.stack([row, col], dim=0),
            edge_labels=edge_labels,
        )
        graphs.append(graph)

    return graphs


def full_event_to_graphs(mppc, pixel):
    all_graphs = []
    mask_mppc = mppc[:, -1] != -1

    mask_pixel = pixel[:, -1] != -1

    positions_mppc = mppc[:, :3][mask_mppc]
    tracks_mppc = mppc[:, 4][mask_mppc]
    layers_mppc = mppc[:, 3][mask_mppc]
    times_mppc = mppc[:, -1][mask_mppc]

    positions_pixel = pixel[:, :3][mask_pixel]
    tracks_pixel = pixel[:, 4][mask_pixel]
    layers_pixel = pixel[:, 3][mask_pixel]
    times_pixel = pixel[:, -1][mask_pixel]



    mppc_cut_times = (times_mppc // 8) * 8

    for t in torch.unique(times_pixel):
        tm_pixel = times_pixel == t
        tracks_t_pixel = tracks_pixel[tm_pixel]
        layers_t_pixel = layers_pixel[tm_pixel]
        nodes_pixel = torch.cat([positions_pixel[tm_pixel],torch.full((tm_pixel.sum(), 1), 1), torch.full((tm_pixel.sum(), 1), t, device=positions_pixel.device)], dim=1)
        
        tm_mppc = mppc_cut_times == t
        tracks_t_mppc = tracks_mppc[tm_mppc]
        layers_t_mppc = layers_mppc[tm_mppc]
        nodes_mppc = torch.cat([positions_mppc[tm_mppc],torch.full((tm_mppc.sum(), 1), 0), times_mppc[tm_mppc].unsqueeze(1)], dim=1)

        nodes = torch.cat([nodes_pixel, nodes_mppc], dim=0)
        tracks_t = torch.cat([tracks_t_pixel, tracks_t_mppc], dim=0)
        layers_t = torch.cat([layers_t_pixel, layers_t_mppc], dim=0)

        num_nodes = nodes.size(0)
        if num_nodes < 2 or torch.unique(tracks_t).numel() < 2:
            continue

        row, col = torch.triu_indices(num_nodes, num_nodes, offset=1, device=nodes.device)
        if layers_t is not None:
            edge_mask = ((layers_t[row] - layers_t[col]).abs()) <= 1 & ~((( layers_t[row] == 2) & (layers_t[col] == 3)) | ((layers_t[row] == 3) & (layers_t[col] == 2)))
            row, col = row[edge_mask], col[edge_mask]
        if row.numel() == 0:
            continue

        edge_labels = ((tracks_t[row] > 0) & (tracks_t[row] == tracks_t[col])).float()

        graph = Data(
            x=nodes,
            edge_index=torch.stack([row, col], dim=0),
            edge_labels=edge_labels,
        )
        all_graphs.append(graph)
    return all_graphs


def batch_full_events_to_graph_set(
    mppc : torch.Tensor,
    pixel : torch.Tensor,
    labels: torch.Tensor,
):
    all_graphs = []
    event_indices = []

    for event_idx in range(mppc.size(0)):
        graphs = full_event_to_graphs(
            mppc[event_idx],
            pixel[event_idx]
        )
        all_graphs.extend(graphs)
        event_indices.extend([event_idx] * len(graphs))

    batch = Batch.from_data_list(all_graphs)
    batch.event_idx = torch.tensor(event_indices, dtype=torch.long)
    batch.y = labels.float() if isinstance(labels, torch.Tensor) else torch.tensor(labels, dtype=torch.float)
    return batch

def batch_full_events_to_graphs(
    mppc : torch.Tensor,
    pixel : torch.Tensor,
    labels: torch.Tensor,
):
    mppc = mppc if isinstance(mppc, torch.Tensor) else torch.tensor(mppc, dtype=torch.float32)
    pixel = pixel if isinstance(pixel, torch.Tensor) else torch.tensor(pixel, dtype=torch.float32)
    labels = labels if isinstance(labels, torch.Tensor) else torch.tensor(labels, dtype=torch.float32)

    all_graphs = []
    event_indices = []

    for event_idx in range(mppc.size(0)):
        graphs = full_event_to_graphs(
            mppc[event_idx],
            pixel[event_idx]
        )
        for graph in graphs:
            graph.y = labels[event_idx] if isinstance(labels, torch.Tensor) else torch.tensor(labels[event_idx], dtype=torch.float)
        all_graphs.extend(graphs)
        event_indices.extend([event_idx] * len(graphs))

    return all_graphs

# ------------------------
# Dataset: per-event graphs
# ------------------------
class GraphSetDataset(Dataset):
    def __init__(self, X, y, track_ids, layer_ids=None, cache_graphs=False):
        super().__init__()
        self.cache_graphs = cache_graphs

        self.X = X if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=torch.float32)
        self.y = y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.float32)
        self.track_ids = (
            track_ids if isinstance(track_ids, torch.Tensor) else torch.tensor(track_ids, dtype=torch.long) 
        )
        self.layer_ids = (
            layer_ids if layer_ids is None or isinstance(layer_ids, torch.Tensor)
            else torch.tensor(layer_ids, dtype=torch.long)
        )
        self.valid_indices = get_viable_events(self.X, self.track_ids, self.layer_ids)
        if self.cache_graphs:
            self.graphs = []
            for idx in range(len(self.X)):
                gs = pixel_hits_to_graphs(
                    self.X[idx],
                    self.track_ids[idx],
                    self.y[idx],
                    self.layer_ids[idx] if self.layer_ids is not None else None,
                    event_idx=idx
                )
                self.graphs.append(gs)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        idx = self.valid_indices[idx]
        if self.cache_graphs:
            return self.graphs[idx]
        else:
            return pixel_hits_to_graphs(
                self.X[idx],
                self.track_ids[idx],
                self.layer_ids[idx] if self.layer_ids is not None else None
            ), self.y[idx]

class GraphSetDataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = torch.arange(len(self.dataset))
        if self.shuffle:
            indices = indices[torch.randperm(len(indices))]
        for start in range(0, len(self.dataset), self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            batch_graphs = []
            event_indices = []
            labels = []
            for event_index, idx in enumerate(batch_indices):
                graphs, label = self.dataset[idx.item()]
                batch_graphs.extend(graphs)
                event_indices.extend([event_index] * len(graphs))
                labels.append(label)
            batch = Batch.from_data_list(batch_graphs)
            batch.event_idx = torch.tensor(event_indices, dtype=torch.long)
            batch.y = torch.tensor(labels, dtype=torch.float)
            yield batch

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size  # Ceiling division
    
class FullEventDataset(Dataset):
    def __init__(self, pixel, mppc, y):
        super().__init__()
        self.pixel_position = pixel[:,:, :3] if isinstance(pixel, torch.Tensor) else torch.tensor(pixel[:,:,:3], dtype=torch.float32)
        self.pixel_layer = pixel[:,:, 3] if isinstance(pixel, torch.Tensor) else torch.tensor(pixel[:,:,3], dtype=torch.long)
        self.pixel_track_id = pixel[:,:, 4] if isinstance(pixel, torch.Tensor) else torch.tensor(pixel[:,:,4], dtype=torch.long)
        self.pixel_time = pixel[:,:, 5] if isinstance(pixel, torch.Tensor) else torch.tensor(pixel[:,:,5], dtype=torch.float32)

        self.mppc_position = mppc[:,:, :3] if isinstance(mppc, torch.Tensor) else torch.tensor(mppc[:,:,:3], dtype=torch.float32)
        self.mppc_layer = mppc[:,:, 3] if isinstance(mppc, torch.Tensor) else torch.tensor(mppc[:,:,3], dtype=torch.long)
        self.mppc_track_id = mppc[:,:, 4] if isinstance(mppc, torch.Tensor) else torch.tensor(mppc[:,:,4], dtype=torch.long)
        self.mppc_time = mppc[:,:, 5] if isinstance(mppc, torch.Tensor) else torch.tensor(mppc[:,:,5], dtype=torch.float32)

        self.y = y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.float32)



    def __len__(self):
        return len(self.pixel)

    def __getitem__(self, idx):
        return full_event_to_graphs(
            self.mppc_position[idx],
            self.pixel_position[idx],
            self.mppc_track_id[idx],
            self.pixel_track_id[idx],
            self.mppc_layer[idx],
            self.pixel_layer[idx]
        ), self.y[idx]
    
class FullEventDataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = torch.arange(len(self.dataset))
        if self.shuffle:
            indices = indices[torch.randperm(len(indices))]
        for start in range(0, len(self.dataset), self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            batch_graphs = []
            event_indices = []
            labels = []
            for event_index, idx in enumerate(batch_indices):
                graphs, label = self.dataset[idx.item()]
                for graph in graphs:
                    graph.y = label
                batch_graphs.extend(graphs)
                event_indices.extend([event_index] * len(graphs))
                labels.append(label)
            batch = Batch.from_data_list(batch_graphs)
            batch.event_idx = torch.tensor(event_indices, dtype=torch.long)
            batch.y = torch.tensor(labels, dtype=torch.float)
            yield batch

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size  # Ceiling division