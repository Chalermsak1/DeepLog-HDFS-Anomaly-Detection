"""
DeepLog Graph Topology (GNN) Module
===================================
Constructs directed execution graphs (DAGs / transition graphs) from log sequences
and detects structural and topological anomalies using Graph Neural Networks (GAT / Gated Graph Conv).

Features:
- `LogGraphBuilder`: Converts sequence of log events and transitions into directed transition
  graphs / DAGs with edge features (transition counts, latencies) using networkx and PyTorch.
- `LogGNNAnomalyDetector`: Graph Neural Network in PyTorch with Graph Attention (GAT) and
  Gated Message Passing layers to compute graph-level representations and detect structural/topological
  anomalies in distributed execution graphs.
"""

from typing import Dict, Any, List, Optional, Tuple, Union, Sequence, Set
import math
import os
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================ #
#                            Graph Construction                                #
# ============================================================================ #

class LogGraphBuilder:
    """
    LogGraphBuilder: Converts log event sequences into directed transition graphs / DAGs
    with rich node and edge attributes, and converts them to PyTorch tensors for GNN processing.
    """

    def __init__(
        self,
        vocab_size: Optional[int] = None,
        add_self_loops: bool = True,
        node_feature_dim: int = 16,
        edge_feature_dim: int = 4,
        time_unit_scale: float = 1.0,
    ):
        """
        Parameters
        ----------
        vocab_size : int, optional
            Size of event vocabulary. If provided, event IDs are expected in [0, vocab_size - 1].
        add_self_loops : bool, default=True
            Whether to ensure every node has at least a self-loop edge for stable GNN message passing.
        node_feature_dim : int, default=16
            Target dimensionality of output node features.
        edge_feature_dim : int, default=4
            Target dimensionality of output edge features.
        time_unit_scale : float, default=1.0
            Scaling factor applied to timestamps / latencies (e.g. 1.0 for seconds, 0.001 for ms -> s).
        """
        self.vocab_size = vocab_size
        self.add_self_loops = add_self_loops
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.time_unit_scale = time_unit_scale

    def build_graph(
        self,
        events: Sequence[Union[int, str]],
        timestamps: Optional[Sequence[float]] = None,
        attributes: Optional[Sequence[Dict[str, Any]]] = None,
        block_id: Optional[str] = None,
    ) -> nx.DiGraph:
        """
        Convert a sequence of log events into a directed transition graph (networkx.DiGraph).

        Parameters
        ----------
        events : Sequence[Union[int, str]]
            Sequence of log event IDs (e.g. [5, 5, 5, 26, 26, 26, 11]).
        timestamps : Sequence[float], optional
            Monotonic timestamps corresponding to each event.
        attributes : Sequence[Dict[str, Any]], optional
            Optional metadata/parameters per event step.
        block_id : str, optional
            Identifier for the log block / session.

        Returns
        -------
        graph : nx.DiGraph
            Directed graph where nodes are unique event types and edges represent observed transitions.
        """
        graph = nx.DiGraph()
        graph.graph["block_id"] = str(block_id) if block_id is not None else "unknown"
        graph.graph["seq_len"] = len(events)

        if len(events) == 0:
            return graph

        seq_len = len(events)
        scaled_times = (
            [float(t) * self.time_unit_scale for t in timestamps]
            if timestamps is not None and len(timestamps) == seq_len
            else None
        )

        # 1. Register Nodes
        for step, ev in enumerate(events):
            ev_id = int(ev) if isinstance(ev, (int, np.integer)) or (isinstance(ev, str) and ev.lstrip("-").isdigit()) else str(ev)
            if not graph.has_node(ev_id):
                graph.add_node(
                    ev_id,
                    event_id=ev_id,
                    count=1,
                    first_pos=step,
                    last_pos=step,
                    is_start=(step == 0),
                    is_end=(step == seq_len - 1),
                )
            else:
                graph.nodes[ev_id]["count"] += 1
                graph.nodes[ev_id]["last_pos"] = step
                if step == seq_len - 1:
                    graph.nodes[ev_id]["is_end"] = True

        # 2. Register Directed Transition Edges
        for step in range(1, seq_len):
            u_raw = events[step - 1]
            v_raw = events[step]
            u = int(u_raw) if isinstance(u_raw, (int, np.integer)) or (isinstance(u_raw, str) and u_raw.lstrip("-").isdigit()) else str(u_raw)
            v = int(v_raw) if isinstance(v_raw, (int, np.integer)) or (isinstance(v_raw, str) and v_raw.lstrip("-").isdigit()) else str(v_raw)

            latency = 0.0
            if scaled_times is not None:
                latency = max(0.0, scaled_times[step] - scaled_times[step - 1])

            if not graph.has_edge(u, v):
                graph.add_edge(
                    u,
                    v,
                    count=1,
                    latencies=[latency],
                    mean_latency=latency,
                    min_latency=latency,
                    max_latency=latency,
                    std_latency=0.0,
                    weight=1.0,
                )
            else:
                edge_data = graph[u][v]
                edge_data["count"] += 1
                edge_data["latencies"].append(latency)
                lats = edge_data["latencies"]
                edge_data["mean_latency"] = float(np.mean(lats))
                edge_data["min_latency"] = float(np.min(lats))
                edge_data["max_latency"] = float(np.max(lats))
                edge_data["std_latency"] = float(np.std(lats)) if len(lats) > 1 else 0.0
                edge_data["weight"] = float(edge_data["count"])

        # 3. Compute normalized transition probabilities
        for u in graph.nodes():
            out_edges = list(graph.out_edges(u, data=True))
            total_out = sum(d.get("count", 1) for _, _, d in out_edges)
            if total_out > 0:
                for _, v, d in out_edges:
                    d["transition_prob"] = float(d.get("count", 1) / total_out)

        return graph

    def graph_to_tensors(
        self,
        graph: nx.DiGraph,
        vocab_size: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Convert a networkx.DiGraph into standard PyTorch GNN tensors:
        node features, COO edge index, edge features, and adjacency.

        Parameters
        ----------
        graph : nx.DiGraph
            Directed graph produced by `build_graph`.
        vocab_size : int, optional
            Event vocabulary size for node embedding lookup.

        Returns
        -------
        tensors : Dict[str, torch.Tensor]
            Dictionary containing:
            - 'x': Node feature tensor (num_nodes, node_feature_dim)
            - 'node_ids': Categorical event index tensor (num_nodes,)
            - 'edge_index': COO edge indices (2, num_edges)
            - 'edge_attr': Edge attribute tensor (num_edges, edge_feature_dim)
            - 'num_nodes': Number of unique nodes
            - 'num_edges': Number of edges
            - 'adjacency': Dense adjacency matrix (num_nodes, num_nodes)
        """
        v_size = vocab_size or self.vocab_size or 64
        nodes = list(graph.nodes())
        num_nodes = len(nodes)

        if num_nodes == 0:
            # Handle empty graph edge case
            return {
                "x": torch.zeros((0, self.node_feature_dim), dtype=torch.float32),
                "node_ids": torch.zeros((0,), dtype=torch.long),
                "edge_index": torch.zeros((2, 0), dtype=torch.long),
                "edge_attr": torch.zeros((0, self.edge_feature_dim), dtype=torch.float32),
                "num_nodes": torch.tensor(0, dtype=torch.long),
                "num_edges": torch.tensor(0, dtype=torch.long),
                "adjacency": torch.zeros((0, 0), dtype=torch.float32),
                "node_map": {},
            }

        node_map = {node_id: idx for idx, node_id in enumerate(nodes)}
        total_seq_len = max(1, graph.graph.get("seq_len", sum(graph.nodes[n].get("count", 1) for n in nodes)))

        # Build Node Features
        node_id_list = []
        node_features_list = []

        for node_id in nodes:
            ndata = graph.nodes[node_id]
            # Convert node_id to an integer vocabulary index
            if isinstance(node_id, int):
                int_id = node_id % v_size
            elif isinstance(node_id, str) and node_id.isdigit():
                int_id = int(node_id) % v_size
            elif isinstance(node_id, str) and node_id.startswith("E") and node_id[1:].isdigit():
                int_id = int(node_id[1:]) % v_size
            else:
                int_id = abs(hash(str(node_id))) % v_size

            node_id_list.append(int_id)

            # Node structural signals
            count = float(ndata.get("count", 1))
            freq = count / float(total_seq_len)
            first_pos = float(ndata.get("first_pos", 0)) / float(total_seq_len)
            last_pos = float(ndata.get("last_pos", 0)) / float(total_seq_len)
            is_start = 1.0 if ndata.get("is_start", False) else 0.0
            is_end = 1.0 if ndata.get("is_end", False) else 0.0
            in_deg = float(graph.in_degree(node_id)) / max(1.0, float(num_nodes))
            out_deg = float(graph.out_degree(node_id)) / max(1.0, float(num_nodes))

            feat = [
                float(int_id) / float(v_size),
                freq,
                first_pos,
                last_pos,
                is_start,
                is_end,
                in_deg,
                out_deg,
            ]
            # Pad or truncate to self.node_feature_dim
            if len(feat) < self.node_feature_dim:
                feat = feat + [0.0] * (self.node_feature_dim - len(feat))
            else:
                feat = feat[: self.node_feature_dim]
            node_features_list.append(feat)

        node_ids_tensor = torch.tensor(node_id_list, dtype=torch.long)
        x_tensor = torch.tensor(node_features_list, dtype=torch.float32)

        # Build Edges
        edge_src = []
        edge_dst = []
        edge_attr_list = []
        seen_edges: Set[Tuple[int, int]] = set()

        for u, v, edata in graph.edges(data=True):
            u_idx = node_map[u]
            v_idx = node_map[v]
            edge_src.append(u_idx)
            edge_dst.append(v_idx)
            seen_edges.add((u_idx, v_idx))

            count = float(edata.get("count", 1))
            prob = float(edata.get("transition_prob", 1.0))
            mean_lat = float(edata.get("mean_latency", 0.0))
            std_lat = float(edata.get("std_latency", 0.0))

            # Edge feature vector: [count_norm, prob, mean_latency_log, std_latency_log]
            attr = [
                min(1.0, count / float(total_seq_len)),
                prob,
                math.log1p(max(0.0, mean_lat)),
                math.log1p(max(0.0, std_lat)),
            ]
            if len(attr) < self.edge_feature_dim:
                attr = attr + [0.0] * (self.edge_feature_dim - len(attr))
            else:
                attr = attr[: self.edge_feature_dim]
            edge_attr_list.append(attr)

        # Add self-loops if requested and missing
        if self.add_self_loops:
            for i in range(num_nodes):
                if (i, i) not in seen_edges:
                    edge_src.append(i)
                    edge_dst.append(i)
                    # Self-loop default features
                    attr = [1.0 / float(total_seq_len), 1.0, 0.0, 0.0]
                    if len(attr) < self.edge_feature_dim:
                        attr = attr + [0.0] * (self.edge_feature_dim - len(attr))
                    else:
                        attr = attr[: self.edge_feature_dim]
                    edge_attr_list.append(attr)

        if len(edge_src) > 0:
            edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, self.edge_feature_dim), dtype=torch.float32)

        # Dense Adjacency Matrix
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
        if edge_index.size(1) > 0:
            adj[edge_index[0], edge_index[1]] = 1.0

        return {
            "x": x_tensor,
            "node_ids": node_ids_tensor,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "num_nodes": torch.tensor(num_nodes, dtype=torch.long),
            "num_edges": torch.tensor(edge_index.size(1), dtype=torch.long),
            "adjacency": adj,
            "node_map": node_map,
        }

    def sequence_to_tensors(
        self,
        events: Sequence[Union[int, str]],
        timestamps: Optional[Sequence[float]] = None,
        block_id: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """Shortcut to build graph from event sequence and convert directly to PyTorch tensors."""
        graph = self.build_graph(events, timestamps=timestamps, block_id=block_id)
        return self.graph_to_tensors(graph)

    @staticmethod
    def collate_graphs(
        graph_dicts: Sequence[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Batch a list of single-graph tensor dictionaries into a single batched graph dictionary.

        Parameters
        ----------
        graph_dicts : Sequence[Dict[str, torch.Tensor]]
            List of tensor dicts produced by `graph_to_tensors`.

        Returns
        -------
        batched : Dict[str, torch.Tensor]
            Batched graph representation with offset edge indices and node batch vector.
        """
        batch_x = []
        batch_node_ids = []
        batch_edge_index = []
        batch_edge_attr = []
        batch_vec = []

        node_offset = 0
        for b_idx, g in enumerate(graph_dicts):
            n_nodes = g["x"].size(0)
            if n_nodes == 0:
                continue

            batch_x.append(g["x"])
            batch_node_ids.append(g["node_ids"])

            e_idx = g["edge_index"].clone()
            e_idx += node_offset
            batch_edge_index.append(e_idx)

            batch_edge_attr.append(g["edge_attr"])
            batch_vec.append(torch.full((n_nodes,), b_idx, dtype=torch.long))

            node_offset += n_nodes

        if len(batch_x) == 0:
            return {
                "x": torch.zeros((0, 16), dtype=torch.float32),
                "node_ids": torch.zeros((0,), dtype=torch.long),
                "edge_index": torch.zeros((2, 0), dtype=torch.long),
                "edge_attr": torch.zeros((0, 4), dtype=torch.float32),
                "batch": torch.zeros((0,), dtype=torch.long),
                "num_graphs": 0,
            }

        return {
            "x": torch.cat(batch_x, dim=0),
            "node_ids": torch.cat(batch_node_ids, dim=0),
            "edge_index": torch.cat(batch_edge_index, dim=1),
            "edge_attr": torch.cat(batch_edge_attr, dim=0),
            "batch": torch.cat(batch_vec, dim=0),
            "num_graphs": len(graph_dicts),
        }

    @staticmethod
    def extract_topological_features(graph: nx.DiGraph) -> Dict[str, Any]:
        """
        Extract structural and topological graph properties (DAG validity, cycles, density).

        Parameters
        ----------
        graph : nx.DiGraph
            Input execution graph.

        Returns
        -------
        metrics : Dict[str, Any]
            Topological properties dictionary.
        """
        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()

        if n_nodes == 0:
            return {
                "num_nodes": 0,
                "num_edges": 0,
                "density": 0.0,
                "is_dag": True,
                "has_cycles": False,
                "cycle_count": 0,
                "cycles": [],
                "branching_factor": 0.0,
            }

        # Check DAG property (acyclic)
        is_dag = nx.is_directed_acyclic_graph(graph)
        cycles = []
        if not is_dag:
            try:
                # Find simple cycles (capped at first 10 for performance)
                for c in nx.simple_cycles(graph):
                    cycles.append(c)
                    if len(cycles) >= 10:
                        break
            except Exception:
                cycles = [["detected_cycle"]]

        density = nx.density(graph)
        out_degrees = [d for _, d in graph.out_degree()]
        branching_factor = float(np.mean(out_degrees)) if out_degrees else 0.0

        return {
            "num_nodes": n_nodes,
            "num_edges": n_edges,
            "density": float(density),
            "is_dag": is_dag,
            "has_cycles": not is_dag,
            "cycle_count": len(cycles),
            "cycles": cycles,
            "branching_factor": branching_factor,
        }

    @staticmethod
    def detect_structural_anomalies(
        graph: nx.DiGraph,
        reference_graph: Optional[nx.DiGraph] = None,
        allowed_transitions: Optional[Set[Tuple[Any, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Compare graph against a reference execution topology to identify
        unexpected transitions, novel cycles, or missing transitions.
        """
        topo = LogGraphBuilder.extract_topological_features(graph)
        novel_edges = []
        missing_edges = []

        if reference_graph is not None:
            ref_edges = set(reference_graph.edges())
            curr_edges = set(graph.edges())
            novel_edges = [list(e) for e in (curr_edges - ref_edges)]
            missing_edges = [list(e) for e in (ref_edges - curr_edges)]
        elif allowed_transitions is not None:
            curr_edges = set(graph.edges())
            novel_edges = [list(e) for e in (curr_edges - allowed_transitions)]

        has_structural_anomaly = bool(topo["has_cycles"] or len(novel_edges) > 0)
        structural_score = 0.0
        if topo["has_cycles"]:
            structural_score += 0.5
        if novel_edges:
            structural_score += min(0.5, 0.15 * len(novel_edges))

        return {
            "has_structural_anomaly": has_structural_anomaly,
            "structural_score": min(1.0, structural_score),
            "has_cycles": topo["has_cycles"],
            "cycle_count": topo["cycle_count"],
            "cycles": topo["cycles"],
            "novel_edges": novel_edges,
            "missing_edges": missing_edges,
            "num_nodes": topo["num_nodes"],
            "num_edges": topo["num_edges"],
            "density": topo["density"],
        }


# ============================================================================ #
#                           GNN Layers & Detector                              #
# ============================================================================ #

class GraphAttentionLayer(nn.Module):
    """
    Multi-Head Graph Attention Network (GAT) layer with edge feature conditioning
    and vectorized attention aggregation via native PyTorch index_add_.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        edge_dim: int = 4,
        nheads: int = 4,
        dropout: float = 0.1,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        assert out_features % nheads == 0, "out_features must be divisible by nheads"
        self.in_features = in_features
        self.out_features = out_features
        self.edge_dim = edge_dim
        self.nheads = nheads
        self.head_dim = out_features // nheads
        self.negative_slope = negative_slope

        # Projections
        self.linear_src = nn.Linear(in_features, out_features, bias=False)
        self.linear_dst = nn.Linear(in_features, out_features, bias=False)
        self.linear_edge = nn.Linear(edge_dim, out_features, bias=False)
        self.linear_val = nn.Linear(in_features, out_features, bias=False)
        self.linear_edge_val = nn.Linear(edge_dim, out_features, bias=False)

        # Attention score parameters
        self.attn_vec = nn.Parameter(torch.Tensor(1, nheads, self.head_dim))
        nn.init.xavier_uniform_(self.attn_vec)

        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_features)

        # Residual shortcut if dimensions match or project
        if in_features != out_features:
            self.res_proj = nn.Linear(in_features, out_features, bias=False)
        else:
            self.res_proj = nn.Identity()

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        h : torch.Tensor of shape (N, in_features)
            Node embeddings.
        edge_index : torch.Tensor of shape (2, E)
            COO edge indices (source -> target).
        edge_attr : torch.Tensor of shape (E, edge_dim), optional
            Edge attribute features.

        Returns
        -------
        out : torch.Tensor of shape (N, out_features)
            Updated node representations.
        attn_weights : torch.Tensor of shape (E, nheads)
            Attention coefficients per edge per head.
        """
        N = h.size(0)
        E = edge_index.size(1)

        if E == 0:
            # No edges: pass through residual
            res = self.res_proj(h)
            return self.layer_norm(res), torch.zeros((0, self.nheads), device=h.device)

        src_idx = edge_index[0]
        dst_idx = edge_index[1]

        # 1. Project node representations
        h_src = self.linear_src(h).view(N, self.nheads, self.head_dim)
        h_dst = self.linear_dst(h).view(N, self.nheads, self.head_dim)
        h_val = self.linear_val(h).view(N, self.nheads, self.head_dim)

        # 2. Project edge representations
        if edge_attr is not None and self.edge_dim > 0:
            e_proj = self.linear_edge(edge_attr).view(E, self.nheads, self.head_dim)
            e_val = self.linear_edge_val(edge_attr).view(E, self.nheads, self.head_dim)
        else:
            e_proj = torch.zeros((E, self.nheads, self.head_dim), device=h.device)
            e_val = torch.zeros((E, self.nheads, self.head_dim), device=h.device)

        # 3. Compute unnormalized attention scores
        # Score per edge: LeakyReLU( (h_src[src] + h_dst[dst] + e_proj) * attn_vec )
        edge_h_src = h_src[src_idx]  # (E, nheads, head_dim)
        edge_h_dst = h_dst[dst_idx]  # (E, nheads, head_dim)
        edge_combined = self.leaky_relu(edge_h_src + edge_h_dst + e_proj)

        # Dot product with attention vector: (E, nheads)
        score = (edge_combined * self.attn_vec).sum(dim=-1)

        # 4. Softmax per destination node
        score_clamped = torch.clamp(score, -15.0, 15.0)
        exp_score = torch.exp(score_clamped)  # (E, nheads)

        # Sum exp per target node
        denom = torch.zeros((N, self.nheads), device=h.device, dtype=torch.float32)
        denom.index_add_(0, dst_idx, exp_score)
        denom = denom + 1e-9

        # Normalized attention coefficients: (E, nheads)
        alpha = exp_score / denom[dst_idx]
        alpha = self.dropout(alpha)

        # 5. Message Aggregation: alpha * (h_val[src] + e_val)
        messages = alpha.unsqueeze(-1) * (h_val[src_idx] + e_val)  # (E, nheads, head_dim)
        messages_flat = messages.view(E, self.out_features)

        # Aggregate messages into target nodes
        out = torch.zeros((N, self.out_features), device=h.device, dtype=torch.float32)
        out.index_add_(0, dst_idx, messages_flat)

        # 6. Residual connection + LayerNorm
        res = self.res_proj(h)
        out = self.layer_norm(out + res)

        return out, alpha


class GatedGraphConvLayer(nn.Module):
    """
    Gated Graph Convolution Layer with edge features and GRU gating.
    Alternative topological message passing architecture.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        edge_dim: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.edge_dim = edge_dim

        self.msg_mlp = nn.Sequential(
            nn.Linear(in_features + edge_dim, out_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, out_features),
        )
        self.gru_cell = nn.GRUCell(out_features, out_features)
        self.layer_norm = nn.LayerNorm(out_features)

        if in_features != out_features:
            self.res_proj = nn.Linear(in_features, out_features, bias=False)
        else:
            self.res_proj = nn.Identity()

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        N = h.size(0)
        E = edge_index.size(1)

        h_proj = self.res_proj(h)
        if E == 0:
            return self.layer_norm(h_proj), torch.zeros((0, 1), device=h.device)

        src_idx = edge_index[0]
        dst_idx = edge_index[1]

        h_src = h[src_idx]
        if edge_attr is not None and self.edge_dim > 0:
            msg_input = torch.cat([h_src, edge_attr], dim=-1)
        else:
            dummy_edge = torch.zeros((E, self.edge_dim), device=h.device)
            msg_input = torch.cat([h_src, dummy_edge], dim=-1)

        messages = self.msg_mlp(msg_input)  # (E, out_features)

        # Aggregate incoming messages
        agg = torch.zeros((N, self.out_features), device=h.device, dtype=torch.float32)
        agg.index_add_(0, dst_idx, messages)

        # Gated GRU update
        h_updated = self.gru_cell(agg, h_proj)
        h_out = self.layer_norm(h_updated)

        return h_out, torch.ones((E, 1), device=h.device)


class GlobalAttentionPooling(nn.Module):
    """
    Learned attention-based graph pooling layer.
    Computes a single vector representation for each graph in the batch:
    h_G = sum_i (softmax_i(w^T h_i) * h_i)
    """

    def __init__(self, in_features: int, hidden_dim: Optional[int] = None):
        super().__init__()
        h_dim = hidden_dim or max(16, in_features // 2)
        self.attn_net = nn.Sequential(
            nn.Linear(in_features, h_dim),
            nn.Tanh(),
            nn.Linear(h_dim, 1),
        )

    def forward(
        self,
        h: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        h : torch.Tensor of shape (N, in_features)
        batch : torch.Tensor of shape (N,), optional
            Batch indices mapping each node to its graph in the batch.

        Returns
        -------
        graph_embed : torch.Tensor of shape (num_graphs, in_features)
        node_weights : torch.Tensor of shape (N, 1)
        """
        N = h.size(0)
        if N == 0:
            return torch.zeros((0, h.size(1)), device=h.device), torch.zeros((0, 1), device=h.device)

        if batch is None:
            batch = torch.zeros(N, dtype=torch.long, device=h.device)

        num_graphs = int(batch.max().item()) + 1 if N > 0 else 1

        # Unnormalized score per node: (N, 1)
        scores = self.attn_net(h)
        scores_clamped = torch.clamp(scores, -15.0, 15.0)
        exp_scores = torch.exp(scores_clamped)

        # Sum exp per graph in batch
        denom = torch.zeros((num_graphs, 1), device=h.device)
        denom.index_add_(0, batch, exp_scores)
        denom = denom + 1e-9

        weights = exp_scores / denom[batch]  # (N, 1)
        weighted_nodes = weights * h  # (N, in_features)

        # Sum per graph
        pooled = torch.zeros((num_graphs, h.size(1)), device=h.device)
        pooled.index_add_(0, batch, weighted_nodes)

        return pooled, weights


class LogGNNAnomalyDetector(nn.Module):
    """
    LogGNNAnomalyDetector: Graph Neural Network for Execution Graph Anomaly Detection.

    Combines:
    - Node embedding projection + structural feature concatenation
    - Stacked Graph Attention (GAT) or Gated Graph Conv layers with edge attributes
    - Multi-scale Graph-level readout (Attention pooling / Mean-Max concat)
    - Topological anomaly classification head predicting graph structural anomaly probability.
    """

    def __init__(
        self,
        node_in_dim: int = 16,
        edge_in_dim: int = 4,
        hidden_dim: int = 64,
        out_dim: int = 2,
        num_layers: int = 2,
        nheads: int = 4,
        dropout: float = 0.1,
        gnn_type: str = "gat",
        pooling: str = "attention",
        vocab_size: Optional[int] = 64,
        use_embedding: bool = True,
        embedding_dim: int = 32,
    ):
        """
        Parameters
        ----------
        node_in_dim : int, default=16
            Dimension of input continuous node features.
        edge_in_dim : int, default=4
            Dimension of edge attribute vectors.
        hidden_dim : int, default=64
            Hidden embedding dimension across GNN layers.
        out_dim : int, default=2
            Output classification dimension (2 for binary normal vs anomaly).
        num_layers : int, default=2
            Number of message passing layers.
        nheads : int, default=4
            Number of attention heads (for GAT).
        dropout : float, default=0.1
            Dropout rate.
        gnn_type : str, default='gat'
            Type of message passing: 'gat' or 'gated'.
        pooling : str, default='attention'
            Readout mechanism: 'attention', 'mean', 'max', or 'concat'.
        vocab_size : int, optional, default=64
            Size of event vocabulary for discrete node ID embedding.
        use_embedding : bool, default=True
            Whether to embed discrete event IDs.
        embedding_dim : int, default=32
            Dimension of discrete event embedding.
        """
        super().__init__()
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.nheads = nheads
        self.gnn_type = gnn_type.lower()
        self.pooling_type = pooling.lower()
        self.vocab_size = vocab_size or 64
        self.use_embedding = use_embedding
        self.embedding_dim = embedding_dim

        # 1. Node Input Projection
        if self.use_embedding:
            self.node_embed = nn.Embedding(self.vocab_size, embedding_dim)
            initial_dim = embedding_dim + node_in_dim
        else:
            self.node_embed = None
            initial_dim = node_in_dim

        self.input_proj = nn.Sequential(
            nn.Linear(initial_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 2. GNN Backbone Layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if self.gnn_type == "gated":
                layer = GatedGraphConvLayer(
                    in_features=hidden_dim,
                    out_features=hidden_dim,
                    edge_dim=edge_in_dim,
                    dropout=dropout,
                )
            else:
                layer = GraphAttentionLayer(
                    in_features=hidden_dim,
                    out_features=hidden_dim,
                    edge_dim=edge_in_dim,
                    nheads=nheads,
                    dropout=dropout,
                )
            self.layers.append(layer)

        # 3. Graph Readout / Pooling
        if self.pooling_type == "attention":
            self.readout = GlobalAttentionPooling(hidden_dim)
            readout_dim = hidden_dim
        elif self.pooling_type == "concat":
            self.readout = None
            readout_dim = hidden_dim * 2
        else:
            self.readout = None
            readout_dim = hidden_dim

        # 4. Anomaly Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(readout_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        node_ids: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through GNN anomaly detector.

        Parameters
        ----------
        x : torch.Tensor of shape (N, node_in_dim)
            Continuous node features.
        edge_index : torch.Tensor of shape (2, E)
            COO edge connectivity.
        edge_attr : torch.Tensor of shape (E, edge_in_dim), optional
            Edge attribute features.
        node_ids : torch.Tensor of shape (N,), optional
            Categorical event IDs.
        batch : torch.Tensor of shape (N,), optional
            Batch assignment vector.

        Returns
        -------
        output : Dict[str, torch.Tensor]
            Dictionary containing:
            - 'logits': (B, out_dim) classification logits
            - 'probs': (B, out_dim) probabilities
            - 'anomaly_score': (B,) anomaly probability
            - 'graph_embedding': (B, hidden_dim) graph representation
            - 'node_embeddings': (N, hidden_dim) updated node representations
            - 'attn_weights': Edge attention weights (if GAT)
        """
        N = x.size(0)
        device = x.device

        if batch is None:
            batch = torch.zeros(N, dtype=torch.long, device=device)

        num_graphs = int(batch.max().item()) + 1 if N > 0 else 1

        # Handle empty graph edge case
        if N == 0:
            return {
                "logits": torch.zeros((num_graphs, self.out_dim), device=device),
                "probs": torch.zeros((num_graphs, self.out_dim), device=device),
                "anomaly_score": torch.zeros((num_graphs,), device=device),
                "graph_embedding": torch.zeros((num_graphs, self.hidden_dim), device=device),
                "node_embeddings": torch.zeros((0, self.hidden_dim), device=device),
                "attn_weights": torch.zeros((0, self.nheads), device=device),
            }

        # 1. Input embedding and projection
        if self.use_embedding and self.node_embed is not None:
            if node_ids is None:
                node_ids = torch.zeros(N, dtype=torch.long, device=device)
            clamped_ids = torch.clamp(node_ids, 0, self.vocab_size - 1)
            emb = self.node_embed(clamped_ids)
            h_in = torch.cat([emb, x], dim=-1)
        else:
            h_in = x

        h = self.input_proj(h_in)

        # 2. Message Passing through GNN layers
        all_attn = []
        for layer in self.layers:
            h, attn = layer(h, edge_index, edge_attr=edge_attr)
            all_attn.append(attn)

        # 3. Readout / Pooling
        node_importance = None
        if self.pooling_type == "attention" and self.readout is not None:
            h_graph, node_importance = self.readout(h, batch=batch)
        elif self.pooling_type == "concat":
            # Concat mean and max
            mean_pool = torch.zeros((num_graphs, self.hidden_dim), device=device)
            counts = torch.zeros((num_graphs, 1), device=device)
            ones = torch.ones((N, 1), device=device)
            mean_pool.index_add_(0, batch, h)
            counts.index_add_(0, batch, ones)
            mean_pool = mean_pool / torch.clamp(counts, min=1.0)

            # Max pool approximation
            max_pool = torch.zeros((num_graphs, self.hidden_dim), device=device)
            max_pool.index_add_(0, batch, torch.relu(h))
            h_graph = torch.cat([mean_pool, max_pool], dim=-1)
        elif self.pooling_type == "max":
            max_pool = torch.zeros((num_graphs, self.hidden_dim), device=device)
            max_pool.index_add_(0, batch, torch.relu(h))
            h_graph = max_pool
        else:
            # Mean pool
            mean_pool = torch.zeros((num_graphs, self.hidden_dim), device=device)
            counts = torch.zeros((num_graphs, 1), device=device)
            ones = torch.ones((N, 1), device=device)
            mean_pool.index_add_(0, batch, h)
            counts.index_add_(0, batch, ones)
            h_graph = mean_pool / torch.clamp(counts, min=1.0)

        # 4. Classification Head
        logits = self.classifier(h_graph)
        probs = F.softmax(logits, dim=-1)
        anomaly_score = probs[:, 1] if self.out_dim >= 2 else torch.sigmoid(logits[:, 0])

        return {
            "logits": logits,
            "probs": probs,
            "anomaly_score": anomaly_score,
            "graph_embedding": h_graph,
            "node_embeddings": h,
            "attn_weights": all_attn[-1] if all_attn else None,
            "node_importance": node_importance,
        }

    def predict_graph(
        self,
        graph_or_tensors: Union[nx.DiGraph, Dict[str, torch.Tensor], Sequence[Union[int, str]]],
        builder: Optional[LogGraphBuilder] = None,
        threshold: float = 0.5,
        reference_graph: Optional[nx.DiGraph] = None,
    ) -> Dict[str, Any]:
        """
        Diagnose whether an execution graph / sequence is an anomaly.

        Parameters
        ----------
        graph_or_tensors : Union[nx.DiGraph, Dict[str, torch.Tensor], Sequence[Union[int, str]]]
            Input graph, tensor dictionary, or raw event sequence.
        builder : LogGraphBuilder, optional
            Builder instance to construct tensors if not already built.
        threshold : float, default=0.5
            Decision threshold for anomaly classification.
        reference_graph : nx.DiGraph, optional
            Normal reference graph for topological comparison.

        Returns
        -------
        result : Dict[str, Any]
            Detailed prediction report with confidence, embedding, and topological flags.
        """
        self.eval()
        gb = builder or LogGraphBuilder(vocab_size=self.vocab_size)

        nx_graph = None
        if isinstance(graph_or_tensors, nx.DiGraph):
            nx_graph = graph_or_tensors
            tensor_dict = gb.graph_to_tensors(nx_graph)
        elif isinstance(graph_or_tensors, dict) and "x" in graph_or_tensors:
            tensor_dict = graph_or_tensors
        elif isinstance(graph_or_tensors, (list, tuple, np.ndarray)):
            nx_graph = gb.build_graph(graph_or_tensors)
            tensor_dict = gb.graph_to_tensors(nx_graph)
        else:
            raise TypeError(f"Unsupported input type: {type(graph_or_tensors)}")

        device = next(self.parameters()).device
        x = tensor_dict["x"].to(device)
        edge_index = tensor_dict["edge_index"].to(device)
        edge_attr = tensor_dict["edge_attr"].to(device) if "edge_attr" in tensor_dict else None
        node_ids = tensor_dict.get("node_ids", None)
        if node_ids is not None:
            node_ids = node_ids.to(device)

        with torch.no_grad():
            out = self.forward(x=x, edge_index=edge_index, edge_attr=edge_attr, node_ids=node_ids)

        anomaly_score = float(out["anomaly_score"][0].cpu().item())
        probs = out["probs"][0].cpu().numpy().tolist()
        is_anomaly = bool(anomaly_score >= threshold)
        confidence = float(max(probs))
        graph_embed = out["graph_embedding"][0].cpu().numpy()

        # Structural flags
        topo_flags = {}
        if nx_graph is not None:
            topo_flags = gb.detect_structural_anomalies(nx_graph, reference_graph=reference_graph)

        # Node importance
        node_imp = {}
        if out["node_importance"] is not None:
            weights = out["node_importance"].cpu().numpy().flatten()
            node_map = tensor_dict.get("node_map", {})
            inv_map = {idx: nid for nid, idx in node_map.items()}
            for idx, w in enumerate(weights):
                nid = inv_map.get(idx, f"node_{idx}")
                node_imp[str(nid)] = float(w)

        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": anomaly_score,
            "confidence": confidence,
            "probabilities": probs,
            "graph_embedding": graph_embed,
            "node_importance": node_imp,
            "topological_flags": topo_flags,
            "num_nodes": int(tensor_dict["num_nodes"].item()) if "num_nodes" in tensor_dict else x.size(0),
            "num_edges": int(tensor_dict["num_edges"].item()) if "num_edges" in tensor_dict else edge_index.size(1),
        }

    def save(self, outfile: str):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(os.path.abspath(outfile)), exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "node_in_dim": self.node_in_dim,
                "edge_in_dim": self.edge_in_dim,
                "hidden_dim": self.hidden_dim,
                "out_dim": self.out_dim,
                "num_layers": self.num_layers,
                "nheads": self.nheads,
                "gnn_type": self.gnn_type,
                "pooling": self.pooling_type,
                "vocab_size": self.vocab_size,
                "use_embedding": self.use_embedding,
                "embedding_dim": self.embedding_dim,
            },
            outfile,
        )

    @classmethod
    def load(cls, infile: str, device: Optional[torch.device] = None, **kwargs) -> "LogGNNAnomalyDetector":
        """Load model checkpoint."""
        checkpoint = torch.load(infile, map_location=device)
        params = {
            "node_in_dim": checkpoint.get("node_in_dim", 16),
            "edge_in_dim": checkpoint.get("edge_in_dim", 4),
            "hidden_dim": checkpoint.get("hidden_dim", 64),
            "out_dim": checkpoint.get("out_dim", 2),
            "num_layers": checkpoint.get("num_layers", 2),
            "nheads": checkpoint.get("nheads", 4),
            "gnn_type": checkpoint.get("gnn_type", "gat"),
            "pooling": checkpoint.get("pooling", "attention"),
            "vocab_size": checkpoint.get("vocab_size", 64),
            "use_embedding": checkpoint.get("use_embedding", True),
            "embedding_dim": checkpoint.get("embedding_dim", 32),
        }
        params.update(kwargs)
        model = cls(**params)
        model.load_state_dict(checkpoint["state_dict"])
        if device is not None:
            model = model.to(device)
        return model
