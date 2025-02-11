import numpy as np
from .graph_utils import compute_hop_distance_matrix, normalize_adjacency_matrix


class GraphBuilder():
    """
    Args:
        skeleton_layout ():
        skeleton_center ():
        partition_strategy (string): must be one of the follow candidates
        - uniform : Uniform Labeling
        - distance : Distance Partitioning
        - spatial : Spatial Configuration
        max_hop_distance (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """
    def __init__(
            self,
            skeleton_layout,
            skeleton_center,
            partition_strategy="uniform",
            max_hop_distance=1,
            dilation=1):
        
        self.skeleton_center = skeleton_center
        self.max_hop_distance = max_hop_distance
        self.dilation = dilation

        self.set_number_of_nodes(skeleton_layout)
        self.get_graph_edges(skeleton_layout)

        self.adjacency_matrix = self.build_adjacency_matrix()
        self.distance_matrix = compute_hop_distance_matrix(
            self.adjacency_matrix, self.max_hop_distance)

        self.valid_hop_distances = self._generate_valid_hop_distances()

        self.compute_label_map(partition_strategy)


    def __str__(self):
        return self.label_map


    def set_number_of_nodes(self, layout):
        self.num_nodes = layout.num_elements()


    def get_graph_edges(self, layout):
        self_loop_edges = self._generate_self_loops()
        layout_edges = self._fetch_layout_connections(layout)

        self.edges = self_loop_edges + layout_edges
    

    def _generate_self_loops(self):
        return [(i, i) for i in range(self.num_nodes)]
    

    def _fetch_layout_connections(self, layout):
        return list(layout.get_connections())


    def build_adjacency_matrix(self):
        adjacency_matrix = self._initialize_matrix()

        for i, j in self.edges:
            adjacency_matrix[i, j] = 1
            adjacency_matrix[j, i] = 1

        return adjacency_matrix


    def _initialize_matrix(self):
        return np.zeros((self.num_nodes, self.num_nodes))
    

    def build_dilated_adjacency_matrix(self):
        dilated_adjacency_matrix = self._initialize_matrix()

        for current_distance in self.valid_hop_distances:
            dilated_adjacency_matrix[
                self.distance_matrix == current_distance
            ] = 1

        return dilated_adjacency_matrix


    def _generate_valid_hop_distances(self):
        return range(0, self.max_hop_distance + 1, self.dilation)


    def compute_label_map(self, strategy):
        dilated_adjacency_matrix = self.build_dilated_adjacency_matrix()
        normalized_matrix = normalize_adjacency_matrix(
            dilated_adjacency_matrix)
        
        self.label_map = self._select_strategy(strategy, normalized_matrix)
        

    def _select_strategy(self, strategy, normalized_matrix):
        strategy_methods = {
            'uniform': self._apply_uniform_strategy,
            'distance': self._apply_distance_strategy,
            'spatial': self._apply_spatial_strategy
        }

        if strategy not in strategy_methods:
            raise ValueError(f"Unknown strategy: {strategy}.")
        
        return strategy_methods[strategy](normalized_matrix)

    
    def _apply_uniform_strategy(self, normalized_matrix):
        return np.expand_dims(normalized_matrix, axis=0)
        

    def _apply_distance_strategy(self, normalized_matrix):
        label_map = np.tile(
            self._initialize_matrix(), (len(self.valid_hop_distances), 1, 1))

        for i, current_distance in enumerate(self.valid_hop_distances):
            distance_mask = self.distance_matrix == current_distance
            label_map[i][distance_mask] = normalized_matrix[distance_mask]

        return label_map


    def _apply_spatial_strategy(self, normalized_matrix):
        label_map = []

        for current_distance in self.valid_hop_distances:
            root = self._initialize_matrix()
            close = self._initialize_matrix()
            further = self._initialize_matrix()

            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    if self.distance_matrix[j, i] == current_distance:
                        
                        if self.distance_matrix[
                            j, self.skeleton_center] == self.distance_matrix[i, self.skeleton_center]:
                            root[j, i] = normalized_matrix[j, i]

                        elif self.distance_matrix[
                            j, self.skeleton_center] > self.distance_matrix[i, self.skeleton_center]:
                            close[j, i] = normalized_matrix[j, i]
                        
                        else:
                            further[j, i] = normalized_matrix[j, i]

            if current_distance == 0:
                label_map.append(root)
            else:
                label_map.append(root + close)
                label_map.append(further)
        
        return np.stack(label_map)