import numpy as np


def compute_hop_distance_matrix(adjacency_matrix, max_distance):
    arrival_matrices = determine_arrival_matrices(
        adjacency_matrix, max_distance)
    
    distance_matrix = assign_hop_distances(
        arrival_matrices, max_distance)

    return distance_matrix


def determine_arrival_matrices(adjacency_matrix, max_distance):
    transfer_matrices = calculate_transfer_matrices(
        adjacency_matrix, max_distance)
    
    return np.stack(transfer_matrices) > 0


def calculate_transfer_matrices(adjacency_matrix, max_distance):
    return [np.linalg.matrix_power(adjacency_matrix, d) for d in range(max_distance + 1)]


def assign_hop_distances(arrival_matrices, max_distance):
    distance_matrix = initialize_distance_matrix(arrival_matrices[0])

    for d in range(max_distance, -1, -1):
        distance_matrix[arrival_matrices[d]] = d

    return distance_matrix


def initialize_distance_matrix(matrix):
    return np.zeros_like(matrix) + np.inf


def normalize_adjacency_matrix(matrix):
    degree_matrix = compute_degree_matrix(matrix)

    return np.dot(np.dot(degree_matrix, matrix), degree_matrix)


def compute_degree_matrix(matrix):
    degree_vector = calculate_degree_vector(matrix)
    
    return np.diag(np.where(degree_vector > 0, degree_vector ** (-0.5), 0))


def calculate_degree_vector(matrix):
    return np.sum(matrix, axis=0)