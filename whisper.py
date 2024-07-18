import numpy as np
import networkx as nx
import random

class Node:
    """
    Most of the functions here are from the Cogworks documentation,
    but I need to fix some stuff before it is fully completed
    """
    def __init__(self, ID, neighbors, descriptor, truth, file_path):
        self.id = ID
        self.label = ID
        self.neighbors = tuple(neighbors)
        self.descriptor = descriptor
        self.truth = truth
        self.file_path = file_path

def cos_dist(a,b):
    return 1 - np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))

def create_adjacency_matrix(nodes, threshold):
    """
    !This has a slight problem to be fixed,
    """
    num_nodes = len(nodes)
    adj_matrix = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            dist = cos_dist(nodes[i].descriptor, nodes[j].descriptor)
            if dist < threshold:
                weight = 1 - dist
                adj_matrix[i, j] = weight
                adj_matrix[j, i] = weight

    return adj_matrix

def connected_components(nodes, adj_matrix):
    graph = nx.Graph()
    #* I still haven't full figured this part out,
    #* still work in progress
    graph.add_nodes_from(range(len(nodes)))
    graph.add_edges_from(zip(*np.where(np.triu(adj_matrix) > 0))) # np.triu is for getting the diagonal thingys
    # components = list(map(to_list, nx.connected_components(graph)))
    components = [list(c) for c in nx.connected_components(graph)]
    return [[nodes[i] for i in comp] for comp in components]

def propagate_label(node, nodes, adj_matrix):
    label_weights = {}
    for neighbor_id in node.neighbors:
        neighbor = nodes[neighbor_id]
        weight = adj_matrix[node.id, neighbor_id]
        label_weights[neighbor.label] = label_weights.get(neighbor.label, 0) + weight

    if label_weights:
        max_label = max(label_weights, key=label_weights.get)
        node.label = max_label

def whispers(nodes, adj_matrix, max_iterations=100):
    num_nodes = len(nodes)
    for i in range(max_iterations):
        node = random.choice(nodes)
        propagate_label(node, nodes, adj_matrix)
        if len(set(node.label for node in nodes)) == 1:
            break

def plot_graph(nodes, adj_matrix):
    """
    *I have to research how to plot nodes(If you know how to; feel free to add the code)
    """
    