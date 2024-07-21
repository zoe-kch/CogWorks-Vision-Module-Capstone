import numpy as np
import networkx as nx
import random
from main import cos_dist
import matplotlib.pyplot as plt

class Node:
    """
    I tested the code, and there was no error.
    So it should be completed
    """
    def __init__(self, ID, neighbors, descriptor, truth, file_path):
        self.id = ID
        self.label = ID
        self.neighbors = tuple(neighbors)
        self.descriptor = descriptor
        self.truth = truth
        self.file_path = file_path

def create_adjacency_matrix(nodes, threshold):
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
    g = nx.Graph()
    for n, node in enumerate(nodes): #nodes should be tuple
        g.add_node(n)

    # construct a network-x graph from the adjacency matrix: a non-zero entry at adj[i, j]
    # indicates that an egde is present between Node-i and Node-j. Because the edges are
    # undirected, the adjacency matrix must be symmetric, thus we only look ate the triangular
    # upper-half of the entries to avoid adding redundant nodes/edges
    g.add_edges_from(zip(*np.where(np.triu(adj_matrix) > 0)))

    # we want to visualize our graph of nodes and edges; to give the graph a spatial representation,
    # we treat each node as a point in 2D space, and edges like compressed springs. We simulate
    # all of these springs decompressing (relaxing) to naturally space out the nodes of the graph
    # this will hopefully give us a sensible (x, y) for each node, so that our graph is given
    # a reasonable visual depiction
    pos = nx.spring_layout(g)

    # make a mapping that maps: node-lab -> color, for each unique label in the graph
    color = list(iter(cm.tab20b(np.linspace(0, 1, len(set(i.label for i in nodes))))))
    color_map = dict(zip(sorted(set(i.label for i in nodes)), color))
    colors = [color_map[i.label] for i in nodes]  # the color for each node in the graph, according to the node's label

    # render the visualization of the graph, with the nodes colored based on their labels!
    fig, ax = plt.subplots()
    nx.draw_networkx_nodes(g, pos=pos, ax=ax, nodelist=range(len(nodes)), node_color=colors)
    nx.draw_networkx_edges(g, pos, ax=ax, edgelist=g.edges())
    return fig, ax
