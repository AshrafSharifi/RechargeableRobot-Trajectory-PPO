import networkx as nx
import matplotlib.pyplot as plt

def filter_edges_by_target(graph_data, target):
    # Filter the edges based on the target node
    filtered_edges = {edge: weight for edge, weight in graph_data['edges'].items() if edge[0] == target}
    # Update the graph data with the filtered edges
    graph_data['edges'] = filtered_edges
    return graph_data

def plotgraph(graph_data, source_node=0, path=None):
    # Create a directed graph to reflect the bidirectional nature
    G = nx.DiGraph()

    # Add nodes to the graph
    G.add_nodes_from(graph_data['nodes'])

    # Add edges with weights to the graph
    for edge, weight in graph_data['edges'].items():
        G.add_edge(edge[0], edge[1], weight=weight)

    # Define positions for each node to match the image layout
    pos = {
        1: (4, 4),
        2: (4, 2),
        3: (3, 2),
        4: (2, 2),
        5: (1, 3),
        6: (2, 4),
        7: (3, 4)
    }

    # Plot the graph
    plt.figure(figsize=(10, 6))

    # Draw the nodes with different sizes and colors
    nx.draw_networkx_nodes(G, pos, node_size=[4500 if n == 5 else 3500 for n in G.nodes],
                           node_color=['pink' if n == source_node else 'green' if n in [3, 5, 7] else 'lightgreen' for n in G.nodes])

    # Draw the edges with different colors for each direction
    nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v in G.edges if graph_data['edges'][(u, v)] == 15],
                           width=2, edge_color='red', arrowsize=20, connectionstyle="arc3")

    nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v in G.edges if graph_data['edges'][(u, v)] == 25],
                           width=2, edge_color='orange', arrowsize=20, connectionstyle="arc3")

    nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v in G.edges if graph_data['edges'][(u, v)] == 35],
                           width=2, edge_color='blue', arrowsize=20, connectionstyle="arc3")

    # Highlight the path
    if path:
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=4, edge_color='yellow', arrowsize=25, connectionstyle="arc3")
        nx.draw_networkx_nodes(G, pos, nodelist=path, node_size=5000, node_color='yellow')

    # Add labels to the nodes
    nx.draw_networkx_labels(G, pos, font_size=16, font_weight='bold')

    # Add edge labels
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black', font_size=12)

    plt.title(f"Shortest Path from Node {source_node}" if path else "Bidirectional Graph with Weights")
    plt.axis('off')  # Turn off the axis
    plt.show()
    
    
# Define the graph data based on the extracted information
graph_data = {
    'nodes': [1, 2, 3, 4, 5, 6, 7],
    'edges': {
        (1, 2): 15, (1, 3): 25, (1, 4): 35, (1, 5): 35, (1, 7): 15,
        (2, 1): 15, (2, 3): 15, (2, 5): 35, (2, 6): 35, (2, 7): 25,
        (3, 1): 25, (3, 2): 15, (3, 4): 15, (3, 5): 25, (3, 6): 25, (3, 7): 15,
        (4, 1): 35, (4, 3): 15, (4, 5): 15, (4, 6): 15, (4, 7): 25,
        (5, 1): 35, (5, 2): 35, (5, 3): 25, (5, 4): 15, (5, 6): 15, (5, 7): 25,
        (6, 2): 35, (6, 3): 25, (6, 4): 15, (6, 5): 15, (6, 7): 15 ,
        (7, 1): 15, (7, 2): 25, (7, 3): 15, (7, 4): 25, (7, 5): 25, (7, 6): 15
    }
}


plotgraph(graph_data,0)


# Create a directed graph
G = nx.DiGraph()

# Add nodes and edges to the graph
G.add_nodes_from(graph_data['nodes'])
for edge, weight in graph_data['edges'].items():
    G.add_edge(edge[0], edge[1], weight=weight)

# Compute all-pairs shortest paths using Floyd-Warshall algorithm
shortest_paths_length = dict(nx.floyd_warshall(G))
shortest_paths = dict(nx.all_pairs_shortest_path(G))

# Print the shortest paths and their lengths between all pairs of nodes
print("Shortest paths and distances between all pairs of nodes:")
for source, target_dict in shortest_paths.items():
    for target, path in target_dict.items():
        if source != target:
            distance = shortest_paths_length[source][target]
            print(f"Shortest path from {source} to {target} is {path} with distance {distance}")
            plotgraph(graph_data, source, path)


