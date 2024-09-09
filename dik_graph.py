import networkx as nx
import matplotlib.pyplot as plt
import json
from variables import variables

def plotgraph(graph_data,pos, source_node=0, path=None):
    # Create a directed graph to reflect the bidirectional nature
    G = nx.DiGraph()

    # Add nodes to the graph
    G.add_nodes_from(graph_data['nodes'])

    # Add edges with weights to the graph
    for edge, weight in graph_data['edges'].items():
        G.add_edge(edge[0], edge[1], weight=weight)


   


    # Plot the graph
    plt.figure(figsize=(10, 6))

    # Draw the nodes with different sizes and colors
    nx.draw_networkx_nodes(G, pos, node_size=[4500 if n == 5 else 3500 for n in G.nodes],
                           node_color=['pink' if n == source_node else 'green' if n in [3, 5, 7] else 'lightgreen' for n in G.nodes])

    # Draw the edges with different colors for each direction
    nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v in G.edges if graph_data['edges'][(u, v)] in [15,30]],
                           width=2, edge_color='red', arrowsize=20, connectionstyle="arc3")

    nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v in G.edges if graph_data['edges'][(u, v)] == 45],
                           width=2, edge_color='orange', arrowsize=20, connectionstyle="arc3")

    nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v in G.edges if graph_data['edges'][(u, v)] == 60],
                           width=2, edge_color='blue', arrowsize=20, connectionstyle="arc3")

    # Highlight the path
    if path:
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=4, edge_color='yellow', arrowsize=30, connectionstyle="arc3")
        nx.draw_networkx_nodes(G, pos, nodelist=path, node_size=5000, node_color='yellow')

    # Add labels to the nodes
    nx.draw_networkx_labels(G, pos, font_size=16, font_weight='bold')

    # Add edge labels
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black', font_size=12)

    plt.title(f"Shortest Path from Node {source_node}" if path else "Bidirectional Graph with Weights")
    plt.axis('off')  # Turn off the axis
    plt.show()


# positions for each node to match the image layout
obj = variables()
pos = obj.pos    

graph_data = obj.graph_data

# Create a directed graph
G = nx.DiGraph()

# Add nodes and edges to the graph
G.add_nodes_from(graph_data['nodes'])
for edge, weight in graph_data['edges'].items():
    G.add_edge(edge[0], edge[1], weight=weight)

# Compute shortest paths using Dijkstra's algorithm
shortest_paths = {}
shortest_paths_length = {}
for node in G.nodes:
    length, path = nx.single_source_dijkstra(G, source=node)
    shortest_paths[node] = path
    shortest_paths_length[node] = length

# Dictionary to save the data
shortest_paths_data = {}
charging_stations = obj.charging_stations
# Collect the shortest paths data
for source, target_dict in shortest_paths.items():
    shortest_paths_data[source] = {}
    for target, path in target_dict.items():
        if source != target:
            distance = shortest_paths_length[source][target]
            shortest_paths_data[source][target] = {
                'path': path,
                'distance': distance, 
            }
            
            if source in charging_stations:
                shortest_paths_data[source]['dis_to_charging_station']=0
                shortest_paths_data[source]['nearest_charging_station']=source
                shortest_paths_data[source]['path_to_charging_station']=[source]
            else:
                ch_dist = []
                for ch in charging_stations:
                    ch_dist.append(shortest_paths_length[source][ch])
                
                nearest_charging_station = ch_dist.index(min(ch_dist))  
                shortest_paths_data[source]['dis_to_charging_station']=ch_dist[nearest_charging_station]
                shortest_paths_data[source]['nearest_charging_station']=charging_stations[nearest_charging_station]
                shortest_paths_data[source]['path_to_charging_station']=shortest_paths[source][charging_stations[nearest_charging_station]]
                
            print(f"Shortest path from {source} to {target} is {path} with distance {distance}")
            plotgraph(graph_data,pos, source, path)

# # Write the data to a JSON file
with open('data/shortest_paths.json', 'w') as f:
    json.dump(shortest_paths_data, f, indent=4)
