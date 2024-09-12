import json
import networkx as nx
import matplotlib.pyplot as plt
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Rectangle

class variables:
    def __init__(self):
        # Define self.positions for each node to match the image layout
        self.pos = {
            1: (4, 4),
            2: (4, 2),
            3: (3, 2),
            4: (2, 2),
            5: (1, 3),
            6: (2, 4),
            7: (3, 4)
        }
        
        # Define the graph data 
        self.graph_data = {
            'nodes': [1, 2, 3, 4, 5, 6, 7],
            'edges': {
                (1, 2): 30, (1, 3): 45, (1, 4): 60, (1, 5): 60, (1, 7): 15,
                (2, 1): 30, (2, 3): 15, (2, 5): 60, (2, 6): 60, (2, 7): 45,
                (3, 1): 45, (3, 2): 15, (3, 4): 15, (3, 5): 45, (3, 6): 45, (3, 7): 30,
                (4, 1): 60, (4, 3): 15, (4, 5): 30, (4, 6): 30, (4, 7): 45,
                (5, 1): 60, (5, 2): 60, (5, 3): 45, (5, 4): 30, (5, 6): 30, (5, 7): 45,
                (6, 2): 60, (6, 3): 45, (6, 4): 30, (6, 5): 30, (6, 7): 15,
                (7, 1): 15, (7, 2): 45, (7, 3): 30, (7, 4): 45, (7, 5): 45, (7, 6): 15
            }
        }
        
        # Define the charging stations
        self.charging_stations = [5,7,3]
        
        self.initialChargingLevel = 300
        
        self.shortest_paths_data = None
        with open('data/shortest_paths.json', 'r') as f:
            self.shortest_paths_data = json.load(f)
            
    def extract_date_time(self,next_state):
        # Extract date and time elements
        year = next_state[0, 1]
        month = next_state[0, 2]
        day = next_state[0, 3]
        hour = next_state[0, 4]
        minute = next_state[0, 5]
        # Create a datetime object
        date_time = datetime(year, month, day, hour, minute)
        # Format the date and time
        formatted_date_time = date_time.strftime("%Y-%m-%d %H:%M")
        return formatted_date_time


    def plotgraph(self, source_node=0, path=None, battery_level=None, reward=None, temperature_difference=None, time=None):
        # Create a directed graph to reflect the bidirectional nature
        G = nx.DiGraph()
    
        # Add nodes to the graph
        G.add_nodes_from(self.graph_data['nodes'])
    
        # Add edges with weights to the graph
        for edge, weight in self.graph_data['edges'].items():
            G.add_edge(edge[0], edge[1], weight=weight)
    
        # Plot the graph
        plt.figure(figsize=(10, 6))
    
        # Draw the nodes with different sizes and colors
        nx.draw_networkx_nodes(G, self.pos, node_size=5000,
                               node_color=['pink' if n == source_node else 'green' if n in [3, 5, 7] else 'lightgreen' for n in G.nodes])
    
        # Draw the edges with different colors for each direction
        nx.draw_networkx_edges(G, self.pos, edgelist=[(u, v) for u, v in G.edges if self.graph_data['edges'][(u, v)] in [15]],
                               width=2, edge_color='green', arrowsize=20, connectionstyle="arc3")
    
        nx.draw_networkx_edges(G, self.pos, edgelist=[(u, v) for u, v in G.edges if self.graph_data['edges'][(u, v)] == 30],
                               width=2, edge_color='orange', arrowsize=20, connectionstyle="arc3")
    
        nx.draw_networkx_edges(G, self.pos, edgelist=[(u, v) for u, v in G.edges if self.graph_data['edges'][(u, v)] == 45],
                               width=2, edge_color='blue', arrowsize=20, connectionstyle="arc3")
        
        nx.draw_networkx_edges(G, self.pos, edgelist=[(u, v) for u, v in G.edges if self.graph_data['edges'][(u, v)] == 60],
                               width=2, edge_color='red', arrowsize=20, connectionstyle="arc3")
    
        # Highlight the path
        if path:
            path_edges = list(zip(path, path[1:]))
            nx.draw_networkx_edges(G, self.pos, edgelist=path_edges, width=4, edge_color='yellow', arrowsize=30, connectionstyle="arc3")
            nx.draw_networkx_nodes(G, self.pos, nodelist=path, node_size=5000, node_color='yellow')
    
            # Annotate the end node with battery level, reward, and temperature difference
            end_node = path[-1]
            
            # Define a color based on the battery level
            if battery_level > 150:
                box_color = 'lightgreen'
            elif 70 < battery_level < 150:
                box_color = 'yellow'
            else:
                box_color = 'red'
            
            annotation_text = f"{time}\n-------------------------\nBattery: {battery_level}\nReward: {reward}\nTemp Diff: {temperature_difference}"
            x, y = self.pos[end_node]
            if end_node in [5, 4, 3, 2]:
                plt.text(x - .8, y - 0.25, annotation_text, fontsize=10, ha='left', va='center', 
                         bbox=dict(facecolor=box_color, alpha=0.8, edgecolor='black'))
            else:
                plt.text(x - .6, y + 0.45, annotation_text, fontsize=10, ha='left', va='center', 
                         bbox=dict(facecolor=box_color, alpha=0.8, edgecolor='black'))
    
        # Add battery symbols to the charging stations
        battery_icon = plt.imread('battery_icon.png')  # Load an image of a battery icon
    
        for node in [5, 3, 7]:
            x, y = self.pos[node]
            imagebox = OffsetImage(battery_icon)
            ab = AnnotationBbox(imagebox, (x - 0.1, y - 0.1), frameon=False)  # Shift the icon to the bottom left
            plt.gca().add_artist(ab)
    
        # Add labels to the nodes
        nx.draw_networkx_labels(G, self.pos, font_size=16, font_weight='bold')
    
        # Add edge labels
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, self.pos, edge_labels=edge_labels, font_color='black', font_size=12)
    
        plt.axis('off')  # Turn off the axis
        plt.show()
 
    def plotgraph__(self, source_node=0, path=None, battery_level=None, reward=None, temperature_difference=None, time=None):
        # Create a directed graph to reflect the bidirectional nature
        G = nx.DiGraph()
    
        # Add nodes to the graph
        G.add_nodes_from(self.graph_data['nodes'])
    
        # Add edges with weights to the graph
        for edge, weight in self.graph_data['edges'].items():
            G.add_edge(edge[0], edge[1], weight=weight)
    
        # Plot the graph
        plt.figure(figsize=(10, 6))
    
        # Draw the nodes with different sizes and colors
        nx.draw_networkx_nodes(G, self.pos, node_size=[4500 if n == 5 else 3500 for n in G.nodes],
                               node_color=['pink' if n == source_node else 'green' if n in [3, 5, 7] else 'lightgreen' for n in G.nodes])
    
        # Draw the edges with different colors for each direction
        nx.draw_networkx_edges(G, self.pos, edgelist=[(u, v) for u, v in G.edges if self.graph_data['edges'][(u, v)] in [15,20]],
                               width=2, edge_color='red', arrowsize=20, connectionstyle="arc3")
    
        nx.draw_networkx_edges(G, self.pos, edgelist=[(u, v) for u, v in G.edges if self.graph_data['edges'][(u, v)] == 30],
                               width=2, edge_color='orange', arrowsize=20, connectionstyle="arc3")
    
        nx.draw_networkx_edges(G, self.pos, edgelist=[(u, v) for u, v in G.edges if self.graph_data['edges'][(u, v)] == 45],
                               width=2, edge_color='blue', arrowsize=20, connectionstyle="arc3")
    
        # Highlight the path
        if path:
            path_edges = list(zip(path, path[1:]))
            nx.draw_networkx_edges(G, self.pos, edgelist=path_edges, width=4, edge_color='yellow', arrowsize=30, connectionstyle="arc3")
            nx.draw_networkx_nodes(G, self.pos, nodelist=path, node_size=5000, node_color='yellow')
    
            # Annotate the end node with battery level, reward, and temperature difference
            end_node = path[-1]
            annotation_text = f"{time}\n-------------------------\nBattery: {battery_level}\nReward: {reward}\nTemp Diff: {temperature_difference}"
            x, y = self.pos[end_node]
            if end_node in [5,4,3,2]:
                plt.text(x - .8, y - 0.25, annotation_text, fontsize=10, ha='left', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
            else:
                plt.text(x - .6, y + 0.45, annotation_text, fontsize=10, ha='left', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
                
    
        # Add labels to the nodes
        nx.draw_networkx_labels(G, self.pos, font_size=16, font_weight='bold')
    
        # Add edge labels
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, self.pos, edge_labels=edge_labels, font_color='black', font_size=12)
    
        # plt.title(f"Shortest Path from Node {source_node}" if path else "Bidirectional Graph with Weights")
        plt.axis('off')  # Turn off the axis
        plt.show()
        
            
            
            
            
            
