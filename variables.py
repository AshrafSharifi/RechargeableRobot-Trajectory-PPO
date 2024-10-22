import json
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Rectangle
import os

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
        # self.graph_data = {
        #     'nodes': [1, 2, 3, 4, 5, 6, 7],
        #     'edges': {
        #         (1, 2): 45, (1, 3): 45, (1, 4): 60, (1, 5): 60, (1, 7): 15,
        #         (2, 1): 45, (2, 3): 15, (2, 5): 60, (2, 6): 60, (2, 7): 45,
        #         (3, 1): 45, (3, 2): 15, (3, 4): 15, (3, 5): 45, (3, 6): 45, (3, 7): 45,
        #         (4, 1): 60, (4, 3): 15, (4, 5): 45, (4, 6): 45, (4, 7): 45,
        #         (5, 1): 60, (5, 2): 60, (5, 3): 45, (5, 4): 45, (5, 6): 45, (5, 7): 45,
        #         (6, 2): 60, (6, 3): 45, (6, 4): 45, (6, 5): 45, (6, 7): 15,
        #         (7, 1): 15, (7, 2): 45, (7, 3): 45, (7, 4): 45, (7, 5): 45, (7, 6): 15
        #     }
        # }
        # self.graph_data = {
        #             'nodes': [1, 2, 3, 4, 5, 6, 7],
        #             'edges': {
        #                 (1, 2): 15, (1, 3): 15, (1, 4): 15, (1, 5): 15, (1, 7): 15,
        #                 (2, 1): 15, (2, 3): 15, (2, 6): 15, (2, 7): 15,
        #                 (3, 1): 15, (3, 2): 15, (3, 4): 15, (3, 5): 15, (3, 6): 15, (3, 7): 15,
        #                 (4, 1): 15, (4, 3): 15, (4, 5): 15, (4, 6): 15, (4, 7): 15,
        #                 (5, 1): 15, (5, 3): 15, (5, 4): 15, (5, 6): 15, (5, 7): 15,
        #                 (6, 2): 15, (6, 3): 15, (6, 4): 15, (6, 5): 15, (6, 7): 15,
        #                 (7, 1): 15, (7, 2): 15, (7, 3): 15, (7, 4): 15, (7, 5): 15, (7, 6): 15
        #             }
        #         }
        
        self.graph_data = {
            'nodes': [1, 2, 3, 4, 5, 6, 7],
            'edges': {
                (1, 2): 45, (1, 3): 45,   (1, 7): 15,
                (2, 1): 45, (2, 3): 45,   (2, 7): 45,
                (3, 1): 45, (3, 2): 45, (3, 4): 45, (3, 5): 15, (3, 6): 45, (3, 7): 45,
                 (4, 3): 45, (4, 5): 15, (4, 6): 45, (4, 7): 15,
                 (5, 3): 15, (5, 4): 15, (5, 6): 15, (5, 7): 15,
                 (6, 3): 45, (6, 4): 45, (6, 5): 15, 
                (7, 1): 15, (7, 2): 45, (7, 3): 45, (7, 4): 15, (7, 5): 15
            }
        }
        
        # Define the charging stations
        self.charging_stations = [3,5,7]
        
        self.initialChargingLevel = 300
 
        self.shortest_paths_data = None
        
     
        

       
            
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
  
    def plot_trajectory(self, complete_path, show_path = True, shortest_path = None):
        # Create a directed graph to reflect the bidirectional nature
        G = nx.DiGraph()
    
        # Add nodes to the graph
        G.add_nodes_from(self.graph_data['nodes'])
    
        # Add edges with weights to the graph
        for edge, weight in self.graph_data['edges'].items():
            G.add_edge(edge[0], edge[1], weight=weight)
    
        # Plot the graph
        plt.figure(figsize=(10, 7))
    
        # Draw the nodes with different sizes and colors
        node_colors = ['green' if n in self.charging_stations else 'lightgreen' for n in G.nodes]
        nx.draw_networkx_nodes(G, self.pos, node_size=3000, node_color=node_colors)
    
        # Define edge colors based on weights
        edge_colors = {15: 'green', 30: 'orange', 45: 'blue', 60: 'red'}
        for weight, color in edge_colors.items():
            nx.draw_networkx_edges(G, self.pos, edgelist=[(u, v) for u, v in G.edges if self.graph_data['edges'][(u, v)] == weight],
                                   width=2, edge_color=color, arrowsize=20, connectionstyle="arc3")
        if show_path:    
            # Process each path in complete_path
            for item in complete_path.values():
        
                path, previous_data, current_data = item[0], item[1], item[2]
                prev_time, prev_battery_level = previous_data
                battery_level, reward, time, dist_to_charging_station = current_data
        
                # Draw the path edges and nodes
                path_edges = list(zip(path, path[1:]))
                nx.draw_networkx_edges(G, self.pos, edgelist=path_edges, width=4, edge_color='yellow', arrowsize=30, connectionstyle="arc3")
                nx.draw_networkx_nodes(G, self.pos, nodelist=path, node_size=3000, node_color='yellow')
        
                # Helper function for setting box color based on battery level
                def get_box_color(battery_level, dist_to_charging_station=None):
                    if battery_level == self.initialChargingLevel:
                        return 'limegreen'
                    elif (.5*self.initialChargingLevel) <= battery_level < self.initialChargingLevel:
                        return 'lightgreen'
                    elif dist_to_charging_station != None and battery_level <= dist_to_charging_station:
                        return 'red'
                    else:
                        return 'yellow'
        
                # Annotate end node
                end_node = path[-1]
                box_color = get_box_color(battery_level, dist_to_charging_station)
                annotation_text = f"End: {time}\n-------------------------\nBattery: {battery_level}\nReward: {reward}"
                x, y = self.pos[end_node]
                plt.text(x-.5 if end_node in [5, 4, 3, 2] else x-.5, y - 0.3 if end_node in [5, 4, 3, 2] else y + 0.35, 
                         annotation_text, fontsize=11,fontweight='bold', ha='left', va='center', bbox=dict(facecolor=box_color, alpha=0.8, edgecolor='black'))
        
                # Annotate start node
                start_node = path[0]
                box_color = get_box_color(prev_battery_level)
                annotation_text = f"Start:\n {prev_time}\n-------------------------\nBattery: {prev_battery_level}"
                x, y = self.pos[start_node]
                plt.text(x-.5 if start_node in [5, 4, 3, 2] else x-.5, y - 0.3 if start_node in [5, 4, 3, 2] else y + 0.35, 
                         annotation_text, fontsize=11,fontweight='bold', ha='left', va='center', bbox=dict(facecolor=box_color, alpha=0.8, edgecolor='black'))
        elif shortest_path:
                path_edges = list(zip(shortest_path, shortest_path[1:]))
                nx.draw_networkx_edges(G, self.pos, edgelist=path_edges, width=4, edge_color='yellow', arrowsize=30, connectionstyle="arc3")
                nx.draw_networkx_nodes(G, self.pos, nodelist=shortest_path, node_size=3000, node_color='yellow')
        
                # Add labels to the nodes
                nx.draw_networkx_labels(G, self.pos, font_size=16, font_weight='bold')
            
                # Add edge labels
                edge_labels = nx.get_edge_attributes(G, 'weight')
                nx.draw_networkx_edge_labels(G, self.pos, edge_labels=edge_labels, font_color='black', font_size=12)
            
                plt.title(f"Shortest Path from Node {shortest_path[0]} to Node {shortest_path[-1]}")
            
        # Add battery symbols to the charging stations
        battery_icon = plt.imread('data/Icons/battery_icon.png')  # Load an image of a battery icon
        for node in self.charging_stations:
            x, y = self.pos[node]
            imagebox = OffsetImage(battery_icon)
            ab = AnnotationBbox(imagebox, (x - 0.1, y), frameon=False)  # Shift the icon to the bottom left
            plt.gca().add_artist(ab)
    
        # Add labels to the nodes
        nx.draw_networkx_labels(G, self.pos, font_size=16, font_weight='bold')
    
        # Add edge labels
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, self.pos, edge_labels=edge_labels, font_color='black', font_size=12)
    
        plt.axis('off')  # Turn off the axis
        plt.show()
    
    def dikestra(self, display = False):
        


        pos = self.pos    

        graph_data = self.graph_data

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
        charging_stations = self.charging_stations
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
                        
                    
                    # plotgraph(graph_data,pos, source, path)
                    if display:
                        print(f"Shortest path from {source} to {target} is {path} with distance {distance}")
                        self.plot_trajectory(None, False, shortest_path = path)
                        
        # # Write the data to a JSON file
      
        return shortest_paths_data
    
    def set_shortest_path(self,path):
        with open(path, 'rb') as f:
            self.shortest_paths_data = json.load(f)

    
