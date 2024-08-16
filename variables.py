class variables:
    def __init__(self):
        # Define positions for each node to match the image layout
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
                (1, 2): 15, (1, 3): 25, (1, 4): 35, (1, 5): 35, (1, 7): 15,
                (2, 1): 15, (2, 3): 15, (2, 5): 35, (2, 6): 35, (2, 7): 25,
                (3, 1): 25, (3, 2): 15, (3, 4): 15, (3, 5): 25, (3, 6): 25, (3, 7): 15,
                (4, 1): 35, (4, 3): 15, (4, 5): 15, (4, 6): 15, (4, 7): 25,
                (5, 1): 35, (5, 2): 35, (5, 3): 25, (5, 4): 15, (5, 6): 15, (5, 7): 25,
                (6, 2): 35, (6, 3): 25, (6, 4): 15, (6, 5): 15, (6, 7): 15,
                (7, 1): 15, (7, 2): 25, (7, 3): 15, (7, 4): 25, (7, 5): 25, (7, 6): 15
            }
        }
        
        # Define the charging stations
        self.charging_stations = [5,7,3]
        
        
        
        
        
        