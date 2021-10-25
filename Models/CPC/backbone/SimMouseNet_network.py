import networkx as nx
import matplotlib.pyplot as plt

class Network:
    def __init__(self):
        self.AREAS_LIST = ['Retina','VISp', 'VISl','VISal','VISpm','VISam'] #'LGN',
        self.AREAS_CONNECTIVITY = [('Retina','VISp'), ('VISp','VISl'), ('VISp','VISal'), ('VISp','VISpm'), ('VISp','VISam')] #('Retina','LGN'), ('LGN','VISp'),
        self.OUTPUT_AREA_LIST = ['VISl','VISal','VISpm','VISam']

    def create_graph(self):

        self.G = nx.DiGraph()

        self.G.add_nodes_from(self.AREAS_LIST)
        self.G.add_edges_from(self.AREAS_CONNECTIVITY)

    def draw_graph(self):

        plt.figure(figsize=(12,12))
        pos = nx.nx_pydot.graphviz_layout(self.G, prog='dot')
        nx.draw(self.G, pos, with_labels=True, font_weight='bold')
        plt.show()


