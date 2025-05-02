import matplotlib.pyplot as plt
import networkx as nx

def create_tree(graph, positions, node_labels=None, node_size=2000, font_size=10):
    """
    Creates and visualizes a tree graph using Matplotlib.

    Args:
        graph: A networkx graph object representing the tree.
        positions: A dictionary specifying node positions.
        node_labels: An optional dictionary specifying node labels.
        node_size: Size of the nodes.
        font_size: Font size for node labels.
    """
    plt.figure(figsize=(8, 6))
    nx.draw(graph, pos=positions, with_labels=True, labels=node_labels, node_size=node_size, node_color="skyblue", font_size=font_size, font_weight="bold", edge_color="gray")
    plt.title("Tree Visualization")
    plt.show()

if __name__ == "__main__":
    # Create a graph
    tree = nx.Graph()
    tree.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'D'), ('B', 'E'), ('C', 'F'), ('C', 'G')])

    # Define node positions 
    positions = {'A': (0, 0), 'B': (-2, -1), 'C': (2, -1), 'D': (-3, -2), 'E': (-1, -2), 'F': (1, -2), 'G': (3, -2)}

    # Define node labels (optional)
    node_labels = {'A': 'Root', 'B': 'Child 1', 'C': 'Child 2', 'D': 'Leaf 1', 'E': 'Leaf 2', 'F': 'Leaf 3', 'G': 'Leaf 4'}

    # Create and visualize the tree
    create_tree(tree, positions, node_labels=node_labels)