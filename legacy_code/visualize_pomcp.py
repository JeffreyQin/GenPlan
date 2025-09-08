# pylint: disable=no-member
import numpy as np
import pygame
from tree_builder import Cell, Action, Node
from generator import Generator
from legacy_code.pomcp import POMCP
import networkx as nx
import matplotlib.pyplot as plt


# Setup colors and cell size
CELL_SIZE = 50
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
GRAY = (200, 200, 200)

# Define the map as a NumPy array
input_maps = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
    [0, 2, 2, 2, 2, 2, 0, 2, 2, 2, 0],
    [0, 2, 0, 0, 0, 2, 0, 0, 2, 2, 0],
    [0, 2, 0, 2, 2, 2, 2, 0, 2, 2, 0],
    [0, 2, 0, 2, 0, 0, 2, 0, 2, 0, 0],
    [0, 2, 2, 2, 0, 2, 2, 2, 2, 0, 0],
    [0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0],
    [0, 2, 2, 2, 2, 2, 0, 2, 2, 2, 0],
    [0, 2, 2, 2, 2, 2, 0, 0, 2, 2, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
])


# Initial parameters
agent_start_pos = (2, 1)
exit_state = (0, 0)  # Set exit state as desired
range_sight = 5
penalty = -1.0
discount = 0  # Discount factor for POMCP

def build_graph(root):
    """
    Traverses the tree starting from root and builds a directed graph.
    """
    G = nx.DiGraph()
    queue = [root]

    while queue:
        node = queue.pop(0)
        for action, child in node.children.items():
            if child is not None:
                G.add_edge(node.id, child.id, weight=child.num_visited)
                G.nodes[child.id]['num_visited'] = child.num_visited
                queue.append(child)
        G.nodes[node.id]['num_visited'] = node.num_visited

    return G

def draw_graph(G):
    """
    Draws the graph with edge thickness based on number of visits.
    """
    pos = nx.spring_layout(G)  # or use graphviz_layout for tree shape
    edges = G.edges(data=True)

    weights = [max(1, data['weight']) for _, _, data in edges]
    labels = {node: G.nodes[node]['num_visited'] for node in G.nodes}

    nx.draw(G, pos, with_labels=True, labels=labels, node_color='lightblue', node_size=500)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), width=weights)
    plt.title("Tree of Node Visits")
    plt.show()


def run_experiment():
    pygame.init()
    rows, cols = input_maps.shape
    screen = pygame.display.set_mode((cols * CELL_SIZE, rows * CELL_SIZE))
    pygame.display.set_caption("POMCP Moves Visualization")
    clock = pygame.time.Clock()

    generator = Generator(input_maps, range_sight, penalty)
    obs, belief = generator.get_init_state(agent_start_pos)

    current_node = Node(agent_start_pos, obs, belief, parent_id="", parent_a=0)

    pomcp_algorithm = POMCP(generator, discount)
    action = pomcp_algorithm.search(current_node)

    root_node:Node = current_node #save the root node so we can draw from it

    running = True
    exit_found = False

    count = 0
    cycle_length = 1

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Press SPACE to get the next move from POMCP
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and not exit_found:
                    # POMCP search returns an action (int)

                    #action_values: list[float] = list()
                    if(count % cycle_length == 0):
                        action = pomcp_algorithm.search(current_node)
                    else:
                        action_values = []
                        for a in range(4):
                            action_values.append(current_node.action_values[a])

                        action :int = action_values.index(max(action_values))
                    
                    print(f"POMCP selected action: {Action(action).name}")

                    # Update the state using the generator's generate function
                    exit_found, new_agent_pos, new_obs, new_belief, reward = generator.generate(exit_state, current_node.agent_pos, current_node.obs, current_node.belief, action)
                    print(f"Agent moved to {new_agent_pos}, reward: {reward}")
                    print(current_node.action_values)

                    # Update the current node with new state values
                    current_node = current_node.children[action]
                    current_node.obs = new_obs
                    current_node.belief = new_belief
                    if exit_found:
                        print("Exit found!")
                    
                    #pomcp_algorithm.depth_limit -= 1

                    count += 1

                    G = build_graph(root_node)
                    draw_graph(G)

        # Drawing routine
        screen.fill(WHITE)
        # Loop over each cell in the map and color it based on its state:
        for r in range(rows):
            for c in range(cols):
                rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                # Use GRAY if cell has been observed; otherwise, white
                cell_color = GRAY if (r, c) in current_node.obs else WHITE
                # Draw walls as BLACK
                if input_maps[r, c] == Cell.WALL.value:
                    cell_color = BLACK
                pygame.draw.rect(screen, cell_color, rect)
                pygame.draw.rect(screen, BLACK, rect, 1)

        # Draw the agent in BLUE
        agent_rect = pygame.Rect(current_node.agent_pos[1] * CELL_SIZE, current_node.agent_pos[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, BLUE, agent_rect)

        # Draw the exit in GREEN
        exit_rect = pygame.Rect(exit_state[1] * CELL_SIZE, exit_state[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, GREEN, exit_rect)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    run_experiment()
