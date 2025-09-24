from structure_based_planner import *
import pygame
from maps import *
import sys
import builtins

# Constants
TILE_SIZE = 30
FPS = 40

# Colors
COLOR_BG = (30, 30, 30)
COLOR_FLOOR = (160, 160, 160)
COLOR_OBSERVED = (210, 210, 210)
COLOR_WALL = (50, 50, 50)
COLOR_AGENT = (255, 0, 0)
COLOR_GOAL = (0, 255, 0)
COLOR_TRAIL = (100, 100, 255)


def draw_arrow(surface, start, end, color):
    """
    Draw just the arrowhead pointing from start to end
    """

    dx, dy = end[0] - start[0], end[1] - start[1]
    base_ang = math.atan2(dy, dx)
    head_len = 12
    spread   = math.radians(30)
    left_ang  = base_ang + math.pi - spread
    right_ang = base_ang + math.pi + spread

    left_pt  = (end[0] + head_len * math.cos(left_ang),
                end[1] + head_len * math.sin(left_ang))
    right_pt = (end[0] + head_len * math.cos(right_ang),
                end[1] + head_len * math.sin(right_ang))

    pygame.draw.polygon(surface, color, (left_pt, right_pt, end))


def visualize_after_checkpoint(map_array, pos_indices, agent_path):
    """
    Visualization tool for agent path, clipped at each checkpoint
    """

    pygame.init()
    rows, cols = map_array.shape
    num_empty_spaces = 0
    for x in range(rows):
        for y in range(cols):
            if map_array[x][y] != Cell.WALL.value:
                num_empty_spaces += 1
                
    screen = pygame.display.set_mode((cols * TILE_SIZE, rows * TILE_SIZE))
    pygame.display.set_caption("POMCP Moves Visualization")
    clock = pygame.time.Clock()

    generator = Generator(map_array)
    generator.get_init_state(agent_path[0])
    pygame.display.set_caption("Agent Full Path")

    font = pygame.font.SysFont(None, 16)

    running = True
    checkpoint = 0
    past_pos = agent_path[0]
    path_index = 0
    action = 0
    past_path = [past_pos]
    observed = set()

    past_check_point = 0
    checkpoint_index = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Press SPACE to get the next move from POMCP
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    
                    for _ in range(pos_indices[checkpoint_index] - past_check_point):

                        path_index += 1

                        if(past_pos[0] > agent_path[path_index][0]):
                            action = 0
                        elif(past_pos[0] < agent_path[path_index][0]):
                            action = 2
                        elif(past_pos[1] > agent_path[path_index][1]):
                            action = 1
                        elif(past_pos[1] < agent_path[path_index][1]):
                            action = 3
                        
                        past_path.append(past_pos)
                        generator.observed = generator.observed.union(generator.get_observation(past_pos))
                        
                        exit_found, new_agent_pos, new_obs, new_belief, reward = generator.generate((0,0), past_pos, set(), set(), action)
                        generator.observed = generator.observed.union(generator.get_observation(past_pos))
                        generator.observed = generator.observed.union(new_obs)
                        past_pos = agent_path[path_index]

                    generator.observed = generator.observed.union(generator.get_observation(past_pos))
                    past_check_point = pos_indices[checkpoint_index]
                    checkpoint_index += 1          

        map_surf = pygame.Surface(screen.get_size())
        for i in range(rows):
            for j in range(cols):
                rect = pygame.Rect(j * TILE_SIZE, i * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                tile = map_array[i][j]
                if tile == 0:
                    col = COLOR_WALL
                elif (i, j) in generator.observed:
                    col = COLOR_OBSERVED
                elif tile == 5 or tile == 3:
                    col = COLOR_GOAL
                else:
                    col = COLOR_FLOOR
                pygame.draw.rect(map_surf, col, rect)
                pygame.draw.rect(map_surf, (0,0,0), rect, 1)

        # 2) Trail surface: draw entire path at once
        trail_surf = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        px_pts = [
            (p[1]*TILE_SIZE + TILE_SIZE//2, p[0]*TILE_SIZE + TILE_SIZE//2)
            for p in past_path
        ]
        if len(px_pts) > 1:
            pygame.draw.lines(trail_surf, (255, 0, 0), False, px_pts, 7)
            draw_arrow(trail_surf, px_pts[-2], px_pts[-1], (255, 0, 0))

        # 3) Annotations
        corner_used = {}  # (i,j) to 'top-right' or 'bottom-left'

        for idx, (i, j) in enumerate(past_path):
            pos = (i, j)
            text = font.render(str(idx), True, (0, 0, 0))
            text_rect = text.get_rect()

            top_right = (j * TILE_SIZE + TILE_SIZE - 2 - text_rect.width, i * TILE_SIZE + 2)
            bottom_left = (j * TILE_SIZE + 2, i * TILE_SIZE + TILE_SIZE - text_rect.height - 2)

            if corner_used.get(pos) != 'top-right':
                trail_surf.blit(text, top_right)
                corner_used[pos] = 'top-right'
            else:
                trail_surf.blit(text, bottom_left)
                corner_used[pos] = 'bottom-left'

        # 4) Final draw
        screen.blit(map_surf, (0,0))
        screen.blit(trail_surf, (0,0))


        end_i, end_j = past_path[-1]
        center = (end_j * TILE_SIZE + TILE_SIZE//2, end_i * TILE_SIZE + TILE_SIZE//2)
        pygame.draw.circle(screen, COLOR_AGENT, center, TILE_SIZE//3)

        start_i, start_j = past_path[0]
        start_center = (start_j * TILE_SIZE + TILE_SIZE//2, start_i * TILE_SIZE + TILE_SIZE//2)
        pygame.draw.circle(screen, (0, 255, 255), start_center, TILE_SIZE//3)

        explored_ratio = len(generator.observed) / num_empty_spaces
        explored_text = font.render(f"Explored: {explored_ratio:.2%}", True, (0, 0, 0))
        screen.blit(explored_text, (10, 10))

        pygame.display.flip()

        clock = pygame.time.Clock()
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


def how_much_observed(map_array, pos_indices, agent_path) -> list[float]:
    """
    Returns a list of exploration percentages at each checkpoint index.
    """

    rows, cols = map_array.shape
    num_empty_spaces = sum(1 for x in range(rows) for y in range(cols)
                           if map_array[x][y] != Cell.WALL.value)

    generator = Generator(map_array)
    generator.get_init_state(agent_path[0])

    past_pos = agent_path[0]
    path_index = 0
    past_check_point = 0
    checkpoint_index = 0

    observed_ratios = []

    while checkpoint_index < len(pos_indices):
        # Simulate movement until reaching this checkpoint
        steps = min(pos_indices[checkpoint_index] - past_check_point, len(agent_path) - 1 - path_index)

        for _ in range(steps):
            path_index += 1

            if past_pos[0] > agent_path[path_index][0]:
                action = 0
            elif past_pos[0] < agent_path[path_index][0]:
                action = 2
            elif past_pos[1] > agent_path[path_index][1]:
                action = 1
            elif past_pos[1] < agent_path[path_index][1]:
                action = 3
            else:
                action = -1  # stay put (shouldn't happen ideally)

            generator.observed |= generator.get_observation(past_pos)

            exit_found, new_agent_pos, new_obs, new_belief, reward = generator.generate(
                (0, 0), past_pos, set(), set(), action
            )
            generator.observed |= generator.get_observation(past_pos)
            generator.observed |= new_obs
            past_pos = agent_path[path_index]

        # After reaching checkpoint
        generator.observed |= generator.get_observation(past_pos)
        past_check_point = pos_indices[checkpoint_index]
        checkpoint_index += 1

        explored_ratio = len(generator.observed) / num_empty_spaces
        observed_ratios.append(explored_ratio)

    return observed_ratios


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python modular_v2_test.py <map_number>")
        sys.exit(1)

    map_number = sys.argv[1]
    map_var = f"map{map_number}"
    fragment_var = f"fragment{map_number}"
    copies_var = f"copies{map_number}"

    try:
        map_data = builtins.globals()[map_var]
        fragment_data = builtins.globals()[fragment_var]
        copies_data = builtins.globals()[copies_var]
    except KeyError as e:
        print(f"Error: Could not find variable {e} for map {map_number}")
        sys.exit(1)
    # Test run_sbp_planner function
    
    map_copy = map_data.copy()
    # Test SBP planner
    print("\nStarting SDB planner...")
    agent_path, checkpoints, escape_rollout_checkpoints, pomcp_rollout_checkpoints,  bridge_rollout_checkpoints, bridge_time_checkpoints, fragment_time_checkpoints, escape_time_checkpoints = run_modular(map_data, fragment_data, copies_data)
    observed =  how_much_observed(map_copy, checkpoints, agent_path)
    print("SBP planning completed successfully!")
    
    print("\nTest completed!")

    with open(f'modular_v2_results/map_{map_number}_naive.txt', 'a') as f:
    # Save map
        np.savetxt(f, map_data, fmt='%d')
        f.write('\n\n')
        f.write("Exploration result with generator call limit\n")

        # Print out arrays
        f.write("agent_path:\n")
        f.write(f"{agent_path}\n\n")

        f.write("checkpoints:\n")
        f.write(f"{checkpoints}\n\n")

        f.write("escape_rollout_checkpoints:\n")
        f.write(f"{escape_rollout_checkpoints}\n\n")

        f.write("pomcp_rollout_checkpoints:\n")
        f.write(f"{pomcp_rollout_checkpoints}\n\n")

        f.write("bridge_rollout_checkpoints:\n")
        f.write(f"{bridge_rollout_checkpoints}\n\n")

        f.write("bridge_time_checkpoints:\n")
        f.write(f"{bridge_time_checkpoints}\n\n")

        f.write("fragment_time_checkpoints:\n")
        f.write(f"{fragment_time_checkpoints}\n\n")

        f.write("escape_time_checkpoints:\n")
        f.write(f"{escape_time_checkpoints}\n\n")

        f.write("observed amount:\n")
        f.write(f"{observed}\n\n")

        rollout_arr = []
        time_arr = []
        for x in range(len(fragment_time_checkpoints)):
            time_increment = (
                fragment_time_checkpoints[x] +
                bridge_time_checkpoints[x] +
                escape_time_checkpoints[x]
            )

            # accumulate time (carry over last value if exists)
            if time_arr:
                time_arr.append(time_arr[-1] + time_increment)
            else:
                time_arr.append(time_increment)
            rollout_arr.append(escape_rollout_checkpoints[x] + pomcp_rollout_checkpoints[x] + bridge_rollout_checkpoints[x])

        f.write(f"time_arr = {time_arr}\n")
        f.write(f"roll_arr = {rollout_arr}\n")


    # Visualize
    visualize_after_checkpoint(map_copy, checkpoints, agent_path)
    