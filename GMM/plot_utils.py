import matplotlib.pyplot as plt
from map_utils import generate_from_partition, find_map_partition


# @TODO: This is a cludge: currently there is 
# - one code used by pattern editor
# - another used by the tree builder
# - and finally grayscale for the plotting utilities
def to_grayscale(pattern):
    grayscale_vals = {
        0: 0.5, # Unobserved cells should be gray
        0.5: 0.25, # Ambiguous values (according to map synthesis) will be dark gray
        1: 0.0, # Walls should be black
        2: 1.0, # Observed cells should be white
    }
    gray_pattern = [[grayscale_vals[entry] for entry in row] for row in pattern]
    return gray_pattern

# plotting - showing one image
def plot_pattern(pattern, title):
      plt.imshow(to_grayscale(pattern), cmap='gray')
      plt.title(title)
      plt.show()

# plotting a list of patterns with titles, helper function
# optionally, save the plot to image_logs/{save_image}.png to use later in slides
def plot_patterns(patterns, titles, figure_title = "", save_image="", show_plots=True):

    fig, axs = plt.subplots(1, len(patterns), figsize=(15, 5))
    for i, pattern in enumerate(patterns):
        axs[i].imshow(to_grayscale(pattern), cmap='gray')
        axs[i].set_title(titles[i])
    
    plt.suptitle(figure_title)
    if (len(save_image) > 0):
        plt.savefig(f"image_logs/{save_image}.png")

    if (show_plots):
        plt.show()

    plt.close()


# plotting input, and an array of proposed fragments
def plot_input_response(input_image, fragments, save_image="", show_plots=True):
    plot_patterns([input_image] + fragments, 
                  ["Input Pattern"] + [f"Fragment {i}" for i in range(len(fragments))], 
                  "Input and Returned Fragment", 
                  save_image, 
                  show_plots)
    


# plotting input, fragment, and output
def plot_input_fragment_output(input_image, fragment, output, figure_title = "", save_image="", show_plots=True):
   plot_patterns([input_image] + [fragment] + [output], 
                 ["Input Pattern"] + ["Fragment"] + ["Output"], 
                 figure_title, save_image, show_plots)


def generate_completion_plot(input_map, f, title, plot_filename, show_plots=True):
    p = find_map_partition(input_map, f)
    output = generate_from_partition(f, p, input_map.shape)
    plot_input_fragment_output(input_map, f, output, title, plot_filename, show_plots=show_plots)
      