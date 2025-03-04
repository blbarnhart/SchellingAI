import os
import openai
import numpy as np
import matplotlib.pyplot as plt
import random

# -----------------------
# Global Parameters
# -----------------------
GRID_SIZE = 20             # Defines the dimensions of the grid (GRID_SIZE x GRID_SIZE).
EMPTY_RATIO = 0.1          # Fraction of cells that will be empty.
THRESHOLD = 0.3            # Default threshold used if DYNAMIC_THRESHOLDS=False.
                           # If DYNAMIC_THRESHOLDS is True, each agent may have its own threshold.

# Toggle this to True/False to enable/disable dynamic thresholds
DYNAMIC_THRESHOLDS = True

# Name of the model to be used with OpenAI's API (e.g. 'gpt-3.5-turbo' or 'gpt-4').
MODEL_NAME = "gpt-3.5-turbo"

# Set the API key from environment variables, if available.
openai.api_key = os.getenv("OPENAI_API_KEY")

# -----------------------
# Personality Options
# -----------------------
PERSONALITY_OPTIONS = [
    "very sociable, loves meeting new people, and embraces diversity",
    "shy and introverted, prefers a quiet neighborhood",
    "adventurous and curious about different cultures",
    "cautious and family-oriented, values close-knit communities",
    "artistic, enjoys creative, lively surroundings",
    "academic, values intellectual discussions with neighbors",
]

# Map each personality to a custom threshold (if dynamic is enabled).
# Interpretation:
#  - Lower threshold => more tolerant of neighbors who are different
#  - Higher threshold => less tolerant (wants more same-type neighbors)
PERSONALITY_THRESHOLD_MAP = {
    "very sociable": 0.1,
    "shy and introverted": 0.5,
    "adventurous": 0.2,
    "cautious and family-oriented": 0.4,
    "artistic": 0.25,
    "academic": 0.3
}

# This dict will store (x, y) -> {"personality": "...", "threshold": value}
# so each occupied cell has a distinct personality and threshold (if dynamic).
AGENT_PROFILES = {}

# -----------------------
# Schelling Model Logic
# -----------------------
def initialize_grid(size, empty_ratio):
    """
    Create a 2D grid of shape (size, size) filled with:
      - 0 for empty cells
      - 1 for Red agents
      - 2 for Blue agents

    Arguments:
        size (int): The width/height of the grid.
        empty_ratio (float): Fraction of cells that should be empty in the grid.

    Returns:
        np.ndarray: A 2D NumPy array (size x size) representing the initial grid configuration.
    """
    num_cells = size * size
    # Determine how many cells should be empty.
    num_empty = int(empty_ratio * num_cells)
    
    # Use np.random.choice to fill the grid. Probability breakdown:
    # empty_ratio => fraction for empty cells (0),
    # (1 - empty_ratio)/2 => fraction for Red (1),
    # (1 - empty_ratio)/2 => fraction for Blue (2).
    agents = np.random.choice(
        [0, 1, 2], 
        num_cells,
        p=[empty_ratio, (1 - empty_ratio)/2, (1 - empty_ratio)/2]
    )
    
    # Reshape the 1D array into a 2D grid.
    return agents.reshape(size, size)

def assign_personalities_and_thresholds(grid):
    """
    Assign each occupied cell in the grid a random personality, and if DYNAMIC_THRESHOLDS is True,
    assign a threshold that matches their personality (based on PERSONALITY_THRESHOLD_MAP).

    Arguments:
        grid (np.ndarray): The 2D array representing the current grid state.
    """
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if grid[x, y] != 0:  # Occupied cell (Red or Blue)
                # Pick a random personality from our list of options.
                personality = random.choice(PERSONALITY_OPTIONS)
                
                # If dynamic thresholds are on, try to match a threshold from the map.
                if DYNAMIC_THRESHOLDS:
                    threshold_candidates = []
                    
                    # Check if any key fragment in PERSONALITY_THRESHOLD_MAP is in the chosen personality string.
                    for key_fragment, thr_value in PERSONALITY_THRESHOLD_MAP.items():
                        if key_fragment in personality:
                            threshold_candidates.append(thr_value)
                    
                    # If we found a matching fragment, pick the first threshold. Otherwise fallback to global THRESHOLD.
                    if threshold_candidates:
                        assigned_threshold = threshold_candidates[0]
                    else:
                        assigned_threshold = THRESHOLD
                else:
                    # If not using dynamic thresholds, just assign the global threshold.
                    assigned_threshold = THRESHOLD

                # Store the chosen personality and threshold in AGENT_PROFILES.
                AGENT_PROFILES[(x, y)] = {
                    "personality": personality,
                    "threshold": assigned_threshold
                }

def get_neighbors(grid, x, y):
    """
    Return a list of neighbors around position (x, y).
    Neighbors are all cells within 1 step horizontally, vertically, or diagonally.

    Arguments:
        grid (np.ndarray): The grid from which we retrieve neighbors.
        x (int): Row index of the target cell.
        y (int): Column index of the target cell.

    Returns:
        np.ndarray: An array of the neighbor cell values (could be empty or agent types).
    """
    size = grid.shape[0]
    neighbors = []
    
    # Loop over the 3x3 block centered on (x, y), skipping the center cell itself.
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            # Ensure (nx, ny) is within grid bounds.
            if 0 <= nx < size and 0 <= ny < size:
                neighbors.append(grid[nx, ny])
    
    return np.array(neighbors)

def get_agent_perspective(grid, x, y):
    """
    Fetch a short narrative from the LLM about the agent at (x, y).
    If the cell is empty, return a simple message indicating emptiness.

    Arguments:
        grid (np.ndarray): The 2D grid of agents (and empty cells).
        x (int): Row index of the cell we're querying.
        y (int): Column index of the cell we're querying.

    Returns:
        str: A string describing the agent's personality, threshold, neighbor fraction,
             and an LLM-generated statement.
    """
    agent = grid[x, y]
    
    # If the cell is empty, return a short text.
    if agent == 0:
        return f"Cell ({x}, {y}) is empty. No agent perspective to show."

    # Determine the agent's color based on whether it's a 1 (Red) or 2 (Blue).
    color = "Red" if agent == 1 else "Blue"
    
    # Gather neighbor info.
    neighbors = get_neighbors(grid, x, y)
    same_type = np.sum(neighbors == agent)        # Number of neighbors with the same agent type.
    total_neighbors = np.sum(neighbors > 0)       # Number of occupied neighbors (ignoring empties).

    # Retrieve the agent's stored personality and threshold from AGENT_PROFILES.
    profile = AGENT_PROFILES.get((x, y), {})
    personality = profile.get("personality", "no particular personality")
    threshold = profile.get("threshold", THRESHOLD)

    # Calculate fraction of neighbors that are the same type as the agent.
    fraction_same = (same_type / total_neighbors) if total_neighbors else 0
    # Determine if the agent is unhappy based on fraction_same < threshold.
    unhappy = (fraction_same < threshold and total_neighbors > 0)

    # Build the system prompt, injecting the agent's personality and threshold.
    system_content = (
        f"You are a {color} agent in a Schelling segregation model.\n"
        f"You have {same_type} same-color neighbors out of {total_neighbors} total.\n"
        f"Your personal threshold is {threshold}. You are currently "
        f"{'unhappy' if unhappy else 'happy'} with your location.\n\n"
        f"Your personality is that you are {personality}.\n\n"
        "Explain your reasoning briefly in first person, "
        "given your personality and your tolerance threshold."
    )

    # Call the LLM. If it fails, we'll catch the exception.
    try:
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": system_content}],
            max_tokens=60,
            temperature=0.7
        )
        text_response = response.choices[0].message["content"].strip()
    except Exception as e:
        text_response = f"(LLM call failed) {e}"

    # Format and return the perspective string.
    perspective = (
        f"Agent at ({x}, {y}): {color}\n"
        f"  Personality: {personality}\n"
        f"  Threshold: {threshold}\n"
        f"  fraction_same: {fraction_same:.2f}\n"
        f"  Unhappy: {unhappy}\n"
        f"LLM says:\n{text_response}"
    )
    return perspective

# -----------------------
# Interactivity
# -----------------------
def on_click(event, grid):
    """
    Called whenever the user clicks on the plot.
    Converts the click coordinate to integer grid positions,
    then calls get_agent_perspective to print the agent's details.

    Arguments:
        event: A Matplotlib event object with xdata, ydata for the click.
        grid (np.ndarray): The grid from which we want agent info.
    """
    # If the click is not within the axes area, these may be None.
    if event.xdata is None or event.ydata is None:
        return
    
    col = int(round(event.xdata))
    row = int(round(event.ydata))
    
    # Check if the click is within the valid grid range.
    if 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]:
        result = get_agent_perspective(grid, row, col)
        print(result)
    else:
        print("Clicked outside the grid.")

def show_grid(grid):
    """
    Display the grid using matplotlib, and attach a click event so the user can
    click cells to see agent perspectives.

    Arguments:
        grid (np.ndarray): The grid to display.
    """
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(grid, cmap='coolwarm', vmin=0, vmax=2)
    ax.set_title("Click a cell to see the agent's perspective!")
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Attach the event handler to detect mouse clicks on the figure.
    fig.canvas.mpl_connect("button_press_event", lambda event: on_click(event, grid))
    plt.show()

def main():
    """
    Main entry point:
    1) Initialize the grid
    2) Assign personalities (and thresholds if DYNAMIC_THRESHOLDS is True)
    3) Display the grid with interactive clicking
    """
    # Initialize the grid with random placement of Red, Blue, or Empty.
    grid = initialize_grid(GRID_SIZE, EMPTY_RATIO)
    
    # Assign each agent a personality and (optionally) a dynamic threshold.
    assign_personalities_and_thresholds(grid)

    # Render the grid and allow user interaction.
    show_grid(grid)

# If this file is run directly, call main().
if __name__ == "__main__":
    main()
