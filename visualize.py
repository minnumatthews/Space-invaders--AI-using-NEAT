import matplotlib.pyplot as plt
from neat.math_util import mean

def plot_stats(statistics, ylog=False, view=False):
    """Plots the evolution statistics from the NEAT run, including fitness and standard deviation."""
    
    generation = range(len(statistics.most_fit_genomes))

    # Get average fitness across generations
    avg_fitness = statistics.get_fitness_mean()

    # Get standard deviation of fitness
    stdev_fitness = statistics.get_fitness_stdev()

    # Plotting fitness stats
    plt.figure(figsize=(10, 6))
    plt.plot(generation, avg_fitness, 'b-', label="Average Fitness")
    plt.plot(generation, stdev_fitness, 'r-', label="Fitness Standard Deviation")
    
    plt.title("Population's Average and Standard Deviation of Fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid(True)
    plt.legend(loc="best")

    if ylog:
        plt.yscale('log')

    if view:
        plt.show()

    # Save the plot to a file
    plt.savefig("avg_fitness.svg")


def plot_species(statistics, view=False):
    """Visualizes speciation throughout the evolution."""
    species_sizes = statistics.get_species_sizes()
    generation = range(len(species_sizes))

    # Plot species sizes across generations
    plt.figure(figsize=(10, 6))
    plt.stackplot(generation, *species_sizes, labels=[f"Species {i+1}" for i in range(len(species_sizes[0]))])

    plt.title("Speciation Over Generations")
    plt.xlabel("Generations")
    plt.ylabel("Species Size")
    plt.grid(True)
    plt.legend(loc="best")

    if view:
        plt.show()

    # Save the plot to a file
    plt.savefig("speciation.svg")


def plot_performance_stats(logfile='game_performance.log', view=False):
    """Plots performance statistics from the game log such as bullets fired, hits, misses, levels, survival time, and total distance moved."""
    
    # Extract performance data from the log file
    bullets_fired = []
    hits = []
    misses = []
    distance_moved = []
    levels = []
    survival_times = []

    with open(logfile, 'r') as log:
        for line in log:
            if "Bullets Fired" in line:
                parts = line.strip().split(',')
                bullets_fired.append(int(parts[0].split(": ")[1]))
                hits.append(int(parts[1].split(": ")[1]))
                misses.append(int(parts[2].split(": ")[1]))
                distance_moved.append(float(parts[3].split(": ")[1]))
                survival_times.append(float(parts[4].split(": ")[1]))
            if "Level" in line:
                level = int(line.strip().split(": ")[1])
                levels.append(level)

    # Plot bullets fired, hits, and misses
    plt.figure(figsize=(12, 8))

    plt.subplot(4, 1, 1)
    plt.plot(bullets_fired, label='Bullets Fired', color='blue')
    plt.plot(hits, label='Hits', color='green')
    plt.plot(misses, label='Misses', color='red')
    plt.title("Bullets Fired, Hits, and Misses Over Time")
    plt.xlabel("Time (Generations)")
    plt.ylabel("Count")
    plt.legend(loc="best")
    plt.grid(True)

    # Total distance moved
    plt.subplot(4, 1, 2)
    plt.plot(distance_moved, label='Total Distance Moved', color='purple')
    plt.title("Total Distance Moved by Player Over Time")
    plt.xlabel("Time (Generations)")
    plt.ylabel("Distance")
    plt.legend(loc="best")
    plt.grid(True)

    # Levels over time
    if levels:
        plt.subplot(4, 1, 3)
        plt.plot(levels, label='Level Progression', color='orange')
        plt.title("Level Progression Over Time")
        plt.xlabel("Time (Generations)")
        plt.ylabel("Level")
        plt.legend(loc="best")
        plt.grid(True)

    # Survival time over time
    plt.subplot(4, 1, 4)
    plt.plot(survival_times, label='Survival Time', color='magenta')
    plt.title("Survival Time Over Time")
    plt.xlabel("Time (Generations)")
    plt.ylabel("Survival Time (seconds)")
    plt.legend(loc="best")
    plt.grid(True)

    # Display the plots
    if view:
        plt.show()

    # Save the plot to a file
    plt.savefig("performance_stats.svg")
