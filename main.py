from sys import argv
import numpy as np

from simulator import RaceTrack, Simulator, plt

if __name__ == "__main__":
    assert(len(argv) == 3)
    racetrack = RaceTrack(argv[1])
    raceline_path = argv[2]
    
    # Load the raceline data
    raceline_data = np.loadtxt(raceline_path, delimiter=',')
    raceline = raceline_data[:, 0:2]  # Extract x, y coordinates
    
    simulator = Simulator(racetrack, raceline)
    simulator.start()
    plt.show()