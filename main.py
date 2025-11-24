from sys import argv
import numpy as np

from simulator import RaceTrack, Simulator, plt

if __name__ == "__main__":
    assert(len(argv) == 3)
    racetrack = RaceTrack(argv[1])
    racetrack.raceline = np.loadtxt(argv[2], comments="#", delimiter=",")[:, 0:2]
    simulator = Simulator(racetrack)
    simulator.start()
    plt.show()