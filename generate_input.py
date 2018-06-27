import collections
import sys, os
import argparse
import copy
import json
import numpy as np

parent_parameters = {
  "seed": 1,
  "size": [6, 6],
  "num_particles": 9,
  "tuning_parameters": [
  ],
  "sweeps": [
    {"count": 10000,  "temperature": 1.0000, "type": "local" },
    {"count": 100,    "temperature": 1.0000, "type": "global"},
    {"count": 10000,  "temperature": 0.5000, "type": "local" },
    {"count": 100,    "temperature": 0.5000, "type": "global"},
    {"count": 10000,  "temperature": 0.1000, "type": "local" },
    {"count": 100,    "temperature": 0.1000, "type": "global"},
    {"count": 10000,  "temperature": 0.0100, "type": "local" },
    {"count": 100,    "temperature": 0.0100, "type": "global"},
    {"count": 10000,  "temperature": 0.0010, "type": "local" },
    {"count": 100,    "temperature": 0.0010, "type": "global"},
    {"count": 50000,  "temperature": 0.0001, "type": "local" }
  ],
  "print_every": 100
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", type=str)
    parser.add_argument("-s", "--split", type=int)
    args = parser.parse_args()

    split = args.split
    prefix = args.prefix

    V = [1.0]
    tuning_parameters = []
    for J1 in np.linspace(-1, 1, 6):
        for J2 in np.linspace(-1, 1, 6):
            for J3 in np.linspace(-1, 1, 6):
                J = [J1, J2, J3]
                tuning_parameters.append({"V": V, "J": J})

    n = len(tuning_parameters)
    groupsize = int(np.ceil(n / split))

    igroup = 0
    while tuning_parameters:
        igroup += 1
        parameters = copy.copy(parent_parameters)
        parameters["tuning_parameters"] = tuning_parameters[:groupsize]
        with open("{prefix}-{igroup}.json".format(prefix=prefix, igroup=igroup), "w") as output_file:
            json.dump(parameters, output_file)
        tuning_parameters = tuning_parameters[groupsize:]

if __name__=='__main__':
    main()
