import collections
import sys, os
import argparse
import copy
import json
import numpy as np

parent_parameters = {
  "seed": 1,
  "size": [6, 6],
  "num_particles": 12,
  "tuning_parameters": [
  ],
  "sweeps": [
    {"count": 10000,  "temperature": 10.000, "type": "local" },
    {"count": 100,    "temperature": 10.000, "type": "global"},
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
    {"count": 20000,  "temperature": 0.0001, "type": "local" },
    {"count": 100,    "temperature": 0.0001, "type": "local" },
    {"count": 20000,  "temperature": 0.0001, "type": "local" },

    {"count": 10000,  "temperature": 10.000, "type": "local" },
    {"count": 100,    "temperature": 10.000, "type": "global"},
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
    {"count": 20000,  "temperature": 0.0001, "type": "local" },
    {"count": 100,    "temperature": 0.0001, "type": "local" },
    {"count": 20000,  "temperature": 0.0001, "type": "local" }
  ],
  "print_every": 100
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", type=str)
    parser.add_argument("-s", "--split", type=int, required=True)
    args = parser.parse_args()

    split = args.split
    prefix = args.prefix

    V = [0.1]
    tuning_parameters = []
    for J1 in [0.5]:     #np.linspace(-1, 1, 11):
        for J2 in np.linspace(-1, 1, 51):
            for J3 in np.linspace(-1, 1, 51):
                J = [J1, J2, J3]
                tuning_parameters.append({"V": V, "J": J})

    n = len(tuning_parameters)
    groupsize = int(np.ceil(n / split))
    ngroups = int(np.ceil(n / groupsize))
    ndigits = 1+int(np.floor(np.log10(ngroups)))

    igroup = 0
    while tuning_parameters:
        igroup += 1
        parameters = copy.copy(parent_parameters)
        parameters["tuning_parameters"] = tuning_parameters[:groupsize]
        format_string = "{{prefix}}-{{igroup:0{ndigits}d}}.json".format(ndigits=ndigits)
        with open(format_string.format(prefix=prefix, igroup=igroup), "w") as output_file:
            json.dump(parameters, output_file)
        tuning_parameters = tuning_parameters[groupsize:]

if __name__=='__main__':
    main()
