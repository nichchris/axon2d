import multiprocessing as mp
import pandas as pd
import axon2d
import pathlib

def grow_simple(fname):
    res_path = 'mp_results_' + pathlib.Path(fname.parts[-1]).stem
    neuron_data = pd.read_csv(fname)

    print("Writing files to: ", res_path)
    scalingfactor = 1000
    neuron_positions = neuron_data.loc[:, 'X':'Y'].to_numpy()
    w = axon2d.grow_network(n_pos=neuron_positions, x_dim=scalingfactor,
                            y_dim=scalingfactor, days=150, r_path=res_path)
    return w


if __name__ == "__main__":
    print("Reading file names.\n")
    current_path = pathlib.Path.cwd()
    names = []

    print("Working directory:", current_path)
    # set file suffix for file top search for. Probably cvs
    for current_file in pathlib.Path(current_path).rglob('O1*'):
        print("Current file: ", pathlib.Path(current_file.parts[-1]).stem)
        names.append(current_file)
        print("Added file to list: ", current_file)

    print("Starting multiprocess")
    with mp.Pool(processes=6) as p:
        p.map(grow_simple, names)
