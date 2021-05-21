import pandas as pd
import axon2d


neuron_data = pd.read_csv('B2_coordinates.csv')

neuron_positions = neuron_data.loc[:, 'X':'Y'].to_numpy()
w = axon2d.grow_network(n_pos = neuron_positions, r_path='results2')
