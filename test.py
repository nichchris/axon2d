import pandas as pd
import axon2d


neuron_data = pd.read_csv('B2_coordinates.csv')

scalingfactor = 1000

neuron_positions = neuron_data.loc[:, 'X':'Y'].to_numpy()
w = axon2d.grow_network(n_pos = neuron_positions, x_dim=scalingfactor,
                        y_dim=scalingfactor, days=20, r_path='results2')
