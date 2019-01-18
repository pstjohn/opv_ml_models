import sys

i_prop = int(sys.argv[1])

props = ['gap', 'homo', 'lumo', 'spectral_overlap', 'homo_extrapolated',
         'lumo_extrapolated', 'gap_extrapolated', 'optical_lumo_extrapolated']

prop = props[i_prop]

model_name = 'b3lyp_schnet2_{}'.format(prop)
print(model_name)


import os
import numpy as np
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

from nfp.preprocessing import MolPreprocessor, GraphSequence, RobustNanScaler

import gzip
import pickle
import pandas as pd
from tqdm import tqdm

# Define Keras model
import keras
import keras.backend as K

from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler

from keras.layers import (Input, Embedding, Dense, BatchNormalization,
                                 Concatenate, Multiply, Add)

from keras.models import Model

from nfp.layers import (MessageLayer, GRUStep, Squeeze, EdgeNetwork,
                               ReduceAtomToMol, ReduceBondToAtom,
                               GatherAtomToBond)
from nfp.models import GraphModel

train = pd.read_csv('data/mol_train.csv.gz')
valid = pd.read_csv('data/mol_valid.csv.gz')

train = train[train[prop].notna()]

# Rescale Y matrix
y_train_raw = train[[prop]].values
y_valid_raw = valid[[prop]].values

y_scaler = RobustNanScaler()
y_train_scaled = y_scaler.fit_transform(y_train_raw)
y_valid_scaled = y_scaler.transform(y_valid_raw)

from rdkit.Chem import MolFromMolBlock

# Transform MOLs into X matrix
preprocessor = MolPreprocessor(n_neighbors=48)

train_inputs = preprocessor.fit((MolFromMolBlock(mol) for _, mol in train.mol.iteritems()))
valid_inputs = preprocessor.fit((MolFromMolBlock(mol) for _, mol in valid.mol.iteritems()))

def rbf_expansion(distances, mu=0, delta=0.2, kmax=150):
    k = np.arange(0, kmax)
    logits = -(np.atleast_2d(distances).T - (-mu + delta * k))**2 / delta
    return np.exp(logits)

def precalc_rbfs(inputs):

    for item in tqdm(inputs):

        item['distance_rbf'] = rbf_expansion(item['distance'])
        del item['distance']

    return inputs

train_inputs = precalc_rbfs(train_inputs)
valid_inputs = precalc_rbfs(valid_inputs)

# Train a quick group-contribution model to get initial values for enthalpies per atom
from collections import Counter
from sklearn.linear_model import LinearRegression

# X = pd.DataFrame([Counter(row['atom']) for row in train_inputs]).fillna(0)

# model = LinearRegression()
# model.fit(X, y_train_scaled)

# atom_contributions = pd.Series(model.coef_.flatten(), index=X.columns)
# atom_contributions = atom_contributions.reindex(np.arange(preprocessor.atom_classes)).fillna(0)

# Construct input sequences
batch_size = 32
train_sequence = GraphSequence(train_inputs, y_train_scaled, batch_size, final_batch=False)
valid_sequence = GraphSequence(valid_inputs, y_valid_scaled, batch_size, final_batch=False)

# Raw (integer) graph inputs
node_graph_indices = Input(shape=(1,), name='node_graph_indices', dtype='int32')
atom_types = Input(shape=(1,), name='atom', dtype='int32')
distance_rbf = Input(shape=(150,), name='distance_rbf', dtype='float32')
connectivity = Input(shape=(2,), name='connectivity', dtype='int32')

squeeze = Squeeze()

snode_graph_indices = squeeze(node_graph_indices)
satom_types = squeeze(atom_types)

# Initialize RNN and MessageLayer instances
atom_features = 64

# Initialize the atom states
atom_state = Embedding(
    preprocessor.atom_classes,
    atom_features, name='atom_embedding')(satom_types)

atomwise_energy = Embedding(
    preprocessor.atom_classes, 1, name='atomwise_energy',
)(satom_types)

bond_state = distance_rbf

def message_block(atom_state, bond_state, connectivity):

    source_atom_gather = GatherAtomToBond(1)
    target_atom_gather = GatherAtomToBond(0)

    source_atom = source_atom_gather([atom_state, connectivity])
    target_atom = target_atom_gather([atom_state, connectivity])

    # Edge update network
    bond_state = Concatenate()([source_atom, target_atom, bond_state])
    bond_state = Dense(2*atom_features, activation='softplus')(bond_state)
    bond_state = Dense(atom_features)(bond_state)

    # message function
    bond_state = Dense(atom_features, activation='softplus')(bond_state)
    bond_state = Dense(atom_features, activation='softplus')(bond_state)
    source_atom = Dense(atom_features)(source_atom)    
    messages = Multiply()([source_atom, bond_state])
    messages = ReduceBondToAtom(reducer='sum')([messages, connectivity])
    
    # state transition function
    messages = Dense(atom_features, activation='softplus')(messages)
    messages = Dense(atom_features)(messages)
    atom_state = Add()([atom_state, messages])
    
    return atom_state, bond_state

for _ in range(3):
    atom_state, bond_state = message_block(atom_state, bond_state, connectivity)

atom_state = Dense(atom_features//2, activation='softplus')(atom_state)
atom_state = Dense(1)(atom_state)
atom_state = Add()([atom_state, atomwise_energy])

output = ReduceAtomToMol(reducer='mean')([atom_state, snode_graph_indices])

model = GraphModel([
    node_graph_indices, atom_types, distance_rbf, connectivity], [output])

lr = 1E-4
epochs = 500

model.compile(optimizer=keras.optimizers.Adam(lr=lr, decay=1E-5), loss='mae')
model.summary()

if not os.path.exists(model_name):
    os.makedirs(model_name)
 
with open('{}/schnet_preprocessor.p'.format(model_name), 'wb') as f:
    pickle.dump(preprocessor, f)
    
filepath = model_name + "/best_model.hdf5"
checkpoint = ModelCheckpoint(filepath, save_best_only=True, period=10, verbose=1)
csv_logger = CSVLogger(model_name + '/log.csv')

hist = model.fit_generator(train_sequence, validation_data=valid_sequence,
                           epochs=epochs, verbose=1, 
                           callbacks=[checkpoint, csv_logger])
