from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
from tqdm import tqdm
scaler = StandardScaler()
with open('mydata/network_porto/porto_edges_new_simplify.pkl', 'rb') as f:
    edgeinfo = pickle.load(f)
with open('mydata/network_porto/porto_nodes_new.pkl', 'rb') as f:
    nodeinfo = pickle.load(f)

data = np.load('mydata/train.npy', allow_pickle=True)


dxdy_samples = []

for trip in tqdm(data):
    for e in trip[1]:        # each edge id
        info = edgeinfo[e]
        xs, ys = nodeinfo[info[2]][0:2]
        xe, ye = nodeinfo[info[3]][0:2]
        dx = xe - xs
        dy = ye - ys
        dxdy_samples.append([dx, dy])

dxdy_samples = np.asarray(dxdy_samples)

dxdy_scaler = StandardScaler().fit(dxdy_samples)
print("dxdy mean :", dxdy_scaler.mean_)
print("dxdy scale:", dxdy_scaler.scale_)
with open("utils/porto_dxdy_scaler.pkl", "wb") as f:
    pickle.dump(dxdy_scaler, f)