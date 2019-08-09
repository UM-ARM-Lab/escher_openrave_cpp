import pickle, IPython, math, sys, getopt, random, os
import numpy as np
# import mkl
# mkl.get_max_threads()
# import faiss
import timeit
from sklearn.neighbors import BallTree
DIM = 65 * 65 + 25 * 252

random.seed(20190806)

def main():
    # tree = faiss.IndexFlatL2(DIM)
    # depth_map_ids = os.listdir('/mnt/big_narstie_data/chenxi/data/ground_truth_p1p2/ground_depth_maps')
    # for depth_map_id in depth_map_ids:
    #     with open('/mnt/big_narstie_data/chenxi/data/ground_truth_p1p2/ground_depth_maps/' + depth_map_id, 'r') as file:
    #         ground_depth_map = pickle.load(file)
    #     with open('/mnt/big_narstie_data/chenxi/data/ground_truth_p1p2/wall_depth_maps/' + depth_map_id, 'r') as file:
    #         wall_depth_map = pickle.load(file)
    #     arr = np.concatenate((np.clip(ground_depth_map.reshape(1,-1), -0.2, 0.2) * 100, wall_depth_map.reshape(1,-1) * 100), 1)
    #     tree.add(arr)
    # print(tree.ntotal)

    # indices = random.sample(range(len(depth_map_ids)), 1000)
    # query_arr = np.zeros((1000, DIM), dtype=float)
    # for i, index in enumerate(indices):
    #     depth_map_id = depth_map_ids[index]
    #     with open('/mnt/big_narstie_data/chenxi/data/ground_truth_p1p2/ground_depth_maps/' + depth_map_id, 'r') as file:
    #         ground_depth_map = pickle.load(file)
    #     with open('/mnt/big_narstie_data/chenxi/data/ground_truth_p1p2/wall_depth_maps/' + depth_map_id, 'r') as file:
    #         wall_depth_map = pickle.load(file)
    #     query_arr[i][:4225] = np.clip(ground_depth_map.reshape(-1,), -0.2, 0.2) * 100
    #     query_arr[i][4225:] = wall_depth_map.reshape(-1,) * 100
    # _, ids = tree.search(np.float32(query_arr), 5)

    # IPython.embed()

    depth_map_ids = os.listdir('/mnt/big_narstie_data/chenxi/data/ground_truth_p1p2/ground_depth_maps')
    X = np.zeros((len(depth_map_ids), DIM), dtype=float)
    for index, depth_map_id in enumerate(depth_map_ids):
        with open('/mnt/big_narstie_data/chenxi/data/ground_truth_p1p2/ground_depth_maps/' + depth_map_id, 'r') as file:
            ground_depth_map = pickle.load(file)
        with open('/mnt/big_narstie_data/chenxi/data/ground_truth_p1p2/wall_depth_maps/' + depth_map_id, 'r') as file:
            wall_depth_map = pickle.load(file)
        X[index][:4225] = ground_depth_map.reshape(-1,) * 100
        X[index][4225:] = wall_depth_map.reshape(-1,) * 100

    tree = BallTree(X)

    query_indices = random.sample(range(len(depth_map_ids)), 1000)
    dists, ids = tree.query(X[query_indices], k=10)
    
    with open('/mnt/big_narstie_data/chenxi/data/ground_truth_p1p2/p2_ddyn', 'r') as file:
        p2_ddyn = pickle.load(file)
    
    p2_set = set()
    for example_id in p2_ddyn:
        p2 = example_id[example_id.rfind('_')+1:]
        p2_set.add(p2)

    large_stds = []
    small_stds = []
    for i in range(1000):
        for p2 in p2_set:
            temp = []
            for j in range(10):
                if (depth_map_ids[ids[i][j]] + '_' + p2) in p2_ddyn:
                    temp.append(p2_ddyn[depth_map_ids[ids[i][j]] + '_' + p2][3])
            if len(temp) != 0 and len(temp) != 1:
                if np.mean(np.array(temp)) < 200:
                    small_stds.append(np.std(np.array(temp)))
                else:
                    large_stds.append(np.std(np.array(temp)))
    IPython.embed()
    



        


if __name__ == "__main__":
    main()

