"""
Save parameter grid to disk as a JSONLINES file (one JSON object per line). 
"""
import jsonlines





# Loop over n_neighbors only
n_neighbors_list = [1, 2, 4, 8, 16]

with jsonlines.open("params/params.jsonl", "w") as writer:
    for n_neighbors in n_neighbors_list:
        obj = {"n_neighbors": n_neighbors}
        writer.write(obj)



# Exercise: Loop over metric_list = ["cityblock", "cosine", "euclidean", "haversine", "l1", "l2", "manhattan", "nan_euclidean"]
# Solution
n_neighbors_list = [1, 2, 4, 8, 16]
metric_list = ["cityblock", "cosine", "euclidean", "haversine", "l1", "l2", "manhattan", "nan_euclidean"]

with jsonlines.open("params/params.jsonl", "w") as writer:
    for n_neighbors in n_neighbors_list:
        for metric in metric_list:
            obj = {"n_neighbors": n_neighbors, "metric": metric}
            writer.write(obj)
