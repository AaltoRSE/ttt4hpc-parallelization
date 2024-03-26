"""
Load preprocessed data.

Load knn parameters (n_neighbors, metric) from disk corresponding to the given parameter settings ID (--params-id).

Fit a standard scaler + knn classifier pipeline to the training data with given parameters (n_neighbors, metric).

Plot the decision boundaries of the fitted model on the complete data.
"""

import argparse
import jsonlines
import pickle 

from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--params-id", type=int, help="Parameter settings ID to use.")
args = parser.parse_args()
params_id = args.params_id

# Load parameter settings from disk
with jsonlines.open("params/params.jsonl") as reader:
    params = [obj for obj in reader]
param = params[params_id]
n_neighbors = param["n_neighbors"]
metric = param["metric"]

# Load preprocessed data from disk
X, X_train, X_test, y, y_train, y_test, features, label_encoder = pickle.loads(open("data/preprocessed/Iris.pkl", "rb"))

# Fit
clf = Pipeline(
    steps=[("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric))]
)
clf.fit(X_train, y_train)

# Plot
disp = DecisionBoundaryDisplay.from_estimator(
    clf,
    X_test,
    response_method="predict",
    plot_method="pcolormesh",
    xlabel=features[0],
    ylabel=features[1],
    shading="auto",
    alpha=0.5,
)
scatter = disp.ax_.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, edgecolors="k")
disp.ax_.legend(
    scatter.legend_elements()[0],
    label_encoder.classes_,
    loc="lower left",
    title="Classes",
)
_ = disp.ax_.set_title(f"3-Class classification\n(k={n_neighbors!r}, metric={metric!r})")

# Save image to disk
Path("results/").mkdir(parents=True, exist_ok=True)
plt.savefig(f"results/n_neighbors={n_neighbors}___metric={metric}.png")

# Close plot
plt.close()