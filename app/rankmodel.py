"""The rank model as plain data, evaluated with numpy alone.

The model is trained on a workstation (research/train_rank_model.py) and scored on a 1 GB server.
A pickled scikit-learn estimator would tie the server to the exact library version it was trained
with - this one was trained with 1.6, the server has 1.9, and unpickling across versions is not
supported - and would load scikit-learn into a process that only needs to add up some numbers.

So the trained gradient-boosting model is exported as its trees: for every node, the feature it
tests, the threshold, which way a missing value goes, its two children, and the leaf value. A
prediction is the baseline plus the leaf value each tree routes the row to, which is exactly what
scikit-learn's own predictor does on raw features (HistGradientBoosting applies the learning rate
to the leaf values when it fits, so nothing is multiplied here).

`export()` refuses a model it cannot reproduce: after writing the trees it evaluates them on the
training rows and requires agreement with scikit-learn's predict() to 1e-9. A model that cannot be
reproduced exactly is not exported, so the server can never run something the workstation did not.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

FORMAT = "rank-hgb-trees/1"


def _trees_from(model):
    trees = []
    for it in model._predictors:
        if len(it) != 1:
            raise ValueError("only single-output regression models can be exported")
        nodes = it[0].nodes
        if nodes["is_categorical"].any():
            raise ValueError("categorical splits are not supported by this exporter")
        trees.append({
            "feature": nodes["feature_idx"].astype(int).tolist(),
            "threshold": nodes["num_threshold"].astype(float).tolist(),
            "missing_left": nodes["missing_go_to_left"].astype(int).tolist(),
            "left": nodes["left"].astype(int).tolist(),
            "right": nodes["right"].astype(int).tolist(),
            "leaf": nodes["is_leaf"].astype(int).tolist(),
            "value": nodes["value"].astype(float).tolist(),
        })
    return trees


def predict(spec: dict, X: np.ndarray) -> np.ndarray:
    """Evaluate an exported model on rows of X (NaN allowed). numpy only."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != len(spec["features"]):
        raise ValueError(f"expected {len(spec['features'])} columns, got {X.shape}")
    out = np.full(len(X), float(spec["baseline"]))
    rows = np.arange(len(X))
    for t in spec["_arrays"]:
        node = np.zeros(len(X), dtype=int)
        for _ in range(t["depth"] + 1):
            leaf = t["leaf"][node]
            if leaf.all():
                break
            x = X[rows, t["feature"][node]]
            go_left = np.where(np.isnan(x), t["missing_left"][node], x <= t["threshold"][node])
            nxt = np.where(go_left, t["left"][node], t["right"][node])
            node = np.where(leaf, node, nxt)
        out += t["value"][node]
    return out


def _arrays(trees):
    arrs = []
    for t in trees:
        a = {k: np.asarray(v) for k, v in t.items()}
        a["leaf"] = a["leaf"].astype(bool)
        a["missing_left"] = a["missing_left"].astype(bool)
        # depth of the deepest leaf, by walking the tree once
        depth, stack = 0, [(0, 0)]
        while stack:
            n, d = stack.pop()
            depth = max(depth, d)
            if not a["leaf"][n]:
                stack += [(int(a["left"][n]), d + 1), (int(a["right"][n]), d + 1)]
        a["depth"] = depth
        arrs.append(a)
    return arrs


def export(model, features, X_check, path, meta=None, tol=1e-9):
    """Write `model` to `path` as JSON, after proving the export reproduces it on X_check."""
    import sklearn

    spec = {"format": FORMAT, "features": list(features),
            "baseline": float(np.ravel(model._baseline_prediction)[0]),
            "trees": _trees_from(model), "sklearn_version": sklearn.__version__,
            "meta": meta or {}}
    spec["_arrays"] = _arrays(spec["trees"])
    want = model.predict(X_check)
    got = predict(spec, X_check)
    worst = float(np.max(np.abs(want - got))) if len(want) else 0.0
    if not worst <= tol:
        raise AssertionError(f"exported trees disagree with scikit-learn by {worst:.3g} "
                             f"(tolerance {tol}); not writing {path}")
    spec.pop("_arrays")
    spec["meta"]["export_check"] = {"rows": int(len(X_check)), "max_abs_diff": worst}
    body = json.dumps(spec, separators=(",", ":"))
    spec_id = hashlib.sha256(body.encode()).hexdigest()[:12]
    spec["model_id"] = spec_id
    Path(path).write_text(json.dumps(spec, separators=(",", ":")), encoding="utf-8")
    return spec_id, worst


def load(path) -> dict:
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    if spec.get("format") != FORMAT:
        raise ValueError(f"{path}: unknown model format {spec.get('format')!r}")
    spec["_arrays"] = _arrays(spec["trees"])
    return spec
