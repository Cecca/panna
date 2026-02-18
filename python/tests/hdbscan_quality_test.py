import sys
import os
from pathlib import Path
sys.path.append(os.path.join(Path(__file__).resolve().parents[2]))
import panna
import panna.datasets
import numpy as np
import joblib
from icecream import ic
import csv
import logging
import fast_hdbscan

if __name__ == "__main__":
    
        paths = [
                 "fashion-mnist-784-euclidean",
                 "glove-100-angular",
                 "nytimes-256-angular",
                 "gist-960-euclidean",
                 "simplewiki-openai-3072-normalized",
                 "sift-128-euclidean",
                 "deep-image-96-angular",
                 "chem",
                 "ht",
                "imagenet-clip-512-normalized",
                "landmark-nomic-768-normalized",
                "census",
                "pamap2",
        ]
        path_prefix = Path(__file__).resolve().parents[2]

        results_folder = os.path.join(path_prefix, "results")        

        with open( os.path.join(results_folder, "hdbscan_quality_results.csv"), "a+") as f_out:
            for path in paths:
                _, data = panna.datasets.load(name=path, pca_dimensions=4 if path == "pamap2" else None)
                # Take a sample of the datasets 
                data = data[:1000]
                
                # Compute the Ɛ-HDBSCAN clustering
                mst_approx, core_approx, _ = panna.EMST(data, epsilon=0.2, delta=0.001).find_mst_dbscan(k = 5)
                
                # Compute exact HDBSCAN clustering
                mst_exact, core_exact, _ = panna.EMST(data, epsilon=0, delta=0.001).find_mst_dbscan(k = 5)
                # exact = fast_hdbscan.HDBSCAN(min_samples=10)
                # result = exact.fit_predict(data)
                # mst_exact = exact._min_spanning_tree
                # core_exact = exact._core_distances
                
                
                # Compute the cophenetic correlation coefficient between the two clusterings
                def _normalize_mst_edges(mst, n_expected):
                        # Normalize to list of (w, u, v) with 0-based u,v in [0, n_expected-1]
                        def as_list_of_triples(x):
                                if x is None:
                                        return []
                                try:
                                        import numpy as _np
                                except Exception:
                                        pass

                                # numpy array with shape (m,3)
                                try:
                                        import numpy as _np
                                        if isinstance(x, _np.ndarray) and x.ndim == 2 and x.shape[1] >= 3:
                                                arr = x[:, :3]
                                                return [tuple(map(float, arr[i])) for i in range(arr.shape[0])]
                                except Exception:
                                        pass

                                # tuple/list of three arrays
                                if isinstance(x, (tuple, list)) and len(x) == 3:
                                        a, b, c = x
                                        try:
                                                import numpy as _np
                                                if hasattr(a, "__len__") and hasattr(b, "__len__") and hasattr(c, "__len__"):
                                                        m = min(len(a), len(b), len(c))
                                                        return [ (float(a[i]), float(b[i]), float(c[i])) for i in range(m) ]
                                        except Exception:
                                                pass

                                # list of tuples
                                if isinstance(x, list) and x and isinstance(x[0], (tuple, list)) and len(x[0]) >= 3:
                                        return [ (float(t[0]), float(t[1]), float(t[2])) for t in x ]

                                # Unsupported, return empty to fail later
                                return []

                        raw = as_list_of_triples(mst)
                        if not raw:
                                raise ValueError("Unrecognized MST edge format")

                        # Determine which two columns are indices and which is weight
                        candidates = [ (0,1,2), (0,2,1), (1,2,0) ]  # (idx_col1, idx_col2, weight_col)
                        chosen = None
                        for i1, i2, iw in candidates:
                                idx1 = [raw[k][i1] for k in range(len(raw))]
                                idx2 = [raw[k][i2] for k in range(len(raw))]
                                # integer-like check
                                if all(float(v).is_integer() for v in idx1) and all(float(v).is_integer() for v in idx2):
                                        mn = int(min(min(idx1), min(idx2)))
                                        mx = int(max(max(idx1), max(idx2)))
                                        # accept 0-based or 1-based within expectations
                                        if 0 <= mn and mx <= n_expected - 1:
                                                chosen = (i1, i2, iw, 0)
                                                break
                                        if 1 <= mn and mx <= n_expected:
                                                chosen = (i1, i2, iw, 1)
                                                break
                        if chosen is None:
                                # Fallback: assume format (w, u, v) and clamp to range if possible
                                chosen = (1, 2, 0, 0)

                        i1, i2, iw, base = chosen
                        edges = []
                        for t in raw:
                                u = int(t[i1]) - (1 if base == 1 else 0)
                                v = int(t[i2]) - (1 if base == 1 else 0)
                                w = float(t[iw])
                                if 0 <= u < n_expected and 0 <= v < n_expected and u != v:
                                        edges.append((w, u, v))
                        if len(edges) == 0:
                                raise ValueError("No valid edges after normalization")
                        return edges

                def _build_lca(n, edges):
                        adj = [[] for _ in range(n)]
                        for w, u, v in edges:
                                u = int(u); v = int(v); w = float(w)
                                if 0 <= u < n and 0 <= v < n:
                                        adj[u].append((v, w))
                                        adj[v].append((u, w))

                        LOG = (n).bit_length()
                        up = [[-1] * n for _ in range(LOG)]
                        mx = [[0.0] * n for _ in range(LOG)]
                        depth = [-1] * n

                        # DFS to set depth, parent (up[0]) and max edge to parent (mx[0])
                        for root in range(n):
                                if depth[root] != -1:
                                        continue
                                depth[root] = 0
                                stack = [root]
                                while stack:
                                        u = stack.pop()
                                        for v, w in adj[u]:
                                                if depth[v] == -1:
                                                        depth[v] = depth[u] + 1
                                                        up[0][v] = u
                                                        mx[0][v] = w
                                                        stack.append(v)

                        # Binary lifting
                        for k in range(1, LOG):
                                upk, upkm1 = up[k], up[k - 1]
                                mxk, mxkm1 = mx[k], mx[k - 1]
                                for v in range(n):
                                        p = upkm1[v]
                                        if p != -1:
                                                upk[v] = upkm1[p]
                                                mxk[v] = mxkm1[v] if mxkm1[v] >= mxkm1[p] else mxkm1[p]

                        return up, mx, depth, LOG


                def _max_edge_on_path(u, v, up, mx, depth, LOG):
                        res = 0.0
                        if depth[u] < depth[v]:
                                u, v = v, u

                        # Lift u to depth of v
                        diff = depth[u] - depth[v]
                        k = 0
                        while diff:
                                if diff & 1:
                                        w = mx[k][u]
                                        if w > res:
                                                res = w
                                        u = up[k][u]
                                diff >>= 1
                                k += 1

                        if u == v:
                                return res

                        # Lift both until just below LCA
                        for k in range(LOG - 1, -1, -1):
                                pu, pv = up[k][u], up[k][v]
                                if pu != -1 and pu != pv:
                                        wu, wv = mx[k][u], mx[k][v]
                                        if wu > res:
                                                res = wu
                                        if wv > res:
                                                res = wv
                                        u, v = pu, pv

                        # One step to LCA
                        wu, wv = mx[0][u], mx[0][v]
                        if wu > res:
                                res = wu
                        if wv > res:
                                res = wv
                        return res


                def cophenetic_correlation(mst1, mst2, n):
                        up1, mx1, depth1, LOG1 = _build_lca(n, mst1)
                        up2, mx2, depth2, LOG2 = _build_lca(n, mst2)

                        c1 = []
                        c2 = []
                        append1 = c1.append
                        append2 = c2.append
                        for i in range(n):
                                for j in range(i + 1, n):
                                        append1(_max_edge_on_path(i, j, up1, mx1, depth1, LOG1))
                                        append2(_max_edge_on_path(i, j, up2, mx2, depth2, LOG2))

                        c1 = np.asarray(c1, dtype=float)
                        c2 = np.asarray(c2, dtype=float)
                        denom = np.std(c1) * np.std(c2)
                        return float(((c1 - c1.mean()) * (c2 - c2.mean())).mean() / denom) if denom > 0 else 0.0


                assert len(core_exact) == len(core_approx)
                n_points = int(len(core_exact))
                mst_exact_norm = _normalize_mst_edges(mst_exact, n_points)
                mst_approx_norm = _normalize_mst_edges(mst_approx, n_points)
                cophenetic_corr = cophenetic_correlation(mst_exact_norm, mst_approx_norm, n_points)
                print(f"Dataset: {path}, Cophenetic Correlation: {cophenetic_corr:.6f}")

                writer = csv.writer(f_out)
                writer.writerow([path, cophenetic_corr])
