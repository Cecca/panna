import panna
import numpy as np
from icecream import ic
import os
from pathlib import Path
from collections import defaultdict

def edges_to_bracket_notation_tree(weights, edges, root=None):
    """
    Converts edge/weight lists into Bracket Notation: {Label{Child1}{Child2}...}
    """
    
    # 1. Build Adjacency List (Graph)
    tree = defaultdict(list)
    nodes = set()
    
    for (u, v), weight in zip(edges, weights):
        tree[u].append((v, weight))
        tree[v].append((u, weight))
        nodes.add(u)
        nodes.add(v)

    # 2. Determine Root
    # If no root is provided, we pick the "smallest" node (lexicographically or numerically)
    # In a real clustering scenario, you might want to pick a specific node.
    if root is None:
        root = min(nodes)

    # 3. Helper to escape special characters
    def escape_label(label):
        s = str(label)
        return s.replace("{", "\\{").replace("}", "\\}")

    # 4. Recursive Builder
    def build_bracket_notation(node, parent):
        # Sort children by weight (or name) to ensure deterministic output
        # We filter out the 'parent' to prevent infinite recursion in undirected graph logic
        children_nodes = [
            neighbor for neighbor, weight in sorted(tree[node], key=lambda x: -x[1])
            if neighbor != parent
        ]
        
        # Generate the string for all children
        children_str = ""
        for child in children_nodes:
            children_str += build_bracket_notation(child, node)
            
        # CRITICAL FIX:
        # Format must be: { LABEL + CHILDREN_STR }
        return "{" + escape_label(node) + children_str + "}"

    return build_bracket_notation(root, None)

    

if __name__ == "__main__":
        paths = [
                "fashion-mnist-784-euclidean",
                "glove-100-angular",
                # "nytimes-256-angular",
                # "gist-960-euclidean",
                # "simplewiki-openai-3072-normalized",
                # "sift-128-euclidean",
                # "deep-image-96-angular",
                # "chem",
                # "ht",
                # "imagenet-clip-512-normalized",
                # "landmark-nomic-768-normalized",
                # "census",
                # "pamap2",
        ]
        path_prefix = Path(__file__).resolve().parents[2]

        results_folder = os.path.join(path_prefix, "results")        

        with open( os.path.join(results_folder, "tree_similarity_results.csv"), "a+") as f_out:
            for path in paths:
                _, data = panna.datasets.load(name=path, pca_dimensions=4 if path == "pamap2" else None)
                data = data[:1000]
                epsilon = 0.2
                
                tree_weights, tree_edges = panna.EMST(data, epsilon=0, delta=0.001, family="e2lsh").find_mst()

                
                emst_approx = panna.EMST(data, epsilon=epsilon, delta=0.001, family="e2lsh").find_mst()
                tree_weights_approx, tree_edges_approx = emst_approx
                
                tree1 = edges_to_bracket_notation_tree(tree_weights, tree_edges)
                tree2 = edges_to_bracket_notation_tree(tree_weights_approx, tree_edges_approx)
                ic(tree1)
                ic(tree2)
                
                # Call from terminal to compute TED
                import subprocess
                with open("tree1.txt", "w") as f1:
                    f1.write(tree1)
                with open("tree2.txt", "w") as f2:
                    f2.write(tree2)
                path_ted = os.path.join(path_prefix, "/tree-similarity/build/ted")
                cmd = [str("." + path_ted), "file", "tree1.txt", "tree2.txt"]
                result = subprocess.run(cmd, capture_output=True, text=True)
                print(result.stdout)
                # Result is formatted as 
                # Size of source tree:1000
                # Size of destination tree:1000
                # Distance TED:264 
                # so we extract the last line and get the integer value
                ted_distance = int(result.stdout.strip().split("\n")[-1].split(":")[-1])
                
                f_out.write(f"{path},{epsilon},{ted_distance}\n")
        print("Done.")
                
                