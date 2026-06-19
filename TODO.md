# Restructuring the EMST algorithm

## Current pros and cons

- We use the LSH forest approach, moving from hashes of length 4 down to 1.
  Shortening the hashes increases the collision probability, allowing to discover more pairs and to confirm older ones. The problem is that the step is too abrupt: especially going from hashes of length 2 to length 1 is a killer for the performance.
- The good thing of the current approach is that (within the same repetition) a pair will be inspected at most once. Furthermore, we keep track of confirmed connected components, avoiding generating already-connected pairs in the first place.
- Another drawback is that we use Kruskal's algorithm as a subroutine for building the EMST incrementally. This means that we are sorting edges all the time. This shows up as a heavy hitter in the collector. We should use something that does not require to sort edges (like Boruvka's algorithm).
- The setting of the LSH scaling (or quantization width) is another problem. Right now in most cases the heuristic (picking a value large enough to have between n/2 and 2n collisions with hashes of length 4) works well enough, but in some datasets some of the edges of the EMST are much longer than others, creating a scaling problem: if we set a small scaling factor we end up with too many iterations to confirm the long edges, if we set it larger we drown in collisions

## A plan

### 1. Switch from Kruskal's to Boruvka's algorithm

Right now we have:
- `kruskal_merge` to merge together (in linear time) two trees
- `kruskal_new_edges` to select from a batch of edges the ones that could improve the tree

Both require to work with a sorted list of edges. Using an approach à la Boruvka we could instead, in the worker:

- pick the batch of edges passed by `table.search_pairs_different_groups`
- run several rounds of Boruvka's algorithm on these edges: in each round we accumulate the per-component minimum outgoing edge, possibly connecting multiple components
- RISK: how do we aggregate all the batches in a single compact update to send to the collector? We might be connecting two components too early with lighter edges coming in later batches.

