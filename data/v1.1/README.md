# V1.1 Truthset Exceptions

During comparison of the v1 and v1.1 truthsets, it was observed that 2 papers in the train split that were not in
PubMed's Open Access subset were subsequently moved over. Since these papers were not in PMC-OA at the time of pipeline
execution for benchmarks, they were not processed, thus they have been manually set to `can_access = False` in the v1.1
truthset. If for any reason this truthset is regenerated, then these papers must again be modified.

The two papers in question are:

37962692
38114583
