"""Global evaluation work partitioning without DistributedSampler padding."""
from __future__ import annotations


def distributed_eval_indices(
    dataset_size: int,
    *,
    world_size: int,
    rank: int,
    max_batches: int,
    batch_size: int,
):
    """Return this rank's indices from one globally bounded evaluation set.

    Unlike ``DistributedSampler``, this partition never pads a small dataset
    by duplicating examples when ``world_size > dataset_size``.
    """
    if dataset_size < 0 or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("invalid distributed evaluation geometry")
    if max_batches < 0 or batch_size < 1:
        raise ValueError("invalid evaluation batch limit")
    global_limit = min(dataset_size, max_batches * batch_size)
    return list(range(rank, global_limit, world_size))
