"""Deterministic candidate layout for distributed NFT sampling."""
from __future__ import annotations

import torch


def build_unsplit_candidate_epoch(
    *, dataset_size: int, num_replicas: int, batch_size: int,
    batches_per_epoch: int, candidates_per_prompt: int, seed: int,
):
    """Return an epoch layout with complete, distinct groups in each batch."""
    global_batch = num_replicas * batch_size
    if global_batch % candidates_per_prompt:
        raise ValueError(
            f"global batch {global_batch} is not divisible by candidate group "
            f"size {candidates_per_prompt}"
        )
    groups_per_batch = global_batch // candidates_per_prompt
    prompt_count = groups_per_batch * batches_per_epoch
    if dataset_size < prompt_count:
        raise ValueError(
            f"dataset has {dataset_size} rows but epoch needs {prompt_count} prompts"
        )
    generator = torch.Generator().manual_seed(seed)
    selected = torch.randperm(dataset_size, generator=generator)[:prompt_count].tolist()
    result = []
    for batch_index in range(batches_per_epoch):
        group_start = batch_index * groups_per_batch
        repeated = [
            index
            for index in selected[group_start:group_start + groups_per_batch]
            for _ in range(candidates_per_prompt)
        ]
        order = torch.randperm(global_batch, generator=generator).tolist()
        shuffled = [repeated[index] for index in order]
        result.append([
            shuffled[rank * batch_size:(rank + 1) * batch_size]
            for rank in range(num_replicas)
        ])
    return result


def build_split_candidate_epoch(
    *, dataset_size: int, num_replicas: int, batch_size: int,
    batches_per_epoch: int, candidates_per_prompt: int, seed: int,
):
    """Return ``[batch][rank][local sample]`` indices for one epoch.

    Candidate groups may cross rank and sampling-batch boundaries, but the
    complete epoch contains every selected prompt exactly ``k`` times.
    """
    total = num_replicas * batch_size * batches_per_epoch
    if total % candidates_per_prompt:
        raise ValueError(
            f"global epoch {total} is not divisible by candidate group "
            f"size {candidates_per_prompt}"
        )
    prompt_count = total // candidates_per_prompt
    if dataset_size < prompt_count:
        raise ValueError(
            f"dataset has {dataset_size} rows but epoch needs {prompt_count} prompts"
        )
    generator = torch.Generator().manual_seed(seed)
    selected = torch.randperm(dataset_size, generator=generator)[:prompt_count].tolist()
    repeated = [index for index in selected for _ in range(candidates_per_prompt)]
    order = torch.randperm(len(repeated), generator=generator).tolist()
    shuffled = [repeated[index] for index in order]
    result = []
    global_batch = num_replicas * batch_size
    for batch_index in range(batches_per_epoch):
        batch_start = batch_index * global_batch
        result.append([
            shuffled[
                batch_start + rank * batch_size:
                batch_start + (rank + 1) * batch_size
            ]
            for rank in range(num_replicas)
        ])
    return result
