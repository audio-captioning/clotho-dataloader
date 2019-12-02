#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import MutableSequence, Union, Tuple, AnyStr
from numpy import ndarray

from torch import cat as pt_cat, zeros as pt_zeros, \
    ones as pt_ones, from_numpy, Tensor

__author__ = 'Konstantinos Drossos'
__docformat__ = 'reStructuredText'
__all__ = ['clotho_collate_fn']


def clotho_collate_fn(batch: MutableSequence[ndarray],
                      nb_t_steps: Union[AnyStr, Tuple[int, int]]) \
        -> Tuple[Tensor, Tensor]:
    """Pads data.

    :param batch: Batch data.
    :type batch: list[numpy.ndarray]
    :param nb_t_steps: Number of time steps to\
                       pad/truncate to. Cab use\
                       'max', 'min', or exact number\
                       e.g. (1024, 10).
    :type nb_t_steps: str|(int, int)
    :return: Padded data.
    :rtype: torch.Tensor, torch.Tensor
    """
    if type(nb_t_steps) == str:
        truncate_fn = max if nb_t_steps.lower == 'max' else min
        in_t_steps = truncate_fn([i[0].shape[0] for i in batch])
        out_t_steps = truncate_fn([i[1].shape[0] for i in batch])
    else:
        in_t_steps, out_t_steps = nb_t_steps

    input_features = batch[0][0].shape[-1]
    eos_token = batch[0][1][-1]

    input_tensor = pt_cat([pt_cat([
        pt_zeros(in_t_steps - i[0].shape[0], input_features).float(),
        from_numpy(i[0]).float()]).unsqueeze(0) for i in batch])

    output_tensor = pt_cat([pt_cat([
        from_numpy(i[1]).long(),
        pt_ones(out_t_steps - len(i[1])).mul(eos_token).long()]).unsqueeze(0)
                            for i in batch])

    return input_tensor, output_tensor

# EOF
