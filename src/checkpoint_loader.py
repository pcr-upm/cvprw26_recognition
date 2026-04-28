#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Ivan Ferre'
__email__ = 'ivan.ferre@upm.es'

import torch


def load_submodel_state_dict(checkpoint_path: str, submodel_prefix: str) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    submodel_state = {
        k.replace(f"{submodel_prefix}.", ""): v for k, v in state_dict.items() if k.startswith(submodel_prefix)
    }
    return submodel_state
