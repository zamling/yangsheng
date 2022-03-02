# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from .casia import build as build_casia

def build_dataset(image_set, args):
    if args.dataset_file == 'casia':
        return build_casia(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')