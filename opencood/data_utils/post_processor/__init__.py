# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


from opencood.data_utils.post_processor.diff_post_processor import DiffPostProcessor

__all__ = {"DiffPostprocessor": DiffPostProcessor}


def build_postprocessor(anchor_cfg, train):
    process_method_name = anchor_cfg["core_method"]
    anchor_generator = __all__[process_method_name](postprocess_args=anchor_cfg, train=train)

    return anchor_generator
