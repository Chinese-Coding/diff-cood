from opencood.data_utils.pre_processor.diff_pre_processor import DiffPreProcessor

__all__ = {"DiffPreprocessor": DiffPreProcessor}

from opencood.data_utils.pre_processor.diff_pre_processor import DiffPreProcessor


def build_preprocessor(preprocess_cfg, train):
    process_method_name = preprocess_cfg["core_method"]
    error_message = (
        f"{process_method_name} is not found. Please add your processor file's name in opencood/data_utils/processor/init.py"
    )
    assert process_method_name in ["DiffPreprocessor"], error_message

    processor = __all__[process_method_name](preprocessor_args=preprocess_cfg.args, train=train)

    return processor
