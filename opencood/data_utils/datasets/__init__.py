from opencood.data_utils.datasets.basedataset.opv2v_basedataset import OPV2VBaseDataset
from opencood.data_utils.datasets.intermediate_heter_fusion_dataset import getIntermediateheterFusionDataset


def build_dataset(dataset_cfg, visualize=False, train=True):
    fusion_name = dataset_cfg["fusion"]["core_method"]
    dataset_name = dataset_cfg["fusion"]["dataset"]

    assert fusion_name in [
        "late",
        "lateheter",
        "intermediate",
        "intermediate2stage",
        "intermediateheter",
        "early",
        "intermediateheterinfer",
    ]
    assert dataset_name in ["opv2v", "v2xsim", "dairv2x", "v2xset"]

    fusion_dataset_func = "get" + fusion_name.capitalize() + "FusionDataset"
    fusion_dataset_func = eval(fusion_dataset_func)
    base_dataset_cls = dataset_name.upper() + "BaseDataset"
    base_dataset_cls = eval(base_dataset_cls)

    dataset = fusion_dataset_func(base_dataset_cls)(params=dataset_cfg, visualize=visualize, train=train)

    return dataset
