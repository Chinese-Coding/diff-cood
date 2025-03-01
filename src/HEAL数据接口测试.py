import os

from opencood.data_utils.datasets import build_dataset


def main(args):
    opencood_train_dataset = build_dataset(args, visualize=False, train=True)

    for batch in opencood_train_dataset:
        print(1)


if __name__ == "__main__":
    from omegaconf import OmegaConf

    args = OmegaConf.load(os.path.expanduser("~/fleet/diff-cood/train_detection_with_heter.yaml"))
    # args.output_dir = "~/Desktop/logs/vis_2025_02_20"
    # args.batch_size = 1
    # args.ratio = 0.1  # 此时生成的 bev_map 的尺寸为 (1024, 1024)
    # args.postprocess_args.ratio = 0.1
    # args.postprocess_args.anchor_args.feature_stride = 8

    main(args)
