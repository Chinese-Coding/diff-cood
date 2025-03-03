def _change_test():
    import torch

    # 生成两个随机的 tensor，形状分别为 (4, 1280, 10, 13) 和 (4, 1280, 32, 16)
    tensor1 = torch.randn(4, 1280, 10, 13)
    tensor2 = torch.randn(4, 1280, 32, 16)

    # 打印初始的左上角 5x5 区域
    print("Before swap:")
    print("Tensor1 top-left 5x5:\n", tensor1[:, :, :5, :5])
    print("Tensor2 top-left 5x5:\n", tensor2[:, :, :5, :5])

    # 获取左上角 5x5 特征图
    top_left_5x5_tensor1 = tensor1[:, :, :5, :5]
    top_left_5x5_tensor2 = tensor2[:, :, :5, :5]

    # 交换左上角 5x5 区域
    tensor1[:, :, :5, :5] = top_left_5x5_tensor2
    tensor2[:, :, :5, :5] = top_left_5x5_tensor1

    # 打印交换后的左上角 5x5 区域，验证是否交换
    print("\nAfter swap:")
    print("Tensor1 top-left 5x5:\n", tensor1[:, :, :5, :5])
    print("Tensor2 top-left 5x5:\n", tensor2[:, :, :5, :5])

    # 进行验证：检查交换后的 tensor1 和 tensor2 左上角是否正确交换
    assert torch.all(torch.eq(tensor1[:, :, :5, :5], top_left_5x5_tensor2)), "Tensor1's top-left 5x5 region is not correct."
    assert torch.all(torch.eq(tensor2[:, :, :5, :5], top_left_5x5_tensor1)), "Tensor2's top-left 5x5 region is not correct."

    print("\nValidation passed! The top-left 5x5 regions are correctly swapped.")


def _torch_load_test():
    import os

    import torch
    from omegaconf import OmegaConf

    args = OmegaConf.load(os.path.expanduser("~/fleet/diff-cood/config.yaml"))
    checkpoints = torch.load(os.path.expanduser(args.lift_splat_shoot_args.pretrained_model_path), weights_only=False)
    print(checkpoints.keys())
    # print(checkpoints["encoder_m2"])
    # print(checkpoints)


if __name__ == "__main__":
    # _change_test()
    _torch_load_test()
