from icecream import ic

from data_related.entity import ObjectBbxData
from data_related.stable_diffusion_dataset import StableDiffusionDataset
from opencood.data_utils.post_processor.diff_base_post_processor import DiffBasePostProcessor
from loguru import logger

if __name__ == "__main__":
    dataset = StableDiffusionDataset("/datasets/OPV2V/train")
    dataset.reinitialize()
    item = dataset[0]
    ic(item.cav_info)  # 打印效果不是很好, 通过 debug 看又看不到. 很烦

    # 测试框生成 (参数都是从 HEAL 里面抄来的)
    cav_lidar_range = [-102.4, -102.4, -3, 102.4, 102.4, 1]
    order = "lwh"  # hwl or lwh
    max_num = 150
    post_processor = DiffBasePostProcessor(order, max_num, cav_lidar_range, cav_lidar_range)
    for item in dataset:
        object_np, mask, object_ids = post_processor.generate_object_center_lidar(item, item.cav_info.lidar_pose)
        object_bbx_data = ObjectBbxData(object_bbx_center=object_np, object_bbx_mask=mask, object_ids=object_ids)
        gt_box = post_processor.generate_gt_bbx(object_bbx_data)
        logger.success(f"共生成了 {gt_box.shape[0]} 个 gt box")
