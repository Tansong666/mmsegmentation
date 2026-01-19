# Copyright (c) OpenMMLab. All rights reserved.
import mmengine.fileio as fileio

from mmseg.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset

@DATASETS.register_module()
class RootSystemDataset(BaseSegDataset):
    """RootSystemDataset dataset.

    In segmentation map annotation for RootSystemDataset, 0 stands for background, which is
    included in 2 categories. ``reduce_zero_label`` is fixed to False. The
    ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '_manual1.png'.
    """
    METAINFO = dict(
        # classes=('root',), 
        # palette=[[255, 0, 0]]) # 只保留对应的调色板

        classes=('background', 'root'),
        # palette=[[120, 120, 120], [6, 230, 230]])
        # 推荐配色方案 (背景色建议设为全黑 [0,0,0] 以实现透明效果)
        # 方案1：高亮红 (经典，对比度最强)
        palette=[[0, 0, 0], [255, 0, 0]])
        
        # 方案2：荧光绿 (科技感，清晰)
        # palette=[[0, 0, 0], [0, 255, 0]])

        # 方案3：金黄色 (柔和高亮)
        # palette=[[0, 0, 0], [255, 215, 0]])

        # 方案4：青色 (原方案优化，背景透明)
        # palette=[[0, 0, 0], [6, 230, 230]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
        assert fileio.exists(
            self.data_prefix['img_path'], backend_args=self.backend_args)
