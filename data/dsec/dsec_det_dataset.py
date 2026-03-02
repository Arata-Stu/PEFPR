import numpy as np

from utils.helper import compute_class_mapping
from .dsec_det.dataset import DSECDet, render_events_on_image, render_object_detections_on_image
from ..utils.augmentor  import init_transforms

class DSECDataset(DSECDet):
    """
    DSECDetを継承し、データ拡張(Transform)機能とモダリティ選択機能を追加したクラス
    """
    def __init__(self, root, split="train", sync="front", debug=False, split_config=None, transforms=None, 
                 use_image=True, use_events=False): 

        super().__init__(root=root, split=split, sync=sync, debug=False, split_config=split_config)
        
        self.debug_augmented = debug 
        self.transforms = transforms
        self.use_image = use_image
        self.use_events = use_events
        
        if self.transforms is not None:
            init_transforms(self.transforms, self.height, self.width)

        self.dataset_name = "DSEC-Det"
        self.time_window = 1000000

        self.target_classes = ["pedestrian", "car", "two-wheeler"]
        self.MAPPING = dict(
            pedestrian="pedestrian",
            rider="two-wheeler",
            car="car",
            bus="car",
            truck="car",
            bicycle="two-wheeler",
            motorcycle="two-wheeler",
            train=None
        )

        original_classes = self.classes
        self.class_remapping = compute_class_mapping(self.target_classes, original_classes, self.MAPPING)
        self.classes = self.target_classes
    
    def preprocess_events(self, events):
        """生のイベントのクリッピングと正規化"""
        mask = events['y'] < self.height
        events = {k: v[mask] for k, v in events.items()}
        
        if len(events['t']) > 0:
            events['t'] = self.time_window + events['t'] - events['t'][-1]
            
        events['p'] = events['p'].reshape((-1, 1)).astype(np.int8)
        
        events['x'] = events['x'].astype(np.float32)
        events['y'] = events['y'].astype(np.float32)
        
        return events

    def generate_event_representation(self, events):
        """
        🌟 拡張用メソッド: 
        将来的にここでヒストグラムやイベント画像、Voxel Gridなどに変換します。
        現在は生のイベント辞書をそのまま返します。
        """
        # 例: return events_to_histogram(events, self.height, self.width)
        return events

    def __getitem__(self, item):
        output = {}
        
        # 1. 画像の読み込み (必要な場合のみ)
        if self.use_image:
            output['image'] = self.get_image(item)
            
        # 2. イベントの読み込みと表現変換 (必要な場合のみ)
        if self.use_events:
            raw_events = self.get_events(item)
            processed_events = self.preprocess_events(raw_events) 
            output['events'] = self.generate_event_representation(processed_events)
        
        # 3. バウンディングボックスの取得とマッピング
        tracks = self.get_tracks(item)
        if tracks is not None and len(tracks) > 0:
            mapped_ids = self.class_remapping[tracks['class_id']]
            valid_mask = mapped_ids != -1
            tracks = tracks[valid_mask]
            tracks['class_id'] = mapped_ids[valid_mask]
        output['tracks'] = tracks

        # 4. データ拡張
        # ※以前実装したAugmentationクラスは 'if "events" in data:' のように
        # キーの存在確認を行っているため、eventsが無くてもエラーになりません！
        if hasattr(self, 'transforms') and self.transforms is not None:
            output = self.transforms(output)

        # 5. デバッグ可視化
        if self.debug_augmented:
            # ベース画像の準備
            if 'image' in output:
                image = (255 * (output['image'].astype("float32") / 255) ** (1/2.2)).astype("uint8")
            else:
                image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            # イベントの重畳描画 (イベントが存在し、かつ生の辞書形式の場合のみ)
            if 'events' in output and isinstance(output['events'], dict) and 'x' in output['events']:
                ev = output['events']
                image = render_events_on_image(image, x=ev['x'], y=ev['y'], p=ev['p'])
                
            output['debug'] = render_object_detections_on_image(image, output['tracks'])

        return output