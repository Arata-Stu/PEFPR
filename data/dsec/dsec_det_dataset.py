import numpy as np
import cv2

from utils.dsec_utils import filter_tracks, map_classes, filter_small_bboxes 
from utils.helper import compute_class_mapping
from .dsec_det.dataset import DSECDet, render_events_on_image, render_object_detections_on_image
from ..utils.augmentor import init_transforms

class DSECDataset(DSECDet):
    def __init__(self, root, split="train", sync="front", debug=False, split_config=None, transforms=None, 
                 use_image=True, use_events=False, min_bbox_diag=30, min_bbox_side=20): 

        super().__init__(root=root, split=split, sync=sync, debug=False, split_config=split_config)
        
        self.debug_augmented = debug 
        self.transforms = transforms
        self.use_image = use_image
        self.use_events = use_events
        self.min_bbox_diag = min_bbox_diag
        self.min_bbox_side = min_bbox_side
        
        if self.transforms is not None:
            init_transforms(self.transforms, self.height, self.width)

        self.dataset_name = "DSEC-Det"
        self.time_window = 1000000

        self.target_classes = ["pedestrian", "car", "two-wheeler"]
        self.MAPPING = dict(
            pedestrian="pedestrian", rider="two-wheeler", car="car", bus="car", 
            truck="car", bicycle="two-wheeler", motorcycle="two-wheeler", train=None
        )

        self.class_remapping = compute_class_mapping(self.target_classes, self.classes, self.MAPPING)
        self.classes = self.target_classes

        print(f"[{self.dataset_name}] Filtering dataset using DAGR's vectorized functions...")
        self.image_index_pairs, self.track_masks = filter_tracks(
            dataset=self, 
            image_width=self.width, 
            image_height=self.height,
            class_remapping=self.class_remapping,
            min_bbox_height=self.min_bbox_side,
            min_bbox_diag=self.min_bbox_diag,
            scale=1, 
            only_perfect_tracks=False
        )
        print(f"[{self.dataset_name}] Filtering complete. Total valid frames: {len(self)}")

    def __len__(self):
        # フィルタリング済みのペア数の合計を返す
        return sum(len(pairs) for pairs in self.image_index_pairs.values())

    def rel_index(self, idx):
        """グローバルなidxを、ディレクトリ名とローカルなインデックスに変換する (DAGR由来)"""
        for folder in self.subsequence_directories:
            name = folder.name
            image_index_pairs = self.image_index_pairs[name]
            directory = self.directories[name]
            track_mask = self.track_masks[name]
            if idx < len(image_index_pairs):
                return directory, image_index_pairs, track_mask, idx
            idx -= len(image_index_pairs)
        raise IndexError

    def preprocess_events(self, events):
        mask = events['y'] < self.height
        events = {k: v[mask] for k, v in events.items()}
        if len(events['t']) > 0:
            events['t'] = self.time_window + events['t'] - events['t'][-1]
        events['p'] = events['p'].reshape((-1, 1)).astype(np.int8)
        events['x'] = events['x'].astype(np.float32)
        events['y'] = events['y'].astype(np.float32)
        return events

    def generate_event_representation(self, events):
        return events

    def __getitem__(self, idx):
        # 1. インデックスの解決
        directory, image_index_pairs, track_mask, rel_idx = self.rel_index(idx)
        
        idx0, idx1 = image_index_pairs[rel_idx]
        dir_name = directory.root.name

        output = {}
        
        # 2. データの取得 (DSECDetのget系メソッドはディレクトリ名を受け取れます)
        if self.use_image:
            output['image'] = self.get_image(idx1, directory_name=dir_name)
            
        if self.use_events:
            raw_events = self.get_events(idx1, directory_name=dir_name)
            processed_events = self.preprocess_events(raw_events) 
            output['events'] = self.generate_event_representation(processed_events)
        
        # 3. トラックの取得と前処理
        tracks = self.get_tracks(idx1, mask=track_mask, directory_name=dir_name)
        
        if tracks is not None and len(tracks) > 0:
            # 取得したトラックのクラスIDを新しいクラスIDにマッピング
            mapped_ids, class_mask = map_classes(tracks['class_id'], self.class_remapping)
            
            size_mask = filter_small_bboxes(tracks['w'], tracks['h'], self.min_bbox_side, self.min_bbox_diag)
            
            valid_mask = class_mask & size_mask
            tracks = tracks[valid_mask]
            tracks['class_id'] = mapped_ids[valid_mask]
            
        output['tracks'] = tracks

        # 4. データ拡張とデバッグ
        if hasattr(self, 'transforms') and self.transforms is not None:
            output = self.transforms(output)

        if self.debug_augmented:
            image = (255 * (output['image'].astype("float32") / 255) ** (1/2.2)).astype("uint8") if 'image' in output else np.zeros((self.height, self.width, 3), dtype=np.uint8)
            if 'events' in output and isinstance(output['events'], dict) and 'x' in output['events']:
                ev = output['events']
                image = render_events_on_image(image, x=ev['x'], y=ev['y'], p=ev['p'])
            output['debug'] = render_object_detections_on_image(image, output['tracks'])

        return output