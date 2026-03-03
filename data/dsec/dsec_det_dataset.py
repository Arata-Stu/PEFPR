import numpy as np
import cv2
 
from utils.helper import compute_class_mapping, format_targets_for_yolox
from .dsec_det.dataset import DSECDet, render_events_on_image, render_object_detections_on_image
from ..utils.augmentor import init_transforms
from ..utils.representation import EventHistogram, VoxelGrid
from .dsec_utils import filter_tracks, map_classes, filter_small_bboxes

class DSECDataset(DSECDet):
    def __init__(self, root, split="train", sync="front", debug=False, split_config=None, transforms=None, 
                 use_image=True, use_events=False, event_repr_config=None, min_bbox_diag=30, min_bbox_side=20): 

        super().__init__(root=root, split=split, sync=sync, debug=False, split_config=split_config)
        
        self.debug_augmented = debug 
        self.transforms = transforms
        self.use_image = use_image
        self.use_events = use_events
        self.min_bbox_diag = min_bbox_diag
        self.min_bbox_side = min_bbox_side
        self.event_repr_config = event_repr_config

        if self.event_repr_config is not None and self.event_repr_config.name == "histogram":
            self.event_repr_func = EventHistogram(height=self.height, width=self.width)
        elif self.event_repr_config is not None and self.event_repr_config.name == "voxel_grid":
            self.event_repr_func = VoxelGrid(height=self.height, width=self.width, num_bins=self.event_repr_config.num_bins)
        else:
            self.event_repr_func = None
        
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

    def _resolve_global_index(self, idx):
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
        if self.event_repr_func is not None:
            return self.event_repr_func(events)
        return events

    def __getitem__(self, idx):
        # 1. インデックスの解決
        directory, image_index_pairs, track_mask, rel_idx = self._resolve_global_index(idx)
        
        idx0, idx1 = image_index_pairs[rel_idx]
        dir_name = directory.root.name

        output = {}
        
        # 2. データの取得
        if self.use_image:
            output['image'] = self.get_image(idx1, directory_name=dir_name)
            
        if self.use_events:
            raw_events = self.get_events(idx1, directory_name=dir_name)
            # 🌟 ここでは前処理(マスクや相対時間化)だけ行い、まだテンソル化しない！
            output['events'] = self.preprocess_events(raw_events) 
        
        # 3. トラックの取得と前処理
        tracks = self.get_tracks(idx1, mask=track_mask, directory_name=dir_name)
        
        if tracks is not None and len(tracks) > 0:
            mapped_ids, class_mask = map_classes(tracks['class_id'], self.class_remapping)
            size_mask = filter_small_bboxes(tracks['w'], tracks['h'], self.min_bbox_side, self.min_bbox_diag)
            
            valid_mask = class_mask & size_mask
            tracks = tracks[valid_mask]
            tracks['class_id'] = mapped_ids[valid_mask]
            
        output['tracks'] = tracks

        # 4. 🌟 データ拡張を先に適用する (生の辞書に対してFlipなどを実行)
        if hasattr(self, 'transforms') and self.transforms is not None:
            output = self.transforms(output)

        # 5. 🌟 データ拡張が終わった後に、イベント表現(Histogram等)に変換する
        if self.use_events and 'events' in output:
            output['events'] = self.generate_event_representation(output['events'])

        output['targets'] = format_targets_for_yolox(output.get('tracks'))

        # 6. デバッグ表示
        if self.debug_augmented:
            image = (255 * (output['image'].astype("float32") / 255) ** (1/2.2)).astype("uint8") if 'image' in output else np.zeros((self.height, self.width, 3), dtype=np.uint8)
            # デバッグ描画は生のイベント座標が必要なため、テンソル化の前に呼び出すか、
            # もしくはこのデバッグブロックを 4 と 5 の間に移動させるのがおすすめです。
            if 'events' in output and isinstance(output['events'], dict) and 'x' in output['events']:
                ev = output['events']
                image = render_events_on_image(image, x=ev['x'], y=ev['y'], p=ev['p'])
            output['debug'] = render_object_detections_on_image(image, output['tracks'])

        return output