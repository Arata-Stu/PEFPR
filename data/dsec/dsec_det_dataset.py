import numpy as np

from .dsec_det.dataset import DSECDet, render_events_on_image, render_object_detections_on_image
from ..utils.augmentor  import init_transforms

class DSECDataset(DSECDet):
    """
    DSECDetを継承し、データ拡張(Transform)機能を追加したクラス
    """
    def __init__(self, root, split="train", sync="front", debug=False, split_config=None, transforms=None):

        super().__init__(root=root, split=split, sync=sync, debug=False, split_config=split_config)
        
        self.debug_augmented = debug 
        self.transforms = transforms
        
        if self.transforms is not None:
            init_transforms(self.transforms, self.height, self.width)

        self.dataset_name = "DSEC-Det"
        self.time_window = 1000000
    
    def preprocess_events(self, events):
        mask = events['y'] < self.height
        events = {k: v[mask] for k, v in events.items()}
        
        if len(events['t']) > 0:
            events['t'] = self.time_window + events['t'] - events['t'][-1]
            
        events['p'] = events['p'].reshape((-1, 1)).astype(np.int8)
        
        events['x'] = events['x'].astype(np.float32)
        events['y'] = events['y'].astype(np.float32)
        
        return events

    def __getitem__(self, item):
        output = {}
        output['image'] = self.get_image(item)
        raw_events = self.get_events(item)
        output['events'] = self.preprocess_events(raw_events) 
        output['tracks'] = self.get_tracks(item)

        if hasattr(self, 'transforms') and self.transforms is not None:
            output = self.transforms(output)

        if self.debug_augmented:
            events = output['events']
            image = (255 * (output['image'].astype("float32") / 255) ** (1/2.2)).astype("uint8")
            
            # イベントとBBoxの重畳
            output['debug'] = render_events_on_image(image, x=events['x'], y=events['y'], p=events['p'])
            output['debug'] = render_object_detections_on_image(output['debug'], output['tracks'])

        return output