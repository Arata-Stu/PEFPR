import torch
import numpy as np

def custom_collate_fn(batch):
    """
    DSECデータセットの出力をモデル(YoloXPEPRDetector)の入力形式に変換し、バッチ化する関数
    """
    images = []
    targets_list = []
    events_list = []

    for item in batch:
        # 1. 画像の処理
        img = item.get('image')
        if img is not None:
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img).float()
                # もし HWC (Height, Width, Channel) なら CHW に変換
                if img.ndim == 3 and img.shape[-1] == 3:
                    img = img.permute(2, 0, 1)
            images.append(img)
        
        # 2. Targets の処理
        targets_list.append(item.get('targets', torch.zeros((0, 5), dtype=torch.float32)))

        # 3. Events の処理
        events_list.append(item.get('events', None))

    # --- バッチ化 (スタック) ---
    out = {}

    # 画像のバッチ化 (画像が存在する場合のみ)
    if len(images) > 0:
        out['x'] = torch.stack(images)

    # BBoxのゼロパディングとバッチ化
    max_targets = max((t.shape[0] for t in targets_list), default=0)
    if max_targets > 0:
        padded_targets = torch.zeros((len(batch), max_targets, 5), dtype=torch.float32)
        for i, t in enumerate(targets_list):
            if t.shape[0] > 0:
                padded_targets[i, :t.shape[0], :] = t
    else:
        padded_targets = torch.zeros((len(batch), 0, 5), dtype=torch.float32)

    out['targets'] = padded_targets
    
    # イベントデータのバッチ化
    if len(events_list) > 0 and events_list[0] is not None:
        if isinstance(events_list[0], torch.Tensor):
            # HistogramやVoxelGridなど、Tensor化されている場合は4Dテンソル (B, C, H, W) にスタック
            out['event_data'] = torch.stack(events_list)
        else:
            # rawイベント(辞書)のままの場合はリストとして渡す
            out['event_data'] = events_list

    return out