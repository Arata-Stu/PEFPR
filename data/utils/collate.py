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
        # 1. 画像の処理 (HWC -> CHW に並べ替え、0~255を維持するか正規化するかはモデル実装に依存)
        img = item['image']
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).float()
            # もし HWC (Height, Width, Channel) なら CHW に変換
            if img.ndim == 3 and img.shape[-1] == 3:
                img = img.permute(2, 0, 1)
        images.append(img)
        
        # 2. Tracks (BBox) の処理
        tracks = item.get('tracks', None)
        if tracks is not None and len(tracks) > 0:
            # 構造化配列 (numpy.void) から各フィールドを取り出して標準のFloat配列にする
            # 一般的なYOLOXは [class_id, x, y, w, h] または [class_id, cx, cy, w, h] を期待します
            cls_id = tracks['class_id'].astype(np.float32)
            x = tracks['x'].astype(np.float32)
            y = tracks['y'].astype(np.float32)
            w = tracks['w'].astype(np.float32)
            h = tracks['h'].astype(np.float32)
            
            # (N, 5) のテンソルに結合
            target_tensor = torch.tensor(np.column_stack([cls_id, x, y, w, h]), dtype=torch.float32)
        else:
            # BBoxがない場合は空のテンソル
            target_tensor = torch.zeros((0, 5), dtype=torch.float32)
            
        targets_list.append(target_tensor)

        # 3. Events の処理
        events_list.append(item.get('events', None))

    # --- バッチ化 (スタック) ---
    # 画像はすべて同じサイズ(Crop等適用済み)である前提
    x_batch = torch.stack(images)

    # BBoxは画像によって数が違うため、最大の数に合わせてゼロパディングする
    # YOLOXは一般的に (Batch, Max_Objects, 5) のテンソルを受け取ります
    max_targets = max((t.shape[0] for t in targets_list), default=0)
    if max_targets > 0:
        padded_targets = torch.zeros((len(batch), max_targets, 5), dtype=torch.float32)
        for i, t in enumerate(targets_list):
            if t.shape[0] > 0:
                padded_targets[i, :t.shape[0], :] = t
    else:
        padded_targets = torch.zeros((len(batch), 0, 5), dtype=torch.float32)

    # モデルの引数名 (x, targets, event_data) に合わせてリネーム
    out = {
        'x': x_batch,
        'targets': padded_targets
    }
    
    # イベントデータも存在すれば追加 (現状はリスト形式)
    if events_list[0] is not None:
        out['event_data'] = events_list

    return out