import sys
sys.path.append('../') 

from pathlib import Path
import numpy as np
import cv2
import argparse
import hdf5plugin

from data.dsec.dsec_det_dataset import DSECDataset, render_object_detections_on_image, render_events_on_image
from data.utils.augmentor import Augmentations, init_transforms
# =========================================================================

def generate_image_panel(images_dict, panel_shape=(1, 3), padding=30):
    """
    複数の画像を並べて1枚のパネル（キャンバス）を作成する
    images_dict: { 'seq_name': image_array, ... }
    """
    seq_names = sorted(list(images_dict.keys()))
    # 描画対象の画像リストを取得
    images = [images_dict[name] for name in seq_names]
    
    if not images:
        return None

    h, w = images[0].shape[:2]
    rows, cols = panel_shape
    
    # パネル全体のサイズを計算
    panel_h = h * rows + padding * (rows - 1)
    panel_w = w * cols + padding * (cols - 1)
    
    panel = np.full((panel_h, panel_w, 3), 255, dtype=np.uint8)

    for i, img in enumerate(images):
        if i >= rows * cols: break
        r, c = divmod(i, cols)
        y = r * (h + padding)
        x = c * (w + padding)
        panel[y:y+h, x:x+w] = img
        
        # シーケンス名を画像上に描画（任意）
        cv2.putText(panel, seq_names[i], (x + 5, y + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return panel

# Transformのパラメータを渡すための簡易クラス
class DummyArgs:
    aug_p_flip = 0.5   # 50%の確率で左右反転
    aug_zoom = 1.2     # 最大1.2倍までズーム
    aug_trans = 0.1    # 10%の範囲でランダム平行移動

def main():
    parser = argparse.ArgumentParser(description="DSEC Dataset Visualizer")
    parser.add_argument("--dsec_root", type=Path, required=True, help="Path to DSEC root")
    parser.add_argument("--output_path", type=Path, help="Path to save images")
    parser.add_argument("--split", type=str, default="train", choices=['train', 'test', 'val'])
    parser.add_argument("--sync", type=str, default="front", choices=['front', 'back'])
    parser.add_argument("--transform", action="store_true", help="Disable augmentations")
    args = parser.parse_args()

    # Datasetの初期化 (Transform後の状態を描画するため、ここでは debug=False にする)
    dataset = DSECDataset(root=args.dsec_root, split=args.split, sync=args.sync, debug=False)

    # ---------------------------------------------------------
    # Transformの初期化設定
    # ---------------------------------------------------------
    aug_args = DummyArgs()
    augmentations = Augmentations(aug_args)
    # train時はデータ拡張あり、test/val時はクロップのみなど使い分けが可能
    transform_pipeline = augmentations.transform_training if args.transform else augmentations.transform_testing
    
    # 画像サイズをTransformクラスに教える
    init_transforms(transform_pipeline, dataset.height, dataset.width)
    print("Transform pipeline initialized.")
    # ---------------------------------------------------------

    # 可視化したいシーケンス名のリスト
    available_seqs = list(dataset.directories.keys())
    print(f"Available sequences: {available_seqs}")
    
    # ここでは例として最初の3つのシーケンスを表示対象にする
    target_seqs = available_seqs[:3]
    print(f"Visualizing: {target_seqs}")

    max_len = max([len(dataset.img_idx_track_idxs[s]) - 1 for s in target_seqs])
    start_idx = 1 if args.sync == "front" else 0

    for i in range(start_idx, max_len):
        display_images = {}
        
        for seq_name in target_seqs:
            try:
                # 1. データの取得
                data = dataset.__getitem__(i) 
                
                # 2. Transform (データ拡張) の適用
                data = transform_pipeline(data)

                # 3. 拡張後のデータを使って描画処理
                events = data['events']
                tracks = data['tracks']
                
                # ★ Numbaエラー回避のため、xとyを四捨五入して整数(int32)にキャストする
                events['x'] = np.round(events['x']).astype(np.int32)
                events['y'] = np.round(events['y']).astype(np.int32)
                
                # 画像のガンマ補正
                image = (255 * (data['image'].astype("float32") / 255) ** (1/2.2)).astype("uint8")
                
                # イベントの描画
                debug_img = render_events_on_image(image, x=events['x'], y=events['y'], p=events['p'])
                # バウンディングボックスの描画
                debug_img = render_object_detections_on_image(debug_img, tracks)

                display_images[seq_name] = debug_img

            except (ValueError, IndexError):
                # データが終了しているシーケンスは黒画像で埋める
                display_images[seq_name] = np.zeros((dataset.height, dataset.width, 3), dtype=np.uint8)

        # パネル作成
        panel = generate_image_panel(display_images, panel_shape=(1, len(target_seqs)))

        if panel is not None:
            cv2.imshow("DSEC Visualization (Events + BBox)", panel)
            
            # 保存処理
            if args.output_path:
                args.output_path.mkdir(parents=True, exist_ok=True)
                save_file = args.output_path / f"frame_{i:06d}.png"
                cv2.imwrite(str(save_file), panel)

            # 'q' キーで終了
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()