import torch

def smart_load_state_dict(model, weights_path):
    """
    YOLOXの学習済み重みを、自作のDetectorモデルに柔軟にロードする。
    """
    checkpoint = torch.load(weights_path, map_location="cpu")
    # "model" キーがある場合は中身を取り出し、なければ全体を state_dict とする
    src_dict = checkpoint.get("model", checkpoint)
    target_dict = model.state_dict()
    
    new_dict = {}
    shape_mismatch = []

    for k, v in src_dict.items():
        clean_k = k
        
        # 1. Backboneの接頭辞正規化 (backbone.backbone -> backbone / dark -> backbone.dark)
        if clean_k.startswith("backbone.backbone."):
            clean_k = clean_k.replace("backbone.backbone.", "backbone.", 1)
        elif any(clean_k.startswith(p) for p in ["dark", "stem"]):
            clean_k = "backbone." + clean_k
            
        # 2. Headの接頭辞正規化 (head -> yolox_head)
        if clean_k.startswith("head."):
            clean_k = clean_k.replace("head.", "yolox_head.", 1)

        # 3. マッチング確認
        if clean_k in target_dict:
            if v.shape == target_dict[clean_k].shape:
                new_dict[clean_k] = v
            else:
                shape_mismatch.append(clean_k)

    # ロード実行 (strict=False にすることで、追加したLSTM層などは初期値のまま維持される)
    missing, unexpected = model.load_state_dict(new_dict, strict=False)
    
    # ロード結果のサマリを表示
    print("\n" + "="*40)
    print(f"✅ ロード成功: {len(new_dict)} 個")
    
    # CNN/Headに関連する未ロード（名前不一致の可能性）を特定
    essential_missing = [m for m in missing if any(x in m for x in ["dark", "stem", "yolox_head"])]
    if essential_missing:
        print(f"⚠️ 予期せぬ未ロード (CNN系): {len(essential_missing)} 個")
    
    if shape_mismatch:
        print(f"❌ サイズ不一致 (Shape Mismatch): {len(shape_mismatch)} 個")
        

    return model

def count_parameters(model, model_name="Model"):
    """
    モデルの総パラメータ数、学習可能パラメータ数、固定パラメータ数をカウントして表示する。
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print("\n" + "="*50)
    print(f"📊 {model_name} Parameter Statistics")
    print("-"*50)
    print(f"Total Parameters     : {total_params:,}")
    print(f"Trainable Parameters : {trainable_params:,}")
    print(f"Frozen Parameters    : {frozen_params:,}")
    
    # 概算（M: Million）の表示
    print(f"Model Size (approx)  : {total_params / 1e6:.2f} M")
    print("="*50 + "\n")
    
    return total_params, trainable_params