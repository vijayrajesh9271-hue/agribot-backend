import os
import io
import json
import base64
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
import time
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

MODEL_DIR = os.path.join("models")
MODEL_FILES = {
    "cotton": "cottonbest.keras",
    "wheat":  "best_model.keras",      # wheat 
    "corn":   "cornbest.keras",
    "rice":   "rice_best_model.keras"  
}

# Load crop classifier
CROP_CLASSIFIER_PATH = os.path.join(MODEL_DIR, "crops_classifier.keras")
CROP_CLASSIFIER = tf.keras.models.load_model(CROP_CLASSIFIER_PATH, compile=False)

with open(os.path.join(MODEL_DIR, "class_indices.json"), "r") as f:
    CROP_CLASS_INDICES = json.load(f)
    # invert mapping 
    if all(not k.isdigit() for k in CROP_CLASS_INDICES.keys()):
        CROP_CLASS_INDICES = {int(v): k for k, v in CROP_CLASS_INDICES.items()}


#  class index files 
CLASS_INDEX_FILES = {
    "cotton": os.path.join(MODEL_DIR, "cotton_class_indices.json"),
    "wheat":  os.path.join(MODEL_DIR, "wheat_class_indices.json"),
    "corn":   os.path.join(MODEL_DIR, "corn_class_indices.json"),
    "rice":   os.path.join(MODEL_DIR, "rice_class_indices.json"),
}

# fallback labels 
FALLBACK_LABELS = {
    "cotton": {0:"bacterial_blight",1:"curl_virus",2:"fussarium_wilt",3:"healthy"},
    "wheat":  {0:"Brown rust",1:"Healthy",2:"Loose Smut",3:"Septoria",4:"Yellow rust"},
    "corn":   {0:"Blight",1:"Common_Rust",2:"Gray_Leaf_Spot",3:"Healthy"},
    "rice":   {0:"bacterial_leaf_blight",1:"brown_spot",2:"healthy",3:"leaf_blast",4:"leaf_scald",5:"narrow_brown_spot"}
}

# helpers

def predict_crop_from_classifier(b64_image: str):
    pil_img = decode_base64_to_pil(b64_image)
    tensor = preprocess_for_model(pil_img, target_size=(224,224), channels=3)

    start = time.perf_counter()
    preds = CROP_CLASSIFIER.predict(tensor, verbose=0)[0]
    end = time.perf_counter()
    print(f"⏱️ Crop classifier took {end - start:.2f} seconds")

    sorted_indices = np.argsort(preds)[::-1]
    top_idx = sorted_indices[0]
    second_idx = sorted_indices[1]
    top_conf = float(preds[top_idx])
    second_conf = float(preds[second_idx])
    gap = top_conf - second_conf

    crop_name = CROP_CLASS_INDICES.get(top_idx, str(top_idx)).lower()
    second_crop = CROP_CLASS_INDICES.get(second_idx, str(second_idx)).lower()

    return crop_name, top_conf, second_crop, second_conf, gap



def _full_model_path(name):
    fn = MODEL_FILES.get(name)
    if not fn:
        raise ValueError(f"No model configured for '{name}'")
    return os.path.join(MODEL_DIR, fn)

def _load_class_indices(name):
    p = CLASS_INDEX_FILES.get(name)
    if p and os.path.exists(p):
        try:
            with open(p, "r") as f:
                d = json.load(f)
            # If mapping is class_name -> index, invert
            if all(not k.isdigit() for k in d.keys()):
                inv = {int(v): k for k, v in d.items()}
                return inv
            # else assume dict of "0":"class_name"
            return {int(k): v for k, v in d.items()}
        except Exception:
            pass
    return FALLBACK_LABELS[name]

def model_target_size_and_channels(model):
    ish = model.input_shape
    if ish is None or len(ish) < 4:
        return (224,224), 3
    h = int(ish[1]) if ish[1] else 224
    w = int(ish[2]) if ish[2] else 224
    c = int(ish[3]) if ish[3] else 3
    return (h, w), c

def _strip_data_url_prefix(b64str: str) -> str:
    if not isinstance(b64str, str):
        raise ValueError("Expected base64 string")
    if b64str.startswith("data:"):
        return b64str.split(",", 1)[1]
    return b64str

def decode_base64_to_pil(b64str: str) -> Image.Image:
    b64_clean = _strip_data_url_prefix(b64str)
    img_bytes = base64.b64decode(b64_clean)
    img = Image.open(io.BytesIO(img_bytes))
    return img

def preprocess_for_model(pil_img: Image.Image, target_size=(224,224), channels=3):
    # Force channel mode
    if channels == 3:
        img = pil_img.convert("RGB")
    elif channels == 1:
        img = pil_img.convert("L")
    else:
        img = pil_img.convert("RGB")

    # Resize to exact size
    img = img.resize(target_size, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)

    # normalize channel dimension
    if channels == 1:
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=-1)
        elif arr.shape[-1] == 4:
            arr = arr[..., 0:1]
    else:
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]

    if arr.shape[-1] != channels:
        raise RuntimeError(f"Preprocessed channel mismatch: got {arr.shape[-1]}, expected {channels}")

    arr = arr / 255.0
    return np.expand_dims(arr, axis=0).astype(np.float32)

# load all models
MODELS = {}
META = {}

for name in ("cotton","wheat","corn","rice"):
    path = _full_model_path(name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found for {name}: {path}")
    print(f"[vision] loading {name} model from {path}")
    m = tf.keras.models.load_model(path, compile=False)
    MODELS[name] = m
    target_size, channels = model_target_size_and_channels(m)
    META[name] = {
        "target_size": target_size,
        "channels": channels,
        "labels": _load_class_indices(name)
    }
    print(f"[vision] {name} input_shape: {m.input_shape}, target_size: {target_size}, channels: {channels}")

#  prediction 
def _predict_generic(name: str, b64_image: str):
    model = MODELS[name]
    meta = META[name]
    labels = meta["labels"]
    target_size = meta["target_size"]
    channels = meta["channels"]

    pil_img = decode_base64_to_pil(b64_image)
    try:
        print(f"[vision] incoming image mode={pil_img.mode}, size={pil_img.size} for {name}")
    except Exception:
        pass

    tensor = preprocess_for_model(pil_img, target_size=target_size, channels=channels)
    print(f"[vision] tensor shape for {name}: {tensor.shape}")

    start = time.perf_counter()
    preds = model.predict(tensor, verbose=0)[0]
    end = time.perf_counter()
    print(f"⏱️ Disease model ({name}) took {end - start:.2f} seconds")

    sorted_indices = np.argsort(preds)[::-1]
    top_idx = sorted_indices[0]
    second_idx = sorted_indices[1]

    conf = float(preds[top_idx])
    second_conf = float(preds[second_idx])
    gap = conf - second_conf
    label = labels.get(top_idx, str(top_idx))

    # Thresholds
    min_conf = 0.60
    min_gap = 0.10

    if conf < min_conf or gap < min_gap:
        status = "Uncertain"
        description = (
            f"The model is not confident in its prediction.\n"
            f"It suggests **{label}**, but the confidence is only {conf:.2f} and the gap between top predictions is {gap:.2f}.\n"
            f"Please confirm the crop type or upload a clearer image."
        )
    else:
        status = "Healthy" if str(label).lower() == "healthy" else "Infected"
        description = (
            f"The {name} leaf looks healthy (confidence {conf:.2f})."
            if status == "Healthy"
            else f"The {name} leaf shows signs of {label} (confidence {conf:.2f})."
        )

    return {
        "crop": name.capitalize(),
        "status": status,
        "label": label,
        "confidence": conf,
        "description": description,
        "uncertain": status == "Uncertain"
    }


# Exposed functions
def predict_cotton_from_base64(b64): return _predict_generic("cotton", b64)
def predict_wheat_from_base64(b64):  return _predict_generic("wheat", b64)
def predict_corn_from_base64(b64):   return _predict_generic("corn", b64)
def predict_rice_from_base64(b64):   return _predict_generic("rice", b64)

def predict_crop_from_base64(b64_image: str, crop_name: str):
    name = (crop_name or "").strip().lower()
    if name == "maize": name = "corn"
    if name not in MODELS:
        raise ValueError(f"No model loaded for {name}")
    return _predict_generic(name, b64_image)


#  Auto-detect & analyze helpers



_ADVICE_HEURISTICS = {
    "healthy": "Leaf appears healthy — continue regular monitoring and ensure adequate irrigation and nutrients.",
    "blight": "Likely blight. Remove heavily infected leaves, avoid overhead watering, and consider a targeted fungicide as per local guidance.",
    "rust": "Likely rust. Increase airflow, remove nearby infected debris, and consider rust-specific fungicide. Rotate crops next season.",
    "smut": "Possible smut infection. Remove infected plants and avoid replanting in the same soil without treatment.",
    "blast": "Possible blast disease. Improve drainage, reduce nitrogen over-application, and consider protective fungicide.",
    "scald": "Possible scald. Avoid prolonged leaf wetness; remove infected material and monitor spread.",
    "spot": "Leaf spots detected. Remove infected tissue, reduce wet foliage contact, and consider a bactericide/fungicide if spreading.",
    "curl": "Curl virus detected. Remove infected plants immediately to prevent spread. Use virus-free seeds and control aphid vectors.",  # FIXED
    "fusarium": "Fusarium wilt detected. Improve soil drainage, avoid overwatering, use resistant varieties, and practice crop rotation.",  # FIXED
    "bacterial": "Bacterial infection detected. Remove infected parts, avoid overhead irrigation, apply copper-based bactericide if needed.",
    "default": "Diagnosis suggests disease — isolate affected plants, take photos, and consult local extension services for treatment recommendations."
}

def _get_advice_for_label(crop_name: str, label: str):
    """
    Convert a predicted label string into an advice string using heuristics.
    Matches keywords (case-insensitive) within the label.
    """
    lab = (label or "").lower()
    if "healthy" in lab or "normal" in lab:
        return _ADVICE_HEURISTICS["healthy"]

    # check common keywords
    for key in ("blight", "rust", "smut", "blast", "scald", "spot", "curl", "fusarium", "bacterial"):  # ADDED curl, fusarium, bacterial
        if key in lab:
            return _ADVICE_HEURISTICS.get(key, _ADVICE_HEURISTICS["default"])

    # fallback per-crop mapping examples (optional customized messages)
    if crop_name == "rice":
        if "bacterial" in lab or "brown" in lab:
            return _ADVICE_HEURISTICS.get("spot", _ADVICE_HEURISTICS["default"])
    if crop_name == "corn":
        if "gray" in lab or "spot" in lab:
            return _ADVICE_HEURISTICS.get("spot", _ADVICE_HEURISTICS["default"])

    return _ADVICE_HEURISTICS["default"]

# ---------------- High-level auto analyze ----------------
def analyze_image_auto(b64_image: str, require_threshold: float = 0.60):
    start = time.perf_counter()
    crop_name, crop_conf, second_crop, second_conf, gap = predict_crop_from_classifier(b64_image)
    print(f"[vision] Crop classifier top: {crop_name} ({crop_conf:.2f}), second: {second_crop} ({second_conf:.2f}), gap={gap:.2f}")

    #  Case 1: Very low confidence (< 60%) → Reject
    if crop_conf < 0.60:
        return {
            "mode": "rejected",
            "status": "Rejected",
            "label": "Unknown crop",
            "description": (
                f"It appears you have entered a wrong image, I can only detect and give advice on crop images "
                f"Please upload a relevant image."
            ),
            "advice": "Try to capture a clear leaf image from a real crop."
        }

    # ⚠️ Case 2: Moderate confidence (60–79%) OR high confidence but small gap (< 0.10) → Ask user to confirm crop
    if crop_conf < 0.80 or gap < 0.10:
        return {
            "mode": "auto_uncertain_dual",
            "status": "Uncertain",
            "crop_1": crop_name,
            "crop_2": second_crop,
            "description": (
                f"I have detected these 2 crops: **{crop_name}** and **{second_crop}**, kindly confirm your crop "
                f"so I can proceed with a more accurate disease diagnosis."
                ),
            "advice": "Please reply with the name of your crop (e.g., 'My crop is wheat') so I can continue."
        }


    # ✅ Case 3: High confidence (≥ 80%) and clear gap → Proceed
    result = _predict_generic(crop_name, b64_image)
    advice = _get_advice_for_label(crop_name, result["label"])
    result["advice"] = advice
    result["mode"] = "auto"
    result["crop_confidence"] = crop_conf
    result["uncertain"] = result.get("uncertain", False)

    end = time.perf_counter()
    print(f"⏱️ Full vision pipeline took {end - start:.2f} seconds")
    return result

######################

#####################


# import os

# # CRITICAL: must be set before `import tensorflow` — cannot be changed afterward.
# # Your Quadro P1000 (compute capability 6.1 / sm_61) is NOT included in this
# # TensorFlow build's compiled GPU kernels (only sm_60/70/80/89/90 are shipped),
# # so TF falls back to JIT-compiled PTX on your card, which is a known source
# # of instability/segfaults on older Pascal GPUs with recent TF + cuDNN 9 builds.
# # These are small, single-image classifiers — CPU inference is plenty fast here,
# # and disabling GPU avoids this whole class of crash entirely.
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# import io
# import json
# import base64
# from pathlib import Path
# import numpy as np
# from PIL import Image
# import tensorflow as tf
# import time

# MODEL_DIR = os.path.join("models")
# MODEL_FILES = {
#     "cotton": "cottonbest.keras",
#     "wheat":  "best_model.keras",      # wheat 
#     "corn":   "cornbest.keras",
#     "rice":   "rice_best_model.keras"  
# }

# # Load crop classifier
# CROP_CLASSIFIER_PATH = os.path.join(MODEL_DIR, "crops_classifier.keras")
# CROP_CLASSIFIER = tf.keras.models.load_model(CROP_CLASSIFIER_PATH, compile=False)

# with open(os.path.join(MODEL_DIR, "class_indices.json"), "r") as f:
#     CROP_CLASS_INDICES = json.load(f)
#     # invert mapping 
#     if all(not k.isdigit() for k in CROP_CLASS_INDICES.keys()):
#         CROP_CLASS_INDICES = {int(v): k for k, v in CROP_CLASS_INDICES.items()}


# #  class index files 
# CLASS_INDEX_FILES = {
#     "cotton": os.path.join(MODEL_DIR, "cotton_class_indices.json"),
#     "wheat":  os.path.join(MODEL_DIR, "wheat_class_indices.json"),
#     "corn":   os.path.join(MODEL_DIR, "corn_class_indices.json"),
#     "rice":   os.path.join(MODEL_DIR, "rice_class_indices.json"),
# }

# # fallback labels 
# FALLBACK_LABELS = {
#     "cotton": {0:"bacterial_blight",1:"curl_virus",2:"fussarium_wilt",3:"healthy"},
#     "wheat":  {0:"Brown rust",1:"Healthy",2:"Loose Smut",3:"Septoria",4:"Yellow rust"},
#     "corn":   {0:"Blight",1:"Common_Rust",2:"Gray_Leaf_Spot",3:"Healthy"},
#     "rice":   {0:"bacterial_leaf_blight",1:"brown_spot",2:"healthy",3:"leaf_blast",4:"leaf_scald",5:"narrow_brown_spot"}
# }

# # helpers

# def predict_crop_from_classifier(b64_image: str):
#     pil_img = decode_base64_to_pil(b64_image)
#     tensor = preprocess_for_model(pil_img, target_size=(224,224), channels=3)

#     start = time.perf_counter()
#     preds = CROP_CLASSIFIER.predict(tensor, verbose=0)[0]
#     end = time.perf_counter()
#     print(f"⏱️ Crop classifier took {end - start:.2f} seconds")

#     sorted_indices = np.argsort(preds)[::-1]
#     top_idx = sorted_indices[0]
#     second_idx = sorted_indices[1]
#     top_conf = float(preds[top_idx])
#     second_conf = float(preds[second_idx])
#     gap = top_conf - second_conf

#     crop_name = CROP_CLASS_INDICES.get(top_idx, str(top_idx)).lower()
#     second_crop = CROP_CLASS_INDICES.get(second_idx, str(second_idx)).lower()

#     return crop_name, top_conf, second_crop, second_conf, gap



# def _full_model_path(name):
#     fn = MODEL_FILES.get(name)
#     if not fn:
#         raise ValueError(f"No model configured for '{name}'")
#     return os.path.join(MODEL_DIR, fn)

# def _load_class_indices(name):
#     p = CLASS_INDEX_FILES.get(name)
#     if p and os.path.exists(p):
#         try:
#             with open(p, "r") as f:
#                 d = json.load(f)
#             # If mapping is class_name -> index, invert
#             if all(not k.isdigit() for k in d.keys()):
#                 inv = {int(v): k for k, v in d.items()}
#                 return inv
#             # else assume dict of "0":"class_name"
#             return {int(k): v for k, v in d.items()}
#         except Exception:
#             pass
#     return FALLBACK_LABELS[name]

# def model_target_size_and_channels(model):
#     ish = model.input_shape
#     if ish is None or len(ish) < 4:
#         return (224,224), 3
#     h = int(ish[1]) if ish[1] else 224
#     w = int(ish[2]) if ish[2] else 224
#     c = int(ish[3]) if ish[3] else 3
#     return (h, w), c

# def _strip_data_url_prefix(b64str: str) -> str:
#     if not isinstance(b64str, str):
#         raise ValueError("Expected base64 string")
#     if b64str.startswith("data:"):
#         return b64str.split(",", 1)[1]
#     return b64str

# def decode_base64_to_pil(b64str: str) -> Image.Image:
#     b64_clean = _strip_data_url_prefix(b64str)
#     img_bytes = base64.b64decode(b64_clean)
#     img = Image.open(io.BytesIO(img_bytes))
#     return img

# def preprocess_for_model(pil_img: Image.Image, target_size=(224,224), channels=3):
#     # Force channel mode
#     if channels == 3:
#         img = pil_img.convert("RGB")
#     elif channels == 1:
#         img = pil_img.convert("L")
#     else:
#         img = pil_img.convert("RGB")

#     # Resize to exact size
#     img = img.resize(target_size, Image.BILINEAR)
#     arr = np.array(img, dtype=np.float32)

#     # normalize channel dimension
#     if channels == 1:
#         if arr.ndim == 2:
#             arr = np.expand_dims(arr, axis=-1)
#         elif arr.shape[-1] == 4:
#             arr = arr[..., 0:1]
#     else:
#         if arr.ndim == 2:
#             arr = np.stack([arr]*3, axis=-1)
#         if arr.shape[-1] == 4:
#             arr = arr[..., :3]

#     if arr.shape[-1] != channels:
#         raise RuntimeError(f"Preprocessed channel mismatch: got {arr.shape[-1]}, expected {channels}")

#     arr = arr / 255.0
#     return np.expand_dims(arr, axis=0).astype(np.float32)

# # load all models
# MODELS = {}
# META = {}

# for name in ("cotton","wheat","corn","rice"):
#     path = _full_model_path(name)
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Model file not found for {name}: {path}")
#     print(f"[vision] loading {name} model from {path}")
#     m = tf.keras.models.load_model(path, compile=False)
#     MODELS[name] = m
#     target_size, channels = model_target_size_and_channels(m)
#     META[name] = {
#         "target_size": target_size,
#         "channels": channels,
#         "labels": _load_class_indices(name)
#     }
#     print(f"[vision] {name} input_shape: {m.input_shape}, target_size: {target_size}, channels: {channels}")

# #  prediction 
# def _predict_generic(name: str, b64_image: str):
#     model = MODELS[name]
#     meta = META[name]
#     labels = meta["labels"]
#     target_size = meta["target_size"]
#     channels = meta["channels"]

#     pil_img = decode_base64_to_pil(b64_image)
#     try:
#         print(f"[vision] incoming image mode={pil_img.mode}, size={pil_img.size} for {name}")
#     except Exception:
#         pass

#     tensor = preprocess_for_model(pil_img, target_size=target_size, channels=channels)
#     print(f"[vision] tensor shape for {name}: {tensor.shape}")

#     start = time.perf_counter()
#     preds = model.predict(tensor, verbose=0)[0]
#     end = time.perf_counter()
#     print(f"⏱️ Disease model ({name}) took {end - start:.2f} seconds")

#     sorted_indices = np.argsort(preds)[::-1]
#     top_idx = sorted_indices[0]
#     second_idx = sorted_indices[1]

#     conf = float(preds[top_idx])
#     second_conf = float(preds[second_idx])
#     gap = conf - second_conf
#     label = labels.get(top_idx, str(top_idx))

#     # Thresholds
#     min_conf = 0.60
#     min_gap = 0.10

#     if conf < min_conf or gap < min_gap:
#         status = "Uncertain"
#         description = (
#             f"The model is not confident in its prediction.\n"
#             f"It suggests **{label}**, but the confidence is only {conf:.2f} and the gap between top predictions is {gap:.2f}.\n"
#             f"Please confirm the crop type or upload a clearer image."
#         )
#     else:
#         status = "Healthy" if str(label).lower() == "healthy" else "Infected"
#         description = (
#             f"The {name} leaf looks healthy (confidence {conf:.2f})."
#             if status == "Healthy"
#             else f"The {name} leaf shows signs of {label} (confidence {conf:.2f})."
#         )

#     return {
#         "crop": name.capitalize(),
#         "status": status,
#         "label": label,
#         "confidence": conf,
#         "description": description,
#         "uncertain": status == "Uncertain"
#     }


# # Exposed functions
# def predict_cotton_from_base64(b64): return _predict_generic("cotton", b64)
# def predict_wheat_from_base64(b64):  return _predict_generic("wheat", b64)
# def predict_corn_from_base64(b64):   return _predict_generic("corn", b64)
# def predict_rice_from_base64(b64):   return _predict_generic("rice", b64)

# def predict_crop_from_base64(b64_image: str, crop_name: str):
#     name = (crop_name or "").strip().lower()
#     if name == "maize": name = "corn"
#     if name not in MODELS:
#         raise ValueError(f"No model loaded for {name}")
#     return _predict_generic(name, b64_image)


# #  Auto-detect & analyze helpers



# _ADVICE_HEURISTICS = {
#     "healthy": "Leaf appears healthy — continue regular monitoring and ensure adequate irrigation and nutrients.",
#     "blight": "Likely blight. Remove heavily infected leaves, avoid overhead watering, and consider a targeted fungicide as per local guidance.",
#     "rust": "Likely rust. Increase airflow, remove nearby infected debris, and consider rust-specific fungicide. Rotate crops next season.",
#     "smut": "Possible smut infection. Remove infected plants and avoid replanting in the same soil without treatment.",
#     "blast": "Possible blast disease. Improve drainage, reduce nitrogen over-application, and consider protective fungicide.",
#     "scald": "Possible scald. Avoid prolonged leaf wetness; remove infected material and monitor spread.",
#     "spot": "Leaf spots detected. Remove infected tissue, reduce wet foliage contact, and consider a bactericide/fungicide if spreading.",
#     "curl": "Curl virus detected. Remove infected plants immediately to prevent spread. Use virus-free seeds and control aphid vectors.",  # FIXED
#     "fusarium": "Fusarium wilt detected. Improve soil drainage, avoid overwatering, use resistant varieties, and practice crop rotation.",  # FIXED
#     "bacterial": "Bacterial infection detected. Remove infected parts, avoid overhead irrigation, apply copper-based bactericide if needed.",
#     "default": "Diagnosis suggests disease — isolate affected plants, take photos, and consult local extension services for treatment recommendations."
# }

# def _get_advice_for_label(crop_name: str, label: str):
#     """
#     Convert a predicted label string into an advice string using heuristics.
#     Matches keywords (case-insensitive) within the label.
#     """
#     lab = (label or "").lower()
#     if "healthy" in lab or "normal" in lab:
#         return _ADVICE_HEURISTICS["healthy"]

#     # check common keywords
#     for key in ("blight", "rust", "smut", "blast", "scald", "spot", "curl", "fusarium", "bacterial"):  # ADDED curl, fusarium, bacterial
#         if key in lab:
#             return _ADVICE_HEURISTICS.get(key, _ADVICE_HEURISTICS["default"])

#     # fallback per-crop mapping examples (optional customized messages)
#     if crop_name == "rice":
#         if "bacterial" in lab or "brown" in lab:
#             return _ADVICE_HEURISTICS.get("spot", _ADVICE_HEURISTICS["default"])
#     if crop_name == "corn":
#         if "gray" in lab or "spot" in lab:
#             return _ADVICE_HEURISTICS.get("spot", _ADVICE_HEURISTICS["default"])

#     return _ADVICE_HEURISTICS["default"]

# # ---------------- High-level auto analyze ----------------
# def analyze_image_auto(b64_image: str, require_threshold: float = 0.60):
#     start = time.perf_counter()
#     crop_name, crop_conf, second_crop, second_conf, gap = predict_crop_from_classifier(b64_image)
#     print(f"[vision] Crop classifier top: {crop_name} ({crop_conf:.2f}), second: {second_crop} ({second_conf:.2f}), gap={gap:.2f}")

#     #  Case 1: Very low confidence (< 60%) → Reject
#     if crop_conf < 0.60:
#         return {
#             "mode": "rejected",
#             "status": "Rejected",
#             "label": "Unknown crop",
#             "description": (
#                 f"It appears you have entered a wrong image, I can only detect and give advice on crop images "
#                 f"Please upload a relevant image."
#             ),
#             "advice": "Try to capture a clear leaf image from a real crop."
#         }

#     # ⚠️ Case 2: Moderate confidence (60–79%) OR high confidence but small gap (< 0.10) → Ask user to confirm crop
#     if crop_conf < 0.80 or gap < 0.10:
#         return {
#             "mode": "auto_uncertain_dual",
#             "status": "Uncertain",
#             "crop_1": crop_name,
#             "crop_2": second_crop,
#             "description": (
#                 f"I have detected these 2 crops: **{crop_name}** and **{second_crop}**, kindly confirm your crop "
#                 f"so I can proceed with a more accurate disease diagnosis."
#                 ),
#             "advice": "Please reply with the name of your crop (e.g., 'My crop is wheat') so I can continue."
#         }


#     # ✅ Case 3: High confidence (≥ 80%) and clear gap → Proceed
#     result = _predict_generic(crop_name, b64_image)
#     advice = _get_advice_for_label(crop_name, result["label"])
#     result["advice"] = advice
#     result["mode"] = "auto"
#     result["crop_confidence"] = crop_conf
#     result["uncertain"] = result.get("uncertain", False)

#     end = time.perf_counter()
#     print(f"⏱️ Full vision pipeline took {end - start:.2f} seconds")
#     return result