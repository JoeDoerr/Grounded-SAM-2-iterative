import io
import json
from typing import Optional

import cv2
import torch
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from utils.track_utils import sample_points_from_masks

from sam2.sam2_video_predictor import load_single_image

import zmq


def compute_area(box):
    x0, y0, x1, y1 = box
    return max(0, x1 - x0) * max(0, y1 - y0)


def intersection_area(boxA, boxB):
    x0 = max(boxA[0], boxB[0])
    y0 = max(boxA[1], boxB[1])
    x1 = min(boxA[2], boxB[2])
    y1 = min(boxA[3], boxB[3])
    return max(0, x1 - x0) * max(0, y1 - y0)


def filter_boxes(boxes, threshold=0.9, max_inside=3):
    keep = []
    for i, A in enumerate(boxes):
        count = 0
        area_A = compute_area(A)
        for j, B in enumerate(boxes):
            if i == j:
                continue
            area_B = compute_area(B)
            inter = intersection_area(A, B)
            if inter / area_B >= threshold:
                count += 1
        if count <= max_inside:
            keep.append(A)
    return np.array(keep) if keep else np.zeros((0, 4), dtype=np.float32)


def _default_server_params():
    return {
        "box_threshold": 0.2,
        "text_threshold": 0.25,
        "min_best_score": 0.35,
        "box_overlap_filter_threshold": 0.9,
        "max_inside": 3,
        "max_frames_in_state": 2,
        "nms_iou_threshold": 0.7,
    }


def merge_gsam_params(base: dict, override: Optional[dict]) -> dict:
    out = dict(base)
    if override:
        out.update(override)
    return out


def first_step(
    processor,
    grounding_model,
    video_predictor,
    image_predictor,
    device,
    text,
    raw_image_inp,
    image_inp,
    video_height,
    video_width,
    p: dict,
):
    """
    GroundingDINO + SAM2 image + video init for one target (best-scoring box only).
    p keys: box_threshold, text_threshold, min_best_score, box_overlap_filter_threshold,
    max_inside, nms_iou_threshold (optional; NMS not applied on single best box path).
    Returns (mask_hw, inference_state, meta). mask_hw is bool (H,W) or None on failure.
    """
    meta = {"ok": False, "reason": "unknown", "best_score": None, "params_used": dict(p)}

    inference_state = video_predictor.non_video_path_init_state(
        image_inp, video_height, video_width
    )
    ann_frame_idx = 0
    image = raw_image_inp

    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=float(p["box_threshold"]),
        text_threshold=float(p["text_threshold"]),
        target_sizes=[image.size[::-1]],
    )

    scores = results[0]["scores"].cpu().numpy()
    input_boxes = results[0]["boxes"].cpu().numpy()
    labels = results[0]["labels"]
    print("objects", labels, "scores", scores, "num_boxes", len(input_boxes))

    if len(input_boxes) == 0:
        meta["reason"] = "no_boxes"
        return None, None, meta

    best_i = int(np.argmax(scores))
    best_score = float(scores[best_i])
    meta["best_score"] = best_score
    if best_score < float(p["min_best_score"]):
        meta["reason"] = "low_score"
        return None, None, meta

    input_boxes = input_boxes[best_i : best_i + 1]
    filtered = filter_boxes(
        input_boxes,
        threshold=float(p["box_overlap_filter_threshold"]),
        max_inside=int(p["max_inside"]),
    )
    if filtered.shape[0] == 0:
        meta["reason"] = "filtered_empty"
        return None, None, meta
    input_boxes = filtered[:1]

    label_one = labels[best_i] if hasattr(labels, "__getitem__") else labels

    PROMPT_TYPE_FOR_VIDEO = "box"

    if PROMPT_TYPE_FOR_VIDEO in ("point", "mask"):
        image_predictor.set_image(np.array(image.convert("RGB")))
        masks, sam_scores, logits = image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        if masks.ndim == 3:
            masks = masks[None]
        elif masks.ndim == 4:
            masks = masks.squeeze(1)
        if masks is None or masks.shape[0] == 0:
            meta["reason"] = "sam_empty"
            return None, None, meta
    else:
        masks = None

    if PROMPT_TYPE_FOR_VIDEO == "point":
        all_sample_points = sample_points_from_masks(masks=masks, num_points=10)
        for object_id, points in enumerate(all_sample_points, start=1):
            labels_pt = np.ones((points.shape[0]), dtype=np.int32)
            video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                points=points,
                labels=labels_pt,
            )
    elif PROMPT_TYPE_FOR_VIDEO == "box":
        for object_id, box in enumerate(input_boxes, start=1):
            video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                box=box,
            )
    elif PROMPT_TYPE_FOR_VIDEO == "mask":
        for object_id, mask in enumerate(masks, start=1):
            video_predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                mask=mask,
            )

    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(
        inference_state
    ):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    if not video_segments:
        meta["reason"] = "propagate_empty"
        return None, None, meta

    _, segments = next(iter(video_segments.items()))
    mask_list = list(segments.values())
    if not mask_list:
        meta["reason"] = "no_masks"
        return None, None, meta

    stacked = np.concatenate(mask_list, axis=0)
    mask_hw = np.any(stacked, axis=0).astype(np.bool_)
    meta["ok"] = True
    meta["reason"] = "ok"
    meta["label"] = str(label_one)
    return mask_hw, inference_state, meta


def update_inference_state(inference_state, frame, max_frames: int):
    print("image shapes", inference_state["images"].shape, frame.shape)
    inference_state["images"] = torch.cat((inference_state["images"], frame), dim=0)
    inference_state["num_frames"] += 1
    while inference_state["num_frames"] > max_frames:
        inference_state["images"] = inference_state["images"][1:]
        inference_state["num_frames"] -= 1
    return inference_state


def new_frame(video_predictor, inference_state, new_frame_tensor, max_frames: int):
    inference_state = update_inference_state(
        inference_state, new_frame_tensor, max_frames
    )
    inference_state_idx = inference_state["num_frames"] - 1
    print("inference_state_idx", inference_state_idx)

    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video_after_start(
        inference_state, start_frame_idx=inference_state_idx
    ):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    if not video_segments:
        return None, inference_state

    _, segments = next(iter(video_segments.items()))
    masks = list(segments.values())
    if not masks:
        return None, inference_state
    stacked = np.concatenate(masks, axis=0)
    mask_hw = np.any(stacked, axis=0).astype(np.bool_)
    return mask_hw, inference_state


def run_segment_with_fallback(
    processor,
    grounding_model,
    video_predictor,
    image_predictor,
    device,
    text,
    image_pil,
    image_prepared,
    video_height,
    video_width,
    base_params: dict,
    allow_fallback: bool,
    fallback_steps: list,
):
    """Try segmentation with base_params, then optional relaxed steps."""
    attempts = [merge_gsam_params(base_params, {})]
    if allow_fallback and fallback_steps:
        for step in fallback_steps:
            attempts.append(merge_gsam_params(attempts[-1], step))

    used_fallback = False
    last_meta = None
    for i, p in enumerate(attempts):
        mask_hw, inference_state, meta = first_step(
            processor,
            grounding_model,
            video_predictor,
            image_predictor,
            device,
            text,
            image_pil,
            image_prepared,
            video_height,
            video_width,
            p,
        )
        last_meta = meta
        if meta.get("ok") and mask_hw is not None:
            if i > 0:
                used_fallback = True
                print(
                    f"GSAM: segmentation succeeded after lowering thresholds (attempt {i + 1}/{len(attempts)}). "
                    f"params_used={meta.get('params_used')}"
                )
            meta["used_fallback"] = used_fallback
            return mask_hw, inference_state, meta
        if i < len(attempts) - 1:
            print(
                f"GSAM: segmentation failed ({meta.get('reason')}); retrying with looser thresholds. "
                f"next_params={attempts[i + 1]}"
            )

    last_meta = last_meta or {"ok": False, "reason": "unknown"}
    last_meta["ok"] = False
    last_meta["used_fallback"] = len(attempts) > 1
    print(
        f"GSAM: segmentation failed after all attempts. Last reason={last_meta.get('reason')} "
        f"best_score={last_meta.get('best_score')}"
    )
    return None, None, last_meta


def _result_meta(meta: dict, mask_hw: Optional[np.ndarray], mask_part_index=None):
    ok = mask_hw is not None and meta.get("ok")
    meta_out = {
        "ok": bool(ok),
        "reason": meta.get("reason", "ok" if ok else "error"),
        "used_fallback": bool(meta.get("used_fallback", False)),
        "params_used": meta.get("params_used"),
        "mask_dtype": "uint8" if ok else None,
        "mask_shape": list(mask_hw.shape) if ok else [],
        "best_score": meta.get("best_score"),
        "label": meta.get("label"),
    }
    if mask_part_index is not None:
        meta_out["mask_part_index"] = mask_part_index
    return meta_out


def _reply(socket, meta: dict, mask_hw: Optional[np.ndarray]):
    if mask_hw is not None and meta.get("ok"):
        mask_bytes = (mask_hw.astype(np.uint8) * 255).tobytes()
        meta_out = _result_meta(meta, mask_hw)
    else:
        mask_bytes = b""
        meta_out = _result_meta(meta, None)
    socket.send_multipart([json.dumps(meta_out).encode("utf-8"), mask_bytes])


def _handle_legacy_request(
    socket,
    message_parts,
    processor,
    grounding_model,
    video_predictor,
    image_predictor,
    device,
    server_defaults: dict,
    inference_state_holder: list,
    target_text_holder: list,
):
    """
    Legacy: [text_utf8, jpeg_bytes, one_byte_flag].
    Flag 0x01 = force new segmentation (old 'do_not_track' True).
    """
    text_data = message_parts[0].decode("utf-8")
    image_data = message_parts[1]
    force_segment = False
    if len(message_parts) >= 3:
        force_segment = message_parts[2] == b"\x01"

    image_pil = Image.open(io.BytesIO(image_data))
    image_prepared, video_height, video_width = load_single_image(
        image_pil, 1024, compute_device=device
    )

    if text_data:
        target_text_holder[0] = text_data
    text = target_text_holder[0]
    if text is None:
        _reply(socket, {"ok": False, "reason": "no_target_text"}, None)
        return

    max_f = int(server_defaults["max_frames_in_state"])
    if force_segment or inference_state_holder[0] is None:
        mask_hw, inf, meta = run_segment_with_fallback(
            processor,
            grounding_model,
            video_predictor,
            image_predictor,
            device,
            text,
            image_pil,
            image_prepared,
            video_height,
            video_width,
            server_defaults,
            allow_fallback=True,
            fallback_steps=[
                {
                    "box_threshold": 0.12,
                    "text_threshold": 0.18,
                    "min_best_score": 0.22,
                },
                {
                    "box_threshold": 0.08,
                    "text_threshold": 0.12,
                    "min_best_score": 0.12,
                },
            ],
        )
        if meta.get("ok"):
            inference_state_holder[0] = inf
        else:
            inference_state_holder[0] = None
        _reply(socket, meta, mask_hw)
        return

    mask_hw, inf = new_frame(
        video_predictor,
        inference_state_holder[0],
        image_prepared,
        max_f,
    )
    inference_state_holder[0] = inf
    if mask_hw is None:
        _reply(
            socket,
            {
                "ok": False,
                "reason": "tracking_failed",
                "params_used": server_defaults,
            },
            None,
        )
        return
    _reply(
        socket,
        {
            "ok": True,
            "reason": "ok",
            "used_fallback": False,
            "params_used": server_defaults,
            "label": None,
        },
        mask_hw,
    )


def main():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://0.0.0.0:8091")

    # Detect backend: ROCm/HIP reports as CUDA in PyTorch but needs different flags.
    _is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        backend = "ROCm/HIP" if _is_rocm else "CUDA"
        print(f"Using device: {device} ({backend}) — {gpu_name}")
    else:
        print("Using device: cpu")

    if torch.cuda.is_available():
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.is_available() and not _is_rocm:
        # tf32 is only available on NVIDIA Ampere+ (compute capability >= 8.0).
        props = torch.cuda.get_device_properties(0)
        if props.major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    sam2_checkpoint = "./checkpoints/sam2.1_hiera_small.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"

    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    image_predictor = SAM2ImagePredictor(sam2_image_model)

    model_id = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_id
    ).to(device)

    server_defaults = _default_server_params()
    # Keyed by target_text so each object label has its own tracking state.
    inference_states: dict = {}
    target_text_holder = [None]

    print("GSAM server ready on tcp://0.0.0.0:8091 (REP)")

    while True:
        print("Waiting for message")
        message_parts = socket.recv_multipart(flags=0)
        if not message_parts:
            continue

        cmd = message_parts[0]

        if cmd == b"set_config" and len(message_parts) >= 2:
            try:
                upd = json.loads(message_parts[1].decode("utf-8"))
                server_defaults = merge_gsam_params(server_defaults, upd)
                socket.send_multipart(
                    [
                        json.dumps(
                            {"ok": True, "applied": server_defaults}
                        ).encode("utf-8")
                    ]
                )
            except Exception as e:
                socket.send_multipart(
                    [
                        json.dumps({"ok": False, "reason": str(e)}).encode(
                            "utf-8"
                        )
                    ]
                )
            continue

        if cmd == b"segment" and len(message_parts) >= 3:
            try:
                spec = json.loads(message_parts[1].decode("utf-8"))
            except Exception as e:
                _reply(socket, {"ok": False, "reason": f"bad_json:{e}"}, None)
                continue

            image_data = message_parts[2]
            image_pil = Image.open(io.BytesIO(image_data))
            image_prepared, video_height, video_width = load_single_image(
                image_pil, 1024, compute_device=device
            )

            mode = spec.get("mode", "segment")
            target_text = spec.get("target_text") or target_text_holder[0]
            if not target_text:
                _reply(socket, {"ok": False, "reason": "no_target_text"}, None)
                continue
            target_text_holder[0] = target_text

            req_params = merge_gsam_params(
                server_defaults, spec.get("params") or {}
            )
            allow_fallback = bool(spec.get("allow_fallback", True))
            fallback_steps = spec.get("fallback_steps") or [
                {
                    "box_threshold": 0.12,
                    "text_threshold": 0.18,
                    "min_best_score": 0.22,
                },
                {
                    "box_threshold": 0.08,
                    "text_threshold": 0.12,
                    "min_best_score": 0.12,
                },
            ]

            max_f = int(req_params["max_frames_in_state"])

            if mode == "track":
                state = inference_states.get(target_text)
                if state is None:
                    _reply(
                        socket,
                        {
                            "ok": False,
                            "reason": "no_state_need_segment",
                            "params_used": req_params,
                        },
                        None,
                    )
                    continue
                mask_hw, inf = new_frame(
                    video_predictor,
                    state,
                    image_prepared,
                    max_f,
                )
                inference_states[target_text] = inf
                if mask_hw is None:
                    _reply(
                        socket,
                        {
                            "ok": False,
                            "reason": "tracking_failed",
                            "params_used": req_params,
                        },
                        None,
                    )
                    continue
                meta = {
                    "ok": True,
                    "reason": "ok",
                    "used_fallback": False,
                    "params_used": req_params,
                    "best_score": None,
                    "label": None,
                }
                _reply(socket, meta, mask_hw)
                continue

            # mode == segment
            mask_hw, inf, meta = run_segment_with_fallback(
                processor,
                grounding_model,
                video_predictor,
                image_predictor,
                device,
                target_text,
                image_pil,
                image_prepared,
                video_height,
                video_width,
                req_params,
                allow_fallback,
                fallback_steps,
            )
            if meta.get("ok"):
                inference_states[target_text] = inf
            else:
                inference_states.pop(target_text, None)
            _reply(socket, meta, mask_hw)
            continue

        if cmd == b"batch_segment" and len(message_parts) >= 3:
            try:
                spec = json.loads(message_parts[1].decode("utf-8"))
            except Exception as e:
                socket.send_multipart(
                    [
                        json.dumps(
                            {"ok": False, "reason": f"bad_json:{e}", "results": []}
                        ).encode("utf-8")
                    ]
                )
                continue

            requests = spec.get("requests") or []
            if not requests:
                socket.send_multipart(
                    [
                        json.dumps(
                            {"ok": False, "reason": "no_requests", "results": []}
                        ).encode("utf-8")
                    ]
                )
                continue

            image_data = message_parts[2]
            image_pil = Image.open(io.BytesIO(image_data))
            image_prepared, video_height, video_width = load_single_image(
                image_pil, 1024, compute_device=device
            )

            results = []
            mask_parts = []
            for req in requests:
                mode = req.get("mode", "segment")
                target_text = req.get("target_text")
                if not target_text:
                    results.append(
                        _result_meta({"ok": False, "reason": "no_target_text"}, None)
                    )
                    continue

                req_params = merge_gsam_params(
                    server_defaults, req.get("params") or {}
                )
                allow_fallback = bool(req.get("allow_fallback", True))
                fallback_steps = req.get("fallback_steps") or [
                    {
                        "box_threshold": 0.12,
                        "text_threshold": 0.18,
                        "min_best_score": 0.22,
                    },
                    {
                        "box_threshold": 0.08,
                        "text_threshold": 0.12,
                        "min_best_score": 0.12,
                    },
                ]
                max_f = int(req_params["max_frames_in_state"])

                if mode == "track":
                    state = inference_states.get(target_text)
                    if state is None:
                        results.append(
                            _result_meta(
                                {
                                    "ok": False,
                                    "reason": "no_state_need_segment",
                                    "params_used": req_params,
                                },
                                None,
                            )
                        )
                        continue
                    mask_hw, inf = new_frame(
                        video_predictor,
                        state,
                        image_prepared,
                        max_f,
                    )
                    inference_states[target_text] = inf
                    if mask_hw is None:
                        results.append(
                            _result_meta(
                                {
                                    "ok": False,
                                    "reason": "tracking_failed",
                                    "params_used": req_params,
                                },
                                None,
                            )
                        )
                        continue
                    meta = {
                        "ok": True,
                        "reason": "ok",
                        "used_fallback": False,
                        "params_used": req_params,
                        "best_score": None,
                        "label": None,
                    }
                else:
                    mask_hw, inf, meta = run_segment_with_fallback(
                        processor,
                        grounding_model,
                        video_predictor,
                        image_predictor,
                        device,
                        target_text,
                        image_pil,
                        image_prepared,
                        video_height,
                        video_width,
                        req_params,
                        allow_fallback,
                        fallback_steps,
                    )
                    if meta.get("ok"):
                        inference_states[target_text] = inf
                    else:
                        inference_states.pop(target_text, None)

                if mask_hw is not None and meta.get("ok"):
                    mask_part_index = len(mask_parts)
                    mask_parts.append((mask_hw.astype(np.uint8) * 255).tobytes())
                    results.append(_result_meta(meta, mask_hw, mask_part_index))
                else:
                    results.append(_result_meta(meta, None))

            socket.send_multipart(
                [
                    json.dumps({"ok": True, "results": results}).encode("utf-8"),
                    *mask_parts,
                ]
            )
            continue

        # Legacy protocol
        _handle_legacy_request(
            socket,
            message_parts,
            processor,
            grounding_model,
            video_predictor,
            image_predictor,
            device,
            server_defaults,
            inference_state_holder,
            target_text_holder,
        )


if __name__ == "__main__":
    main()


def summarize_dict(data):
    def summarize_value(value):
        if isinstance(value, torch.Tensor):
            return f"Tensor shape: {tuple(value.shape)}"
        elif isinstance(value, np.ndarray):
            return f"Array shape: {value.shape}"
        elif isinstance(value, dict):
            return summarize_dict(value)
        elif isinstance(value, list):
            return len(value)
        elif isinstance(value, tuple):
            return tuple(summarize_value(item) for item in value)
        else:
            return value

    return {key: summarize_value(val) for key, val in data.items()}


def convert_to_jpeg(png_path, jpeg_path, background_color=(255, 255, 255)):
    with Image.open(png_path) as img:
        if img.mode in ("RGBA", "LA") or (
            img.mode == "P" and "transparency" in img.info
        ):
            background = Image.new("RGB", img.size, background_color)
            background.paste(img, mask=img.split()[-1])
            img = background
        else:
            img = img.convert("RGB")
        img.save(jpeg_path, "JPEG")
    return jpeg_path


"""
Step 5: Visualize the segment results across the video and save them


save_dir = "./tracking_results"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS, start=1)}
for frame_idx, segments in video_segments.items():
    img = cv2.imread(os.path.join(video_dir, frame_names[frame_idx]))

    object_ids = list(segments.keys())
    masks = list(segments.values())
    masks = np.concatenate(masks, axis=0)

    detections = sv.Detections(
        xyxy=sv.mask_to_xyxy(masks),  # (n, 4)
        mask=masks, # (n, h, w)
        class_id=np.array(object_ids, dtype=np.int32),
    )
    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=[ID_TO_OBJECTS[i] for i in object_ids])
    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    cv2.imwrite(os.path.join(save_dir, f"annotated_frame_{frame_idx:05d}.jpg"), annotated_frame)



#Step 6: Convert the annotated frames to video


output_video_path = "./tracking_demo_video.mp4"
create_video_from_images(save_dir, output_video_path)
"""
