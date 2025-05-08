import os
import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images

import io
from sam2.sam2_video_predictor import _load_img_bytes_as_tensor, load_single_image
from torchvision.ops import nms

"""
For the first step we are running GroundingDINO on the first frame and setting the inference frame up.
This means we need the processor, grounding_model, video_predictor, text prompt, the input image (to be prepared)
"""
def apply_nms(boxes, scores, iou_threshold=0.7):
    keep_indices = nms(boxes, scores, iou_threshold)
    return boxes[keep_indices], scores[keep_indices], keep_indices


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
    return np.array(keep)

def first_step(processor, grounding_model, video_predictor, image_predictor, device, text, raw_image_inp, image_inp, video_height, video_width):

    # setup the input image and text prompt for SAM 2 and Grounding DINO
    # VERY important: text queries need to be lowercased + end with a dot
    #text = "car."

    # init video predictor state
    inference_state = video_predictor.non_video_path_init_state(image_inp, video_height, video_width)

    ann_frame_idx = 0  # the frame index we interact with


    """
    Step 2: Prompt Grounding DINO and SAM image predictor to get the box and mask for specific frame
    """

    # prompt grounding dino to get the box coordinates on specific frame
    #img_path = os.path.join(video_dir, frame_names[ann_frame_idx])
    #image = Image.open(img_path)
    #image = Image.open(image_inp_path)
    image = raw_image_inp

    # run Grounding DINO on the image
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.2,
        text_threshold=0.7,
        target_sizes=[image.size[::-1]]
    )

    # prompt SAM image predictor to get the mask for the object
    image_predictor.set_image(np.array(image.convert("RGB")))
    # process the detection results
    scores = results[0]["scores"].cpu().numpy()
    max_index = np.argmax(scores) #scores.index(max(scores))
    input_boxes = results[0]["boxes"].cpu().numpy()
    OBJECTS = results[0]["labels"]
    print("objects", OBJECTS, "scores", scores, "num_boxes", len(input_boxes))
    if len(input_boxes) == 0:
        print("target object not detected")
        return None, None
    #non_overlapping_boxes, scores_torch, _ = apply_nms(results[0]["boxes"], results[0]["scores"])
    #scores = scores_torch.cpu().numpy()
    #input_boxes = non_overlapping_boxes.cpu().numpy()
    input_boxes = filter_boxes(input_boxes)
    print("after overlap filter boxes", len(input_boxes))

    # prompt SAM 2 image predictor to get the mask for the object
    masks, scores, logits = image_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    # convert the mask shape to (n, H, W)
    if masks.ndim == 3:
        masks = masks[None]
        scores = scores[None]
        logits = logits[None]
    elif masks.ndim == 4:
        masks = masks.squeeze(1)

    """
    Step 3: Register each object's positive points to video predictor with separate add_new_points call
    """

    PROMPT_TYPE_FOR_VIDEO = "box" # or "point"

    assert PROMPT_TYPE_FOR_VIDEO in ["point", "box", "mask"], "SAM 2 video predictor only support point/box/mask prompt"

    # If you are using point prompts, we uniformly sample positive points based on the mask
    if PROMPT_TYPE_FOR_VIDEO == "point":
        # sample the positive points from mask for each objects
        all_sample_points = sample_points_from_masks(masks=masks, num_points=10)

        for object_id, (label, points) in enumerate(zip(OBJECTS, all_sample_points), start=1):
            labels = np.ones((points.shape[0]), dtype=np.int32)
            _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                points=points,
                labels=labels,
            )
    # Using box prompt
    elif PROMPT_TYPE_FOR_VIDEO == "box":
        for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes), start=1):
            _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                box=box,
            )
    # Using mask prompt is a more straightforward way
    elif PROMPT_TYPE_FOR_VIDEO == "mask":
        for object_id, (label, mask) in enumerate(zip(OBJECTS, masks), start=1):
            labels = np.ones((1), dtype=np.int32)
            _, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                mask=mask
            )
    else:
        raise NotImplementedError("SAM 2 video predictor only support point/box/mask prompts")


    """
    Step 4: Propagate the video predictor to get the segmentation results for each frame
    """
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    for _, segments in video_segments.items():
        masks = list(segments.values())
        masks = np.concatenate(masks, axis=0)

        #There is only one value in the loop so return here
        return masks, inference_state, input_boxes, max_index

#Just concatenate the frame to the batch dimension of the inference state and increment the number of frames
def update_inference_state(inference_state, frame, video_predictor):
    #inference_state["images"] = process frame correctly and pass it in here, it is img = torch.zeros(batch, 3, image_size, image_size, dtype=torch.float32)
    print("image shapes", inference_state["images"].shape, frame.shape)
    inference_state["images"] = torch.cat((inference_state["images"], frame), dim=0)
    inference_state["num_frames"] += 1
    if inference_state["num_frames"] > 5:
        inference_state["images"] = inference_state["images"][1:]
        inference_state["num_frames"] -= 1
    return inference_state

#Update the inference state then run propagate in video now without the preflight in a custom function "propagate in video after start"
def new_frame(video_predictor, inference_state, new_frame):
    #fix inference state
    inference_state = update_inference_state(inference_state, new_frame, video_predictor) #video_predictor.update_inference_state(inference_state, new_frame)
    inference_state_idx = inference_state['num_frames'] - 1 #Only propagate the newest frame
    print("inference_state_idx", inference_state_idx)

    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video_after_start(inference_state, start_frame_idx=inference_state_idx):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    for _, segments in video_segments.items():
        masks = list(segments.values())
        masks = np.concatenate(masks, axis=0)

        #There is only one value in the loop so return here
        return masks, inference_state


import zmq
import json
def main():
    context = zmq.Context()
    socket = context.socket(zmq.REP) #REP sends replies when it gets something and is paired with a REQ socket that sends requests
    #socket.bind("tcp://*:8091") #Yuhan self.socket.bind(f"tcp://*:{port}")
    socket.bind("tcp://0.0.0.0:8091")

    #while True:
    #    if socket.poll(timeout=500):
    #        print("socket poll == True")

    #Their code:
    """
    Step 1: Environment settings and model initialization
    """
    # use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # init sam image predictor and video predictor model
    sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
    image_predictor = SAM2ImagePredictor(sam2_image_model)


    # init grounding dino model from huggingface
    model_id = "IDEA-Research/grounding-dino-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    #This results in video_predictor, image_predicotr, processor, grounding_model

    #My code:
    print("Server is ready...")
    inference_state = None
    target_text = None
    input_boxes = None
    max_index = None
    while True:
        print("Waiting for message")
        message_parts = socket.recv_multipart(flags=0) #Receiving text and image bytes
        text_data = message_parts[0].decode() #No need to specify size using zmq
        print("string received", text_data)
        #Second get the image inp
        image_data = message_parts[1] #No need to specify size using zmq
        ground = message_parts[2] == True.to_bytes(length=1, byteorder='big')
        image_pil = Image.open(io.BytesIO(image_data))
        image_prepared, video_height, video_width = load_single_image(image_pil, 1024)
        print("image loaded")
        #(processor, grounding_model, video_predictor, image_predictor, device, text, raw_image_inp, image_inp, video_height, video_width)
        if text_data:
            target_text = text_data
        if ground or (inference_state is None and target_text is not None):
            print("first step")
            masks, inference_state, input_boxes, max_index = first_step(processor, grounding_model, video_predictor, image_predictor, device, target_text, image_pil, image_prepared, video_height, video_width)
        else:
            masks = None
            if inference_state is not None:
                print("another frame")
                masks, inference_state = new_frame(video_predictor, inference_state, image_prepared)

        #masks=masks.cpu().numpy() don't need this as it already is a np array
        if masks is None:
            mask_bytes = b'\x00'
            dtype = None
            shape = [0]
        else:
            mask_bytes = masks.tobytes()
            dtype = str(masks.dtype)
            shape = masks.shape
        metadata = {
            "dtype": dtype,
            "shape": shape
        }
        
        metadata_json = json.dumps(metadata)

        input_boxes_bytes = input_boxes.tobytes()
        input_boxes_dtype = str(input_boxes.dtype)
        input_boxes_shape = input_boxes.shape
        metadata_input_boxes = {
            "dtype": input_boxes_dtype,
            "shape": input_boxes_shape
        }
        json_input_boxes = json.dumps(metadata_input_boxes)

        #Serialized mask back with its metadata first
        send_message_parts = [metadata_json.encode(), mask_bytes, str(max_index).encode('utf-8')]#, json_input_boxes.encode(), input_boxes_bytes]
        #socket.send_json(metadata)
        #socket.send(mask_bytes) 
        socket.send_multipart(send_message_parts)

#Take the image in, laod it properly and save it into a list of images, pass the images in individually one by one
def reference():
    #some loop for waiting for a signal from the server
    #Remember not to use snipping tool with different sizes as they must be the same size
    images = []
    for o in range(1, 4):
        #image_path = convert_to_jpeg(f"0000{o}.png", f"0000{o}.jpg")
        image_path = f"0000{o}.jpg"
        # image = Image.open(image_path).convert("RGB")
        # buffer = io.BytesIO()
        # image.save(buffer, format="JPEG")
        # img_bytes = buffer.getvalue()
        # buffer.close()
        # images.append(img_bytes)
        image, video_height, video_width = load_single_image(image_path, 1024)
        images.append(image)

    video_predictor, inference_state, masks1, segments1 = first_step("car.", images[0], "00001.jpg", video_height, video_width)
    print("mask shapes", masks1.shape)
    print("inference state 1", summarize_dict(inference_state))
    video_predictor, inference_state, masks2, segments2 = new_frame(video_predictor, inference_state, images[1])
    print("mask shapes", masks2.shape)
    print("inference state 2", summarize_dict(inference_state))
    video_predictor, inference_state, masks3, segments3 = new_frame(video_predictor, inference_state, images[2])
    print("mask shapes", masks3.shape)
    print("inference state 3", summarize_dict(inference_state))

    masks_list = [masks1, masks2, masks3]
    print(torch.equal(torch.tensor(masks1), torch.tensor(masks2)), torch.equal(torch.tensor(masks2), torch.tensor(masks3)))
    segments_list = [segments1, segments2, segments3]

    for o in range(1, 4):
        masks = masks_list[o-1]
        segments = segments_list[o-1]
        img = cv2.imread(f"0000{o}.jpg")
        object_ids = list(segments.keys())

        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks),  # (n, 4)
            mask=masks, # (n, h, w)
            class_id=np.array(object_ids, dtype=np.int32),
        )
        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        cv2.imwrite(f"output0000{o}.jpg", annotated_frame)

    # for o in range(1, 4):
    #     img = cv2.imread(f"track{o}.jpg")

    #     print("o", o)
    #     print("mask shape", masks[2].shape)
    #     for mask in masks[o-1]:
    #         mask = (mask > 0.5).astype(np.uint8)
    #         print("mask shape", mask.shape)
    #         color = np.random.randint(0, 255, (3,), dtype=np.uint8)  # Random color
    #         colored_mask = np.zeros_like(img, dtype=np.uint8)
    #         print("colored mask shape in full", colored_mask.shape)
    #         for c in range(3):
    #             print("colored mask shape", colored_mask[:, :, c].shape, "color shape", color[c].shape)
    #             colored_mask[:, :, c] = mask * color[c]
    #         img = cv2.addWeighted(img, 0.9, colored_mask, 0.1, 0)
    #     cv2.imwrite(f"trackresult{o}.jpg", img)


if __name__ == "__main__":
    main()
    

def summarize_dict(data):
    """
    Recursively summarize a dictionary, replacing tensor or array values with their shapes.
    
    Args:
        data (dict): The dictionary to process.
        
    Returns:
        dict: A summarized dictionary.
    """
    def summarize_value(value):
        if isinstance(value, torch.Tensor):  # Check for PyTorch tensors
            return f"Tensor shape: {tuple(value.shape)}"
        elif isinstance(value, np.ndarray):  # Check for NumPy arrays
            return f"Array shape: {value.shape}"
        elif isinstance(value, dict):  # Recursively process nested dictionaries
            return summarize_dict(value)
        elif isinstance(value, list):
            return len(value)
        elif isinstance(value, tuple):
            return tuple(summarize_value(item) for item in value)
        else:
            return value  # Leave other types unchanged
    
    return {key: summarize_value(val) for key, val in data.items()}

def convert_to_jpeg(png_path, jpeg_path, background_color=(255, 255, 255)):
    with Image.open(png_path) as img:
        # Handle transparency by adding a background
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            background = Image.new("RGB", img.size, background_color)
            background.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
            img = background
        else:
            img = img.convert("RGB")  # Ensure RGB mode
        img.save(jpeg_path, "JPEG")
    return jpeg_path

#def visualize()
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
