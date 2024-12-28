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

"""
For the first step we are running GroundingDINO on the first frame and setting the inference frame up.
This means we need the processor, grounding_model, video_predictor, text prompt, the input image (to be prepared)
"""
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
        box_threshold=0.25,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )

    # prompt SAM image predictor to get the mask for the object
    image_predictor.set_image(np.array(image.convert("RGB")))

    # process the detection results
    input_boxes = results[0]["boxes"].cpu().numpy()
    OBJECTS = results[0]["labels"]

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
        return masks, inference_state

#Just concatenate the frame to the batch dimension of the inference state and increment the number of frames
def update_inference_state(inference_state, frame, video_predictor):
    #inference_state["images"] = process frame correctly and pass it in here, it is img = torch.zeros(batch, 3, image_size, image_size, dtype=torch.float32)
    print("image shapes", inference_state["images"].shape, frame.shape)
    inference_state["images"] = torch.cat((inference_state["images"], frame), dim=0)
    inference_state["num_frames"] += 1
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
def main():
    context = zmq.Context()
    socket = context.socket(zmq.REP) #REP sends replies when it gets something and is paired with a REQ socket that sends requests
    socket.bind("tcp://*:8090") #Yuhan self.socket.bind(f"tcp://*:{port}")
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
    model_id = "IDEA-Research/grounding-dino-tiny"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    #This results in video_predictor, image_predicotr, processor, grounding_model

    print("Server is ready...")
    first = False
    inference_state = None
    while True:
        #First get the text inp
        text_data = socket.recv_string() #No need to specify size using zmq
        #Second get the image inp
        image_data = socket.recv() #No need to specify size using zmq
        image_pil = Image.open(io.BytesIO(image_data))
        image_prepared, video_height, video_width = load_single_image(image_pil, 1024)
        print("image loaded")
        #(processor, grounding_model, video_predictor, image_predictor, device, text, raw_image_inp, image_inp, video_height, video_width)
        if first == False:
            masks, inference_state = first_step(processor, grounding_model, video_predictor, image_predictor, device, text_data, image_pil, image_prepared, video_height, video_width)
        else:
            masks, inference_state = new_frame(video_predictor, inference_state, image_prepared)

        mask_bytes = masks.tobytes()
        metadata = {
            "dtype": str(masks.dtype),
            "shape": masks.shape
        }
        
        #Serialized mask back with its metadata first
        socket.send_json(metadata)
        socket.send(mask_bytes) 


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