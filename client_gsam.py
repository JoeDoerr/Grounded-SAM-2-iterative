import zmq
from PIL import Image
import io
import numpy as np
import cv2
import json
import time
import random
from pathlib import Path

#Get two segmentations of the same scene. Check each mask against each other mask. Remove masks that overlap too much from the all-mask. 
import numpy as np

def track_two(image_1, image_2):
    pass

#Send it in both directions, keeping T-W and W-T. Then compare overlaps between the two T images and two W images.
#The ids that overlap above a threshold will have their masks put together as a single object for both T and W. 
#Mask by bits

def remove_target_mask(instance_masks, target_mask, threshold=0.9):
    """
    Removes any mask in instance_masks that overlaps with the target_mask 
    by >= threshold (default 90%).

    Parameters:
    - instance_masks: np.ndarray of shape [num_masks, height, width]
                      Binary masks for each instance.
    - target_mask: np.ndarray of shape [height, width]
                   Binary mask to compare against.
    - threshold: float, overlap ratio to trigger removal.

    Returns:
    - np.ndarray: Filtered instance_masks with overlapping masks removed.
    """
    keep_masks = []
    target_area = np.sum(target_mask)

    for mask in instance_masks:
        intersection = np.sum(mask * target_mask)
        overlap_ratio = intersection / target_area if target_area > 0 else 0

        if overlap_ratio < threshold:
            keep_masks.append(mask)

    return np.stack(keep_masks) if keep_masks else np.zeros((0, *target_mask.shape), dtype=instance_masks.dtype)

def send_one(image, send_string, socket, do_not_track=0, boxes_use=False):
    #image = Image.open(image_path)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image_stream = io.BytesIO()
    image.save(image_stream, format="JPEG")  #Save image in JPEG format
    image_data = image_stream.getvalue()  #Get byte data
    #socket.send(image_data) #Send the serialized image

    message = [send_string.encode(), image_data, do_not_track.to_bytes(length=1, byteorder='big')]
    socket.send_multipart(message) #SENDING text and image
    print("sent")

    #To receive it blocks until it receives
    message_parts = socket.recv_multipart() #RECEIVING metadata for masks and mask bytes
    metadata = message_parts[0].decode()  # Decode as string
    metadata = json.loads(metadata)
    print(f"Received metadata: {metadata}")
    #Bytes
    mask_bytes = message_parts[1]
    # Deserialize the mask
    masks = np.frombuffer(mask_bytes, dtype=metadata["dtype"]).reshape(metadata["shape"])

    #Boxes
    if boxes_use == True:
        box_metadata = json.loads(message_parts[2].decode())
        box_bytes = message_parts[3]
        boxes = np.frombuffer(box_bytes, dtype=box_metadata["dtype"]).reshape(box_metadata["shape"]) #[x0, y0, x1, y1] format
        return masks, boxes
    else:
        return masks

def instance_and_target_masks_to_one_mask(instance_mask, target_mask):
    """
    instance_mask = [masks, height, width] bool values
    target_mask = [height, width] bool values

    For each masks value, put it as the corresponding bit of a [height, width] as 1
    All values at mask0 move 0+1
    All values at mask1 move 1+1
    etc
    Keep the first as the target value

    return encoded_instance_mask = [masks] of 32bit values
    """
    encoded_instance_mask = np.zeros((instance_mask.shape[1], instance_mask.shape[2]), dtype=np.int32) #[height, width]
    for i in range(instance_mask.shape[0]):
        encoded_instance_mask |= (instance_mask[i].astype(np.int32) << (i+1))
    encoded_instance_mask |= (target_mask.astype(np.int32))

    return encoded_instance_mask

def send_instance_and_target(img, tar_string, socket):
    mask_instance = send_one(img, "Object.", socket, do_not_track=True)
    mask_target = send_one(img, tar_string, socket, do_not_track=True)
    mask_instance = remove_target_mask(mask_instance, mask_target)
    encoded_instance_mask = instance_and_target_masks_to_one_mask(mask_instance, mask_target)
    return mask_instance, mask_target, encoded_instance_mask

def prep_and_send():
    context = zmq.Context()
    socket = context.socket(zmq.REQ) #Make sure to use REQ
    print("trying to connect to port")
    #socket.connect("tcp://:8091")
    socket.connect("tcp://0.0.0.0:8091")
    print("Client connected")
    send_strings = ["Object.", "Object."]
    paths = ["test_segmentation_images/scene7_wrist.jpg", "test_segmentation_images/scene7_head.jpg"]
    for i in range(len(paths)):
        image_path = paths[i]
        send_string = send_strings[i]
        image = Image.open(image_path)
        do_not_track = False
        if i % 2 == 0: #Purposely want it to run object detection again when it is the first run, so I can keep the gsam running and continue restarting the object detection from there
            do_not_track = True
        #Do not track means that for this given image don't track it from the previous one
        masks, boxes = send_one(image, send_string, socket, do_not_track=do_not_track)
        make_seg_img(masks, image_path, tag=i)

def make_seg_img(masks, image_path, tag=0, boxes=None):
    img = cv2.imread(image_path)
    overlay = np.zeros_like(img, dtype=np.uint8)
    i = -1
    for mask in masks:
        i += 1
        mask = (mask > 0.5).astype(np.uint8)
        bold_colors = [
            (255, 0, 0),      # Red
            (0, 255, 0),      # Green
            (0, 0, 255),      # Blue
            (255, 255, 0),    # Yellow
            (255, 0, 255),    # Magenta
            (0, 255, 255),    # Cyan
            (255, 128, 0),    # Orange
            (128, 0, 255),    # Purple
            (0, 128, 255),    # Light Blue
            (128, 255, 0),    # Lime
            #(255, 0, 128),    # Pink
        ]
        color = np.array(random.choice(bold_colors), dtype=np.uint8)

        if i < len(bold_colors):
            color = np.array(bold_colors[i], dtype=np.uint8)
        else:
            color = tuple(np.random.randint(0, 256, size=3).tolist())
            color = np.array(color, dtype=np.uint8)
        #if i >= len(masks) - 1:
        #    print("Pink color")
        #    color = np.array((255, 0, 128), dtype=np.uint8) #Pink
        colored_mask = np.zeros_like(img, dtype=np.uint8)
        
        for c in range(3):
            colored_mask[:, :, c] = mask * color[c]
        
        overlay = np.where(mask[..., None], overlay + colored_mask, overlay)

        if boxes is not None:
            for box in boxes:
                x0, y0, x1, y1 = map(int, box)
                cv2.rectangle(overlay, (x0, y0), (x1, y1), color=(0, 0, 0), thickness=2)

    # Clip values to avoid overflow after summation
    overlay = np.clip(overlay, 0, 255)

    # Blend once at the end
    blended = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)
    
    cv2.imwrite(f"./result_images/outputfromclient_{tag}.jpg", blended)

def main():
    prep_and_send()
    return
    context = zmq.Context()
    socket = context.socket(zmq.REQ) #Make sure to use REQ
    print("trying to connect to port")
    #socket.connect("tcp://:8091")
    socket.connect("tcp://0.0.0.0:8091")
    print("Client connected")

    num_pics=1#3
    send_string_original = "Object." #"wares. shapes. objects. items. products. inventory."
    #"car." #mustard bottle. green can. black box. #household object. #"cracker box. pringles can. mustard bottle. robot arm. robot gripper."
    do_not_track=True
    #for o in range(1, num_pics+1):
    png_dir = Path("./test_segmentation_images/")
    o=-1
    paths = png_dir.glob("*.png")
    mask_instance = None
    mask_target = None
    target_run = 0
    for image_path in png_dir.glob("*.png"):
        for double_try in range(2):
            send_string = send_string_original
            if double_try == target_run:
                send_string = "shoe."
            #image_path = paths[q]
            o+=1
            t0 = time.time()
            #image_path = f"0000{o}.jpg"
            #image_path = "0005-color.jpg" #"d455_color.jpg" #"0005-color.jpg" #"d455_color.jpg" #"seg_overlay_1.png" #"test_depth_accuracy.png"

            #socket.send_string("car.") #Make sure to use send_string

            image = Image.open(image_path)
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            image_stream = io.BytesIO()
            image.save(image_stream, format="JPEG")  #Save image in JPEG format
            image_data = image_stream.getvalue()  #Get byte data
            #socket.send(image_data) #Send the serialized image

            message = [send_string.encode(), image_data, do_not_track.to_bytes(length=1, byteorder='big')]
            socket.send_multipart(message) #SENDING text and image
            print("sent")

            #To receive it blocks until it receives
            message_parts = socket.recv_multipart() #RECEIVING metadata for masks and mask bytes
            metadata = message_parts[0].decode()  # Decode as string
            metadata = json.loads(metadata)
            print(f"Received metadata: {metadata}")
            #Bytes
            mask_bytes = message_parts[1]
            # Deserialize the mask
            masks = np.frombuffer(mask_bytes, dtype=metadata["dtype"]).reshape(metadata["shape"])
            if double_try == target_run:
                mask_target = masks
            else:
                mask_instance = masks
            print("time took to do entire one image", time.time() - t0)

            #Boxes
            box_metadata = json.loads(message_parts[2].decode())
            box_bytes = message_parts[3]
            boxes = np.frombuffer(box_bytes, dtype=box_metadata["dtype"]).reshape(box_metadata["shape"]) #[x0, y0, x1, y1] format
            print("boxes", boxes)

            #Saving images to verify this worked
            # img = cv2.imread(image_path)
            # for mask in masks:
            #     mask = (mask > 0.5).astype(np.uint8)
            #     print("mask shape", mask.shape)
            #     color = np.random.randint(0, 255, (3,), dtype=np.uint8)  # Random color
            #     colored_mask = np.zeros_like(img, dtype=np.uint8)
            #     print("colored mask shape in full", colored_mask.shape)
            #     for c in range(3):
            #         print("colored mask shape", colored_mask[:, :, c].shape, "color shape", color[c].shape)
            #         colored_mask[:, :, c] = mask * color[c]
            #     img = cv2.addWeighted(img, 0.5, colored_mask, 0.5, 0)
            # cv2.imwrite(f"outputfromclient0000{o}.jpg", img)
            if double_try == 1:
                print("Mask shapes", mask_target.shape, mask_instance.shape)
                mask_instance = remove_target_mask(mask_instance, mask_target)
                print("Masks after removal", len(mask_instance))
                masks = np.concatenate((mask_instance, mask_target), axis=0)
                for j in range(3):
                    img = cv2.imread(image_path)
                    overlay = np.zeros_like(img, dtype=np.uint8)  # Combined overlay

                    if j == 1:
                        masks = []
                        img_height, img_width = img.shape[:2]
                        for box in boxes:
                            x0, y0, x1, y1 = map(int, box)
                            mask = np.zeros((img_height, img_width), dtype=np.uint8)
                            mask[y0:y1, x0:x1] = 1
                            masks.append(mask)

                    if j == 2:
                        masks = mask_instance
                    
                    i = -1
                    for mask in masks:
                        i += 1
                        mask = (mask > 0.5).astype(np.uint8)
                        bold_colors = [
                            (255, 0, 0),      # Red
                            (0, 255, 0),      # Green
                            (0, 0, 255),      # Blue
                            (255, 255, 0),    # Yellow
                            (255, 0, 255),    # Magenta
                            (0, 255, 255),    # Cyan
                            (255, 128, 0),    # Orange
                            (128, 0, 255),    # Purple
                            (0, 128, 255),    # Light Blue
                            (128, 255, 0),    # Lime
                            #(255, 0, 128),    # Pink
                        ]
                        color = np.array(random.choice(bold_colors), dtype=np.uint8)

                        if i < len(bold_colors):
                            color = np.array(bold_colors[i], dtype=np.uint8)
                        else:
                            color = tuple(np.random.randint(0, 256, size=3).tolist())
                            color = np.array(color, dtype=np.uint8)
                        if i >= len(masks) - 1:
                            print("Pink color")
                            color = np.array((255, 0, 128), dtype=np.uint8) #Pink
                        colored_mask = np.zeros_like(img, dtype=np.uint8)
                        
                        for c in range(3):
                            colored_mask[:, :, c] = mask * color[c]
                        
                        overlay = np.where(mask[..., None], overlay + colored_mask, overlay)

                    if j == 1:
                        for box in boxes:
                            x0, y0, x1, y1 = map(int, box)
                            cv2.rectangle(overlay, (x0, y0), (x1, y1), color=(0, 0, 0), thickness=2)

                    # Clip values to avoid overflow after summation
                    overlay = np.clip(overlay, 0, 255)

                    # Blend once at the end
                    blended = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)
                    
                    cv2.imwrite(f"./result_images/outputfromclient0000{o}{j}.jpg", blended)

if __name__ == "__main__":
    main()