import zmq
from PIL import Image
import io
import numpy as np
import cv2
import json
import time
import random
from pathlib import Path

def main():
    context = zmq.Context()
    socket = context.socket(zmq.REQ) #Make sure to use REQ
    print("trying to connect to port")
    #socket.connect("tcp://:8091")
    socket.connect("tcp://0.0.0.0:8091")
    print("Client connected")

    num_pics=1#3
    send_string = "Object." #"wares. shapes. objects. items. products. inventory."
    #"car." #mustard bottle. green can. black box. #household object. #"cracker box. pringles can. mustard bottle. robot arm. robot gripper."
    do_not_track=True
    #for o in range(1, num_pics+1):
    png_dir = Path("./test_segmentation_images/")
    o=-1
    for image_path in png_dir.glob("*.png"):
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
        for j in range(2):
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
                    (255, 0, 128),    # Pink
                ]
                color = np.array(random.choice(bold_colors), dtype=np.uint8)

                if i < len(bold_colors):
                    color = np.array(bold_colors[i], dtype=np.uint8)
                else:
                    color = tuple(np.random.randint(0, 256, size=3).tolist())
                    color = np.array(color, dtype=np.uint8)
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