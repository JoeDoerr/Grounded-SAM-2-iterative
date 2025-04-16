import zmq
from PIL import Image
import io
import numpy as np
import cv2
import json
import time
import random

def main():
    context = zmq.Context()
    socket = context.socket(zmq.REQ) #Make sure to use REQ
    print("trying to connect to port")
    #socket.connect("tcp://:8091")
    socket.connect("tcp://0.0.0.0:8091")
    print("Client connected")

    num_pics=1#3
    send_string = "mustard bottle. red can. box. red can tall. robot arm. shelf."#"car."
    do_not_track=True
    for o in range(1, num_pics+1):
        t0 = time.time()
        #image_path = f"0000{o}.jpg"
        image_path = f"grasping1.png"

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
        img = cv2.imread(image_path)
        overlay = np.zeros_like(img, dtype=np.uint8)  # Combined overlay

        for mask in masks:
            mask = (mask > 0.5).astype(np.uint8)
            bold_colors = [
                (255, 0, 0),    # Red
                (0, 255, 0),    # Green
                (0, 0, 255),    # Blue
                (255, 255, 0),  # Yellow
                (255, 0, 255),  # Magenta
                (0, 255, 255),  # Cyan
                (255, 128, 0),  # Orange
            ]
            color = np.array(random.choice(bold_colors), dtype=np.uint8)
            colored_mask = np.zeros_like(img, dtype=np.uint8)
            
            for c in range(3):
                colored_mask[:, :, c] = mask * color[c]
            
            overlay = np.where(mask[..., None], overlay + colored_mask, overlay)

        # Clip values to avoid overflow after summation
        overlay = np.clip(overlay, 0, 255)

        # Blend once at the end
        blended = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)

        cv2.imwrite(f"outputfromclient0000{o}.jpg", blended)

if __name__ == "__main__":
    main()