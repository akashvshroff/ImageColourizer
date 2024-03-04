import requests
import modelbit as mb
import math
from PIL import Image
import numpy as np
import base64
import io


def load_lottieurl(
    url="https://lottie.host/7af58fa9-62dc-4373-9464-40e087294535/pUrdD895QL.json",
):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


def decode_b64(encoded_b64):
    """
    Decode the encoded numpy data
    """
    buffer = io.BytesIO(base64.b64decode(encoded_b64))
    data = np.load(buffer)
    return data


def process_image(path):
    with Image.open(path) as img:
        img = img.convert("RGB")
        width, height = img.size
        scale = min(1, 512 / width, 512 / height)
        new_width = 2 ** int(math.log(width * scale, 2))
        new_height = 2 ** int(math.log(height * scale, 2))

        resized_img = img.resize((new_width, new_height), Image.LANCZOS)

        return np.array(resized_img)


def colourize_image_with_path(path):
    try:
        image_arr = process_image(path)
        response = mb.get_inference(
            region="us-east-1",
            workspace="akashshroff",
            deployment="colourize_image",
            data={"image_arr": image_arr},
        )
        colourized_b64 = response["data"]["colourized_image_b64"]
        colourized_arr = decode_b64(colourized_b64)
        return colourized_arr
    except Exception as e:
        print(f"uh oh, an error occured. {e}")
        return None
