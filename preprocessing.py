import numpy as np
from PIL import Image, ImageOps

def prepare_image(uploaded_image):
    image = uploaded_image.convert("L")
    pixels = np.array(image)
    border_pixels = np.concatenate([
        pixels[0, :],
        pixels[-1, :],
        pixels[:, 0],
        pixels[:, -1]
    ])
    if border_pixels.mean() > 127:
        image = ImageOps.invert(image)

    #Converting image into NumPy array
    pixels = np.array(image)

    #Detecting handwritten strokes
    stroke_mask = pixels > 30

    #Rejecting blank images
    if pixels.max() - pixels.min() < 20:
        raise ValueError("The uploaded image is blank.")
    if np.count_nonzero(stroke_mask) < 10:
        raise ValueError("No handwritten digit detected.")

    #Finding digit boundaries
    mask_image = Image.fromarray(
        (stroke_mask * 255).astype(np.uint8)
    )
    bounding_box = mask_image.getbbox()
    if bounding_box is None:
        raise ValueError("No digit detected.")

    #Cropping handwritten digit
    image = image.crop(bounding_box)
    width, height = image.size
    side = int(max(width, height) * 1.2)
    canvas = Image.new("L", (side, side), 0)

    #Centering handwritten digit
    position = (
        (side - width) // 2,
        (side - height) // 2
    )
    canvas.paste(image, position)
    prepared_image = canvas.resize(
        (8, 8),
        Image.Resampling.LANCZOS
    )
    image_array = np.array(
        prepared_image,
        dtype=np.float32
    ) / 255.0
    return prepared_image, image_array
