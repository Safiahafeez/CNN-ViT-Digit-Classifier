
import streamlit as st
import torch
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from models import CNN, ViT
from preprocessing import prepare_image


#Setting the device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

#Output folder
output_folder = Path(__file__).parent / "module4_outputs"


#Loading the saved CNN model
cnn_checkpoint = torch.load(
    output_folder / "cnn_model.pth",
    map_location=device,
    weights_only=True
)

cnn_model = CNN()
cnn_model.load_state_dict(cnn_checkpoint["model_state_dict"])
cnn_model.to(device)
cnn_model.eval()


#Loading the saved ViT model
vit_checkpoint = torch.load(
    output_folder / "vit_model.pth",
    map_location=device,
    weights_only=True
)

vit_model = ViT()
vit_model.load_state_dict(vit_checkpoint["model_state_dict"])
vit_model.to(device)
vit_model.eval()


#Streamlit App
st.title("Handwritten Digit Prediction")
st.write(
    "Upload an image of a handwritten digit (0-9) "
    "to get predictions from CNN and ViT models."
)
#Uploading an image
uploaded_file = st.file_uploader(
    "Upload a handwritten digit",
    type=["png", "jpg", "jpeg"]
)
if uploaded_file is not None:
    try:
        # Opening the uploaded image
        original_image = Image.open(uploaded_file)
        original_image.load()

        #Displaying the uploaded image
        st.subheader("Uploaded Image")
        st.image(original_image, caption="Original Image")

        #Preparing the image for prediction
        prepared_image, image_array = prepare_image(original_image)

        #Displaying the prepared 8x8 image
        st.subheader("Prepared 8x8 Image")

        preview_image = prepared_image.resize(
            (256, 256),
            Image.Resampling.NEAREST
        )
        st.image(
            preview_image,
            caption="Processed Image (8x8)"
        )
        #Converting image into PyTorch tensor
        image_tensor = torch.tensor(
            image_array,
            dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)
        image_tensor = image_tensor.to(device)

        #Making predictions using both models
        with torch.no_grad():
            cnn_output = cnn_model(image_tensor)
            vit_output = vit_model(image_tensor)
            # Getting predicted digits
            cnn_prediction = torch.argmax(
                cnn_output, dim=1
            ).item()
            vit_prediction = torch.argmax(
                vit_output, dim=1
            ).item()

        #Displaying CNN prediction
        st.subheader("Prediction Results")

        st.success(
            f"CNN Predicted Digit: {cnn_prediction}"
        )

        #Displaying ViT prediction
        st.success(
            f"ViT Predicted Digit: {vit_prediction}"
        )
    except (UnidentifiedImageError, OSError, ValueError) as error:
        st.error(f"Image processing error: {error}")
else:
    st.info("Please upload an image to get predictions.")
