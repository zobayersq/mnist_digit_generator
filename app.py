import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io
import zipfile
import base64 # For favicon, if needed

# --- Configuration (MUST match training script) ---
latent_dim = 100
num_classes = 10
image_size = 28 # MNIST image size

# --- Generator Network (MUST match training script's Generator) ---
class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, img_shape=(1, 28, 28)):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.latent_dim = latent_dim

        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.fc = nn.Linear(latent_dim + num_classes, 128 * 7 * 7)

        self.main = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), # Output: (batch_size, 64, 14, 14)
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),  # Output: (batch_size, 1, 28, 28)
            nn.Tanh() # Output pixels are in [-1, 1]
        )

    def forward(self, noise, labels):
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        out = self.fc(gen_input)
        out = out.view(out.size(0), 128, 7, 7)
        img = self.main(out)
        return img

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="MNIST Digit Generator",
    page_icon="🔢",
    layout="wide"
)

# --- Model Loading (Cached to avoid reloading on every rerun) ---
@st.cache_resource
def load_generator_model():
    # Load model on CPU, as Streamlit apps usually don't have GPU access
    device = torch.device('cpu')
    generator = Generator(latent_dim=latent_dim, num_classes=num_classes).to(device)

    try:
        # Load the trained model weights. Ensure 'generator_cgan.pth' is in the same directory.
        generator.load_state_dict(torch.load('generator_cgan.pth', map_location=device))
        generator.eval() # Set model to evaluation mode
        return generator, device
    except FileNotFoundError:
        st.error("Model file 'generator_cgan.pth' not found. Please train the model using the provided training script and upload the 'generator_cgan.pth' file to the same directory as this Streamlit app.")
        st.stop() # Stop execution if model is not found
    except Exception as e:
        st.error(f"Error loading model: {e}. Ensure the Generator architecture matches the trained model.")
        st.stop()

# --- Image Generation Function ---
def generate_digit_images(generator, device, digit, num_samples=5):
    generator.eval() # Ensure model is in evaluation mode
    with torch.no_grad(): # Disable gradient calculations for inference
        # Generate `num_samples` images for the specified digit
        noise = torch.randn(num_samples, latent_dim, device=device)
        labels = torch.full((num_samples,), digit, dtype=torch.long, device=device) # All labels are the selected digit

        fake_imgs = generator(noise, labels)

        # Convert to numpy and denormalize from [-1, 1] to [0, 1]
        imgs = fake_imgs.cpu().numpy()
        imgs = (imgs + 1) / 2
        imgs = np.clip(imgs, 0, 1) # Ensure values are strictly within [0, 1]

        return imgs

# --- Utility to convert numpy array to PIL Image ---
def numpy_to_pil(img_array):
    # img_array shape is (C, H, W). Convert to (H, W) for grayscale PIL 'L' mode.
    # Assuming C is 1 for grayscale MNIST images.
    img_uint8 = (img_array[0] * 255).astype(np.uint8)
    return Image.fromarray(img_uint8, mode='L')

# --- Streamlit UI ---
def main():
    st.title("🔢 MNIST Handwritten Digit Generator")
    st.markdown("Generate handwritten digits using a **Conditional GAN** model.")

    # Load the generator model
    generator, device = load_generator_model()

    # Sidebar for controls
    st.sidebar.header("Generation Controls")

    # Digit selection
    selected_digit = st.sidebar.selectbox(
        "Select digit to generate:",
        options=list(range(num_classes)),
        index=0 # Default to digit 0
    )

    # Generate button
    if st.sidebar.button("🎲 Generate New Images", type="primary"):
        # Trigger image generation
        st.session_state.generate_new = True

    # Initialize session state for consistent display
    if 'generate_new' not in st.session_state:
        st.session_state.generate_new = True
    if 'current_digit' not in st.session_state:
        st.session_state.current_digit = selected_digit
    if 'generated_imgs' not in st.session_state:
        st.session_state.generated_imgs = None

    # Check if digit changed or if new generation is requested
    if st.session_state.current_digit != selected_digit:
        st.session_state.generate_new = True
        st.session_state.current_digit = selected_digit

    # Perform generation if triggered or if no images are loaded yet
    if st.session_state.generate_new or st.session_state.generated_imgs is None:
        with st.spinner(f"Generating digit {selected_digit} images..."):
            try:
                generated_imgs = generate_digit_images(generator, device, selected_digit, 5)
                st.session_state.generated_imgs = generated_imgs
                st.session_state.generate_new = False
            except Exception as e:
                st.error(f"Error generating images: {str(e)}")
                # Reset generation flag to allow retry
                st.session_state.generate_new = False

    # Display generated images
    if st.session_state.generated_imgs is not None:
        st.header(f"Generated Images for Digit: {selected_digit}")

        # Create columns for image display
        cols = st.columns(5) # 5 images

        for i in range(5):
            with cols[i]:
                img_pil = numpy_to_pil(st.session_state.generated_imgs[i])

                # Resize for better display (optional, but makes them larger)
                img_pil_resized = img_pil.resize((150, 150), Image.NEAREST)

                st.image(
                    img_pil_resized,
                    caption=f"Sample {i+1}",
                    use_column_width=False # Set to False to control width with resize
                )

        # --- Download functionality ---
        st.subheader("💾 Download Generated Images")

        # Create a zip file with all images
        if st.button("📦 Download All Images as ZIP"):
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for i in range(5):
                    img_pil = numpy_to_pil(st.session_state.generated_imgs[i])
                    img_buffer = io.BytesIO()
                    img_pil.save(img_buffer, format='PNG')
                    img_buffer.seek(0)

                    zip_file.writestr(f"digit_{selected_digit}_sample_{i+1}.png", img_buffer.getvalue())

            zip_buffer.seek(0)

            st.download_button(
                label="⬇️ Download ZIP",
                data=zip_buffer.getvalue(),
                file_name=f"generated_digit_{selected_digit}_samples.zip",
                mime="application/zip"
            )

    # --- Information section in sidebar ---
    st.sidebar.markdown("---")
    st.sidebar.header("ℹ️ About This App")
    st.sidebar.markdown("""
    This application utilizes a **Conditional Generative Adversarial Network (cGAN)**
    trained on the MNIST dataset to generate synthetic handwritten digits.

    **Key Features:**
    - **Select a Digit:** Choose any digit from 0 to 9.
    - **Generate Samples:** The model generates 5 unique, MNIST-like images of the selected digit.
    - **Download:** Download all generated images as a ZIP archive.

    **Technical Details:**
    - **Framework:** PyTorch for the model, Streamlit for the web interface.
    - **Dataset:** MNIST (28x28 grayscale images).
    - **Model Type:** Conditional GAN (cGAN) allows generation of specific classes/digits.
    - **Training Environment:** Designed to be trained on Google Colab with a T4 GPU.
    """)

if __name__ == "__main__":
    main()

