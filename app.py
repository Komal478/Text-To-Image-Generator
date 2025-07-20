import gradio as gr
import torch
from diffusers import StableDiffusionPipeline
import datetime

model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

def generate_image(prompt):
    image = pipe(prompt, guidance_scale=7.5).images[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = f"generated_image_{timestamp}.png"
    image.save(image_path)
    return image_path

gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(lines=2, placeholder="e.g. CT scan of lungs or Heart in Van Gogh style"),
    outputs=gr.Image(type="filepath"),
    title="🧠 Medical & Artistic Text-to-Image Generator",
    description="Enter a prompt and generate medical or artistic visuals using Stable Diffusion."
).launch()