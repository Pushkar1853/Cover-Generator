# Install the required libraries
# !nvidia-smi
# !pip install transformers -q
# !pip install torchaudio -q
# !pip install nltk -q
# !pip install pydub -q
# !pip install diffusers==0.11.1 -q
# !pip install transformers scipy ftfy accelerate -q

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from diffusers import LDMTextToImagePipeline
import gradio as gr
import PIL.Image
import numpy as np
import random
import torch
import subprocess
from transformers import AutoModelWithLMHead, AutoModelForCausalLM, AutoTokenizer
from transformers import WhisperForConditionalGeneration, WhisperConfig, WhisperProcessor
import torchaudio
import nltk
from pydub import AudioSegment
import re
from datasets import load_dataset
from transformers import AutoModelWithLMHead, AutoTokenizer, set_seed, pipeline
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DPMSolverMultistepScheduler, LMSDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm
from torch import autocast
from PIL import Image
torch_device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_lyrics(sample):
    model_name = "openai/whisper-tiny.en"
    model_config = WhisperConfig.from_pretrained(model_name)
    processor = WhisperProcessor.from_pretrained(model_name)
    asr_model = WhisperForConditionalGeneration.from_pretrained(model_name, config=model_config)
    asr_model.eval()
    input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features 
    transcript = asr_model.generate(input_features)
    predicted_ids = asr_model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    lyrics = transcription[0]
    return lyrics

def generate_summary(lyrics):
    summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")
    summary = summarizer(lyrics)
    return summary

def generate_prompt(summary):
    # model_name = "gpt2"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelWithLMHead.from_pretrained(model_name)
    # Set up GPT-2 model and tokenizer
    model_name = 'gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    # Set the device to GPU if available
    model = model.to(device)
    # Generate prompt text using GPT-2
    prompt = f"Create an image that represents the feeling of '{summary}'"
    # Generate the image prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    output = model.generate(input_ids, do_sample=True, max_length=100, temperature=0.7)
    prompt_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return prompt_text

def generate_image(prompt,
        height = 512,                        # default height of Stable Diffusion
        width = 512 ,                        # default width of Stable Diffusion
        num_inference_steps = 50  ,          # Number of denoising steps
        guidance_scale = 7.5 ,               # Scale for classifier-free guidance
        generator = torch.manual_seed(32),   # Seed generator to create the inital latent noise
        batch_size = 1,):
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    pipe = pipe.to(torch_device)
    # 1. Load the autoencoder model which will be used to decode the latents into image space. 
    vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
    # 2. Load the tokenizer and text encoder to tokenize and encode the text. 
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    # 3. The UNet model for generating the latents.
    unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")
    scheduler = DPMSolverMultistepScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
    vae = vae.to(torch_device)
    text_encoder = text_encoder.to(torch_device)
    unet = unet.to(torch_device)
    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    latents = torch.randn((batch_size, unet.in_channels, height // 8, width // 8), generator=generator,)
    latents = latents.to(torch_device)
    scheduler.set_timesteps(num_inference_steps)
    latents = latents * scheduler.init_noise_sigma
    for t in tqdm(scheduler.timesteps):
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        # predict the noise residual
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample
    # scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    # DPM Solver Multistep scheduler
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    f_images = pil_images
    return f_images

def predict(lyrics, steps=100, seed=42, guidance_scale=6.0):

    # print(subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT).decode("utf8"))
    generator = torch.manual_seed(seed)
    summary_1 = generate_summary(lyrics)
    prompt_text_1 = generate_prompt(summary_1[0]['summary_text'])
    images = generate_image(prompt= prompt_text_1, generator= generator, num_inference_steps=steps, guidance_scale=guidance_scale)
    # images = ldm_pipeline([prompt], generator=generator, num_inference_steps=steps, eta=0.3, guidance_scale=guidance_scale)["images"]
    # print(subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT).decode("utf8"))
    return images[0]

random_seed = random.randint(0, 2147483647)
gr.Interface(
    predict,
    inputs=[
        gr.inputs.Textbox(label='Text', default='a chalk pastel drawing of a llama wearing a wizard hat'),
        gr.inputs.Slider(1, 100, label='Inference Steps', default=50, step=1),
        gr.inputs.Slider(0, 2147483647, label='Seed', default=random_seed, step=1),
        gr.inputs.Slider(1.0, 20.0, label='Guidance Scale - how much the prompt will influence the results', default=6.0, step=0.1),
    ],
    outputs=gr.Image(shape=[256,256], type="pil", elem_id="output_image"),
    css="#output_image{width: 256px}",
    title="Cover Generator (text-to-image)",
    description="Application of OpenAI tools such as Whisper, ChatGPT, and DALL-E to produce covers for the given text",
).launch()

# if __name__ == "__main__":
#     ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
#     sample = ds[0]["audio"]
#     lyrics = generate_lyrics(sample)
#     summary_1 = generate_summary(lyrics)
#     prompt_text_1 = generate_prompt(summary_1[0]['summary_text'])
#     image = generate_image(prompt= prompt_text_1)