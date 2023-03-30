import os
import random
from stable_diffusion_videos import StableDiffusionWalkPipeline

from diffusers.models import AutoencoderKL
from diffusers.schedulers import LMSDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers import UNet2DConditionModel
import torch
from sys import argv
from loguru import logger
import time
from accelerate import Accelerator


# usage: python make_music_video.py <model_path> <mp3_path>


def make_video(model_path="runwayml/stable-diffusion-v1-5",
               audio_filepath="music/thoughts.mp3",
               output_dir="/home/ling/Dropbox/AIDraw-Photos/sd_videos"):
    start_time = time.monotonic()

    gradient_accumulation_steps = 1
    torch.backends.cuda.matmul.allow_tf32 = True
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision="fp16",
        log_with=None,
        logging_dir="./logs",
    )

    torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32

    torch.backends.cudnn.benchmark = True
    logger.info(f"Loading pretrained unet: {model_path}")
    revision = "fp16"
    unet = UNet2DConditionModel.from_pretrained(
        model_path,
        subfolder="unet",
        revision=revision,
        torch_dtype=torch.float16,
    )

    logger.info(f"Done loading pretrained unet: {time.monotonic() - start_time:.2f}s")

    pipe = StableDiffusionWalkPipeline.from_pretrained(
        model_path,
        unet=accelerator.unwrap_model(unet),
        vae=AutoencoderKL.from_pretrained(
            model_path,
            subfolder="vae",
            revision=revision,
            torch_dtype=torch_dtype
        ),
        torch_dtype=torch.float16,
        revision="fp16",
        safety_checker=None,
        # scheduler=LMSDiscreteScheduler(
        #     beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
        # )
    )

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.to("cuda")

    logger.info(f"Done initializing pipeline: {time.monotonic() - start_time:.2f}s")

    # I give you permission to scrape this song :)
    # youtube-dl -f bestaudio --extract-audio --audio-format mp3 --audio-quality 0 -o "music/thoughts.%(ext)s" https://soundcloud.com/nateraw/thoughts

    # Seconds in the song. Here we slice the audio from 0:07-0:16
    # Should be same length as prompts/seeds.
    audio_offsets = [7, 10, 13, 20, 25, 30, 35, 37, 40]

    # Output video frames per second.
    # Use lower values for testing (5 or 10), higher values for better quality (30 or 60)
    fps = 30

    prompts = [
        "night sky",
        "cute looking picture of olis person, with enlarged eyes and a big smile, highly detailed, by Polina Shchuklina, watercolor, watercolor pencils",
        "cyberpunk portrait of olis person as a cyborg, diffuse lighting, fantasy, intricate, elegant, highly detailed, lifelike, photorealistic, digital painting, artstation, illustration, concept art, smooth, sharp focus, art by john collier and albert aublet and krenz cushart and skunkyfly and alphonse mucha",
        "Closeup portrait of olis person as a Hobbit, small, big brown eyes, green and brown clothing, detailed facial features, small feet, wispy hair, fantasy concept art, artstation trending, highly detailed, art by John Howe, Alan Lee, and Weta Workshop, earthy colors, looking into camera.",
        "a line drawing of a olis person, in the style of victorian line art, black and white only",
        "a painting of a olis person wearing sunglasses, a pointillism painting by LeRoy Neiman, reddit contest winner, funk art, detailed painting, acrylic art, made of beads and yarn",
        "a painting of a olis person, in the style of van gogh",
        "sunset in the style of van gogh",
        "night sky",
    ]

    # Convert seconds to frames
    # This array should be `len(prompts) - 1` as its steps between prompts.
    num_interpolation_steps = [(b - a) * fps for a, b in zip(audio_offsets, audio_offsets[1:])]

    # prompts = [
    #     'Baroque oil painting anime key visual concept art of wanderer above the sea of fog 1 8 1 8 with anime maid, brutalist, dark fantasy, rule of thirds golden ratio, fake detail, trending pixiv fanbox, acrylic palette knife, style of makoto shinkai studio ghibli genshin impact jamie wyeth james gilleard greg rutkowski chiho aoshima',
    #     'the conscious mind entering the dark wood window into the surreal subconscious dream mind, majestic, dreamlike, surrealist, trending on artstation, by gustavo dore ',
    #     'Chinese :: by martine johanna and simon stålenhag and chie yoshii and casey weldon and wlop :: ornate, dynamic, particulate, rich colors, intricate, elegant, highly detailed, centered, artstation, smooth, sharp focus, octane render, 3d',
    #     'Chinese :: by martine johanna and simon stålenhag and chie yoshii and casey weldon and wlop :: ornate, dynamic, particulate, rich colors, intricate, elegant, highly detailed, centered, artstation, smooth, sharp focus, octane render, 3d',
    # ]

    if len(prompts) != len(audio_offsets):
        raise ValueError("len(prompts) != len(audio_offsets)")

    negative_prompt = "stock photo,((((ugly)))), (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers, mutated hands, ((distorted face)), (((mutation))), (((deformed))), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), (fused fingers), (too many fingers), (((long neck))), (framed), bad composition, contorted,signature,(framed),[out of frame],bad composition"

    seeds = [
        6954010,
        8092009,
        1326004,
        5019608,
        5019608,
        5019608,
        5019608,
    ]
    seeds = [random.randint(1, 10000000) for _ in range(len(prompts))]
    video_path = pipe.walk(
        prompts=prompts,
        negative_prompt=negative_prompt,
        seeds=seeds,
        num_interpolation_steps=num_interpolation_steps,
        fps=fps,
        audio_filepath=audio_filepath,
        audio_start_sec=audio_offsets[0],
        batch_size=16,
        num_inference_steps=25,
        guidance_scale=7.5,
        margin=1.0,
        smooth=0.2,
        output_dir=output_dir,
    )
    end_time = time.monotonic()
    logger.info(f"Completed creating {audio_offsets[-1]}seconds of video in: {end_time - start_time:2f}s")
    logger.info(f"Video saved to: {video_path}")


if __name__ == "__main__":
    mp3 = "music/thoughts.mp3"
    model_path = "/mnt/5950x/4d55c3e4-6e82-4695-97ef-8fbee61414ed_output_olis/2400"
    make_video(model_path, mp3)
    if len(argv) > 1:
        model_path = argv[1]
    if len(argv) > 2:
        mp3 = argv[2]
    else:
        for sd in os.listdir("/mnt/5950x"):
            for f in os.listdir(f"/mnt/5950x/{sd}"):
                model_path = f"/mnt/5950x/{sd}/{f}"
                logger.info(f"Making video for {model_path}")
                make_video(model_path, mp3)
