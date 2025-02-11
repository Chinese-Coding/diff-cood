import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

model_path = "/home/zfq/fleet/diff-cood/src/tests/原版_text_to_image/sd-naruto-model-2025-2-09"
unet = UNet2DConditionModel.from_pretrained(model_path + "/checkpoint-15000/unet", torch_dtype=torch.float16)
unet=unet,
pipe = StableDiffusionPipeline.from_pretrained("/home/zfq/Desktop/stable-diffusion-2-1-base", unet=unet, torch_dtype=torch.float16)
pipe.to("cuda")

# image = pipe(prompt="yoda").images[0]
image = pipe(prompt="").images[0]
image.save("yoda-naruto.png")
