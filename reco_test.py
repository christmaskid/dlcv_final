# Reference : https://huggingface.co/j-min/reco_sd14_laion
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "j-min/reco_sd14_coco",
    torch_dtype=torch.float16
)
prompt = "A cat on the <|endoftext|> <bin514> <bin575> <bin741> <bin765> <|startoftext|> and a dog on the <|endoftext|> <bin237> <bin517> <bin520> <bin784>."
pipe.to("cuda")