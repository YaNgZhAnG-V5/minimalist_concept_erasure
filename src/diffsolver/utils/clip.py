import open_clip
import torch
import torch.nn.functional as F
from open_clip.model import CLIP
from PIL import Image

SIMPLE_IMAGENET_TEMPLATES = (
    lambda c: f"itap of a {c}.",
    lambda c: f"a bad photo of the {c}.",
    lambda c: f"a origami {c}.",
    lambda c: f"a photo of the large {c}.",
    lambda c: f"a {c} in a video game.",
    lambda c: f"art of the {c}.",
    lambda c: f"a photo of the small {c}.",
)


def get_clip_encoders(
    backbone: str = "ViT-B-16",
    pretrained: str = "datacomp_xl_s13b_b90k",
    only_model: bool = False,
):
    """
    extracted the pretrained clip model, tokenizer and hyperparameters setting
    """
    model: CLIP
    try:
        model, transform, preprocess = open_clip.create_model_and_transforms(backbone, pretrained=pretrained)
    except Exception as e:
        print(f"Error {e}, select the correct pretrained model")

    # change clip image encoder hyperparameters for few-shot learning
    model = model.eval()

    tokenizer = open_clip.get_tokenizer(model_name=backbone)

    logit_scale = model.logit_scale
    clip_config = open_clip.get_model_config(backbone)

    if only_model:
        output = model.eval()
    else:
        output = {
            "clip_model": model,
            "preprocess": preprocess,
            "tokenizer": tokenizer,
            "logit_scale": logit_scale,
            "clip_config": clip_config,
            "transform": transform,
        }
    return output


@torch.no_grad()
def calculate_clip_score(clip_dict, image, text, device, single_word=False, negative_word=None, keepmodel=False):
    global SIMPLE_IMAGENET_TEMPLATES

    clip_model = clip_dict["clip_model"]
    transform = clip_dict["transform"]
    tokenizer = clip_dict["tokenizer"]

    clip_model = clip_model.to(device)
    clip_model.eval()

    if isinstance(image, list):
        raise ValueError("Image should be a single image path or a PIL image")

    # preprocess image and text
    if isinstance(image, str):
        image = transform(Image.open(image)).unsqueeze(0).to(device)
    else:
        import pdb

        pdb.set_trace()
        image = transform(image).unsqueeze(0).to(device)

    # calculate CLIP score
    if single_word:
        prompt_list = [t(text) for t in SIMPLE_IMAGENET_TEMPLATES]
        text = tokenizer(prompt_list).to(device)
        text_embed = clip_model.encode_text(text)
        text_embed = text_embed.reshape(len(SIMPLE_IMAGENET_TEMPLATES), -1).mean(dim=0)
        text_embed /= text_embed.norm(dim=-1, keepdim=True)

        if negative_word:
            prompt_list = [t(negative_word) for t in SIMPLE_IMAGENET_TEMPLATES]
            negative_text = tokenizer(prompt_list).to(device)
            negative_text_embed = clip_model.encode_text(negative_text)
            negative_text_embed = negative_text_embed.reshape(len(SIMPLE_IMAGENET_TEMPLATES), -1).mean(dim=0)
            negative_text_embed /= negative_text_embed.norm(dim=-1, keepdim=True)
    else:
        text = tokenizer(text).to(device)
        text_embed = clip_model.encode_text(text)
        text_embed /= text_embed.norm(dim=-1, keepdim=True)

    image_embed = clip_model.encode_image(image)
    image_embed /= image_embed.norm(dim=-1, keepdim=True)

    score = image_embed @ text_embed.t()

    if negative_word:
        neg_score = image_embed @ negative_text_embed.t()
        # normal score with negative score
        score = F.softmax(torch.cat([score, neg_score], dim=-1), dim=-1)[0]
    score = score.cpu().detach().item()

    # delete the unused tensor
    # release GPU memory
    del image, text, text_embed, image_embed
    if not keepmodel:
        del clip_model
    return score
