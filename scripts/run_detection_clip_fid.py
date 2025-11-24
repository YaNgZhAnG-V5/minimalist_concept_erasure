import glob
import os

import torch
from alive_progress import alive_it
from PIL import Image
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
)

import argparse
from diffsolver.utils import calculate_clip_score, get_clip_encoders

DINO_MODEL_ID = "IDEA-Research/grounding-dino-base"
LLAVA_MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"


def get_conversation(concept):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"Does the image contain {concept} or in {concept} style?, give me Yes or No"},
            ],
        },
    ]


def get_parser():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Early stop mask finder")
    parser.add_argument(
        "--clip_backbone", type=str, default="ViT-L-14", help="CLIP model type. Available options: ViT-B-16, ViT-L-14"
    )
    parser.add_argument(
        "--clip_pretrained",
        type=str,
        default="datacomp_xl_s13b_b90k",
        help="Pretrained CLIP model (e.g., datacomp_xl_s13b_b90k)",
    )
    parser.add_argument(
        "--early_stop_threshold", type=float, default=0.5, help="Threshold for CLIP score to trigger early stopping"
    )
    parser.add_argument("--concept", "-c", type=str, default="Hulk", help="Concept for the prompt")
    parser.add_argument("--device", "-d", type=str, default="cuda:0", help="Device to run the model")
    parser.add_argument("--seed", type=int, default=48, help="random seed")
    parser.add_argument("--save_dir", type=str, default="./images/ip_hulk/ip/flux_unlearn")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose mode")
    parser.add_argument("--style", "-s", type=str, default="ip", help="ip or art")
    return parser.parse_args()


def concept_detection(processor, model, image, text, device):
    if isinstance(image, str):  # make it path object to avoid errors
        image = Image.open(image)
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, box_threshold=0.1, text_threshold=0.1, target_sizes=[image.size[::-1]]
    )
    if len(results[0]["scores"]) == 0:
        return 0
    else:
        score = results[0]["scores"].detach().cpu()[0].item()
        return score


def art_detection(processor, model, image, text, device):
    if isinstance(image, str):  # make it path object to avoid errors
        image = Image.open(image)
    prompt = processor.apply_chat_template(get_conversation(text), add_generation_prompt=True)
    inputs = processor(image, prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=100)
    output_text = processor.decode(output[0], skip_special_tokens=True)
    output_text = output_text[len(prompt) - 5 :]

    print(output_text)

    if "Yes" in output_text:
        return 1
    elif "No" in output_text:
        return 0
    else:
        raise ValueError("The model should return either yes or no")


def main(args):
    assert args.style in ["ip", "art"], "style should be either ip or art"
    if args.style == "ip":
        print("loading ground dino model")
        # load groudingdino detection
        processor = AutoProcessor.from_pretrained(DINO_MODEL_ID)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL_ID).to(args.device)
    elif args.style == "art":
        print("loading llava model")
        # load llava model
        processor = LlavaNextProcessor.from_pretrained(LLAVA_MODEL_ID)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            LLAVA_MODEL_ID, torch_dtype=torch.float16, low_cpu_mem_usage=True
        ).to(args.device)

    print("loading CLIP model")
    # load CLIP model
    clip_dict = get_clip_encoders(backbone=args.clip_backbone, pretrained=args.clip_pretrained)

    # define metrics
    clip_score, accuracy = 0, 0

    # get all the images from the directory
    image_path_list = glob.glob(os.path.join(args.save_dir, "*.png"))

    print("start evaluation with, Accuracy, and CLIP score")
    for idx, p in enumerate(alive_it(image_path_list)):
        score = calculate_clip_score(
            clip_dict=clip_dict, image=p, text=args.concept, device=args.device, single_word=False, keepmodel=True
        )
        if args.style == "art":
            detection_score = art_detection(
                processor=processor, model=model, image=p, text=args.concept, device=args.device
            )
        elif args.style == "ip":
            detection_score = concept_detection(
                processor=processor, model=model, image=p, text=args.concept, device=args.device
            )
        else:
            raise ValueError("style should be either ip or art")
        clip_score += score
        accuracy += detection_score
        if args.verbose:
            print(f"Image: {os.path.basename(p)}, CLIP score: {score}, Detection score: {detection_score}")

    mean_clip_score = clip_score / len(image_path_list)
    print(f"CLIP score: {mean_clip_score}")
    mean_accuracy = accuracy / len(image_path_list)
    print(f"Accuracy: {mean_accuracy}")

    # save the results in the direction
    with open(os.path.join(args.save_dir, "results.txt"), "w") as f:
        f.write(f"CLIP score: {mean_clip_score}")
        f.write(f"Accuracy: {mean_accuracy}")


if __name__ == "__main__":
    main(get_parser())
