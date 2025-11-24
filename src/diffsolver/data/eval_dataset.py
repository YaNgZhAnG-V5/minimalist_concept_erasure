import glob
import json
import os

import gdown
import pandas as pd
import webdataset as wds
from PIL import Image

DATASET = {
    "laion": "https://drive.google.com/drive/folders/1pv_5ER2zeqooOjmUW3f0S7WTwzH0G7dt?usp=share_link",
}


class EvalDataset:
    """
    diffusion pruning dataset for semantic and quality evaluation
    including the following datasets:
    - subset of laion400m (15k image)
    - MS COCO 2017
    - Flickr30k

    all the datasets are download via kaggle API
    save all the downloaded dataset in the datasets folder
    for MS COCO check the following link: https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset
    for flickr30k check the following link: https://www.kaggle.com/datasets/adityajn105/flickr30k

    # no need for image and text transformation
    # provided text will be used for diffusion model to generating images

    """

    def __init__(
        self,
        data_dir="./datasets",
        dataset_name="laion",
        image_size=256,
        max_size=1000,  # max size of the dataset for saving memory
        interpolation=Image.Resampling.BICUBIC,
    ):
        assert dataset_name in ["laion", "coco", "flickr"], "dataset not supported"
        self.dataset_name = dataset_name
        self.image_size = image_size
        self.max_size = max_size
        self.interpolation = interpolation

        # load dataset in webdataset format:w
        if dataset_name == "laion":
            dataset_dir = os.path.join(data_dir, "laion400m")
            if not os.path.exists(dataset_dir):
                os.mkdir(dataset_dir)
            if len(list(os.listdir(dataset_dir))) == 0:
                url = DATASET[dataset_name]
                gdown.download_folder(url, output=dataset_dir, quiet=False)

            tar_list = glob.glob(os.path.join(dataset_dir, "*.tar"))
            self.dataset = list(
                (
                    wds.WebDataset(
                        tar_list,
                    )
                    .decode("pil")
                    .rename(image="jpg;png;jpeg", text="cls;text;txt", keep=False)
                    .to_tuple("image", "text")
                    .batched(1)
                )
            )
        elif dataset_name == "coco":
            data_dir = os.path.join(data_dir, "COCO", "coco2017")
            with open(f"{data_dir}/annotations/captions_train2017.json", "r") as f:
                data = json.load(f)
                data = data["annotations"]

            img_cap_pairs = []

            for sample in data:
                img_name = "%012d.jpg" % sample["image_id"]
                img_cap_pairs.append([img_name, sample["caption"]])

            captions = pd.DataFrame(img_cap_pairs, columns=["image", "caption"])

            # remove the duplicates
            captions = captions.drop_duplicates(subset="image", keep="first")

            print("Number of images before filtering:", len(captions))

            captions["image"] = captions["image"].apply(lambda x: f"{data_dir}/train2017/{x}")
            # remove the image with false image size
            # remove the image with size can not be divided by 8, for SDXL model
            captions["image_size"] = captions["image"].apply(lambda x: Image.open(x).size)
            captions["compatible_size"] = captions["image_size"].apply(lambda x: x[0] % 8 == 0 and x[1] % 8 == 0)
            captions = captions[captions["compatible_size"]]
            print("Number of images after filtering:", len(captions))
            self.captions = captions.iloc[: self.max_size]
        elif dataset_name == "flickr":
            dataset_dir = os.path.join(data_dir, "flickr")
            captions = pd.read_csv(os.path.join(dataset_dir, "Images", "results.csv"), delimiter="|")
            captions.columns = captions.columns.str.strip()
            # remove the duplicates
            captions = captions.drop_duplicates(subset="image_name", keep="first")
            # change the column name from image_name to image, comment to caption to make it consistent with COCO
            captions = captions.rename(columns={"image_name": "image", "comment": "caption"})
            captions = captions.dropna(subset=["comment_number"])
            # modify the image path
            captions["image"] = captions["image"].apply(lambda x: f"{dataset_dir}/Images/{x}")
            self.captions = captions.iloc[: self.max_size]

        else:
            raise ValueError("dataset not supported")

    def __len__(self):
        if self.dataset_name == "laion":
            return len(self.dataset)
        else:
            return len(self.captions)

    def __getitem__(self, idx):
        if self.dataset_name == "laion":
            sample = self.dataset[idx]
            data = {
                "image": sample[0][0],
                "text": sample[1][0],
            }
        elif self.dataset_name in ["coco", "flickr"]:
            meta = self.captions.iloc[idx]
            image = Image.open(meta["image"])
            # image = meta["image"]
            data = {
                "image": image,
                "text": meta["caption"],
            }
        else:
            raise ValueError("dataset not supported")
        return data
