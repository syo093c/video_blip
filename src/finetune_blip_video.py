from PIL import Image
import sys
sys.path.append("./")
from blip_2_video import Blip2Processor, Blip2ForConditionalGeneration
import torch
from torch.utils.data import Dataset
import transformers
import yaml
import deepspeed
import fire
import pandas as pd
import numpy as np
import time


class VideoCaptionDataset(Dataset):
    """Dataset for video captioning"""

    def __init__(
            self,
            model_name: str,
            data_df: pd.DataFrame,
            data_dir: str,
            max_length: int,
            n_frames: int
    ):
        super(VideoCaptionDataset, self).__init__()
        self.data_df = data_df
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.max_length = max_length
        self.n_frames = n_frames
        self.data_dir = data_dir

    def __len__(self) -> int:
        return len(self.data_df)
    
    def preprocess_image(self, images):
        return self.processor(images=images, return_tensors="pt")["pixel_values"]
    
    def tokenize(self, text):
        return self.processor(text=text, return_tensors="pt", padding='max_length', max_length=self.max_length, truncation=True)

    def __getitem__(self, index) -> dict:
        t0 = time.time()
        row = self.data_df.iloc[index]
        answer = row["action"] + " " + row["justification"] + "."
        prompt = f"Question: Could you describe the driving image? Answer: {answer}"
        tokenized = self.tokenize(prompt)
        tokenized_prompt = tokenized["input_ids"][0]
        prompt_attn_mask = tokenized["attention_mask"][0]
        base_filename = row['img_path'][:-4].split("_")[0]
        index = int(row['img_path'][:-4].split("_")[1])
        
        t1 = time.time()
        images = []
        for i in range(self.n_frames):
            tmp_index = index - 15 * (self.n_frames - i)
            tmp_img_path = f"{self.data_dir}/images/%s_%05d.jpg" % (base_filename, tmp_index)
            tmp_image = Image.open(tmp_img_path)
            images.append(tmp_image)

        return_dict = {
            "input_ids": tokenized_prompt,
            "labels": tokenized_prompt,
            "attention_mask": prompt_attn_mask,
            "pixel_values": self.preprocess_image(images)
        }
        t2 = time.time()

        #print("dataset __getitem__", t2 - t1, t1 - t0)
        return return_dict


def train_val_splitter(scene_id: str, train_ratio: float = 0.9):
    partition = 100
    return hash(scene_id) % partition < partition * train_ratio


def main(config_file: str, model_name: str, data_dir: str):
    deepspeed.init_distributed()
    data_df = pd.read_csv(f"{data_dir}/df_v_and_l.csv")
    data_df = data_df[data_df.apply(lambda x: x["img_path"][:-4].split("_")[1] > "00100", axis=1)]
    data_df = data_df.assign(scene_id=data_df.img_path.apply(lambda x: x.split("_")[0]))
    train_mask = data_df.scene_id.apply(lambda scene_id: train_val_splitter(scene_id))
    train_df, eval_df = data_df[train_mask], data_df[~train_mask]

    train_dataset = VideoCaptionDataset(model_name, train_df, data_dir, max_length=64, n_frames=5)
    eval_dataset = VideoCaptionDataset(model_name, eval_df, data_dir, max_length=64, n_frames=5)
    model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16)
    
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    training_args = transformers.TrainingArguments(**config["training"])
    print("training_args", training_args)

    for name, param in model.named_parameters():
        if "vision_model" in name:
            param.requires_grad = False
        if "language_model" in name:
            param.requires_grad = False

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    with torch.autocast("cuda"):
        result = trainer.train()


if __name__=="__main__":
    fire.Fire(main)
