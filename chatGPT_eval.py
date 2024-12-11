import os
import sys
import pandas as pd
import torch
from torchvision import transforms
from tqdm import tqdm
import wandb

import base64
from io import BytesIO
import requests
import openai
from openai import OpenAI
import time
from retry import retry
import copy

from config import get_config
from DataLoader import get_loaders
from DataLoader import CLASSES

from train_utils import binary_metrics

METRIC_TYPE = ["exact", "contains", "clip"]
SYNONYMS = ["manipulated", "synthetic", "altered"]

api_key = "YOUR_KEY"
openai.api_key = api_key
os.environ['OPENAI_API_KEY'] = api_key

headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}"
}

@retry(Exception, tries=5, delay=1)
def chatgpt(payload):
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response


def encode_image(pil):
    buffered = BytesIO()
    pil.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def api_call(loader, transforms, prompt, iterations=None):
    video_id = list()
    gt_lst = list()
    desc_lst = list()
    for batch_idx, (vid, inputs, target) in enumerate(tqdm(loader)):
        inputs = transforms(inputs[0])
        base64_image = encode_image(inputs)
        row = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text":  "USER: <image>\nIs this image {}? a) Yes b) No \nASSISTANT:".format(prompt)
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }
        response = chatgpt(row)
        if "choices" not in response.json():
            time.sleep(5)
            response = chatgpt(row)
        try:
            description = response.json()["choices"][0]["message"]["content"]
            desc_lst.append(description)
            video_id.extend(vid)
            gt_lst.append(target)
        except Exception as e:
            print(e)
            pass
        if iterations:
            if batch_idx >= iterations:
                break
    gt_lst = torch.cat(gt_lst)
    results_df = pd.DataFrame({
        'ground_truth': gt_lst.numpy().tolist(),
        'prompts': [prompt for _ in range(len(video_id))],
        'descriptions': desc_lst,
        'rationale': ['' for _ in range(len(video_id))],
        'vqa_followup': ['' for _ in range(len(video_id))]
    })
    return results_df


if __name__ == "__main__":
    cnf = get_config(sys.argv)
    cnf_dict = vars(cnf)

    ROOT_FOLDER = os.path.join(cnf.wandb.log_dir, 'checkpoints')
    EXP_FOLDER = os.path.join(ROOT_FOLDER, cnf.exp_name)
    PREDS_FOLDER = os.path.join(EXP_FOLDER, 'preds')
    if not os.path.exists(PREDS_FOLDER):
        os.makedirs(PREDS_FOLDER)

    cnf.training.batch_size = 1
    _, test_loader = get_loaders(cnf)
    test_transform = transforms.Compose([transforms.Resize(cnf.image.input_shape)])
    client = OpenAI()
    if cnf.wandb.log:
        wand_run = wandb.init(project='ffVQA', notes='', config=cnf_dict, name=cnf.exp_name)
        binary_table = wandb.Table(columns=["prompt", "metric", "metric_type", "binary"])
        columns = ["prompt", "metric", "metric_type"]
        wb_predictions = wandb.Artifact(name='predictions', type='folder')
    for synonym in SYNONYMS:
        test_table = api_call(
            loader=test_loader,
            transforms=test_transform,
            prompt=synonym,
            iterations=10
        )
        if cnf.wandb.log:
            test_table.to_csv(
                os.path.join(
                    PREDS_FOLDER,
                    '{}_{}_{}_all.csv'.format(cnf.dataset.dataset_name, synonym, cnf.model.architecture)
                )
            )
            wb_predictions.add_file(
                local_path=os.path.join(
                    PREDS_FOLDER,
                    '{}_{}_{}_all.csv'.format(cnf.dataset.dataset_name, synonym, cnf.model.architecture)
                )
            )
            for i, mtype in enumerate(METRIC_TYPE[:2]):
                test_table, bmets = binary_metrics(test_table, CLASSES, synonym=synonym, strict=(mtype == "exact"))
                metrics = ["accuracy", "auc", "recall", "f1"]
                for i, met in enumerate(metrics):
                    row = [prompt_template, met, mtype, bmets[i]]
                    binary_table.add_data(*row)
    if cnf.wandb.log:
        wand_run.log({"binary_evaluation": binary_table})
        wandb.log_artifact(wb_predictions)
