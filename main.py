import os
import sys
import pandas as pd
import datetime

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn

from transformers import AutoTokenizer
from transformers import CLIPTextModel

import wandb

from config import get_config
from DataLoader import get_loaders
from DataLoader import CLASSES
from DataLoader import CLASS_SYNONYMS
from architecture import get_model

from train_utils import eval_model
from train_utils import binary_metrics
from train_utils import finegrained_metrics
from train_utils import clip_distance
from train_utils import bert_score

# region global variable setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.manual_seed(1312)
torch.backends.cudnn.enabled = False

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: ", device)

# endregion

METRIC_TYPE = ["exact", "contains", "clip"]

if __name__ == "__main__":
    cnf = get_config(sys.argv)

    ROOT_FOLDER = os.path.join(cnf.wandb.log_dir, 'checkpoints')
    EXP_FOLDER = os.path.join(ROOT_FOLDER, cnf.exp_name)
    MODELS_FOLDER = os.path.join(EXP_FOLDER, 'models')
    PREDS_FOLDER = os.path.join(EXP_FOLDER, 'preds')
    cnf_dict = vars(cnf)
    # region ddp
    if cnf.DDP:
        cnf.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(cnf.local_rank)
        cnf.is_master = cnf.local_rank == 0
        cnf.device = torch.cuda.device(cnf.local_rank)
        cnf.world_size = int(os.environ['WORLD_SIZE'])
        os.environ['NCCL_BLOCKING_WAIT'] = '0'
        dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=7200))
        df_lst = ["" for _ in range(cnf.world_size)]
        #if 'iterations' in cnf.training.iterations:
        #    cnf.training.iterations = int(cnf.training.iterations // cnf.world_size)
    else:
        os.environ['WORLD_SIZE'] = "1"
        df_lst = [""]
        cnf.local_rank = 0
        cnf.is_master = True
    # endregion

    # region dir set_up
    if cnf.is_master:
        if not os.path.exists(MODELS_FOLDER):
            os.makedirs(MODELS_FOLDER)
        if not os.path.exists(PREDS_FOLDER):
            os.makedirs(PREDS_FOLDER)
    # endregion

    _, test_loader = get_loaders(cnf)
    model, processor = get_model(cnf.model)
    # region CLIP eval
    clip_eval = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32", torch_dtype=torch.float16)
    clip_eval.eval()
    clip_token = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    # endregion

    if cnf.wandb.log and cnf.is_master:
        wand_run = wandb.init(project='ffVQA', notes='', config=cnf_dict, name=cnf.exp_name)
        binary_table = wandb.Table(columns=["prompt", "metric", "metric_type", "binary"])
        columns = ["prompt", "metric", "metric_type"]
        bert_table = wandb.Table(columns=["synonym", "metric", "score"])
        columns.extend(CLASSES)
        fg_table = wandb.Table(columns=columns)
        vqa_table = wandb.Table(columns=columns)
        sample_table = wandb.Table(columns=['video_id', 'ground_truth', 'prediction', 'prompts', 'descriptions', 'rationale', 'vqa_followup'])

    if cnf.DDP:
        #dist.barrier()
        model = model.to(device)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(
            model,
            device_ids=[cnf.local_rank],
            output_device=cnf.local_rank
        )
    for synonym in cnf.prompts.q1.synonyms:
        model = model.to(device)
        prompt_template = cnf.prompts.q1.template.format(synonym)
        followup_template = cnf.prompts.q2.format(synonym) if not cnf.dataset.binary else None
        vqa_template = cnf.prompts.q3.format(', '.join(CLASSES), synonym) if not cnf.dataset.binary else None
        test_table = eval_model(
            cnf,
            loader=test_loader,
            model=model,
            prompt=prompt_template,
            followup=followup_template,
            vqa=vqa_template,
            processor=processor,
            num_classes=len(CLASSES),
            master=cnf.is_master
        )
        model = model.cpu()
        if cnf.wandb.log and cnf.is_master:
            test_table.to_csv(
                os.path.join(
                    PREDS_FOLDER,
                    '{}_{}_{}_all.csv'.format(cnf.dataset.dataset_name, synonym, cnf.model.architecture)
                )
            )
            for i, mtype in enumerate(METRIC_TYPE[:2]):
                test_table, bmets = binary_metrics(test_table, CLASSES, synonym=synonym, strict=(mtype=="exact"))
                metrics = ["accuracy", "auc", "recall", "f1"]
                for i, met in enumerate(metrics):
                    row = [prompt_template, met, mtype, bmets[i]]
                    binary_table.add_data(*row)
                for i in range(10):
                    sample_table.add_data(*test_table.iloc[i][['video_id', 'original', 'prediction', 'prompts', 'descriptions', 'rationale', 'vqa_followup']].tolist())

        if cnf.wandb.log and cnf.is_master and not cnf.dataset.binary:
            bert_mets = bert_score(test_table, CLASSES, synonym=synonym)
            clip_eval = clip_eval.to(device)
            clip_mets = clip_distance(test_table, CLASS_SYNONYMS, CLASSES,clip_eval, clip_token)
            clip_eval = clip_eval.cpu()
            fmets = finegrained_metrics(test_table, CLASSES, CLASS_SYNONYMS)
            test_table["rationale"] = test_table["vqa_followup"]
            vqa_mets = finegrained_metrics(test_table, CLASSES, CLASS_SYNONYMS)
            metrics = ["precision", "auc", "recall", "f1"]
            for i, met in enumerate(metrics):
                row = [synonym, met, bert_mets[i]]
                bert_table.add_data(*row)
                row = [followup_template, met, "contains"]
                row.extend(fmets[i])
                fg_table.add_data(*row)
                row = [followup_template, met, "clip"]
                row.extend(clip_mets[i])
                fg_table.add_data(*row)
                row = [vqa_template, met, "contains"]
                row.extend(vqa_mets[i])
                vqa_table.add_data(*row)

    if cnf.wandb.log and cnf.is_master:
        wand_run.log({"bert_score": bert_table})
        wand_run.log({"binary_evaluation": binary_table})
        wand_run.log({"finegraned_evaluation": fg_table})
        wand_run.log({"vqa_evaluation": vqa_table})
        wand_run.log({"sample": sample_table})
        wb_predictions = wandb.Artifact(name='predictions', type='folder')
        for synonym in cnf.prompts.q1.synonyms:
            wb_predictions.add_file(
                local_path=os.path.join(
                    PREDS_FOLDER,
                    '{}_{}_{}_all.csv'.format(cnf.dataset.dataset_name, synonym, cnf.model.architecture)
                )
            )
        wandb.log_artifact(wb_predictions)
    if cnf.DDP:
        dist.destroy_process_group()
