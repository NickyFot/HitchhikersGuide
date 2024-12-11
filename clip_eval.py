import os
import sys
import datetime

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn


from transformers import AutoProcessor
from transformers import CLIPModel
from peft import PeftModel

import wandb

from config import get_config
from DataLoader import get_loaders
from DataLoader import CLASSES

from train_utils import clip_prompts
from train_utils import binary_metrics
from train_utils import clip_eval
from train_utils import clip_retrieve_words

# region global variable setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.manual_seed(1312)
torch.backends.cudnn.enabled = False

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: ", device)

# endregion

METRIC_TYPE = ["exact", "contains", "clip"]

imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]

OG_SYN = ['real', 'original', 'unaltered']
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
    processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    if cnf.model.peft_weights:
        peft_model_id = cnf.model.peft_weights
        model.vision_model = PeftModel.from_pretrained(model.vision_model, peft_model_id)
    # endregion

    if cnf.wandb.log and cnf.is_master:
        wand_run = wandb.init(project='ffVQA', notes='', config=cnf_dict, name=cnf.exp_name)
        binary_table = wandb.Table(columns=["prompt", "metric", "binary"])
        columns = ["prompt", "metric"]
        columns.extend(CLASSES)
        fg_table = wandb.Table(columns=columns)
        sample_table = wandb.Table(columns=['video_id', 'ground_truth', 'pred'])
    if cnf.DDP:
        #dist.barrier()
        model = model.to(device)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(
            model,
            device_ids=[cnf.local_rank],
            output_device=cnf.local_rank
        )
    model.eval()
    prompt_template = list()
    for synonym in OG_SYN:
        prompt_template.extend([p.format(synonym + ' face') for p in imagenet_templates])
    for synonym in ['manipulated', 'synthetic', 'altered']: #cnf.prompts.q1.synonyms:
        prompt_template.extend([p.format(synonym + ' face') for p in imagenet_templates])
    prompt_logits = clip_prompts(prompt_template, processor, model).reshape(len(imagenet_templates)*len(OG_SYN), 2, -1).mean(0)
    test_table = clip_eval(prompt_logits, test_loader, processor, model, len(CLASSES), cnf.is_master)
    if cnf.wandb.log and cnf.is_master:
        for i in range(10):
            sample_table.add_data(*test_table[['video_id', 'ground_truth', 'predictions']].iloc[i].tolist())
        test_table, bmets = binary_metrics(
            test_table,
            classes=['original'],
            synonym=synonym,
            binary_pred=torch.tensor(test_table['predictions'].tolist()).reshape(-1)
        )
        metrics = ["accuracy", "auc", "f1"]
        for i, met in enumerate(metrics):
            row = [synonym, met, bmets[i]]
            binary_table.add_data(*row)
        wand_run.log({"clip_words": clip_retrieve_words(test_table, model, processor)})
        wand_run.log({"binary_evaluation": binary_table})
        wand_run.log({"sample": sample_table})
        test_table['original'] = test_table['original'].astype(str)
        wand_run.log({"embeddings": test_table[['original', 'embeddings']]})
    if cnf.DDP:
        dist.destroy_process_group()
