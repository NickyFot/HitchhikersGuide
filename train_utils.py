import os
import gc
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import torch.distributed as dist
from torch.cuda.amp import autocast
from torcheval.metrics import BinaryAccuracy
from torcheval.metrics import BinaryRecall
from torcheval.metrics import BinaryAUROC
from torcheval.metrics import BinaryF1Score
from torcheval.metrics import MultilabelAUPRC
from torcheval.metrics import MulticlassF1Score

from bert_score import score

device = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def eval_model(
        cnf,
        loader,
        model,
        prompt,
        processor,
        followup=None,
        vqa=None,
        num_classes: int = 1,
        master: bool = False
):
    all_vid = torch.zeros(len(loader.sampler), dtype=torch.int).to(device)
    vid_lst = [all_vid for i in range(int(os.environ['WORLD_SIZE']))]
    all_gt = torch.zeros((len(loader.sampler), num_classes), dtype=torch.int).to(device)
    gt_lst = [all_gt for i in range(int(os.environ['WORLD_SIZE']))]

    all_descriptions = torch.zeros((len(loader.sampler), 100), dtype=torch.int).to(device)
    descriptions_lst = [all_descriptions for i in range(int(os.environ['WORLD_SIZE']))]
    all_rationale = torch.zeros((len(loader.sampler), 100), dtype=torch.int).to(device)
    rationale_lst = [all_rationale for i in range(int(os.environ['WORLD_SIZE']))]
    all_answers = torch.zeros((len(loader.sampler), 100), dtype=torch.int).to(device)
    vqa_lst = [all_answers for i in range(int(os.environ['WORLD_SIZE']))]
    model.eval()
    for batch_idx, (vid, inputs, target) in enumerate(tqdm(loader)):
        vid, target = vid.to(device), target.to(device)

        start_idx = loader.batch_size * batch_idx
        end_idx = start_idx + loader.batch_size
        end_idx = end_idx if end_idx <= all_vid.shape[0] else all_vid.shape[0]

        all_vid[start_idx:end_idx], all_gt[start_idx:end_idx] = vid[:], target[:]
        p = [prompt for _ in range(len(inputs))]
        generated_text = get_output(cnf, processor, inputs, p, model)
        all_descriptions[start_idx:end_idx, :generated_text.shape[1]] = generated_text[:]
        if followup:
            follow_up = [followup for _ in range(len(inputs))]
            rationale = get_output(cnf, processor, inputs, follow_up, model)
            all_rationale[start_idx:end_idx, :rationale.shape[1]] = rationale[:]
        if vqa:
            vqa_in = [vqa for _ in range(len(inputs))]
            vqa_answer = get_output(cnf, processor, inputs, vqa_in, model)
            all_answers[start_idx:end_idx, :vqa_answer.shape[1]] = vqa_answer[:]
        gc.collect()
    dist.all_gather(vid_lst, all_vid)
    dist.all_gather(gt_lst, all_gt)
    dist.all_gather(descriptions_lst, all_descriptions); dist.all_gather(rationale_lst, all_rationale); dist.all_gather(vqa_lst, all_answers)
    vid_lst, gt_lst = torch.cat(vid_lst), torch.cat(gt_lst)
    descriptions_lst, rationale_lst, vqa_lst = torch.cat(descriptions_lst), torch.cat(rationale_lst), torch.cat(vqa_lst)
    descriptions_lst = decode_sequences(loader.batch_size, processor, descriptions_lst)
    rationale_lst = decode_sequences(loader.batch_size, processor, rationale_lst)
    vqa_lst = decode_sequences(loader.batch_size, processor, vqa_lst)
    if master:
        video_id = loader.dataset.get_img_path(vid_lst.cpu().numpy().tolist())
        results_df = pd.DataFrame({
            'video_id': video_id,
            'ground_truth': gt_lst.cpu().numpy().tolist(),
            'prompts': [prompt for _ in range(len(vid_lst))],
            'descriptions': descriptions_lst,
            'rationale': rationale_lst,
            'vqa_followup': vqa_lst
        })
        return results_df


@torch.no_grad()
def get_output(cnf, processor, inputs, prompts, model):
    with autocast(dtype=torch.bfloat16):
        if cnf.model.architecture == "CogVLM":
            inputs_by_model = list()
            for image, query in zip(inputs, prompts):
                model_in = model.module.build_conversation_input_ids(
                    processor.tokenizer,
                    query=query,
                    history=[],
                    images=[image]
                )
                inputs_by_model.append(model_in)

            def recur_move_to(item, tgt, criterion_func):
                if criterion_func(item):
                    device_copy = item.to(tgt)
                    return device_copy
                elif isinstance(item, list):
                    return [recur_move_to(v, tgt, criterion_func) for v in item]
                elif isinstance(item, tuple):
                    return tuple([recur_move_to(v, tgt, criterion_func) for v in item])
                elif isinstance(item, dict):
                    return {k: recur_move_to(v, tgt, criterion_func) for k, v in item.items()}
                else:
                    return item

            def collate_fn(features, tokenizer) -> dict:
                images = [feature.pop('images') for feature in features]
                tokenizer.padding_side = 'left'
                padded_features = tokenizer.pad(features)
                inputs = {**padded_features, 'images': images}
                return inputs

            input_batch = collate_fn(inputs_by_model, processor.tokenizer)
            inputs = recur_move_to(input_batch, device, lambda x: isinstance(x, torch.Tensor))
        else:
            inputs = processor(images=inputs, text=prompts, return_tensors='pt')
        if hasattr(model, 'module'):
            outputs = model.module.generate(
                **inputs.to(device),
                num_beams=2,
                max_new_tokens=30,
                min_length=1,
                repetition_penalty=1.5,
                early_stopping=True,
                length_penalty=1.0,
                temperature=1,
                return_dict=True,
                #pad_token_id=processor.tokenizer.pad_token_id,
                #eos_token_id=processor.tokenizer.eos_token_id,
            )
        else:
            outputs = model.generate(
                **inputs.to(device),
                num_beams=2,
                max_new_tokens=30,
                min_length=1,
                repetition_penalty=1.5,
                early_stopping=True,
                length_penalty=1.0,
                temperature=1,
                return_dict=True,
                #pad_token_id=processor.tokenizer.eos_token_id
            )
        torch.cuda.empty_cache()
        gc.collect()
        if cnf.model.architecture == "CogVLM":
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
        return outputs


def decode_sequences(batch_size, processor, response):
    text = list()
    for i in range(0, len(response), batch_size):
        end_idx = i + batch_size
        end_idx = end_idx if end_idx <= len(response) else len(response)
        generated_text = processor.batch_decode(response[i:end_idx], skip_special_tokens=True)
        generated_text = [x.split("ASSISTANT:")[-1].split("Answer:")[-1].strip().split('<s>')[0].split("</s>")[0] for x in generated_text]
        text.extend(generated_text)
    return text


def frame_acc(all_logits_class: torch.tensor, all_targets_class: torch.tensor):
    metric = BinaryAccuracy()
    metric.update(all_logits_class[:, -1].reshape(-1), all_targets_class[:, -1].reshape(-1))
    local_compute_result = metric.compute()
    return local_compute_result.cpu()


def vid_acc(all_logits_class: torch.tensor, all_targets_class: torch.tensor, vidx):
    metric = BinaryAccuracy()
    for idx in range(vidx):
        predictions = torch.where((all_logits_class[:, 0] == idx), True, False)
        gt = torch.where((all_targets_class[:, 0] == idx), True, False)
        metric.update(all_logits_class[predictions, -1].reshape(-1), all_targets_class[gt, -1].reshape(-1))
    local_compute_result = metric.compute()
    return local_compute_result.cpu()


def binary_metrics(df, classes, synonym, strict: bool = False, binary_pred: torch.tensor = None):
    df[classes] = df['ground_truth'].tolist()
    binary_gt = torch.tensor(1 - df['original'].to_numpy())
    if strict and (binary_pred is None):
        binary_pred = torch.tensor(
            df['descriptions'].apply(
                lambda x: (x.lower().strip(' ab)(.,') == 'yes') or (x.lower().strip(' )(.,') == 'a')
            ).astype(int).tolist()
        )
    elif (not strict) and (binary_pred is None):
        binary_pred = torch.tensor(
            df['descriptions'].apply(
                lambda x: ('yes' in x.lower()) or (synonym in x.lower()) or ('a' in x.lower())
            ).astype(int).tolist()
        )
    df['prediction'] = binary_pred
    binary_acc = BinaryAccuracy()
    binary_acc.update(binary_pred, binary_gt)
    binary_recall = BinaryRecall()
    binary_recall.update(binary_pred, binary_gt)
    recall = binary_recall.compute()
    acc = binary_acc.compute()
    binary_auc = BinaryAUROC()
    binary_auc.update(binary_pred, binary_gt)
    auc = binary_auc.compute()
    binary_f1 = BinaryF1Score()
    binary_f1.update(binary_pred, binary_gt)
    f1 = binary_f1.compute()
    return df, [acc, auc, recall, f1]


def bert_score(df, classes, synonym):
    finegrained_gt = df[classes].to_numpy()
    refs_gt = df[classes].apply(
        lambda x: 'The areas that are ' + synonym + 'are ' + ' '.join(
            classes[i] for i in range(len(classes)) if x[classes[i]] == 1
        )
        , axis=1
    )
    finegrained_pred = df["rationale"]
    msk = finegrained_gt[:, 0] != 1
    finegrained_gt, finegrained_pred = refs_gt[msk].tolist(), finegrained_pred[msk].tolist()
    P, R, F = score(finegrained_pred, finegrained_gt, lang="en")
    return P.mean(), torch.zeros_like(P.mean()), R.mean(), F.mean()


def finegrained_metrics(df, classes, class_synonyms, finegrained_pred: torch.tensor = None):
    finegrained_gt = torch.tensor(df[classes].to_numpy())
    if finegrained_pred is None:
        finegrained_pred = list()
        for i, cls in enumerate(class_synonyms):
            finegrained_pred.append(
                torch.tensor(df["rationale"].apply(lambda x: any([c in x.lower() for c in cls])).astype(int).tolist())
            )
        finegrained_pred = torch.stack(finegrained_pred, dim=1)
    msk = finegrained_gt[:, 0] != 1
    # msk = torch.ones(len(finegrained_gt), dtype=torch.int)
    finegrained_gt, finegrained_pred = finegrained_gt[msk], finegrained_pred[msk]
    binary_ap = MultilabelAUPRC(num_labels=len(classes), average=None)
    binary_auc = BinaryAUROC(num_tasks=len(classes))
    binary_auc.update(finegrained_pred.T, finegrained_gt.T)
    binary_ap.update(finegrained_pred, finegrained_gt)

    f1 = torch.zeros(len(classes))
    recall = torch.zeros(len(classes))
    for i in range(len(classes)):
        f1_score = BinaryF1Score()
        f1_score.update(finegrained_pred[:, i], finegrained_gt[:, i])
        f1[i] = f1_score.compute()

        bin_recall = BinaryRecall()
        bin_recall.update(finegrained_pred[:, i], finegrained_gt[:, i])
        recall[i] = bin_recall.compute()
    auc = binary_auc.compute()
    mAP = binary_ap.compute()
    return mAP.tolist(), auc.tolist(), recall.tolist(), f1.tolist()


@torch.no_grad()
def clip_distance(df, classes_synonyms, classes, clip_model, clip_token):
    C = len(classes)
    S = len(classes_synonyms[0])
    classes_synonyms = [x for xs in classes_synonyms for x in xs]
    class_tokens = clip_token(classes_synonyms, padding=True, return_tensors="pt")
    with autocast(dtype=torch.float16):
        class_embs = clip_model(**class_tokens.to(clip_model.device)).pooler_output
        class_embs = class_embs.reshape(C, S, -1).mean(1)
    finegrained_gt = torch.tensor(df[classes].to_numpy())
    binary_accuracy = MultilabelAUPRC(num_labels=C, average=None) # Average Precision
    binary_auc = BinaryAUROC(num_tasks=C)
    binary_f1 = [BinaryF1Score() for i in range(C)]
    binary_recall = [BinaryRecall() for i in range(C)]
    f1 = torch.zeros(C)
    recall = torch.zeros(C)
    finegrained_pred = df["rationale"].tolist()
    for i in range(0, len(finegrained_pred), 64):
        end_i = i+64 if i+64 <= len(finegrained_pred) else len(finegrained_pred)
        finegrained_tokens = clip_token(finegrained_pred[i: end_i], padding=True, return_tensors="pt", truncation=True).to(clip_model.device)
        with autocast(dtype=torch.float16):
            pred = clip_model(**finegrained_tokens).pooler_output
        sim = cosine_similarity(pred, class_embs, temp=1/0.5).cpu()
        binary_accuracy.update(sim, finegrained_gt[i: end_i])
        binary_auc.update(sim.T, finegrained_gt[i: end_i].T)
        for j in range(C):
            binary_f1[j].update(sim[:, j], finegrained_gt[i: end_i, j])
            binary_recall[j].update(sim[:, j], finegrained_gt[i: end_i, j])
        torch.cuda.empty_cache()
        gc.collect()
    acc = binary_accuracy.compute()
    auc = binary_auc.compute()
    for i in range(C):
        f1[i] = binary_f1[i].compute()
        recall[i] = binary_recall[i].compute()
    return acc.tolist(), auc.tolist(), recall.tolist(), f1.tolist()


def cosine_similarity(mat_a, mat_b, temp):
    mat_a = mat_a / mat_a.norm(dim=-1, keepdim=True)
    mat_b = mat_b / mat_b.norm(dim=-1, keepdim=True)
    sim = torch.einsum('bd,cd->bc', temp * mat_a, mat_b)
    return sim


@torch.no_grad()
def clip_prompts(text, processor, model):
    with autocast(dtype=torch.float16):
        text_tokens = processor.tokenizer(text=text, padding=True, truncation=True, return_tensors='pt').to(device)
        text_logits = model.module.get_text_features(**text_tokens)
    return text_logits

@torch.no_grad()
def clip_eval(
        text_logits,
        loader,
        processor,
        model,
        num_classes: int = 1,
        master: bool = False
):
    all_vid = torch.zeros(len(loader.sampler), dtype=torch.int).to(device)
    vid_lst = [all_vid for i in range(int(os.environ['WORLD_SIZE']))]
    all_gt = torch.zeros((len(loader.sampler), 1), dtype=torch.int).to(device)
    gt_lst = [all_gt for i in range(int(os.environ['WORLD_SIZE']))]
    all_pd = torch.zeros((len(loader.sampler), 1), dtype=torch.int).to(device)
    pd_lst = [all_pd for i in range(int(os.environ['WORLD_SIZE']))]
    all_emb = torch.zeros((len(loader.sampler), 768), dtype=torch.int).to(device)
    emb_lst = [all_emb for i in range(int(os.environ['WORLD_SIZE']))]

    for batch_idx, (vid, inputs, target) in enumerate(tqdm(loader)):
        vid, target = vid.to(device), target.to(device)

        start_idx = loader.batch_size * batch_idx
        end_idx = start_idx + loader.batch_size
        end_idx = end_idx if end_idx <= all_vid.shape[0] else all_vid.shape[0]
        if num_classes > 1:
            target = target[:, 0].reshape(-1, 1)
        all_vid[start_idx:end_idx], all_gt[start_idx:end_idx, :] = vid[:], target
        with autocast(dtype=torch.float16):
            images = processor.image_processor(inputs, return_tensors='pt').to(device)
            image_logits = model.module.get_image_features(**images)
            all_emb[start_idx:end_idx] = image_logits[:]
            temp = model.module.logit_scale.exp()
            logits_per_image = cosine_similarity(image_logits, text_logits, temp=temp)
            logits_per_image = logits_per_image.softmax(-1)
        all_pd[start_idx:end_idx, :] = logits_per_image.argmax(dim=-1).reshape(-1, 1)
    dist.all_gather(vid_lst, all_vid)
    dist.all_gather(gt_lst, all_gt)
    dist.all_gather(pd_lst, all_pd)
    dist.all_gather(emb_lst, all_emb)
    vid_lst, gt_lst, pd_lst = torch.cat(vid_lst), torch.cat(gt_lst), torch.cat(pd_lst)
    emb_lst = torch.cat(emb_lst)
    if master:
        video_id = loader.dataset.get_img_path(vid_lst.cpu().numpy().tolist())
        results_df = pd.DataFrame({
            'video_id': video_id,
            'ground_truth': gt_lst.cpu().numpy().tolist(),
            'predictions': pd_lst.cpu().numpy().tolist(),
            'embeddings': emb_lst.cpu().numpy().tolist()
        })
        return results_df

@torch.no_grad()
def clip_retrieve_words(
        df,
        model,
        processor,
        topk: int = 10
):
    gt = df['original']
    embs = df['embeddings']
    embs = embs.apply(lambda x: torch.tensor(x, dtype=torch.float)).tolist()
    embs = torch.stack(embs, dim=1).permute(1, 0)
    og = embs[gt == 1]
    ff = embs[gt == 0]
    og_proto = og.mean(0)
    ff_proto = ff.mean(0)
    token_embedding = model.module.text_model.embeddings.token_embedding.weight.cpu()

    # og class keywords
    #print('inputs:', og_proto.unsqueeze(0).shape, token_embedding.float().shape)
    og_distance = torch.cdist(og_proto.unsqueeze(0), token_embedding.float())
    #print('distance:', og_distance.shape)
    og_idxs = torch.argsort(og_distance, dim=1)
    #print('idx:', og_idxs.shape)
    og_idxs = og_idxs[:, :topk] # 1, B
    og_words = processor.batch_decode(og_idxs.squeeze(0))

    # ff class keywords
    ff_distance = torch.cdist(ff_proto.unsqueeze(0), token_embedding)
    ff_idxs = torch.argsort(ff_distance, dim=1)
    ff_idxs = ff_idxs[:, :topk]  # 1, B -> B
    ff_words = processor.batch_decode(ff_idxs.squeeze(0))
    #print(og_distance.shape, og_idxs.shape, og_distance[:, og_idxs.squeeze(0)].shape ) #, og_distance[og_idxs].squeeze(0).shape)
    og_df = pd.DataFrame({'original': [1 for _ in range(topk)], 'words': og_words, 'distance': og_distance[:, og_idxs.squeeze(0)].squeeze(0).numpy().tolist()})
    ff_df = pd.DataFrame({'original': [0 for _ in range(topk)], 'words': ff_words, 'distance': ff_distance[:, ff_idxs.squeeze(0)].squeeze(0).numpy().tolist()})
    return pd.concat([og_df, ff_df])

