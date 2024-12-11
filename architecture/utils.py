def convert_models_to_fp32(model):
    for p in model.parameters():
        p.dataset = p.dataset.float()
    return model
