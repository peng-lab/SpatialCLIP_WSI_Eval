import timm

def optimus():
    model = timm.create_model(
        "hf-hub:bioptimus/H-optimus-0", pretrained=False, init_values=1e-5, dynamic_img_size=False
        )
    return model
    