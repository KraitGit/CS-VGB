import torch
from torchvision.transforms import Normalize

from facer.face_attribute.farl import FaRLFaceAttribute, pretrain_settings
from facer.farl.classification import (
    FACTransformer, VITClassificationHeadV0, VisualTransformer,
    add_method, get_clip_encode_func,
)


class AttributeNetwork(FACTransformer):
    def __init__(self, setting):
        # Upstream constructors force CUDA; keep their architecture and forward.
        torch.nn.Module.__init__(self)
        self.backbone = VisualTransformer(
            input_resolution=224, patch_size=16, width=768,
            layers=12, heads=12, output_dim=512,
        )
        del self.backbone.proj
        del self.backbone.ln_post
        add_method(self.backbone, "forward", get_clip_encode_func(setting["layers"]))
        self.head = VITClassificationHeadV0(
            num_features=len(setting["layers"]), channel=768,
            num_labels=setting["num_classes"], norm=True,
        )


class FaceAttribute(FaRLFaceAttribute):
    def __init__(self, checkpoint, device):
        torch.nn.Module.__init__(self)
        self.conf_name = "celeba/224"
        setting = pretrain_settings[self.conf_name]
        self.labels = setting["classes"]
        self.net = AttributeNetwork(setting)
        self.net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        self.net.to(device)
        self.normalize = Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        )
        self.eval()
