import torch
import torch.nn.functional as F


class HPSv2:
    def __init__(self, checkpoint, device):
        import open_clip

        self.device = torch.device(device)
        self.network, _, _ = open_clip.create_model_and_transforms(
            "ViT-H-14", pretrained=None, precision="amp", device=self.device,
            force_quick_gelu=False, output_dict=True,
        )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.network.load_state_dict(payload["state_dict"])
        self.network.eval().requires_grad_(False)
        self.tokenizer = open_clip.get_tokenizer("ViT-H-14")
        self.mean = torch.tensor((.48145466, .4578275, .40821073), device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor((.26862954, .26130258, .27577711), device=self.device).view(1, 3, 1, 1)
        self.text = None

    @torch.no_grad()
    def set_prompt(self, prompt):
        tokens = self.tokenizer([prompt]).to(self.device)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            self.text = self.network.encode_text(tokens, normalize=True).float()

    def __call__(self, images):
        pixels = F.interpolate(images, size=(224, 224), mode="bicubic", align_corners=False, antialias=True)
        pixels = (pixels - self.mean) / self.std
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            features = self.network.encode_image(pixels, normalize=True)
        return (features.float() @ self.text.T).squeeze(1) * 100
