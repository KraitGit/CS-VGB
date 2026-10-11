import logging

import torch



logger = logging.getLogger(__name__)


def _processor_notice_filter(record):
    return not record.getMessage().startswith((
        "Using a slow image processor as `use_fast` is unset",
        "You have video processor config saved in `preprocessor.json` file which is deprecated.",
    ))


for name in ("transformers.models.auto.image_processing_auto",
             "transformers.processing_utils", "transformers.video_processing_utils"):
    logging.getLogger(name).addFilter(_processor_notice_filter)


class VQAScorer:
    def __init__(self, model_name, device, cache_dir):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.device = torch.device(device)
        logger.info("Loading VQA processor: %s", model_name)
        self.processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir, use_fast=False)
        logger.info("Loading VQA weights: %s", model_name)
        self.model = (
            Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                cache_dir=cache_dir,
            )
            .to(self.device)
            .eval()
            .requires_grad_(False)
        )
        answer_ids = self.processor.tokenizer.encode("Yes", add_special_tokens=False)
        if len(answer_ids) != 1:
            raise ValueError("VQAScore requires Yes to be one tokenizer token")
        self.answer_id = answer_ids[0]

    @torch.inference_mode()
    def score(self, image, prompt):
        question = f'Does this figure show "{prompt}"? Please answer Yes or No.'
        message = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question},
                    ],
                }
            ]
        ]
        text = [
            self.processor.apply_chat_template(
                item, tokenize=False, add_generation_prompt=True
            )
            for item in message
        ]
        inputs = self.processor(
            text=text, images=[image], padding=True, return_tensors="pt"
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            logits = self.model(**inputs, use_cache=False).logits[:, -1].float()
        return float((logits[:, self.answer_id] - logits.logsumexp(-1)).exp()[0])


class Metric:
    def __init__(self, device, clip_iqa=True, vqa=True):
        self.iqa = self.vqa = None
        if clip_iqa:
            logger.info("Loading CLIP-IQA...")
            from torchmetrics.multimodal import CLIPImageQualityAssessment

            self.iqa = CLIPImageQualityAssessment(model_name_or_path="clip_iqa").to(device).eval()
            logger.info("CLIP-IQA ready.")
        if vqa:
            self.vqa = VQAScorer("Qwen/Qwen2.5-VL-3B-Instruct", device, None)
            logger.info("VQA ready.")

    @torch.inference_mode()
    def score(self, images, condition, reward):
        reward.set_prompt(condition)
        pixels = reward.model.decode(images)
        scores = torch.cat([reward.scorer(batch) for batch in pixels.split(reward.batch_size)])
        metrics = {"hpsv2": scores.mean().item()}
        if self.iqa is not None:
            metrics["clip_iqa"] = self.iqa(pixels).mean().item()
            self.iqa.reset()
        if self.vqa is not None:
            from PIL import Image

            rgb = pixels.mul(255).add(.5).clamp(0, 255).to(torch.uint8)
            scores = [self.vqa.score(Image.fromarray(image.permute(1, 2, 0).cpu().numpy()),
                                     condition["prompt"]) for image in rgb]
            metrics["vqa"] = sum(scores) / len(scores)
            metrics["vqa_model"] = "Qwen/Qwen2.5-VL-3B-Instruct"
        return metrics
