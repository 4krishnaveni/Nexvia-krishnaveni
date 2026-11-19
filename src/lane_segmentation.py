"""Step 3b: segmentation-based road area (SegFormer pretrained on Cityscapes; id 0 = road).
Stretch goal: fine-tune on BDD100K drivable-area labels."""
import cv2, numpy as np, torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

class RoadSegmenter:
    def __init__(self, name="nvidia/segformer-b0-finetuned-cityscapes-512-1024", device=None):
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.proc = SegformerImageProcessor.from_pretrained(name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(name).to(self.dev).eval()

    @torch.no_grad()
    def __call__(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inp = self.proc(images=rgb, return_tensors="pt").to(self.dev)
        logits = self.model(**inp).logits
        logits = torch.nn.functional.interpolate(logits, size=bgr.shape[:2], mode="bilinear")
        return (logits.argmax(1)[0] == 0).cpu().numpy().astype(np.uint8) * 255

def overlay_mask(bgr, mask, colour=(255, 0, 0)):
    tint = bgr.copy(); tint[mask > 0] = colour
    return cv2.addWeighted(tint, 0.4, bgr, 0.6, 0)
