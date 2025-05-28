import os

import numpy as np
import torch

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

torch.cuda.empty_cache()

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

model.eval()

# Load and preprocess example images (replace with your own image paths)
image_names = []
for img in os.listdir("images"):
    if img.endswith((".jpg", ".jpeg", ".png")):
        image_names.append(os.path.join("images", img))

images = load_and_preprocess_images(image_names).to(device)

with torch.no_grad():
    with torch.amp.autocast("cuda", dtype=dtype):
        # Predict attributes including cameras, depth maps, and point maps.
        predictions = model(images)


def convert_vggt_to_sonata(predictions, confidence_threshold=0.5):
    import numpy as np
    import torch

    world_points = predictions["world_points"]

    coords = world_points[0, 0].reshape(-1, 3).cpu().numpy()
    conf = predictions["world_points_conf"][0, 0].reshape(-1).cpu().numpy()
    valid_mask = conf > confidence_threshold
    coords = coords[valid_mask]

    # Normals
    if "world_points_normal" in predictions:
        normals = predictions["world_points_normal"][0, 0].reshape(-1, 3).cpu().numpy()
        normals = normals[valid_mask]
    else:
        normals = np.zeros_like(coords)

    sonata_dict = {"coord": coords, "normal": normals}

    # Color
    if "images" in predictions:
        img = predictions["images"][0, 0].permute(1, 2, 0).cpu().numpy().reshape(-1, 3)
        sonata_dict["color"] = img[valid_mask]

    # Convert all to torch tensors
    for k, v in sonata_dict.items():
        if isinstance(v, np.ndarray):
            sonata_dict[k] = torch.from_numpy(v)

    print(coords.shape)

    return sonata_dict


# Usage example:
sonata_data = convert_vggt_to_sonata(predictions)
torch.save(sonata_data, "predictions_sonata.pt")

print(sonata_data.keys())
print("Sonata formatted predictions saved to predictions_sonata.pt")
