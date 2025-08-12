from PIL import Image
from transformers import AutoModelForImageSegmentation
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

def get_prob_mask(image: Image, device):
    model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4", trust_remote_code = True)
    model.to(device)

    image = np.array(image)
    image_size = image.shape[:2] # (H, W)
    image_tensor = torch.tensor(image, dtype = torch.float32)
    image_tensor = image_tensor.permute(2, 0, 1) # convert (H, W, C) to (C, H, W)
    image_tensor = F.interpolate(torch.unsqueeze(image_tensor, 0), size = image_size, mode = 'bilinear') # make sure the shape is (N, C, H, W)

    image = torch.divide(image_tensor, 255.0) # convert (0, 255) to (0, 1)
    image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]).to(device)
    
    with torch.no_grad():
        res = model(image)[0][0]
        res = res.squeeze().cpu()

    # get the probability mask ~ (0, 1)
    res_min = torch.min(res)
    res_max = torch.max(res)
    res = (res - res_min) / (res_max - res_min)

    return res

def apply_mask(image, mask):
    image_np = np.array(image)
    image_np = image_np / image_np.max() # (0, 1)

    image_np = np.expand_dims(mask, axis=-1) * image_np
    image_np[mask < mask.mean()] = 1

    image = (image_np / image_np.max() * 255).astype(np.uint8)
    res = Image.fromarray(image)
    return res
