import cv2
import itertools
import math
import numpy as np
from os.path import join
import torch
import torchvision.transforms as tvf

from .main import VPRModel


# Code from https://github.com/jacobgil/vit-explain/
def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = 0.7 * heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


class MixVPRFeatureExtractor:

    def __init__(self, root, content, pipeline=False):
        self.device = "cuda" if content["cuda"] else "cpu"
        self.saved_state, self.agg_dim = self.load_model(root, content, pipeline)
        self.img_transform = tvf.Resize(self.saved_state["img_size"], antialias=None, interpolation=tvf.InterpolationMode.BICUBIC)
    
    def load_model(self, root, content, pipeline):
        # Load weights at specified path, add # of epochs trained, best recall rate, and input image size if missing
        saved_state = torch.load(join(root, join(content["ckpt_path"], "model_best.pth") if pipeline
                                     else content["ckpt_path"]), map_location=self.device)
        if saved_state.keys() != {"epoch", "best_score", "state_dict", "img_size"}:
            saved_state = {"epoch": 0, "best_score": 0, "state_dict": saved_state, "img_size": content["pt_img_size"]}
        # Remove module prefix from state dict
        state_dict_keys = list(saved_state["state_dict"].keys())
        for state_key in state_dict_keys:
            if state_key.startswith("module"):
                new_key = state_key.removeprefix("module.")
                saved_state["state_dict"][new_key] = saved_state["state_dict"][state_key]
                del saved_state["state_dict"][state_key]
        img_size = saved_state["img_size"] if "img_size" not in content else content["img_size"]
        ignore_agg = img_size[0] != saved_state["img_size"][0] or img_size[1] != saved_state["img_size"][1]

        # Initialize the VPR model with the desired image resolution
        self.model = VPRModel(backbone_arch='resnet50',
                        layers_to_crop=[4],
                        agg_arch='MixVPR',
                        agg_config={'in_channels': 1024,
                                    'in_h': math.ceil(img_size[0] / 16),
                                    'in_w': math.ceil(img_size[1] / 16),
                                    'out_channels': 1024,
                                    'mix_depth': 4,
                                    'mlp_ratio': 1,
                                    'out_rows': 4},
                        )

        # If desired image resolution differs from training resolution, discard aggregator weights
        if ignore_agg:
            agg_parts = [weight_name for weight_name in saved_state["state_dict"] if "aggregator" in weight_name]
            for weight_name in agg_parts:
                del saved_state["state_dict"][weight_name]
            missing_keys, unexpected_keys = self.model.load_state_dict(saved_state["state_dict"], strict=False)
            if (agg_set:=set(agg_key.removeprefix("_orig_mod") for agg_key in agg_parts)) != (missing_set:=set(missing_keys)) or len(unexpected_keys) > 0:
                raise RuntimeError(f"MixVPR state dict missing keys: {missing_set.difference(agg_set)}\nUnexpected keys: {unexpected_keys}")
        else:
            self.model.load_state_dict(saved_state["state_dict"])
        self.model.eval().to(self.device)
        print(f"Loaded MixVPR model for input images at {img_size} "
              f"{f'''(Originally {saved_state['img_size']})''' if ignore_agg else ''} from {content['ckpt_path']} Successfully!")
        saved_state["img_size"] = img_size
        return saved_state, self.model.agg_config["out_rows"] * self.model.agg_config["out_channels"]
    
    def __call__(self, images):
        encodings, descriptors = self.model(self.img_transform(images))
        return encodings, descriptors
    
    def generate_heatmap(self, image, input_tensor, num_mask=30):
        resized = self.img_transform(input_tensor).to(self.device)
        _, unmasked_desc = self.model(resized)
        height_bounds = torch.linspace(0, resized.shape[-2], num_mask + 1, dtype=torch.int)
        width_bounds = torch.linspace(0, resized.shape[-1], num_mask + 1, dtype=torch.int)
        masked_imgs = torch.empty((num_mask ** 2, *resized.shape[1:])).to(self.device)
        for img_ind, (h_ind, w_ind) in enumerate(itertools.product(range(num_mask), repeat=2)):
            masked_imgs[img_ind] = resized[0]
            top_bound, bot_bound = height_bounds[h_ind], height_bounds[h_ind + 1]
            left_bound, right_bound = width_bounds[w_ind], width_bounds[w_ind + 1]
            masked_imgs[img_ind, :, top_bound : bot_bound, left_bound : right_bound] = 0
        masked_descs = torch.empty((masked_imgs.shape[0], self.feature_length)).to(self.device)
        for i in range(0, masked_imgs.shape[0], 200):
            _, descs = self.model(masked_imgs[i: min(i + 200, masked_imgs.shape[0])])
            masked_descs[i: min(i + 200, masked_imgs.shape[0])] = descs
        diff_with_unmasked = torch.nn.PairwiseDistance()(unmasked_desc.expand(masked_descs.shape), masked_descs)
        diff_grid = diff_with_unmasked.reshape(num_mask, num_mask).cpu().numpy()
        activations = diff_grid / np.max(diff_grid)
        activation_base = image.resize(self.saved_state["img_size"])
        np_img = np.array(activation_base)[:, :, ::-1]
        mask = cv2.resize(activations, (np_img.shape[1], np_img.shape[0]))
        activations_map = show_mask_on_image(np_img, mask)
        return activations_map
    
    def set_train(self, is_train):
        self.model.train(is_train)
    
    def torch_compile(self, **compile_args):
        self.model = torch.compile(self.model, **compile_args)
    
    def set_parallel(self):
        self.model = torch.nn.DataParallel(self.model)
    
    def set_float32(self):
        self.model.to(torch.float32)
    
    def save_state(self, save_path, new_state):
        new_state["state_dict"] = self.model.state_dict()
        new_state["img_size"] = self.saved_state["img_size"]
        torch.save(new_state, save_path)
    
    @property
    def last_epoch(self): return self.saved_state["epoch"]

    @property
    def best_score(self): return self.saved_state["best_score"]

    @property
    def parameters(self): return self.model.parameters()

    @property
    def feature_length(self): return self.agg_dim
