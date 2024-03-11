from os.path import join
import torch
import torchvision.transforms as tvf

from .main import VPRModel

class MixVPRFeatureExtractor:

    def __init__(self, root, content):
        self.in_h, self.in_w = [size // 16 for size in content["resized_img_size"]]
        self.device = "cuda" if content["cuda"] else "cpu"
        self.agg_dim = self.load_model(root, content)
        self.img_transform = tvf.Resize((320, 320), interpolation=tvf.InterpolationMode.BICUBIC)
    
    def load_model(self, root, content):
        # Note that images must be resized to 320x320
        self.model = VPRModel(backbone_arch='resnet50',
                        layers_to_crop=[4],
                        agg_arch='MixVPR',
                        agg_config={'in_channels': 1024,
                                    'in_h': self.in_h,
                                    'in_w': self.in_w,
                                    'out_channels': 1024,
                                    'mix_depth': 4,
                                    'mlp_ratio': 1,
                                    'out_rows': 4},
                        )

        state_dict = torch.load(join(root, content["ckpt_path"]), map_location=self.device)
        self.model.load_state_dict(state_dict)

        self.model.eval().to(self.device)
        print(f"Loaded MixVPR model from {content['ckpt_path']} Successfully!")
        return self.model.agg_config["out_rows"] * self.model.agg_config["out_channels"]
    
    def __call__(self, images):
        with torch.no_grad():
            return self.model(self.img_transform(images)).detach().cpu()
    
    @property
    def feature_length(self): return self.agg_dim
