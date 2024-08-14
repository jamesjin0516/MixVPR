from os.path import join
import torch
import torchvision.transforms as tvf

from .main import VPRModel

class MixVPRFeatureExtractor:

    def __init__(self, root, content, pipeline=False):
        self.in_h, self.in_w = [size // 16 for size in content["resized_img_size"]]
        self.device = "cuda" if content["cuda"] else "cpu"
        self.saved_state, self.agg_dim = self.load_model(root, content, pipeline)
        self.img_transform = tvf.Resize((320, 320), interpolation=tvf.InterpolationMode.BICUBIC)
    
    def load_model(self, root, content, pipeline):
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

        saved_state = torch.load(join(root, join(content["ckpt_path"], "model_best.pth") if pipeline
                                     else content["ckpt_path"]), map_location=self.device)
        if saved_state.keys() != {"epoch", "best_score", "state_dict"}:
            saved_state = {"epoch": 0, "best_score": 0, "state_dict": saved_state}
        self.model.load_state_dict(saved_state["state_dict"])

        self.model.eval().to(self.device)
        print(f"Loaded MixVPR model from {content['ckpt_path']} Successfully!")
        return saved_state, self.model.agg_config["out_rows"] * self.model.agg_config["out_channels"]
    
    def __call__(self, images):
        encodings, descriptors = self.model(self.img_transform(images))
        return encodings, descriptors
    
    def set_train(self, is_train):
        self.model.train(is_train)
    
    def torch_compile(self, float32=False, **compile_args):
        self.model = torch.compile(self.model, **compile_args)
        if float32:
            self.model.to(torch.float32)
    
    def save_state(self, save_path, new_state):
        new_state["state_dict"] = self.model.state_dict()
        torch.save(new_state, save_path)
    
    @property
    def last_epoch(self): return self.saved_state["epoch"]

    @property
    def best_score(self): return self.saved_state["best_score"]

    @property
    def parameters(self): return self.model.parameters()

    @property
    def feature_length(self): return self.agg_dim
