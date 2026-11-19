"""Offline full-resolution SDR metrics, using fixed original LPIPS weights.

LPIPS v0.1 AlexNet formula follows richzhang/PerceptualSimilarity's lpips.py,
pretrained_networks.py and normalize_tensor. Functional PyTorch evaluation
avoids a second torchvision installation and performs no training/autograd.
"""
import hashlib
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class QualityMetrics:
    def __init__(self, public_weights):
        import torch
        root = Path(public_weights)
        backbone = root/"torch/hub/checkpoints/alexnet-owt-7be5be79.pth"
        calibration = root/"lpips/0.1.4/alex-v0.1.pth"
        expected = ["7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02",
                    "df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0"]
        self.identities = []
        for path, identity in zip([backbone, calibration], expected):
            actual = sha256(path)
            if actual != identity:
                raise ValueError("LPIPS original weight identity differs: "+str(path))
            self.identities.append({"path": str(path), "sha256": actual})
        state = torch.load(backbone, map_location="cpu", weights_only=True)
        self.weights = {key: value for key,value in state.items() if key.startswith("features.")}
        self.calibration = torch.load(calibration, map_location="cpu", weights_only=True)
        self.shift = torch.tensor([-.030,-.088,-.188]).reshape(1,3,1,1)
        self.scale = torch.tensor([.458,.448,.450]).reshape(1,3,1,1)

    def features(self, image):
        import torch
        import torch.nn.functional as f
        x = torch.from_numpy(image).permute(2,0,1).unsqueeze(0).float()
        x = (2*x-1-self.shift)/self.scale
        output = []
        for stage,(index,stride,padding) in enumerate([(0,4,2),(3,1,2),(6,1,1),(8,1,1),(10,1,1)]):
            if stage in (1,2):
                x = f.max_pool2d(x,3,2)
            x = f.relu(f.conv2d(x,self.weights[f"features.{index}.weight"],
                               self.weights[f"features.{index}.bias"],stride=stride,padding=padding))
            output.append(x/(torch.sqrt(torch.sum(x**2,dim=1,keepdim=True))+1e-10))
        return output

    def pair(self, image, reference):
        import numpy as np
        import torch
        import torch.nn.functional as f
        if image.shape != reference.shape or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("metrics require equal native HWC RGB shapes")
        a,b = image.astype(np.float64),reference.astype(np.float64)
        mse = np.mean((a-b)**2)
        coordinate=np.arange(-5,6,dtype=np.float64)
        gaussian=np.exp(-.5*(coordinate/1.5)**2);gaussian/=gaussian.sum()
        horizontal=torch.from_numpy(gaussian.copy()).reshape(1,1,1,11).repeat(3,1,1,1)
        vertical=horizontal.transpose(2,3).contiguous()
        def blur(x):
            image=torch.from_numpy(np.ascontiguousarray(x.transpose(2,0,1))).unsqueeze(0)
            with torch.inference_mode():
                out=f.conv2d(f.conv2d(image,horizontal,groups=3),vertical,groups=3)
            return out[0].permute(1,2,0).numpy()
        # Valid separable filtering is the standard Gaussian11 result after
        # discarding its five-pixel boundary, with no SciPy dependency.
        ua,ub = blur(a),blur(b)
        va,vb,cov = blur(a*a)-ua*ua,blur(b*b)-ub*ub,blur(a*b)-ua*ub
        score = ((2*ua*ub+.01**2)*(2*cov+.03**2))/((ua*ua+ub*ub+.01**2)*(va+vb+.03**2))
        ssim = float(score.mean())
        with torch.inference_mode():
            left,right = self.features(np.ascontiguousarray(image,dtype=np.float32)),self.features(np.ascontiguousarray(reference,dtype=np.float32))
            total = 0
            for index,(x,y) in enumerate(zip(left,right)):
                total = total+f.conv2d((x-y)**2,self.calibration[f"lin{index}.model.1.weight"]).mean()
        return {"psnr_db": float(-10*np.log10(mse)) if mse else None,
                "numerically_identical": bool(mse==0),"ssim":ssim,"lpips_alex_v01":float(total),"mse":float(mse)}
