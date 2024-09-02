from torchvision.models import resnet18
from torch import nn

class LagendaAge(nn.Module):
    def __init__(
        self,
        timm_backbone: str = "resnet18",
        embed_dim: int = 1280,
        num_classes_age: int = 15,
        num_classes_gender: int = 1,
        bias: bool = True,
        train_backbone: bool = False,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()

        if timm_backbone == "resnet18":
            self.backbone = resnet18(weights="IMAGENET1K_V1", progress=True)
            self.backbone.fc = nn.Identity()

        self.backbone = self.set_parameter_requires_grad(self.backbone, train_backbone)

        self.fc_age = nn.Linear(
            embed_dim, num_classes_age, bias=bias
        )

    def forward(self, x_face):
        x = self.backbone(x_face)
        age_output = self.fc_age(x)
        return age_output

    def set_parameter_requires_grad(self, model, grad=False):
        for param in model.parameters():
            param.requires_grad = grad
        return model
