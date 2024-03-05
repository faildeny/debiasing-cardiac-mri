import torch
import torch.nn as nn
import torchvision
import monai


class ResNet3D_18(nn.Sequential):
    def __init__(
        self,
        pretrained,
        in_ch,
        out_ch,
        linear_ch=512,
        seed=None,
        early_layers_learning_rate=0,
    ):
        """
        in_ch = 1 or 3
        early_layers can be 'freeze' or 'lower_lr'
        """
        super(ResNet3D_18, self).__init__()
        if seed != None:
            print(f"Seed set to {seed}")
            torch.manual_seed(seed)

        self.model = torchvision.models.video.r3d_18(pretrained=pretrained)
        if not early_layers_learning_rate:  #
            print("Freezing layers")
            for p in self.model.parameters():
                p.requires_grad = False
        elif early_layers_learning_rate:
            print(
                f"Early layers will use a learning rate of {early_layers_learning_rate}"
            )
        # Reshape
        print(f"Initializing network for {in_ch} channel input")
        if in_ch != 3:
            self.model.stem[0] = nn.Conv3d(
                in_ch,
                64,
                kernel_size=(3, 7, 7),
                stride=(1, 2, 2),
                padding=(1, 3, 3),
                bias=False,
            )
        self.model.fc = nn.Linear(linear_ch, out_ch)
        print(f"Linear layer initialized with {linear_ch} number of channels.")
        if out_ch == 1:
            self.out = nn.Sigmoid()
        else:
            self.out = nn.Softmax(dim=1)
        super(ResNet3D_18, self).__init__(self.model, self.out)


def get_model(config):
    n_classes = len(config["dataset"]["icd_code"])
    MODEL = config["params"]["model"]
    VIDEO = config["params"]["video"]
    VOLUME = config["params"]["volume"]
    GRAY2RGB = config["params"]["gray2rgb"]
    FINE_TUNING = config["params"]["fine_tuning"]
    EDES = config["params"]["edes"]

    if MODEL == "swin3d_t":
        model = torchvision.models.video.swin3d_t(
            weights=torchvision.models.video.Swin3D_T_Weights.KINETICS400_V1
        )
        model.head = nn.Linear(model.num_features, n_classes)
    else:
        if MODEL == "resnet3D":
            # model = models.ResNet3D_18(pretrained = True, in_ch=1, out_ch=n_classes)
            model = torchvision.models.video.r3d_18(
                weights=torchvision.models.video.R3D_18_Weights.KINETICS400_V1
            )
        elif MODEL == "resnet(2+1)D":
            model = torchvision.models.video.r2plus1d_18(
                weights=torchvision.models.video.R2Plus1D_18_Weights.KINETICS400_V1
            )

        elif MODEL == "mc3":
            model = torchvision.models.video.mc3_18(
                weights=torchvision.models.video.MC3_18_Weights.KINETICS400_V1
            )

        elif MODEL == "resnet18":
            # model = torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2")
            model = torch.hub.load(
                "pytorch/vision", "resnet18", weights="IMAGENET1K_V1"
            )
            if not GRAY2RGB:
                conv_weights = model.conv1.weight.data
                model.conv1.in_channels = 1
                model.conv1.weight = nn.Parameter(
                    torch.sum(conv_weights, dim=1, keepdim=True)
                )

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, n_classes)

    if MODEL == "densenet121":
        model = monai.networks.nets.DenseNet121(
            spatial_dims=2, in_channels=3, out_channels=n_classes, pretrained=True
        )

    return model
