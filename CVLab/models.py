import torch
import torchvision

class Base_Class(torch.nn.Module):
    
    def __init__(self, norm: str='batch', activation: str='relu'):
        super().__init__()
        self.norm = norm
        self.activation = activation
        
        if self.norm == 'batch':
            self._get_norm_layer = torch.nn.BatchNorm2d
        elif self.norm == "layer":
            self._get_norm_layer = torch.nn.LayerNorm
        else:
            raise ValueError("Invalid norm type.")
        
        if self.activation == 'relu':
            self._get_activation = torch.nn.ReLU
        elif self.activation == 'leakyrelu':
            self._get_activation = torch.nn.LeakyReLU
        else:
            raise ValueError("Invalid activation type.")

class VGG16D_Block(Base_Class):
    
    def __init__(self, input_dim: int, output_dim: int, layers: int, norm: str='batch', activation: str='relu', conv_down: bool=False):
        super().__init__(norm, activation)
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.layers = layers
        self.conv_down = conv_down
        
        architecture = [
            torch.nn.Conv2d(self.input_dim, self.output_dim, kernel_size=3, padding=1),
            self._get_norm_layer(self.output_dim),
            self._get_activation()
        ]
        for _ in range(1, layers):
            architecture.append(torch.nn.Conv2d(self.output_dim, self.output_dim, kernel_size=3, padding=1))
            architecture.append(self._get_norm_layer(self.output_dim))
            architecture.append(self._get_activation())
        
        self.block = torch.nn.Sequential(*architecture)
        
        if self.conv_down:
            self.down_sample = torch.nn.Conv2d(self.output_dim, self.output_dim, kernel_size=2, stride=2)
        else:
            self.down_sample = torch.nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        before_sample = self.block(x)
        x = self.down_sample(before_sample)
        return before_sample, x

class VGG16D(Base_Class):
    
    def __init__(self, input_dim: int, norm: str='batch', activation: str='relu', conv_down: bool=False):
        super().__init__(norm, activation)
        self.input_dim = input_dim
        self.output_dim = 512
        self.conv_down = conv_down
        
        self.blocks = torch.nn.ModuleList([
            VGG16D_Block(self.input_dim, 64, 2, self.norm, self.activation, self.conv_down),
            VGG16D_Block(64, 128, 2, self.norm, self.activation, self.conv_down),
            VGG16D_Block(128, 256, 3, self.norm, self.activation, self.conv_down),
            VGG16D_Block(256, 512, 3, self.norm, self.activation, self.conv_down),
            VGG16D_Block(512, 1024, 3, self.norm, self.activation, self.conv_down)
        ])
    
    def forward(self, x):
        encoder_stacks = []
        for block in self.blocks:
            before_sample, x = block(x)
            encoder_stacks.append(before_sample)
        return encoder_stacks, x

class UNet3plus(Base_Class):
    
    def __init__(self, input_dim: int, output_dim: int, norm: str='batch', activation: str='relu', **kwargs):
        super().__init__(norm, activation)
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.kwargs = kwargs
        # Keyword Arguments: conv_down: bool=False
        
        # Hardcoded VGG16D encoder
        self.encoder = VGG16D(self.input_dim, self.norm, self.activation, self.kwargs.get('conv_down', False))
        self.num_encoder_blocks = len(self.encoder.blocks)
        
        # Skip Convolutions
        self.skip_convolutions = torch.nn.ModuleList([
            torch.nn.Conv2d(i, 64, kernel_size=3, padding=1) for i in [block.output_dim for block in self.encoder.blocks]
        ])
        
        self.decoder = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(64 * len(self.encoder.blocks), 64, kernel_size=3, padding=1),
                self._get_norm_layer(64),
                self._get_activation()
            )
            for _ in range(self.num_encoder_blocks - 1)
        ])
        
        self.final_conv = torch.nn.Conv2d(64, self.output_dim, kernel_size=1)
        
        # Deep supervision probably does not make a lot of sense for this task
        # if self.kwargs.get('deep_supervision', False):
        #     self.deep_supervision = torch.nn.ModuleList([
        #         torch.nn.Dropout2d(0.5),
        #         torch.nn.Conv2d(self.encoder.blocks[-1].output_dim, 1, kernel_size=1),
        #         torch.nn.AdaptiveAvgPool2d(1),
        #         torch.nn.Sigmoid()
        #     ])
    
    def resize(self, x: torch.Tensor, scale: float) -> torch.Tensor:
        """
        Resize the input tensor by a given scale.
        """
        # TODO: Maybe use different way of interpolation, maybe use maxpooling
        return torch.nn.functional.interpolate(x, scale_factor=scale)
    
    def forward(self, x):
        encoder_stacks, x = self.encoder(x)
        
        skip_stacks = []
        for encoder_stack, skip_conv in zip(encoder_stacks, self.skip_convolutions):
            skip_stacks.append(skip_conv(encoder_stack))
        
        # Hardcoded for VGG16D
        decoder_stack_1 = self.decoder[0](
            torch.cat([
                self.resize(skip_stacks[0], 1./8),
                self.resize(skip_stacks[1], 1./4),
                self.resize(skip_stacks[2], 1./2),
                skip_stacks[3],
                self.resize(skip_stacks[-1], 2)
            ], dim=1)
        )
        
        decoder_stack_2 = self.decoder[1](
            torch.cat([
                self.resize(skip_stacks[0], 1./4),
                self.resize(skip_stacks[1], 1./2),
                skip_stacks[2],
                self.resize(decoder_stack_1, 2),
                self.resize(skip_stacks[-1], 4)
            ], dim=1)
        )
        
        decoder_stack_3 = self.decoder[2](
            torch.cat([
                self.resize(skip_stacks[0], 1./2),
                skip_stacks[1],
                self.resize(decoder_stack_2, 2),
                self.resize(decoder_stack_1, 4),
                self.resize(skip_stacks[-1], 8)
            ], dim=1)
        )
        
        decoder_stack_4 = self.decoder[3](
            torch.cat([
                skip_stacks[0],
                self.resize(decoder_stack_3, 2),
                self.resize(decoder_stack_2, 4),
                self.resize(decoder_stack_1, 8),
                self.resize(skip_stacks[-1], 16)
            ], dim=1)
        )
        
        x = self.final_conv(decoder_stack_4)
                
        return x
    
    def init_encoder(self):
        """
        Initialize the encoder part of the network.
        """
        if self.kwargs.get('conv_down', False):
            raise NotImplementedError("Convolutional downsampling for loading of pretrained weights not implemented yet.")
        
        vgg16 = torchvision.models.vgg16(weights="DEFAULT")
        vgg16_conv_modules = [module for module in vgg16.modules() if isinstance(module, torch.nn.Conv2d)]
        encoder_conv_modules = [module for module in self.encoder.modules() if isinstance(module, torch.nn.Conv2d)]
        for i, (vgg16_pretrained_conv, encoder_conv) in enumerate(zip(vgg16_conv_modules, encoder_conv_modules)):
            if vgg16_pretrained_conv.weight.data.shape == encoder_conv.weight.data.shape:
                encoder_conv.weight.data = vgg16_pretrained_conv.weight.data
                encoder_conv.bias.data = vgg16_pretrained_conv.bias.data
            else:
                print(f"First {i} convolutions of encoder initialized with pretrained weights.")
                break
        
        # Comparing loaded weights and initialized model weights 
        test_init = [torch.allclose(vgg16_conv_modules[j].weight.data, [module.weight.data for module in self.encoder.modules() if isinstance(module, torch.nn.Conv2d)][j]) for j in range(i)]
        if all(test_init):
            print("Test passed.")
        else:
            print("Test failed.")

class GModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, kernel_size=3, padding=1)

    def forward(self, image: torch.Tensor, temp: torch.Tensor) -> torch.Tensor:
        temp = temp.view(-1, 1, 1, 1)
        temp = temp.expand(-1, 1, image.shape[2], image.shape[3])
        forward = torch.concat([image, temp], dim=1)
        forward = self.conv(forward)
        return forward

class GUNET3plus(torch.nn.Module):

    def __init__(self, input_dim: int, output_dim: int, norm: str='batch', activation: str='relu', **kwargs):
        super().__init__()
        self.unet = UNet3plus(input_dim, output_dim, norm, activation, **kwargs)
        self.GM = GModule()
        self.informed_conv = torch.nn.Conv2d(5, 1, kernel_size=3, padding=1)
    
    def forward(self, image: torch.Tensor, temp: torch.Tensor) -> torch.Tensor:
        unet_out = self.unet(image)
        gm_out = self.GM(image, temp)
        informed_conv_out = self.informed_conv(torch.concat([unet_out, gm_out], dim=1))
        return informed_conv_out



















from typing import Tuple, Optional

class UNetBlock(torch.nn.Module):
    """
    Implementation of a basic U-Net block. Intended for use in plain UNet.
    """

    def __init__(self, input_features: int, output_features: int, down: bool=True, use_batchnorm: bool=True,
                 conv_down: bool=False, **kwargs):
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.down = down
        self.use_batchnorm = use_batchnorm
        self.conv_down = conv_down
        
        if self.down:
            self.conv1 = torch.nn.Conv2d(input_features, output_features, kernel_size=3, padding=1, **kwargs)
            self.conv2 = torch.nn.Conv2d(output_features, output_features, kernel_size=3, padding=1, **kwargs)
        else:
            self.conv1 = torch.nn.Conv2d(input_features, output_features, kernel_size=3, padding=1, **kwargs)
            self.conv2 = torch.nn.Conv2d(output_features, output_features, kernel_size=3, padding=1, **kwargs)

        if self.use_batchnorm:
            if self.down:
                self.bn1 = torch.nn.BatchNorm2d(output_features)
                self.bn2 = torch.nn.BatchNorm2d(output_features)
            else:
                self.bn1 = torch.nn.BatchNorm2d(output_features)
                self.bn2 = torch.nn.BatchNorm2d(output_features)
                self.bn3 = torch.nn.BatchNorm2d(output_features)

        if self.down:
            if self.conv_down:
                self.sample = torch.nn.Conv2d(output_features, output_features, kernel_size=2, stride=2)
            else:
                self.sample = torch.nn.MaxPool2d(2, stride=2)
        else:
            # doubles the spatial size
            self.sample = torch.nn.ConvTranspose2d(input_features, output_features, kernel_size=2, stride=2)
    
    def forward(self, x: torch.Tensor, side: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self.down:
            x = self.sample(x)
            if self.use_batchnorm:
                x = self.bn3(x)
            torch.nn.ReLU(True)(x)
            x = torch.cat([x, side], dim=1)
        x = self.conv1(x)
        if self.use_batchnorm:
            x = self.bn1(x)
        torch.nn.functional.relu(x, True)
        x = self.conv2(x)
        if self.use_batchnorm:
            x = self.bn2(x)
        # Save the input for the skip connection
        if self.down: # down, always relu
            torch.nn.functional.relu(x, True)
            before_sample = x
            x = self.sample(x)
            return x, before_sample
        else:
            return x

class UNet(torch.nn.Module):
    """
    Plain U-Net implementation with added optional BatchNorm.
    """

    def __init__(self, input_features: int=3, num_classes: int=1, use_batchnorm: bool=True,
                 use_dropout: bool=False, stages: int=4, conv_down: bool=False):
        super().__init__()
        self.input_features = input_features
        self.output_features = num_classes
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        self.stages = stages
        self.num_classes = num_classes

        # Encoder
        down_block = []
        stage_sizes = [self.input_features] + [2**i for i in range(5, 5 + self.stages)]
        for i in range(len(stage_sizes) - 1):
            down_block.append(UNetBlock(stage_sizes[i], stage_sizes[i + 1], down=True, use_batchnorm=self.use_batchnorm,
                                        conv_down=conv_down))
        self.down_block = torch.nn.ModuleList(down_block)

        # Final layer
        self.final_block = torch.nn.Sequential(
            torch.nn.Conv2d(stage_sizes[-1], stage_sizes[-1] * 2, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(stage_sizes[-1] * 2),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(stage_sizes[-1] * 2, stage_sizes[-1] * 2, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(stage_sizes[-1] * 2),
            torch.nn.ReLU(True)
        )

        # Decoder
        up_block = []
        up_stages = list(reversed(stage_sizes[1:] + [stage_sizes[-1] * 2]))
        for i in range(len(up_stages) - 1):
            up_block.append(UNetBlock(up_stages[i], up_stages[i + 1], down=False, use_batchnorm=self.use_batchnorm))
        self.up_block = torch.nn.ModuleList(up_block)

        self.final_conv = torch.nn.Conv2d(stage_sizes[1], self.num_classes, kernel_size=3, padding=1)
        if self.use_batchnorm:
            self.bn_final = torch.nn.BatchNorm2d(stage_sizes[1])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        skip_cache = []
        for block in self.down_block:
            x, before_sample = block(x, None)
            skip_cache.append(before_sample)

        # Final layer
        x = self.final_block(x)

        # Decoder
        for i, block in enumerate(self.up_block):
            x = block(x, skip_cache[-i - 1])
            # we dont want relu on the last layer
            torch.nn.functional.relu(x, True)
        if self.use_batchnorm:
            x = self.bn_final(x)
        x = self.final_conv(x)
        return x