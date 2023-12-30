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