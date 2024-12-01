from main import VPRModel

# Note that these models have been trained with images resized to 320x320
# Also, either use BILINEAR or BICUBIC interpolation when resizing.
# The model with 4096-dim output has been trained with images resized with bicubic interpolation
# The model with 8192-dim output with bilinear interpolation
# ConvAP works with all image sizes, but best performance can be achieved when resizing to the training resolution

model = VPRModel(backbone_arch='resnet50', 
                 layers_to_crop=[],
                 agg_arch='ConvAP',
                 agg_config={'in_channels': 2048,
                            'out_channels': 1024,
                            's1' : 2,
                            's2' : 2},
                )


state_dict = torch.load('./LOGS/resnet50_ConvAP_1024_2x2.ckpt')
model.load_state_dict(state_dict)
model.eval()