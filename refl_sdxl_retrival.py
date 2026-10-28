'''
@File       :   ReFL_SDXL.py
@Description:   ReFL Algorithm using StableDiffusionXLPipeline.
* Based on diffusers code base
* https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_sdxl.py
'''

"""Fine-tuning script for Stable Diffusion XL for text2image."""

from models import ReFL_SDXL
import torch    
if __name__ == "__main__":
    print("start training ReFL_SDXL")
    args = ReFL_SDXL.parse_args()
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    trainer = ReFL_SDXL.Trainer(args=args) 
    trainer.train(args=args)
    
    