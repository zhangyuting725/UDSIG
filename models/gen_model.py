

import torch
import torch.nn as nn
import torch.nn.functional as F
import data_utils
from transformers import CLIPVisionModelWithProjection
from diffusers import DiffusionPipeline
import numpy as np
import pickle
import os
from PIL import Image
class PreferenceConditionedDiffusion(nn.Module):

    def __init__(self,  args):#uncertain_model=None,data_path=None,image_features=None):
        super(PreferenceConditionedDiffusion, self).__init__()
        self.device = args.device
        
        self.weight_dtype = torch.float16
        self.top_k = args.top_k

        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            args.ipadapter_path,
            subfolder="models/image_encoder",
            torch_dtype=self.weight_dtype,
        )
        
        self.diffusion_model = DiffusionPipeline.from_pretrained(
                args.sdxl_path,
            torch_dtype=self.weight_dtype, use_safetensors=True,image_encoder=image_encoder).to(self.device)
       
        # Load IP-Adapter (always enabled by default)
        self.diffusion_model.load_ip_adapter(
                args.ipadapter_path,
                subfolder="sdxl_models",
                weight_name=["ip-adapter-plus_sdxl_vit-h.safetensors"]
            )
        self.diffusion_model.set_ip_adapter_scale([0.7])

        # Disable progress bar
        self.diffusion_model.set_progress_bar_config(disable=True)
        
        self.generator = torch.Generator(device=self.device).manual_seed(0)
        
        # Get diffusion model components
        self.vae = self.diffusion_model.vae
        self.unet = self.diffusion_model.unet

        #Text encoders
        self.tokenizer = self.diffusion_model.tokenizer
        self.text_encoder = self.diffusion_model.text_encoder
        self.text_encoder_2 = self.diffusion_model.text_encoder_2
        self.tokenizer_2 = self.diffusion_model.tokenizer_2

        # Freeze diffusion model parameters
        for param in self.vae.parameters():
            param.requires_grad_(False)
        for param in self.unet.parameters():
            param.requires_grad_(False)
        for param in self.text_encoder.parameters():
            param.requires_grad_(False)
        for param in self.text_encoder_2.parameters():
            param.requires_grad_(False)


    
   
        

    
       
    

    def uncertain_retrival(self,his_iids,target_iid,process_image =True):
        '''
        Get semantically similar images + low uncertainty interaction images based on target_iid.

        Thin wrapper around the shared implementation in data_utils, so that inference and
        training (models/ReFL_SDXL.py) always rank candidates identically.
        '''
        return data_utils.uncertainty_aware_retrieval(
            self.near_images, self.near_images_sim, self.image_features, self.uncertain_model,
            self.image_paths, self.data_path, his_iids, target_iid, self.top_k, self.device,
            process_image=process_image,
        )
    def set_retrival_attr(self,data_path,image_features,uncertain_model):
        '''
        Set up various components needed for retrieval, such as uncertain_model, preprocessed near_images, near_images_sim, image_paths, etc.
        '''
        self.image_features = image_features
        self.uncertain_model = uncertain_model
        
        self.data_path = data_path
        self.near_images = np.load(os.path.join(self.data_path, 'near_images.npy'))
        self.near_images_sim = np.load(os.path.join(self.data_path, 'near_images_sim.npy'))
        print("load_near_images_success")
        with open(os.path.join(self.data_path, "image_path.pkl"), "rb") as file:
            self.image_paths = pickle.load(file)
        print("load_retrival_attr_success!")
    def generate_image(self, prompt, sparse_iids, target_iid, inference_steps=30, guidance_scale=5.0):
        """
        Generate a personalized image for one (user history, target item) pair.

        The reference images fed to IP-Adapter are NOT the user's history images: they are
        produced by uncertainty-aware retrieval (self.uncertain_retrival), which selects the
        top-k semantic neighbours of target_iid with the lowest interaction uncertainty.
        """
        with torch.no_grad():
            # Inference runs with batch_size=1 (see inference.py), so uncertain_retrival returns
            # exactly one list of top-k PIL images, for the single prompt being generated.
            uncertain_retrival_images,_,_,_ = self.uncertain_retrival(sparse_iids,target_iid)
            result =self.diffusion_model(
                prompt=prompt,
                ip_adapter_image=[uncertain_retrival_images[0]], 
                negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
                generator=self.generator,
                num_inference_steps=inference_steps,
                guidance_scale=guidance_scale,
            ).images
        return result