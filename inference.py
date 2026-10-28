import torch
import numpy as np
import pickle
import os
from PIL import Image
from tqdm import tqdm
import data_utils
from torch.utils.data import DataLoader
from models.gen_model import PreferenceConditionedDiffusion
from models.preference_model import SASRec,DIN
import argparse
import logging
data_utils.setup_logger()
logger = logging.getLogger(__name__)





def generate_images_with_preference(args, test_data, diffusion_model):
    """
    Generate images using trained preference model
    
    Args:
        args: Configuration parameters
        test_data: Test data
        preference_model_path: Path to trained preference model
        image_features: Image features
        
    """
    # Prepare data
    gen_test = data_utils.IPAdapterDataset(args,test_data)
    # Use fixed random seed to ensure reproducible shuffle results
    gen_dataloader = DataLoader(gen_test, batch_size=1, shuffle=True, drop_last=False, num_workers=4, 
                               generator=torch.Generator().manual_seed(args.random_seed),collate_fn=data_utils.custom_collate_fn)
    
    print("🚀 Starting image generation based on preference model...")
    


    with open(os.path.join(args.data_path, "image_path.pkl"), "rb") as file:
        all_image_paths = pickle.load(file)
    # Create output directory
    output_dir = args.save_folder
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    
    print(f"📊 Start processing {len(gen_test)} test samples...")

    # only_save_data = True
    # save_name = 'cold_test_data.pkl'
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(gen_dataloader, desc="Generating images", unit="batch")):
            user_batch = batch['uids']#.clone()
            
            image_prompt = batch['prompt']
            
            batch_size = user_batch.shape[0]
            
            
                
            generated_images_batch = diffusion_model.generate_image(
                prompt=image_prompt,
                sparse_iids=batch['sparse_iids'],
                target_iid=batch['iids'],
                inference_steps=30,
                guidance_scale=7.5,
            )
            
            for idx in range(batch_size):
                
                uid = batch['uids'][idx].item()
                iid = batch['iids'][idx].item()
                
                # Get generated image for current sample
                
                generated_image = generated_images_batch[idx]
                
                # Get target image size and resize generated image
                target_img_path = os.path.join(args.data_path, 'poster', all_image_paths[iid])
                if os.path.exists(target_img_path):
                    target_image = Image.open(target_img_path)
                    target_width, target_height = target_image.size
                    # Resize generated image to target image size
                    generated_image = generated_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
                # if args.scene == 'pog':
                #     generated_image = extract_foreground(generated_image)
                # 4. Save generated image
                user_dir = os.path.join(output_dir, str(uid))
                os.makedirs(user_dir, exist_ok=True)
                
                # Save generated image
                generated_path = os.path.join(user_dir, f"{iid}_generated.jpg")
                #print("generated_path",generated_path)
                generated_image.save(generated_path)
                
                
               
                
                target_img_path = os.path.join(args.data_path, 'poster', all_image_paths[iid])
                    
                # Record results
                if uid not in all_results:
                    all_results[uid] = {}
                
                all_results[uid][iid] = {
                    'gen_image_path': generated_path,
                    'target_image_path': target_img_path,
                    'his_iids': batch['his_iids'][idx].tolist(),
                }
    # Save results


    results_path = os.path.join(output_dir, 'all_res.npy')
    np.save(results_path, all_results)
    print(f"🎉 Image generation completed!")
    print(f"   Total generated: {sum(len(v) for v in all_results.values())} images")
    print(f"   Output directory: {output_dir}")
    print(f"   Results file: {results_path}")
def parse_args():
    parser = argparse.ArgumentParser(description="Configuration for model and training parameters")
    parser.add_argument("--device", type=str, default="cpu", help="Specify the device, e.g., 'cuda' or 'cpu'")

    # Dataset configuration
    parser.add_argument("--retrival_images", action='store_true', help="Whether to use retrival images")
    parser.add_argument("--data_name", type=str)
    parser.add_argument("--data_path", type=str, default="movielens_v1", help="Path to the dataset")
    parser.add_argument("--save_folder", type=str, default="results", help="Folder to save results")
    parser.add_argument("--item_num", type=int, default=1000, help="Number of items")

    # Pretrained Model paths 
    parser.add_argument("--sdxl_path", type=str, default="stabilityai/stable-diffusion-xl-base-1.0", help="Path to SDXL model (local dir or HF hub id)")
    parser.add_argument("--xl_vae_path", type=str, default="madebyollin/sdxl-vae-fp16-fix", help="Path to the fp16 SDXL VAE (local dir or HF hub id)")
    parser.add_argument("--diffusion_clip_path",type =str,default ="openai/clip-vit-large-patch14", help="CLIP image encoder used for item image features")
    parser.add_argument("--clip_pretrained_path",type=str, default="", help="(unused in inference) open_clip ViT-H-14 weights, kept for eval compatibility")
    parser.add_argument("--ipadapter_path",type=str, default="h94/IP-Adapter", help="Path to IP-Adapter model (needs sdxl_models/ and models/image_encoder/)")
    parser.add_argument("--pref_model_type", type=str, default="DIN", help="Type of preference model")
    parser.add_argument("--pref_model_path", type=str, default="", help="Path to preference model checkpoint from Stage 1")
    parser.add_argument("--generation_model_path", type=str, default="", help="Path to generation model checkpoint from Stage 2")
    # Feature paths
    parser.add_argument("--image_feat_path", type=str, help="Path to image features")
    

    parser.add_argument("--hidden_dim", type=int, default=128, help="hidden dimension for preference model")
    # SASRec-CTR specific parameters (must match the values used in Stage 1 training)
    parser.add_argument("--SASRec_num_layers", type=int, default=2, help="Number of Transformer layers for SASRec")
    parser.add_argument("--SASRec_num_heads", type=int, default=4, help="Number of attention heads for SASRec")
    parser.add_argument("--SASRec_dropout", type=float, default=0.2, help="Dropout rate for SASRec")
    parser.add_argument("--max_seq_len", type=int, default=15, help="Maximum sequence length for SASRec-CTR")
    

    parser.add_argument("--top_k", type=int, default=3, help="Number of reference images retrieved per target and fed to IP-Adapter (must match Stage 2 training)")

    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for training")
    parser.add_argument("--use_wandb", action='store_true', help="Whether to use wandb for logging")
    
    parser.add_argument("--scene", type=str, default='movie',  help="dataset scene")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to checkpoint")
    args = parser.parse_args()
    return args
def load_generation_model(diffusion_model, path: str, device: str):
    """
    Load IP-Adapter parameters to generation model
    Consistent with ReFL_SDXL.load_trainable_params_only logic
    """
    checkpoint = torch.load(path, map_location=device)
    
    # Get UNet object
    unet = diffusion_model.diffusion_model.unet
    
    # Load encoder_hid_proj parameters (with safety check)
    if 'image_proj_state_dict' in checkpoint:
        if hasattr(unet, "encoder_hid_proj") and unet.encoder_hid_proj is not None:
            unet.encoder_hid_proj.load_state_dict(checkpoint['image_proj_state_dict'])
            print("✅ encoder_hid_proj parameters loaded")
        else:
            print("⚠️  Warning: UNet doesn't have encoder_hid_proj, skipping")
    
    # Load IP-Adapter related parameters (to_k_ip, to_v_ip)
    if 'ip_adapter_state_dict' in checkpoint:
        loaded_ip_params = 0
        for name, module in unet.named_modules():
            if name.endswith("to_k_ip") or name.endswith("to_v_ip"):
                if name in checkpoint['ip_adapter_state_dict']:
                    module.load_state_dict(checkpoint['ip_adapter_state_dict'][name])
                    loaded_ip_params += 1
        
        if loaded_ip_params > 0:
            print(f"✅ Loaded {loaded_ip_params} IP-Adapter parameters")
        else:
            print("⚠️  Warning: No matching IP-Adapter parameters found")
    
    print(f"✅ model {path} loaded")
def load_preference_model(args,pref_model_type,pref_model_path,image_features):
    if pref_model_type == 'SASRec':
        pref_model = SASRec(
            item_image_feature_dim=image_features.size(1),
            hidden_dim=args.hidden_dim,
            num_layers=args.SASRec_num_layers,
            num_heads=args.SASRec_num_heads,
            max_seq_len=args.max_seq_len,
            dropout=args.SASRec_dropout
        )
        logger.info(f"Created SASRec model: hidden_dim={args.hidden_dim}, "
                   f"num_layers={args.SASRec_num_layers}, num_heads={args.SASRec_num_heads}, max_seq_len={args.max_seq_len}")
    elif pref_model_type == 'DIN':
        # Create DIN model
        pref_model = DIN(
            item_image_feature_dim=image_features.size(1),
            hidden_dim=args.hidden_dim
        )
        logger.info(f"Created DIN model: hidden_dim={args.hidden_dim}")
    checkpoint = torch.load(pref_model_path, map_location=args.device)
    pref_model.load_state_dict(checkpoint['model_state_dict'])
    pref_model = pref_model.to(args.device)  # Move model to device
    pref_model.eval()  # Set to evaluation mode
    print(f"✅ Pretrained reward model loaded successfully! Model type: {args.pref_model_type}")
    return pref_model
def main():
    args = parse_args()
    data_utils.set_random_seeds(args.random_seed)
    args.device = torch.device("cuda") if torch.cuda.is_available() else  torch.device("cpu")
    os.makedirs(args.save_folder, exist_ok=True)
    image_features = data_utils.get_image_feature(args).to(args.device)
    assert image_features.shape[0]==args.item_num
    uncertain_model = load_preference_model(args,args.pref_model_type,args.pref_model_path,image_features)
    
    PerGenModel = PreferenceConditionedDiffusion(
       args
    ).to(args.device)
    load_generation_model(PerGenModel,args.generation_model_path,args.device)
    PerGenModel.set_retrival_attr(args.data_path,image_features,uncertain_model)
    _,_,test_data = data_utils.load_data(args)
    generate_images_with_preference(args, test_data, PerGenModel)
if __name__ == "__main__":
    main()


