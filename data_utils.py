import pickle
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import numpy as np
import torch
import random
from transformers import CLIPProcessor, CLIPVisionModelWithProjection

SEP = "\n"
IMG_TOKEN = "<img>"
EMB_TOKEN = "<emb>"

from datetime import datetime
import numpy as np
import torch
import random
import re
from torch.utils.data.dataloader import default_collate
import logging
def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    if not logger.handlers:
        logger.addHandler(console_handler)
def custom_collate_fn(batch):
    """Custom collate function to handle PIL images"""
    result = {}
    for key in batch[0].keys():
        if key in ['target_image','his_images']:
            # Return PIL images as list directly
            result[key] = [item[key] for item in batch]
        else:
            # Let official default_collate handle other fields
            result[key] = default_collate([item[key] for item in batch])
    return result

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path, patience=5, verbose=False, delta=0,mode = 'max'):
        """
        Args:
            save_path : Model save folder
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        
        self.mode = mode
        self.counter = 0
        self.best_score = -np.inf
        self.early_stop = False
        
        self.delta = delta
        
    def __call__(self, score, model):

        
        if self.mode == 'min':
            score = -score
        if score <= self.best_score + self.delta: #means not yet reached the best result
            
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            
            self.save_checkpoint(score, model)
            self.best_score = score
            self.counter = 0
        
    def save_checkpoint(self, score, model):
        '''Saves model when validation loss decrease.'''
        print(f'Validation score improved ({self.best_score:.6f} --> {score:.6f}).  Saving model ...')
        
        path = self.save_path
        #torch.save(model, path)
        torch.save(model.state_dict(), path)	# This will store the best model parameters so far

def now_time():
    # Get current time
    now = datetime.now()

    return now.strftime("%Y-%m-%d-%H-%M-%S")
    #print(formatted_time)
def set_random_seeds(seed=42):
    # Set PyTorch random seed
    torch.manual_seed(seed)
    # Set NumPy random seed
    np.random.seed(seed)
    # Set Python random seed
    random.seed(seed)
    # Set CUDA random seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Set CUDA deterministic mode
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



class IPAdapterDataset(Dataset):
    def __init__(self, args, data, train=True):
        self.data = data
        with open(os.path.join(args.data_path, "image_path.pkl"), "rb") as file:
            self.paths = pickle.load(file)
        
        # Load CLIP processor for image preprocessing
        self.movies_info = np.load(os.path.join(args.data_path, "movie_info.npy"), allow_pickle=True).item()
        #print("---------movies_info",self.movies_info)

        self.train = train
        self.data_path = args.data_path
        self.item_num = args.item_num
    def __len__(self):
        return len(self.data["uids"])


    def __getitem__(self, idx):
        #tgt_iid = self.data['iids'][idx]

        tgt_iid = random.randint(0, self.item_num - 1) #Select a reference item
        prompt = self.movies_info[tgt_iid]['description']
        uid = self.data['uids'][idx]
        his_iids = self.data['his_iids'][idx]
        # Cold-start simulation: keep only 1-3 of the user's interactions as the sparse history
        # (sparse_iids). Re-sampled on every __getitem__, so each epoch sees a different history.
        k = random.randint(1, 3)
        
        random_his_iids = random.sample(his_iids, min(len(his_iids), k))


        
        #random_his_iids = his_iids
        # Load target image
        target_image = Image.open(os.path.join(self.data_path, 'poster', self.paths[tgt_iid])).convert("RGB")
        
        # Load user history image list
        user_history_images = []
        for his_iid in random_his_iids:
            his_image = Image.open(os.path.join(self.data_path, 'poster', self.paths[his_iid])).convert("RGB")
            user_history_images.append(his_image)
            
        assert len(user_history_images) == len(random_his_iids),"Number of input images does not equal number of randomly selected cold_start_iids"
        sample = {}
        sample['iids'] = tgt_iid
        sample['uids'] = uid
        sample['prompt'] = prompt
        #-------- Get image input --------
        sample['target_image'] = target_image
        sample['his_images'] = user_history_images
        sample['image_path'] = os.path.join(self.data_path, 'poster', self.paths[tgt_iid])
        sample['his_iids'] = torch.tensor(his_iids, dtype=torch.long)
        sample['sparse_iids'] = torch.tensor(random_his_iids, dtype=torch.long)
        return sample
    







def get_top_k_similar_images(args,data_path,k_r=100):
    data_path = args.data_path
    if os.path.exists(os.path.join(data_path, 'near_images.npy')) and os.path.exists(os.path.join(data_path, 'near_images_sim.npy')):
        return
    movies_info = np.load(os.path.join(data_path, "movie_info.npy"), allow_pickle=True).item()
    if os.path.exists(os.path.join(data_path, 'image_cap_feat.pth')):
        embeddings = torch.load(os.path.join(data_path, 'image_cap_feat.pth'))
    else:
        from transformers import  CLIPTextModel, CLIPTokenizer
        model_path = getattr(args, "text_clip_path", None) or "openai/clip-vit-large-patch14"
        text_tokenizer = CLIPTokenizer.from_pretrained(model_path)
        text_model = CLIPTextModel.from_pretrained(model_path).to(args.device)
        text_model = text_model.eval()
        text_bsz = 32
        latents = []
        
        for i in range(0,args.item_num,text_bsz):
            batch_texts = []
            print("process",i,"to",i+text_bsz)
            for idx in range(i,min(i+text_bsz,args.item_num)):
                batch_texts.append(movies_info[idx]['description'])
            
            with torch.no_grad():
                # Get input embeddings
                inputs = text_tokenizer(batch_texts, 
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                max_length=77).to(args.device)
                input_embeddings = text_model(inputs.input_ids)[0].detach().cpu() #text_model.embed_tokens(inputs.input_ids)  # [B, L, D]

            
            latents.append(input_embeddings)
        embeddings = torch.cat(latents, dim=0)  # [N, hidden_dim]
        print("embeddings",embeddings.shape)
        torch.save(embeddings, os.path.join(data_path, 'image_cap_feat.pth'))
    
        
    # Calculate cosine similarity - batch optimized version
    import math 
    import torch.nn.functional as F
    import time
    
    print("Starting batch similarity calculation...")
    batch_start = time.time()
    
    # Memory-optimized similarity calculation - row-by-row version
    embeddings_flat = embeddings.flatten(start_dim=1).to(args.device)  # [N, D] Move to GPU
    print(f"embeddings_flat shape: {embeddings_flat.shape}, device: {embeddings_flat.device}")
    
    # Don't store entire similarity matrix, calculate row-by-row and process directly
    print("Starting row-by-row similarity calculation...")
    
    
    
    near_images = []
    near_sims = []
    for i in range(0, args.item_num):
        if i % 100 == 0:
            print(f"Processing progress: {i}/{args.item_num}")
            
        # Calculate similarity row-by-row to avoid storing entire matrix
        query_embedding = embeddings_flat[i:i+1]  # [1, D]
        sim = F.cosine_similarity(query_embedding, embeddings_flat, dim=1)  # [N]
        topk = torch.topk(sim, k_r, largest=True)
        top_kr_indices = topk.indices.cpu().tolist()  # Move to CPU then convert to list
        top_kr_sims = topk.values.cpu().tolist()
        
        # different IDs may refer to the same image
        assert top_kr_indices[0] == i or math.isclose(sim[i].cpu().item(), 1, abs_tol = 1e-6), "The most similar image should be itself"
        
        if i % 100 == 0:  # Reduce print frequency
            epsilon = 1e-6
            filtered_sim = sim[sim < (1.0 - epsilon)].cpu()  # Move to CPU for statistics
            print(f"Similarity range: [{filtered_sim.min():.4f}, {filtered_sim.max():.4f}]")
            print(f"Similarity median: {filtered_sim.median():.4f}, mean: {filtered_sim.mean():.4f}")
        
        near_images.append(top_kr_indices)
        near_sims.append(top_kr_sims)
        
        # Clean GPU memory
        if i % 1000 == 0:
            torch.cuda.empty_cache()
    np.save(os.path.join(data_path,'near_images.npy'), near_images)
    np.save(os.path.join(data_path,'near_images_sim.npy'), near_sims)


def uncertainty_aware_retrieval(near_images, near_images_sim, image_features, uncertain_model,
                                image_paths, data_path, his_iids, target_iid, top_k, device,
                                process_image=True):
    """
    Semantic-aware Low-uncertainty Retrieval.

    Shared by training (models/ReFL_SDXL.py) and inference (models/gen_model.py) so that both use
    exactly the same ranking and cannot drift apart: for each target item, its precomputed
    semantic neighbours (near_images / near_images_sim, built by get_top_k_similar_images) are
    re-ranked by

        uncertain_score = CTR(history, candidate) * semantic_similarity

    and the top-k are returned as the reference images fed to IP-Adapter.

    Args:
        near_images: Semantic neighbours per item, indexed by item id
        near_images_sim: Cosine similarity of those neighbours
        image_features: Item image feature tensor, shape [item_num, feature_dim]
        uncertain_model: Preference model used as the CTR scorer
        image_paths: List of image file names, indexed by item id
        data_path: Dataset root (images are read from <data_path>/poster)
        his_iids: Historical interaction item ids per sample [batch_size, seq_len]
        target_iid: Target item ids [batch_size]
        top_k: Number of reference images to retrieve per target
        device: Device the scoring runs on
        process_image: Whether to load and return PIL images

    Returns:
        process_image=True : (batch_retrival_images, batch_retrival_iids, batch_ctr, batch_sim)
        process_image=False: (batch_retrival_iids, batch_ctr, batch_sim)
    """
    bsz = len(target_iid)
    batch_retrival_images = []
    batch_retrival_iids = []
    batch_ctr = []
    batch_sim = []

    with torch.no_grad():
        for i in range(bsz):
            # The target id may arrive as a tensor (collated batch) or as a plain int.
            target_id = int(target_iid[i].item()) if torch.is_tensor(target_iid[i]) else int(target_iid[i])
            if not 0 <= target_id < len(near_images):
                raise KeyError(
                    f"Target ID {target_id} not found in near_images! Please ensure all target "
                    "IDs are in the precomputed similarity index."
                )

            near_image_indices = near_images[target_id]  # [num_candidates]
            near_image_sims = near_images_sim[target_id]  # [num_candidates]
            num_candidate = len(near_image_indices)

            # Score every candidate with the preference model: (history, candidate) -> CTR.
            his_features = image_features[his_iids[i]].to(device)
            candidate_features = image_features[near_image_indices].to(device).reshape(num_candidate, -1)
            his_features = his_features.unsqueeze(0).repeat(num_candidate, 1, 1)  # [num_candidate, seq_len, feature_dim]
            score, _ = uncertain_model(his_features, candidate_features)  # [num_candidate, 1]

            # Higher uncertain_score means lower interaction uncertainty.
            sim = torch.tensor(near_image_sims, device=device).unsqueeze(1)  # [num_candidate, 1]
            uncertain_score = score * sim

            _, top_k_indices = torch.topk(uncertain_score, k=min(top_k, num_candidate), dim=0)
            top_k_indices_np = top_k_indices.cpu().numpy()

            batch_ctr.append(score[top_k_indices_np])
            batch_sim.append(sim[top_k_indices_np])

            retrival_image_index = near_image_indices[top_k_indices_np.flatten()]  # [top_k]
            batch_retrival_iids.append(retrival_image_index)

            if process_image:
                batch_retrival_images.append([
                    Image.open(os.path.join(data_path, 'poster', str(image_paths[idx]))).convert("RGB")
                    for idx in retrival_image_index
                ])

    if process_image:
        return batch_retrival_images, batch_retrival_iids, batch_ctr, batch_sim
    return batch_retrival_iids, batch_ctr, batch_sim




def get_image_feature(args):
    # If feature file already exists, load directly
    if os.path.exists(args.image_feat_path):
        latents = torch.load(args.image_feat_path)
        print("load existing clip image_feature", latents.shape)
        return latents

    # 加载 CLIP 模型和处理器
    device = args.device  # Ensure args.device is defined (e.g., 'cuda' or 'cpu')
    model = CLIPVisionModelWithProjection.from_pretrained(args.diffusion_clip_path).to(device)
    processor = CLIPProcessor.from_pretrained(args.diffusion_clip_path)

    # Load image path list
    with open(os.path.join(args.data_path, "image_path.pkl"), "rb") as file:
        paths = pickle.load(file)

    # Batch process images
    image_bsz = 8  # Adjust based on GPU memory
    all_embeddings = []

    for i in range(0, args.item_num, image_bsz):
        print(f"Processing batch: {i} to {min(i + image_bsz, args.item_num) - 1}")
        batch_images = []
        for idx in range(i, min(i + image_bsz, args.item_num)):
            path = os.path.join(args.data_path, 'poster', paths[idx])
            
            image = Image.open(path).convert("RGB")
            batch_images.append(image)
        # Preprocess and encode
        inputs = processor(images=batch_images, return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            
            embeddings = model(inputs.pixel_values).image_embeds
            #print("embeddings",embeddings.shape)

        all_embeddings.append(embeddings.cpu())  # 保存to CPU

    
    latents = torch.cat(all_embeddings, dim=0)  # shape: [num_images, hidden_size]

    print(f"Encoding complete, total embeddings: {len(latents)}, shape: {latents.shape}")
    torch.save(latents, args.image_feat_path)  # 保存to指定路径
    return latents


def process_data(
    args,
    df
):    
    
    if args.scene == 'pog' or args.scene == 'ser' or args.scene == 'movie':
        def sliding_window(row, max_seq_length=None):
            sequences = []
            item_seq = row['movie_seq']
            
            # Start the window at the 6th interaction so every sample has at least 5 history
            # items; each window is then truncated to the last max_seq_length items below.
            for i in range(6,len(item_seq)):
                if max_seq_length is not None:
                    start = max(0, i - max_seq_length) 
                else:
                    start = 0
                sequences.append(item_seq[start:i]) #Truncate
            return sequences
        
        df['movie_seq'] = df.apply(sliding_window, max_seq_length=16,axis=1) 
        
    uids = []
    his_iids =[] 
    target_iids = []
    df_len =len(df)
    
    for i in range(df_len):
        hist_seqs = df['movie_seq'][i]
        uid = df['userId'][i]
        for hist_seq in hist_seqs:
            his_iids.append(hist_seq[:-1])
            tgt_iid = hist_seq[-1] 

            target_iids.append(tgt_iid)
            
            uids.append(uid)
    data= {"uids": uids, 
    'iids':target_iids, 
    'his_iids':his_iids, 
    }
    return data
def load_data(args):
    '''
    For train/valid/test.pkl
    userId: user ID
    movie_seq: store user's interacted item sequence,
    '''
    

    with open(os.path.join(args.data_path, "train.pkl"), "rb") as file:
        train_seq = pickle.load(file)
    with open(os.path.join(args.data_path, "valid.pkl"), "rb") as file:
        valid_seq = pickle.load(file)
    with open(os.path.join(args.data_path, "test.pkl"), "rb") as file:
        test_seq = pickle.load(file)
    

    train_data = process_data(args,train_seq)
    valid_data = process_data(args,valid_seq)
    test_data = process_data(args,test_seq)


    return train_data,valid_data,test_data









