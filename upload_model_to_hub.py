from huggingface_hub import upload_folder, upload_file
import sys
import os

d = 'tmp-sd15-newnash_1half/checkpoint-190-loop-tau0p025-eta0p05/unet'
for fname in ['config.json', 'diffusion_pytorch_model.safetensors']:
    print(f"Uploading {fname}...")
    upload_file(path_or_fileobj=os.path.join(d, fname),
                path_in_repo=fname,
                repo_id = "Brownight/sd15-1half-190-final", # please don't overwrite other files in repo
                token="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" # Add your HF token here or just feed in via command line
               )
