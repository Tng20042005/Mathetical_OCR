import os
import glob
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    VisionEncoderDecoderModel,
    VisionEncoderDecoderConfig,
    SwinConfig,
    BartConfig,
    NougatProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator
)
import albumentations as A

# --- CẤU HÌNH ---
BASE_DIR = os.getcwd()
DATASET_DIR = os.path.join(BASE_DIR, "processed_dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "Mathematical Expression OCR")

# Kích thước ảnh chuẩn
HEIGHT = 896
WIDTH = 672
MAX_LENGTH = 3584

# --- 1. TỰ DỰNG KIẾN TRÚC MODEL (MODEL ARCHITECTURE) ---
def create_model_from_scratch():
    # A. Cấu hình Encoder (Mắt - Swin Transformer)
    # Đây là thông số chuẩn của Nougat Small
    encoder_config = SwinConfig(
        image_size=(HEIGHT, WIDTH),
        patch_size=4,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        drop_path_rate=0.1,
    )

    # B. Cấu hình Decoder (Miệng - mBART)
    # Đây là thông số chuẩn của mBART
    decoder_config = BartConfig(
        vocab_size=50265, # Kích thước từ điển chuẩn của mBART
        d_model=1024,
        encoder_layers=12,
        decoder_layers=12,
        encoder_attention_heads=16,
        decoder_attention_heads=16,
        encoder_ffn_dim=4096,
        decoder_ffn_dim=4096,
        activation_function="gelu",
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.1,
        max_position_embeddings=MAX_LENGTH + 2, # +2 cho bos/eos tokens
        scale_embedding=True,
    )

    # C. Ghép 2 cái lại thành VisionEncoderDecoder
    config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
    
    # Một số setting quan trọng để 2 mạng hiểu nhau
    config.decoder_start_token_id = 2 # Token bắt đầu (<s>)
    config.pad_token_id = 1           # Token đệm (<pad>)
    config.eos_token_id = 2           # Token kết thúc (</s>)
    config.max_length = MAX_LENGTH
    config.beam_num = 1
    config.encoder.image_size = (HEIGHT, WIDTH) # Set cứng kích thước ảnh

    # D. Khởi tạo Model với trọng số ngẫu nhiên (Random Weights)
    model = VisionEncoderDecoderModel(config)
    
    return model

# --- 2. DATA AUGMENTATION ---
def get_train_transforms():
    return A.Compose([
        A.GaussianBlur(p=0.1),
        A.GaussNoise(p=0.1),
        A.ToGray(p=1.0),
        A.Normalize(mean=[0.5], std=[0.5]),
    ])

# --- 3. DATASET CLASS ---
class Mathematical_Expression_OCR(Dataset):
    def __init__(self, dataset_dir, processor, split="train"):
        self.dataset_dir = dataset_dir
        self.processor = processor
        self.split = split
        self.data_pairs = []
        
        print(f"Loading {split} data...")
        paper_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
        
        for pid in paper_dirs:
            txt_dir = os.path.join(dataset_dir, pid, "pages_text")
            img_dir = os.path.join(dataset_dir, pid, "images")
            if not os.path.exists(txt_dir): continue
            
            mmd_files = glob.glob(os.path.join(txt_dir, "*.mmd"))
            for mmd_path in mmd_files:
                fname = os.path.basename(mmd_path)
                img_name = fname.replace(".mmd", ".png")
                img_path = os.path.join(img_dir, img_name)
                if os.path.exists(img_path):
                    self.data_pairs.append({"img": img_path, "txt": mmd_path})

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        item = self.data_pairs[idx]
        image = Image.open(item["img"]).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze()
        
        with open(item["txt"], "r", encoding="utf-8") as f:
            text = f.read()
            
        labels = self.processor.tokenizer(
            text,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        return {"pixel_values": pixel_values, "labels": labels}

# --- 4. TRAINING MAIN ---
# --- 4. TRAINING MAIN ---
# --- 4. TRAINING MAIN ---
def main():
    # Load Processor
                                                                                        #processor = NougatProcessor.from_pretrained("facebook/nougat-small")
    
    # KHỞI TẠO MODEL TỪ ĐẦU
    print(">>> Đang khởi tạo model từ con số 0 (Random Init)...")
    model = create_model_from_scratch()
    
    # Kiểm tra tham số
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model Size: {num_params / 1e6:.1f}M tham số")

    full_dataset = Mathematical_Expression_OCR(DATASET_DIR, processor)
    
    # Chia tập train/val
    train_size = int(0.95 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # Fix lỗi nếu dataset quá nhỏ
    if val_size == 0 and len(full_dataset) > 1:
        val_size = 1
        train_size = len(full_dataset) - 1
        
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    print(f"Training on {len(train_dataset)} samples, Validating on {len(val_dataset)} samples.")

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2, 
        gradient_accumulation_steps=4,
        learning_rate=1e-4,            
        num_train_epochs=50, 
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch", # Đã sửa eval_strategy
        save_total_limit=2,
        fp16=True,
        dataloader_num_workers=4,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        # --- SỬA DÒNG NÀY ---
        # Cũ (Sai): tokenizer=processor.feature_extractor,
        # Mới (Đúng):
        tokenizer=processor.tokenizer, 
        # --------------------
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset, 
        data_collator=default_data_collator,
    )

    print(">>> START TRAINING...")
    trainer.train()
    
    trainer.save_model(os.path.join(OUTPUT_DIR, "nougat_scratch_final"))
    processor.save_pretrained(os.path.join(OUTPUT_DIR, "nougat_scratch_final"))
    print(">>> DONE!")

if __name__ == "__main__":
    main()