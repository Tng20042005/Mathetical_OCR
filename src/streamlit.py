import streamlit as st
import torch
from PIL import Image
from transformers import NougatProcessor, VisionEncoderDecoderModel
import time
import re

# --- CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="Mathematical Expression OCR",
    page_icon="🔬",
    layout="wide"
)

# --- 1. HÀM CHUYỂN ĐỔI MARKDOWN -> LATEX ---
def markdown_to_latex_converter(md_text):
    """
    Chuyển đổi cú pháp Markdown của Nougat sang LaTeX code hoàn chỉnh.
    """
    latex_out = r"""\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{geometry}
\geometry{a4paper, margin=1in}
\usepackage{hyperref}

\title{Nougat OCR Output}
\date{\today}

\begin{document}
\maketitle

"""
    lines = md_text.split('\n')
    processed_lines = []
    in_abstract = False
    
    for line in lines:
        line = line.strip()
        if not line:
            processed_lines.append("")
            continue
            
        # Xử lý Header
        if line.startswith('# '): line = f"\\section{{{line[2:]}}}"
        elif line.startswith('## '): line = f"\\subsection{{{line[3:]}}}"
        elif line.startswith('### '): line = f"\\subsubsection{{{line[4:]}}}"
        
        # Xử lý Bold/Italic
        line = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', line)
        line = re.sub(r'\*(.*?)\*', r'\\textit{\1}', line)
        
        # Xử lý Abstract
        if "Abstract" in line and "\\textbf" in line:
            line = "\\begin{abstract}\n" + line.replace("\\textbf{Abstract}", "").strip()
            in_abstract = True
            
        processed_lines.append(line)

    if in_abstract:
         processed_lines.append("\\end{abstract}")

    latex_out += "\n".join(processed_lines) + "\n\n\\end{document}"
    return latex_out

# --- 2. LOAD MODEL ---
@st.cache_resource
def load_model():
    print("Đang tải model Nougat...")
    model_name = "facebook/nougat-small"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        processor = NougatProcessor.from_pretrained(model_name)
        model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
    except Exception as e:
        st.error(f"Lỗi tải model: {e}")
        return None, None, None
    return processor, model, device

# --- 3. HÀM SUY LUẬN ---
def predict(image, processor, model, device):
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    
    outputs = model.generate(
        pixel_values,
        min_length=1,
        max_length=3584,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
        output_scores=True,
        stopping_criteria=[],
    )
    
    generated_text = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
    generated_text = processor.post_process_generation(generated_text, fix_markdown=False)
    return generated_text

# --- 4. GIAO DIỆN CHÍNH ---
def main():
    st.title("📄 Mathematical Expression OCR    : Chuyển đổi 3 Dạng")
    st.markdown("Tool chuyển đổi ảnh tài liệu sang: **Render View**, **Raw Markdown** và **LaTeX Source**.")

    with st.spinner("Đang khởi động AI..."):
        processor, model, device = load_model()

    if model is None: return

    st.sidebar.markdown(f"**Thiết bị chạy:** `{device.upper()}`")
    
    uploaded_file = st.file_uploader("Tải ảnh lên (PNG, JPG)", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1.2]) # Cột phải rộng hơn chút để hiển thị text
        
        with col1:
            st.subheader("🖼️ Ảnh gốc")
            image = Image.open(uploaded_file).convert("RGB")
            # Sửa warning use_column_width
            st.image(image, use_container_width=True)

        generate_btn = st.sidebar.button("🚀 Xử lý ngay", type="primary")

        if generate_btn:
            with col2:
                with st.spinner("Đang đọc và dịch sang LaTeX..."):
                    start_time = time.time()
                    
                    # 1. Lấy kết quả Markdown từ Model
                    md_text = predict(image, processor, model, device)
                    
                    # 2. Convert sang LaTeX
                    latex_code = markdown_to_latex_converter(md_text)
                    
                    end_time = time.time()
                
                st.success(f"Hoàn tất trong {end_time - start_time:.2f} giây!")
                
                # --- TẠO 3 TAB HIỂN THỊ ---
                tab_render, tab_markdown, tab_latex = st.tabs([
                    "👁️ Xem trước (Render)", 
                    "📝 Markdown Result", 
                    "💻 LaTeX Output"
                ])
                
                # Tab 1: Render (Xem đẹp mắt)
                with tab_render:
                    st.markdown("### Bản xem trước:")
                    st.markdown("---")
                    st.markdown(md_text)
                    st.markdown("---")
                
                # Tab 2: Markdown (Code gốc của Nougat)
                with tab_markdown:
                    st.markdown("Copy đoạn này nếu dùng Obsidian/Notion:")
                    st.text_area("Raw Markdown", md_text, height=500)
                
                # Tab 3: LaTeX (Code để biên dịch)
                with tab_latex:
                    st.markdown("Copy đoạn này nếu dùng Overleaf/TeXShop:")
                    st.code(latex_code, language='latex')
                    st.download_button(
                        label="📥 Tải file .tex",
                        data=latex_code,
                        file_name="output.tex",
                        mime="text/plain"
                    )

if __name__ == "__main__":
    main()