
import os
import fitz  # PyMuPDF
import re
from tqdm import tqdm
from markdownify import markdownify as md
from rapidfuzz import fuzz

# --- CẤU HÌNH ---
BASE_DIR = os.getcwd()
PDF_DIR = os.path.join(BASE_DIR, "math_papers_pdf")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_dataset")

def clean_for_matching(text):
    """
    Chuẩn hóa text để so sánh: xóa ký tự lạ, xuống dòng, khoảng trắng thừa.
    Giúp việc so sánh giữa PDF và Markdown chính xác hơn.
    """
    return re.sub(r'\W+', '', text).lower()

def find_best_split(markdown_text, pdf_end_anchor, start_search_idx, search_window=2000):
    """
    Tìm vị trí trong markdown_text khớp nhất với pdf_end_anchor.
    Chỉ tìm trong khoảng [start_search_idx, start_search_idx + search_window]
    """
    # 1. Xác định vùng tìm kiếm
    end_search_idx = min(len(markdown_text), start_search_idx + search_window)
    search_area = markdown_text[start_search_idx:end_search_idx]
    
    # Nếu anchor quá ngắn (trang trống/ít chữ), bỏ qua fuzzy match
    clean_anchor = clean_for_matching(pdf_end_anchor)
    if len(clean_anchor) < 10:
        return -1

    # 2. Chiến thuật trượt cửa sổ (Sliding Window) kết hợp Fuzzy
    # Tuy nhiên, chạy sliding window trên từng ký tự rất chậm.
    # Mẹo: Tìm kiếm chuỗi gần đúng (approximate search)
    
    # Tìm vị trí xuất hiện của 20 ký tự cuối cùng của anchor trong vùng tìm kiếm
    # Đây là "Hard Anchor" để định vị nhanh
    mini_anchor = pdf_end_anchor[-20:].strip()
    
    # Tìm tất cả các vị trí xuất hiện của mini_anchor trong search_area
    # (Cho phép sai số nhẹ)
    best_score = 0
    best_relative_pos = -1
    
    # Để tối ưu tốc độ: Ta chia search_area thành các câu hoặc đoạn nhỏ
    # Nhưng đơn giản nhất: Quét từng đoạn ngắt dòng
    candidates = [m.start() for m in re.finditer(r'\n', search_area)]
    if not candidates: candidates = [len(search_area)]

    # Chỉ check tại các điểm xuống dòng để tiết kiệm CPU
    # So sánh 100 ký tự trước dấu xuống dòng với anchor của PDF
    for pos in candidates:
        # Lấy đoạn text trước dấu xuống dòng
        chunk_start = max(0, pos - len(pdf_end_anchor) - 50)
        chunk = search_area[chunk_start:pos]
        
        # So sánh độ trùng khớp
        score = fuzz.partial_ratio(clean_for_matching(chunk), clean_anchor)
        
        if score > best_score:
            best_score = score
            best_relative_pos = pos

    # Ngưỡng chấp nhận: > 70% trùng khớp
    if best_score > 70:
        return start_search_idx + best_relative_pos
    
    return -1

def align_paper(paper_id):
    paper_dir = os.path.join(PROCESSED_DIR, paper_id)
    html_path = os.path.join(paper_dir, "content.html")
    pdf_path = os.path.join(PDF_DIR, f"{paper_id}.pdf")
    
    if not os.path.exists(html_path): return

    # --- BƯỚC 1: CHUẨN BỊ MARKDOWN ---
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
    except: return

    # Convert sang Markdown
    full_markdown = md(html_content, heading_style="ATX", strip=['a', 'img'])
    full_markdown = re.sub(r'\n{3,}', '\n\n', full_markdown).strip()
    
    # --- BƯỚC 2: CHUẨN BỊ PDF TEXT ---
    doc = fitz.open(pdf_path)
    pdf_texts = []
    total_pdf_chars = 0
    for page in doc:
        text = page.get_text()
        pdf_texts.append(text)
        total_pdf_chars += len(text)
        
    if total_pdf_chars == 0: return # PDF rỗng

    # --- BƯỚC 3: CẮT (HYBRID METHOD) ---
    pages_dir = os.path.join(paper_dir, "pages_text")
    os.makedirs(pages_dir, exist_ok=True)
    
    current_md_idx = 0
    total_md_len = len(full_markdown)
    
    page_contents = []
    
    for i, page_text in enumerate(pdf_texts):
        # A. Tính điểm cắt dự kiến (Ratio Method) - Để khoanh vùng
        page_len = len(page_text)
        ratio = page_len / total_pdf_chars
        # Ước lượng độ dài Markdown (thường dài hơn PDF text 10-20%)
        estimated_len = int((page_len / total_pdf_chars) * total_md_len)
        
        search_center = current_md_idx + estimated_len
        search_start = max(current_md_idx, search_center - 1000) # Tìm lùi 1000 ký tự
        
        # B. Lấy "Mỏ neo" (Anchor) - Là đoạn cuối của trang PDF
        # Bỏ qua 10 ký tự cuối cùng vì thường là số trang (Footer) gây nhiễu
        clean_page = page_text.strip()
        if len(clean_page) > 50:
            # Lấy 100 ký tự gần cuối (tránh số trang)
            anchor = clean_page[-150:-5] 
        else:
            anchor = clean_page
            
        # C. Tìm vị trí chính xác (Fuzzy Match)
        # Nếu là trang cuối cùng thì lấy hết
        if i == len(pdf_texts) - 1:
            actual_split = len(full_markdown)
        else:
            found_split = find_best_split(full_markdown, anchor, search_start, search_window=2500)
            
            if found_split != -1:
                # [SUCCESS] Tìm thấy anchor khớp
                actual_split = found_split
            else:
                # [FALLBACK] Nếu fuzzy thất bại, dùng Ratio + tìm dấu xuống dòng gần nhất
                fallback_split = current_md_idx + estimated_len
                # Tìm dấu xuống dòng gần nhất
                rel_break = full_markdown[fallback_split-200:fallback_split+200].rfind('\n\n')
                if rel_break != -1:
                    actual_split = (fallback_split - 200) + rel_break
                else:
                    actual_split = fallback_split

        # D. Cắt và Lưu
        content = full_markdown[current_md_idx:actual_split].strip()
        
        # Fix lỗi: Nếu cắt bị lẹm, đảm bảo content không rỗng
        if not content and i < len(pdf_texts)-1:
            content = "[MISSING_TEXT]" # Đánh dấu để biết
            
        page_contents.append(content)
        
        # Lưu file
        save_path = os.path.join(pages_dir, f"page_{i+1:02d}.mmd")
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(content)
            
        # Cập nhật con trỏ cho vòng lặp sau
        current_md_idx = actual_split

    print(f"   [OK] {paper_id}: Xử lý xong {len(page_contents)} trang.")

def main():
    paper_ids = [d for d in os.listdir(PROCESSED_DIR) if os.path.isdir(os.path.join(PROCESSED_DIR, d))]
    print(f"Bắt đầu Alignment (Hybrid Fuzzy) cho {len(paper_ids)} bài báo...")
    
    for pid in tqdm(paper_ids):
        align_paper(pid)

if __name__ == "__main__":
    main()