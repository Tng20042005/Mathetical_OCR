import os
import glob
import tarfile
import subprocess
import shutil
from pdf2image import convert_from_path
from tqdm import tqdm

# --- CẤU HÌNH ĐƯỜNG DẪN (Dựa trên cấu trúc của bạn) ---
BASE_DIR = os.getcwd()
PDF_DIR = os.path.join(BASE_DIR, "math_papers_pdf")
TEX_DIR = os.path.join(BASE_DIR, "math_papers_tex")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed_dataset")

# --- HÀM HỖ TRỢ ---

def find_main_tex_file(source_dir):
    """
    Tìm file .tex chính trong folder giải nén.
    File chính thường chứa lệnh \documentclass.
    """
    tex_files = glob.glob(os.path.join(source_dir, "**/*.tex"), recursive=True)
    if not tex_files:
        return None
    
    # Ưu tiên 1: Tìm file có chữ "\documentclass"
    for tex in tex_files:
        try:
            with open(tex, 'r', encoding='latin-1') as f:
                if "\\documentclass" in f.read():
                    return tex
        except:
            continue
            
    # Ưu tiên 2: Nếu không tìm thấy, lấy file .tex có kích thước lớn nhất
    return max(tex_files, key=os.path.getsize)

def process_paper(pdf_filename):
    # Lấy ID bài báo (Ví dụ: từ "2512.01154v1.pdf" -> "2512.01154v1")
    paper_id = pdf_filename.replace(".pdf", "")
    
    # Tạo thư mục đầu ra
    save_dir = os.path.join(OUTPUT_DIR, paper_id)
    os.makedirs(save_dir, exist_ok=True)
    images_dir = os.path.join(save_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    print(f"\n>>> Đang xử lý: {paper_id}")

    # ==========================================
    # PHẦN 1: CHUYỂN PDF THÀNH ẢNH (Rasterize)
    # ==========================================
    pdf_path = os.path.join(PDF_DIR, pdf_filename)
    
    # Kiểm tra xem đã làm chưa để tránh làm lại (Resume)
    if not os.listdir(images_dir): 
        try:
            print(f"   [1/2] Đang cắt ảnh từ PDF...")
            # DPI=96 là chuẩn của Nougat paper
            images = convert_from_path(pdf_path, dpi=96, fmt="png")
            for i, img in enumerate(images):
                img.save(os.path.join(images_dir, f"page_{i+1:02d}.png"))
        except Exception as e:
            print(f"   [LỖI PDF] Không thể đọc PDF: {e}")
            return
    else:
        print(f"   [1/2] Ảnh đã có sẵn, bỏ qua.")

    # ==========================================
    # PHẦN 2: CHUYỂN TEX THÀNH HTML (LaTeXML)
    # ==========================================
    html_out_path = os.path.join(save_dir, "content.html")
    
    if os.path.exists(html_out_path):
        print(f"   [2/2] HTML đã có sẵn, bỏ qua.")
        return

    # Tìm file tar.gz tương ứng
    # Logic: File tar thường bắt đầu bằng ID (VD: 2512.01154v1....tar.gz)
    tex_archives = glob.glob(os.path.join(TEX_DIR, f"{paper_id}*.tar.gz"))
    
    if not tex_archives:
        print(f"   [CẢNH BÁO] Không tìm thấy file code .tar.gz cho bài này!")
        return
        
    tex_archive_path = tex_archives[0]
    temp_extract_dir = os.path.join(save_dir, "temp_source") # Thư mục tạm để giải nén
    
    try:
        # A. Giải nén
        with tarfile.open(tex_archive_path) as tar:
            # Dùng filter='data' để tránh lỗi bảo mật trên Python mới, hoặc bỏ qua nếu Python cũ
            if hasattr(tarfile, 'data_filter'):
                tar.extractall(path=temp_extract_dir, filter='data')
            else:
                tar.extractall(path=temp_extract_dir)
        
        # B. Tìm file main.tex
        main_tex = find_main_tex_file(temp_extract_dir)
        if not main_tex:
            print("   [LỖI] Giải nén xong nhưng không tìm thấy file .tex nào.")
            return

        # C. Chạy lệnh LaTeXML
        print(f"   [2/2] Đang biên dịch LaTeX sang HTML...")
        
        # Lệnh command line gọi latexmlc
        # --format=html5: Xuất ra HTML5
        # --mathtex: Giữ nguyên công thức toán dạng TeX (để sau này model học)
        # --timeout=60: Nếu file nào lỗi quá 60s thì bỏ qua
        cmd = [
            "latexmlc",
            main_tex,
            "--dest", html_out_path,
            "--format=html5",
            "--mathtex", 
            "--quiet"
        ]
        
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
        
        if os.path.exists(html_out_path):
            print("   [THÀNH CÔNG] Đã tạo content.html")
        else:
            print("   [THẤT BẠI] LaTeXML chạy nhưng không tạo ra file html (có thể do lỗi code LaTeX).")

    except subprocess.TimeoutExpired:
        print("   [TIMEOUT] File này quá nặng hoặc bị treo, bỏ qua.")
    except Exception as e:
        print(f"   [LỖI] Có vấn đề khi xử lý LaTeX: {e}")
    finally:
        # D. Dọn dẹp thư mục tạm (quan trọng để tiết kiệm ổ cứng)
        if os.path.exists(temp_extract_dir):
            shutil.rmtree(temp_extract_dir)

# --- HÀM MAIN ---
def main():
    # Tạo folder output
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # Lấy danh sách PDF
    pdf_files = sorted([f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")])
    print(f"Tìm thấy tổng cộng {len(pdf_files)} bài báo.")
    
    # Chạy vòng lặp với thanh tiến trình
    for pdf_file in tqdm(pdf_files):
        process_paper(pdf_file)

if __name__ == "__main__":
    main()