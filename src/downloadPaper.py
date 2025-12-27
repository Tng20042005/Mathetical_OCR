import arxiv
import os
import time

categories = [
    "math.AG", "math.PR", "math.ST", "math.AP",
    "math.NT", "math.GR", "math.GM", "math.CO",
    "math.OC", "math.FA"
]

os.makedirs("math_papers_pdf", exist_ok=True)
os.makedirs("math_papers_tex", exist_ok=True)

for cat in categories:
    print("Downloading category:", cat)
    
    search = arxiv.Search(
        query=f"cat:{cat}",
        max_results=5,   # bạn có thể tăng lên 5000
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    for paper in search.results():
        paper_id = paper.get_short_id()

        # ----- TẢI PDF -----
        try:
            paper.download_pdf(
                dirpath="math_papers_pdf",
                filename=f"{paper_id}.pdf"
            )
            print("[PDF]  ", paper_id, paper.title)
        except:
            print("[PDF ERROR]", paper_id)
            pass

        # ----- TẢI TeX -----
        try:
            paper.download_source(
                dirpath="math_papers_tex"
            )
            print("[TEX]  ", paper_id, paper.title)
        except:
            print("[TEX ERROR]", paper_id)
            pass

        time.sleep(0.5)   # tránh bị arXiv chặn rate-limit
