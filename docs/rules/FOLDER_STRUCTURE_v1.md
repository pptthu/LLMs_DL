# Cấu trúc thư mục bài Presentation LLMs (v1)

- Tác giả: Toàn (AI hỗ trợ) | Tạo: 2026-11-15T16:30:00+07:00
- Trạng thái: **[đề xuất]**, chờ Thư duyệt
- Dựa trên: `Lab-DL/labs/lab1/docs/rules/FOLDER_STRUCTURE_v2.md` (gom gốc về `data/` vào, `outputs/` ra, `docs/` tài liệu) và góp ý của Thư ngày 15/11: repo riêng, không có mã nguồn riêng, cấu trúc gọn.

## 1. Cây thư mục repo `LLMs_DL`

    LLMs_DL/
    ├── README.md                 # Setup + cách chạy: thư viện, môi trường, lấy model ở đâu, thời gian chạy
    ├── project_description.md    # Define + Design: đề, 10 mục, phạm vi, phân công
    ├── progress_status.md        # Tiến độ, bằng chứng chạy, nhật ký
    ├── AI_usage.md               # Khai báo dùng AI (mỗi người một cột)
    ├── AGENTS.md                 # Hướng dẫn cho AI agent
    ├── CONTRIBUTING.md           # Quy tắc Git
    ├── requirements.txt          # Thư viện; ghim phiên bản sau lần chạy chính thức
    ├── .gitignore
    ├── .github/                  # Mẫu PR, mẫu issue
    ├── notebook/
    │   ├── llm_presentation.ipynb   # Notebook nộp: phần 0–10, bản đã Save & Run All (Toàn viết toàn bộ code)
    │   └── parts/                   # Không bắt buộc: notebook thử riêng một phần
    ├── data/                     # Bộ câu nhóm tự soạn + danh sách ID (commit được, dung lượng nhỏ)
    ├── outputs/
    │   ├── results/              # CSV, JSON từng phần + env.json (lần chạy chính thức)
    │   ├── figures/              # PNG chèn vào slide
    │   └── _smoke/               # Đầu ra của smoke run, không dùng để báo cáo (không commit)
    ├── slides/                   # llm_presentation_vN.pptx, bản final .pdf
    └── docs/                     # Xem §2

Không có thư mục mã nguồn và thư mục test riêng (góp ý của Thư): code và các `assert` tự kiểm nằm trong notebook.

## 2. Cây thư mục `docs/`

    docs/
    ├── 00_README_v1.md                    # Mục lục, đọc đầu tiên
    ├── presentation_description_v1.md     # Desc cả bài (khuôn DESC_TEMPLATE_v1)
    ├── references.md                      # Danh sách tài liệu tham khảo (sản phẩm nộp số 4)
    ├── rules/                             # Luật luôn bật: RULES_v1.md, file này
    ├── templates/                         # Khuôn để copy
    ├── plans/                             # Kế hoạch, detail plan có version
    ├── parts/                             # Desc từng mục: muc01_<tên>_v1.md … muc10_<tên>_v1.md (người phụ trách viết)
    ├── research/                          # Ghi chú research có nguồn, nội dung chi tiết từng mục
    ├── models/                            # Model card của model dùng (không train)
    ├── experiments/                       # experiment_log.md, error_analysis_v1.md
    ├── scripts/                           # Kịch bản thuyết trình từng người (tạo khi có file đầu tiên)
    ├── qa/                                # Q&A từng người (tạo khi có file đầu tiên)
    └── bugs/                              # Ghi chép lỗi (tạo khi có file đầu tiên)

## 3. Loại kết quả → đường dẫn

| Loại | Đường dẫn | Ghi chú |
|---|---|---|
| Dữ liệu UIT-VSFC | Không lưu; notebook tải lúc chạy (Parquet trên Hugging Face) | Không commit |
| Trọng số model, tokenizer | Không lưu; `from_pretrained` tải về bộ nhớ đệm của Hugging Face | Không commit |
| Bộ câu tự soạn | `data/<tên>.csv` hoặc `.json` | Commit |
| ID 500 câu test, 3 bộ ví dụ few-shot | `data/test_500_ids.json`, `data/fewshot_pools.json` | Commit, để tái lập |
| Môi trường chạy | `outputs/results/env.json` | Phiên bản Python, thư viện, GPU, dtype, ngày chạy |
| Kết quả từng phần | `outputs/results/<phần>_<tên>.csv` / `.json` | Tên cụ thể ở `project_description.md` §2.3 |
| Hình | `outputs/figures/<tên>.png` | |
| Smoke run | `outputs/_smoke/` | Không commit, không báo cáo |
| Notebook nộp | `notebook/llm_presentation.ipynb` | Bản đã chạy, có output |
| Slide | `slides/` | |
| Ghi chép lần chạy | `docs/experiments/experiment_log.md` | Theo `RUN_LOG_TEMPLATE_v1.md` |
| Phân tích lỗi mục 10 | `docs/experiments/error_analysis_v1.md` | Ví dụ là câu thật trong CSV |

## 4. Sản phẩm nộp (theo đề) → đường dẫn

| Sản phẩm | Đường dẫn |
|---|---|
| (1) Slide PDF hoặc PPTX | `slides/` |
| (2) Notebook + hướng dẫn chạy | `notebook/llm_presentation.ipynb`, `README.md`, `requirements.txt` |
| (3) Bộ câu thử nghiệm, kết quả, phân tích lỗi | `data/`, `outputs/results/`, `docs/experiments/error_analysis_v1.md` |
| (4) Tài liệu tham khảo | `docs/references.md` (và slide cuối) |

## 5. Khác với Lab 1 và vì sao

| Lab 1 | Bài presentation | Vì sao |
|---|---|---|
| 8 folder module, mỗi folder 1 script | Không có mã nguồn riêng; code trong notebook | Cô hỏi setup hay code phần nào thì chỉ ngay cell đó, không phải mở file khác (góp ý của Thư) |
| `tests/` với pytest | `assert` trong cell ngay sau hàm | Không còn file `.py` để test |
| `outputs/models/`, `logs/`, `demo/` | Chỉ `results/`, `figures/`, `_smoke/` | Không train, không có checkpoint |
| `data/raw/`, `data/splits/` | `data/` phẳng | Dữ liệu gốc tải lúc chạy; chỉ lưu file nhỏ nhóm soạn |
| Nằm trong repo Lab-DL | Repo riêng `LLMs_DL` | Bài presentation có GitHub riêng |
