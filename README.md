# LLMs_DL — Presentation: Tìm hiểu mô hình ngôn ngữ lớn (LLMs)

Bài thuyết trình nhóm môn Học sâu (UTH). Nhóm: Muội, Phúc, Toàn (thực hiện), Thư (hỗ trợ: review, Q&A, bấm giờ).

> Trạng thái: **[đề xuất]**, khung repo, chưa chạy notebook. Mọi số liệu sẽ lấy từ lần chạy chính thức trên Kaggle. Người phụ trách README: Muội.

## Nội dung
10 mục theo đề của cô; mỗi mục có demo trên model thật, cùng một notebook. Code notebook: Toàn. Cột "Người" là người phụ trách lý thuyết và thuyết trình:

| # | Mục | Người |
|---|---|---|
| 1 | Giới thiệu: LM, LLM, chatbot | Muội |
| 2 | Kiến trúc: Transformer, decoder-only, causal attention | Muội |
| 3 | Tokenization | Muội |
| 4 | Nguyên lý học: next-token prediction, cross-entropy | Phúc |
| 5 | Các giai đoạn huấn luyện: pretraining, SFT, RLHF/DPO | Phúc |
| 6 | Sinh văn bản: greedy, temperature, top-p | Phúc |
| 7 | Khả năng: in-context learning, zero-shot vs few-shot | Toàn |
| 8 | Prompting, fine-tuning, RAG | Muội |
| 9 | Hạn chế: hallucination, thiên lệch, giới hạn ngữ cảnh | Phúc |
| 10 | Thực nghiệm có kiểm soát và kết luận | Toàn |

## Setup

### 1. Môi trường
- **Kaggle Notebook, GPU T4.** Settings → Accelerator → GPU T4; bật **Internet** để tải model.
- Muốn bật GPU và Internet, tài khoản Kaggle phải **xác minh số điện thoại**.
- **Kiểu số:** `torch.float16`. T4 không hỗ trợ bf16 bằng phần cứng.
- Phiên bản thật của Python, thư viện, GPU được in ở **phần 0** của notebook và lưu `outputs/results/env.json`.

### 2. Thư viện
| Thư viện | Dùng làm gì |
|---|---|
| `torch` | Chạy model trên GPU |
| `transformers` (≥ 4.37.0) | Tải model, tokenizer; `generate()` |
| `accelerate` | `device_map="auto"` |
| `pandas` | Đọc UIT-VSFC (Parquet), lưu CSV |
| `scikit-learn` | Accuracy, confusion matrix, TF-IDF |
| `statsmodels` | McNemar test |
| `matplotlib` | Vẽ hình |

Kaggle thường cài sẵn. Cài trên máy khác: `pip install -r requirements.txt`.

### 3. Model, tokenizer, dữ liệu: lấy ở đâu, như thế nào
Cả ba model đều công khai, **không cần token Hugging Face**.

| Thứ | Lấy ở đâu | Giấy phép | Cách lấy |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct (model chính) | https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct | Apache-2.0 | `AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", torch_dtype=torch.float16, device_map="auto")` |
| Qwen2.5-1.5B (base, mục 5) | https://huggingface.co/Qwen/Qwen2.5-1.5B | Apache-2.0 | `AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", ...)` |
| GPT-2 tokenizer (mục 3) | https://huggingface.co/openai-community/gpt2 | MIT | `AutoTokenizer.from_pretrained("openai-community/gpt2")` |
| UIT-VSFC (mục 7, 10) | https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback | Ghi "unknown" → trích bài báo gốc, không commit | `pd.read_parquet(".../resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet")` |

- `from_pretrained` tự tải file về bộ nhớ đệm của Hugging Face; lần sau không tải lại.
- Mỗi model Qwen khoảng 3.1 GB.
- Chi tiết và cách dự phòng khi không tải được: `docs/plans/plan_presentation_llm_v3.md` §3 và `docs/models/*_v2.md` (có sơ đồ kiến trúc).

## Cách chạy
1. Mở `notebook/llm_presentation.ipynb` trên Kaggle (Upload notebook hoặc import từ GitHub).
2. Bật GPU T4 và Internet.
3. **Chạy thử:** đặt `SMOKE = True` ở cell đầu → Run All. Mỗi phần chạy vài câu, kết quả ghi vào `outputs/_smoke/`.
4. **Chạy chính thức:** `SMOKE = False` → **Save & Run All**. Kết quả ghi vào `outputs/results/` và `outputs/figures/`; log ghi vào `outputs/logs/<YYYYMMDD_HHMMSS>_normal/`.
5. Thời gian chạy dự kiến: _chưa đo_ (điền sau lần chạy chính thức).

## Cấu trúc thư mục
```text
LLMs_DL/
├── README.md, project_description.md, progress_status.md, AI_usage.md, AGENTS.md, CONTRIBUTING.md
├── requirements.txt, .gitignore, .github/
├── notebook/      # llm_presentation.ipynb (bản nộp, Toàn viết code) + parts/ (không bắt buộc)
├── data/          # bộ câu nhóm tự soạn, ID 500 câu test, 3 bộ ví dụ few-shot
├── outputs/       # results/ (CSV, JSON, env.json), figures/ (PNG), logs/<YYYYMMDD_HHMMSS>_normal/ (log mỗi lần chạy), _smoke/ (không commit)
├── slides/        # PPTX, PDF
└── docs/          # luật, khuôn, kế hoạch, research, model card, sổ thí nghiệm, tài liệu tham khảo
```
Chi tiết: `docs/rules/FOLDER_STRUCTURE_v2.md`. Đọc tài liệu bắt đầu từ `docs/00_README_v2.md`.

## Sản phẩm nộp
| Sản phẩm | Đường dẫn |
|---|---|
| Slide PDF / PPTX | `slides/` |
| Notebook + hướng dẫn chạy | `notebook/llm_presentation.ipynb`, file này |
| Bộ câu thử nghiệm, kết quả, phân tích lỗi | `data/`, `outputs/results/`, `docs/experiments/error_analysis_v1.md` |
| Tài liệu tham khảo | `docs/references.md` |
