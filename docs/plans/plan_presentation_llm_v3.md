# Plan bài Presentation LLMs: cấu trúc thư mục và plan code (v3)

- Người viết: Toàn (AI hỗ trợ) | Tạo v2: 2026-11-15 | Cập nhật v3: 2026-11-16T01:40:17+07:00 | Người duyệt: Toàn (Thư review)
- Trạng thái: **Duyệt: Toàn, 2026-11-16**.
- Thay cho `plan_presentation_llm_v2.md` (v1, v2 giữ nguyên, không dùng nữa).

## Thay đổi so với v2
| # | Thay đổi | Lý do |
|---|---|---|
| 1 | **Toàn viết toàn bộ code** phần 0–10, viết thẳng vào `llm_presentation.ipynb`; Muội, Phúc làm lý thuyết, slide, thuyết trình và nhận bàn giao code (§4, §7) | Quyết định của Toàn ngày 15/11 |
| 2 | Thêm log mỗi lần chạy trong `outputs/logs/<YYYYMMDD_HHMMSS>_<smoke\|normal>/` (§5) | Yêu cầu log rõ ràng; khuôn `templates/RUN_LOG_TEMPLATE_v2.md` |
| 3 | Mục 2 thêm so sánh LLM với RNN/LSTM, BERT, VLM (§4) | Điểm cộng, Toàn quyết định 16/11 |
| 4 | §8 bỏ câu hỏi "viết riêng rồi ghép" (đã chốt ở #1) | |

## Thay đổi so với v1 (theo góp ý của Thư)

| # | Góp ý | Quyết định trong v2 |
|---|---|---|
| 1 | Bài present có **GitHub riêng** | Thư mục riêng, **không** đặt trong repo Lab-DL; không dùng `shared/paths.py` của Lab-DL |
| 2 | **Không dùng mã nguồn `src/`**, code hết trong notebook | Mọi code nằm trong 1 notebook. Cô hỏi phần nào thì chỉ ngay cell đó, không phải mở file khác. Bỏ `src/`, `tests/`, Kaggle Dataset |
| 3 | Cấu trúc thư mục **quá lớn** | Rút còn 5 thư mục + README (§2) |
| 4 | **Ưu tiên bày cách setup**: GPT-2, Qwen2.5 lấy ở đâu, lấy như thế nào | Phần 0 của notebook và README ghi rõ từng thứ: tên, link, giấy phép, dung lượng, câu lệnh lấy về (§3) |
| 5 | Task code vẫn chia cho Muội và Phúc | Giữ nguyên phân công: mỗi người viết phần notebook của mình, Toàn ghép (§4) |

Các điểm v1 đã giải quyết xong (không cần chốt nữa): vị trí thư mục, rule notebook, `shared/paths.py`, Kaggle Dataset.

---

## 1. Tóm tắt
- **Một notebook duy nhất** `llm_presentation.ipynb`, chạy trên Kaggle (GPU T4), gồm phần 0–10.
- **Phần 0 = setup:** thư viện, môi trường, và **cách lấy từng model / tokenizer / dữ liệu**, viết thành từng bước để cô hỏi là chỉ được ngay.
- Mỗi cell code có markdown giải thích phía trước, comment từng dòng không hiển nhiên, và cell "Phân tích" sau output.
- Notebook **vừa giữ output** (cô yêu cầu) **vừa ghi kết quả ra file** trong `outputs/` để chép số lên slide.
- Mọi số trên slide lấy từ **1 lần Save & Run All**.

---

## 2. Cấu trúc thư mục (repo GitHub riêng)

```text
LLMs_DL/                          # repo pptthu/LLMs_DL (chốt 16/11)
├── README.md                     # Setup + cách chạy: thư viện, môi trường, lấy model ở đâu, thời gian chạy (Muội)
├── notebook/
│   └── llm_presentation.ipynb    # TOÀN BỘ code, phần 0–10, bản đã Save & Run All (Toàn ghép)
├── data/                         # Dữ liệu nhỏ nhóm tự soạn (commit được)
│   ├── tasks_prompts.json        # mục 1           (Muội)
│   ├── base_vs_instruct.json     # mục 5           (Phúc)
│   ├── rag_questions.csv         # mục 8           (Muội)
│   ├── hallucination.csv         # mục 9           (Phúc)
│   ├── bias_prompts.json         # mục 9           (Phúc)
│   ├── test_500_ids.json         # mục 7, 10       (Toàn) — ID 500 câu test, để tái lập
│   └── fewshot_pools.json        # mục 7, 10       (Toàn) — 3 bộ ví dụ
├── outputs/                      # Kết quả do notebook ghi ra; slide lấy số từ đây
│   ├── results/                  # CSV, JSON từng phần + env.json
│   └── figures/                  # PNG chèn vào slide
├── slides/                       # llm_presentation_vN.pptx, bản final .pdf (Muội)
└── docs/                         # Tài liệu nhóm: plan, research, kịch bản, Q&A, references.md
```

- **Không commit:**
  - UIT-VSFC: notebook tự tải lúc chạy;
  - slide môn học dùng cho RAG;
  - trọng số model.
- **Không có `src/`, `tests/`:** hàm nào cần dùng lại thì định nghĩa trong cell của notebook, ngay phần dùng nó.
- **Sản phẩm nộp → đường dẫn:**

| Sản phẩm (theo đề) | Đường dẫn |
|---|---|
| Slide PDF/PPTX | `slides/` |
| Notebook + hướng dẫn chạy | `notebook/llm_presentation.ipynb` + `README.md` |
| Bộ câu thử nghiệm, kết quả, phân tích lỗi | `data/`, `outputs/results/`, `docs/error_analysis.md` |
| Tài liệu tham khảo | `docs/references.md` |

---

## 3. Phần 0 — Setup (ưu tiên số 1)

**Mục tiêu:** cô hỏi "thư viện gì, chạy ở đâu, model lấy ở đâu, lấy thế nào" thì mở **phần 0** chỉ ngay. README chép lại cùng nội dung.

### 3.1 Môi trường
| Thứ | Giá trị | Ghi chú |
|---|---|---|
| Nơi chạy | Kaggle Notebook, **GPU T4** | Settings → Accelerator → GPU T4. Bật **Internet** để tải model. Muốn bật GPU và Internet, tài khoản Kaggle phải xác minh số điện thoại |
| Kiểu số | `torch.float16` | T4 không hỗ trợ bf16 bằng phần cứng. Nếu output fp16 lạ thì dùng `float32` (1.5B × 4 byte ≈ 6.2 GB, vẫn vừa 16 GB) |
| In ra trong notebook | Phiên bản Python, `torch`, `transformers`, `datasets`/`pandas`, tên GPU, CUDA, RAM | Ghi vào `outputs/results/env.json` → chép lên slide 2 |

### 3.2 Thư viện
| Thư viện | Dùng làm gì | Mục |
|---|---|---|
| `torch` | Chạy model trên GPU | Tất cả |
| `transformers` (≥ 4.37.0, model card Qwen yêu cầu) | Tải model, tokenizer; `generate()` | Tất cả |
| `accelerate` | `device_map="auto"` để đặt model lên GPU | 0 |
| `pandas` | Đọc UIT-VSFC (Parquet), lưu CSV | 7, 10 |
| `scikit-learn` | Accuracy, confusion matrix; TF-IDF cho RAG | 8, 10 |
| `statsmodels` | McNemar test | 10 |
| `matplotlib` | Vẽ causal mask, confusion matrix | 2, 10 |

Kaggle thường cài sẵn phần lớn các thư viện này. Notebook in phiên bản thật ra, **không** ghi phiên bản theo trí nhớ.

### 3.3 Lấy model, tokenizer và dữ liệu ở đâu, như thế nào

Cả ba model đều **công khai, không cần token Hugging Face** (`gated: false` trên Hub, kiểm ngày 15/11/2026).

| Thứ | Lấy ở đâu | Giấy phép | Dung lượng | Dùng cho |
|---|---|---|---|---|
| **Qwen2.5-1.5B-Instruct** | Hugging Face Hub: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct | Apache-2.0 | khoảng 3.1 GB (1.54 tỉ tham số × 2 byte) | Model chính, mọi mục |
| **Qwen2.5-1.5B** (bản base) | https://huggingface.co/Qwen/Qwen2.5-1.5B | Apache-2.0 | khoảng 3.1 GB | Mục 5: so base với instruct |
| **GPT-2 tokenizer** | https://huggingface.co/openai-community/gpt2 (chỉ lấy tokenizer, không lấy model) | MIT | vài MB (`vocab.json`, `merges.txt`, `tokenizer.json`) | Mục 3: so số token với Qwen |
| **UIT-VSFC** | Hugging Face: https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback (đọc bản Parquet) | Trang ghi "unknown" → trích bài báo gốc, không commit dữ liệu | Nhỏ | Mục 7, 10 |

**Cách lấy** (mẫu cell phần 0; **chưa chạy thử**, kiểm ở buổi chạy thử Ngày 1):

```python
# 0.3 Lấy model và tokenizer từ Hugging Face Hub
# from_pretrained(tên) tự tải file từ huggingface.co/<tên> về bộ nhớ đệm (~/.cache/huggingface),
# lần sau đọc lại từ bộ nhớ đệm, không tải lại.
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_INSTRUCT = "Qwen/Qwen2.5-1.5B-Instruct"   # model chính
MODEL_BASE = "Qwen/Qwen2.5-1.5B"                # bản base, cho mục 5
TOKENIZER_GPT2 = "openai-community/gpt2"        # chỉ lấy tokenizer, cho mục 3

tok = AutoTokenizer.from_pretrained(MODEL_INSTRUCT)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_INSTRUCT,
    torch_dtype=torch.float16,   # T4 không có bf16 phần cứng
    device_map="auto",           # tự đặt lên GPU
)
model.eval()                     # chỉ suy luận, không train
print(model.generation_config)   # bản Instruct mặc định do_sample=True → phải truyền do_sample=False khi cần greedy

gpt2_tok = AutoTokenizer.from_pretrained(TOKENIZER_GPT2)
```

```python
# 0.4 Lấy UIT-VSFC: đọc thẳng file Parquet Hugging Face tự chuyển đổi
# (script tải dữ liệu cũ không chạy với thư viện datasets >= 4.0)
import pandas as pd
BASE = ("https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback/"
        "resolve/refs%2Fconvert%2Fparquet/default")
test_df = pd.read_parquet(f"{BASE}/test/0000.parquet")    # kỳ vọng 3,166 dòng
train_df = pd.read_parquet(f"{BASE}/train/0000.parquet")  # kỳ vọng 11,426 dòng
```

- **Bản base** (mục 5): tải ở phần 5, không tải ở phần 0. Nếu thiếu bộ nhớ thì `del model` + `torch.cuda.empty_cache()` trước khi tải.
- **Phần 2 cần trọng số attention:** tải lại với `attn_implementation="eager"`. Kiểu mặc định (SDPA) không trả trọng số.
- **Dự phòng nếu không tải được** (Internet tắt hoặc lỗi mạng): tải trước file model và dữ liệu, đưa lên **Kaggle Dataset** của nhóm, rồi đổi đường dẫn trong `from_pretrained` / `read_parquet` sang `/kaggle/input/...`.

### 3.4 Ghi lại môi trường (cell cuối phần 0)
In ra và lưu `env.json`:
- phiên bản Python, `torch`, `transformers`, `pandas`, `scikit-learn`, `statsmodels`;
- `torch.cuda.get_device_name()`, CUDA, RAM;
- tên model và dtype đã dùng;
- ngày chạy.

---

## 4. Khung notebook và phân công (code: Toàn; lý thuyết theo kế hoạch nhóm)

| Phần | Nội dung | Lý thuyết + thuyết trình (code: Toàn) | File dữ liệu / kết quả |
|---|---|---|---|
| **0. Setup** | §3: môi trường, thư viện, lấy model / tokenizer / dữ liệu, `env.json`, mở log | Toàn (lý thuyết + code, chốt 16/11) | `outputs/results/env.json` |
| 1. Tác vụ mẫu | 4 tác vụ trên bản instruct | Muội | `data/tasks_prompts.json` → `tasks_outputs.csv` |
| 2. Kiến trúc | In config, causal mask, attention thật (eager), top-5 token; so sánh LLM với RNN/LSTM, BERT, VLM, có demo (`research/noi_dung_muc02_kien_truc_v2.md` §11.3) | Muội | `top5_next_tokens.csv`, `figures/causal_mask.png` |
| 3. Tokenization | Tokenize Việt, Anh; so GPT-2 và Qwen; embedding | Muội | `token_counts.csv` |
| 4. Nguyên lý học | Cặp ngữ cảnh → đích; loss từng token; so với `outputs.loss` | Phúc | `per_token_loss.csv` |
| 5. Base vs instruct | Cùng prompt, cùng `do_sample=False` | Phúc | `data/base_vs_instruct.json` → `base_vs_instruct.csv` |
| 6. Sinh văn bản | Sinh từng bước; 5 cấu hình, đặt đủ tham số | Phúc | `decoding_samples.csv` |
| 7. Zero / few-shot | Prompt, 3 bộ ví dụ, chạy 500 câu × 4 | **Toàn** | `data/test_500_ids.json`, `fewshot_pools.json` → `icl_*.csv` |
| 8. RAG | Chia đoạn, TF-IDF, không RAG vs có RAG | Muội | `data/rag_questions.csv` → `rag_results.csv` |
| 9. Hạn chế | Hallucination, thiên lệch, ngữ cảnh dài | Phúc | `data/hallucination.csv`, `bias_prompts.json` → `*_results.csv` |
| 10. Tổng hợp | Metric, McNemar, confusion matrix, bảng tổng hợp mọi phần | **Toàn** | `icl_metrics.json`, `summary_table.csv`, `figures/confusion_matrix_*.png` |

**Cách làm song song không đụng nhau:**
- Toàn viết thẳng vào `notebook/llm_presentation.ipynb`; chỉ một người sửa nên không xung đột khi merge.
- Muội, Phúc viết dàn ý desc mục mình (`docs/parts/`) để Toàn code đúng ý, soạn file trong `data/`.
- Toàn code xong phần nào thì **bàn giao** cho người phụ trách mục (giải thích từng cell, chỉ log và file kết quả), ghi `progress_status.md`.

---

## 5. Quy ước viết notebook (để giải thích được từng dòng)
1. Mỗi phần mở đầu bằng markdown: `## Phần X — <tên> (mục X, lý thuyết: <người>; code: Toàn)`.
2. Mỗi cell code có **markdown đứng trước**: mục đích, đầu vào, đầu ra, mục lý thuyết liên quan.
3. Mỗi dòng không hiển nhiên có **comment tiếng Việt**.
4. Sau mỗi output có **cell "Phân tích"**: 2–3 câu (quan sát gì, vì sao, liên hệ lý thuyết).
5. Hàm dùng lại (ví dụ `parse_label`, `build_prompt`) **định nghĩa trong cell ngay trước chỗ dùng**, có docstring ngắn, kèm 2–3 dòng `assert` kiểm nhanh. Đây là phần thay cho `tests/`.
6. Đầu notebook có `SMOKE = True/False`. Chạy thử với `True` (vài câu), chạy chính thức với `False`.
7. Mỗi phần ghi kết quả ra `outputs/results/` (CSV/JSON) và `outputs/figures/` (PNG).
8. Mỗi lần chạy ghi log vào `outputs/logs/<YYYYMMDD_HHMMSS>_<smoke|normal>/`: `run.log`, `config.json`, `env.json`, `partNN_<tên>.log`, `partNN_<tên>_records.csv` (khuôn `docs/templates/RUN_LOG_TEMPLATE_v2.md`). Smoke run ghi vào `outputs/_smoke/logs/`.

---

## 6. Chú ý kỹ thuật (từ research, bắt buộc làm theo)

| # | Chú ý | Phần |
|---|---|---|
| 1 | Qwen2.5-Instruct mặc định `do_sample=True`, `temperature 0.7`, `top_p 0.8`, `top_k 20`, `repetition_penalty 1.1`. Greedy phải truyền `do_sample=False`; mục 6 phải đặt đủ tham số cho từng cấu hình | 1, 5, 6, 7, 8, 9 |
| 2 | Dùng `float16`, không dùng `torch_dtype="auto"` (sẽ nạp bf16, chạy chậm trên T4) | 0 |
| 3 | Cần trọng số attention thì tải với `attn_implementation="eager"` | 2 |
| 4 | UIT-VSFC đọc bằng Parquet; có 3 nhãn → bỏ neutral; test từ split test, few-shot từ split train, kiểm trùng câu | 7, 10 |
| 5 | Bản instruct dùng `apply_chat_template(..., add_generation_prompt=True)`; tránh thêm token đặc biệt hai lần | 1, 5, 7, 8, 9 |
| 6 | Bản base không dùng chat template; ghi rõ trong markdown | 5 |
| 7 | Đếm token với `add_special_tokens=False` khi so GPT-2 và Qwen | 3 |
| 8 | TF-IDF mặc định bỏ từ 1 ký tự ("ở"); dùng `ngram_range=(1, 2)` cho tiếng Việt | 8 |

Chi tiết và nguồn: `research_presentation_llm_v1.md`, các file `noi_dung_mucXX_*_v1.md`.

---

## 7. Mốc thời gian

| Ngày | Việc |
|---|---|
| 1 | Chốt plan v3. **Toàn:** phần 0 setup + log, smoke run tải 2 model Qwen + tokenizer GPT-2 + UIT-VSFC trên Kaggle, ghi `env.json` |
| 2 | Muội, Phúc: dàn ý desc mục mình (`docs/parts/`), lý thuyết + slide, soạn file trong `data/`. Toàn: lý thuyết mục 7, 10; duyệt desc |
| 3 | Toàn: code phần 1, 2, 3, 4, 6 (mỗi phần smoke run); bàn giao cho Muội, Phúc |
| 4 | Toàn: code phần 5, 8, 9; phần 7 (hàm + `assert`, chạy thử 10 câu), chốt prompt trên split validation; bàn giao |
| 5 | Toàn: chạy 500 câu × 4, metric, McNemar, phân tích lỗi; phần 10 |
| 6 | Toàn: smoke run toàn notebook → Save & Run All (normal run, log), xuất HTML dự phòng; audit. Muội: README. Thư: review |
| 7 | Tập trình bày (mỗi người giải thích code phần mình trình bày), nộp |

Ngày 1 = 15/11/2026. Lịch giữ nguyên như bảng (Toàn chốt 16/11).

---

## 8. Còn cần chốt
1. ~~Tên repo, chủ repo~~: `pptthu/LLMs_DL` (https://github.com/pptthu/LLMs_DL), chủ repo: Thư (chốt 16/11).
2. ~~Mỗi người viết notebook riêng rồi ghép~~: đã chốt, Toàn viết toàn bộ trong một notebook (15/11).
3. ~~Cách chấm RAG và hallucination~~: đã chốt 16/11, tự động kiểu SQuAD + 2 người chấm mù (RAG theo RGB/RAGAS, hallucination theo SimpleQA/FActScore); xem `presentation_description_v2.md` mục Đánh giá.
4. ~~Bonferroni ở mục 10~~: đã chốt 16/11, dùng Bonferroni (0.05/3 ≈ 0.017), báo cả p chưa hiệu chỉnh.
