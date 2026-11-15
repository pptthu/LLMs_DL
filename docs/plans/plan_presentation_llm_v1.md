# Plan bài Presentation LLMs: cấu trúc thư mục và plan code (v1)

- Người gửi: Toàn | Tạo: 2026-11-14T16:30:00+07:00 | Người duyệt: Thư
- Trạng thái: **[đề xuất]**, chờ Thư duyệt. Chỗ ghi **[cần chốt]** là chỗ nhóm phải quyết trước khi tạo thư mục.
- Dựa trên: `ke_hoach_presentation_llms.md`, `khung_presentation_llms.md` (bản 2026-11-14), quy ước repo Lab-DL và Lab 1.
- Thay đổi so với v0: thêm plan code (chữ ký hàm, luồng dữ liệu, test, cách chạy), lý do dùng Hugging Face, giải thích các điểm cần chốt.

---

## 1. Tóm tắt

- Đặt bài presentation vào repo Lab-DL, ở thư mục `labs/llm_presentation/` **[cần chốt]**, để dùng lại `shared/paths.py`, `shared/seed.py`, `pytest` và quy tắc Git.
- Gốc thư mục gom thành `data/` (vào), `outputs/` (ra), `docs/` (tài liệu), `src/` (code có test), `notebook/`, `slides/`, `tests/`. Không chia mỗi phần một folder, vì cô từng chê Lab 1 "phân rã quá" (`labs/lab1/docs/rules/FOLDER_STRUCTURE_v2.md` dòng 61).
- **Hàm tính số liệu** (dựng prompt, parse nhãn, accuracy, McNemar, chia đoạn RAG) để trong `src/`, có test. **Code gọi model và in output** viết thẳng trong notebook, có comment từng dòng.
- Mọi số trên slide lấy từ **1 lần Save & Run All trên Kaggle**. Lab 1 từng mất điểm vì code, notebook và tài liệu ra 3 bộ số khác nhau (`labs/lab1/NHAN_XET_LAB1.md`).

---

## 2. Rule và khuôn dùng lại từ repo

| File | Dùng gì |
|---|---|
| `AGENTS.md`, `docs/agent/dl-lab-workflow/SKILL.md` | Mỗi lần làm 1 module; chạy rồi mới nói chạy được; dòng 8 của SKILL: **giải thích được từng dòng** mình nộp |
| `docs/workflow.md` | Quy trình của cô: Define → Design → Break down → Build MVP → Implement → Test → Refactor → Commit → Document |
| `CONTRIBUTING.md` | Nhánh, commit `<type>(<scope>): ...`, PR, không commit dữ liệu |
| `labs/lab1/docs/rules/RULES_v1.md` | Version `_vN` (dòng 12); smoke run trước normal run (dòng 14); số liệu phải có file kết quả và log (dòng 22); Thư duyệt 4 cổng (dòng 28) |
| `labs/lab1/docs/rules/FOLDER_STRUCTURE_v2.md` | Mẫu `data/`, `outputs/`, `docs/` |
| `labs/lab1/docs/templates/DESC_TEMPLATE_v1.md`, `DESC_TEMPLATE_v3.md` | Desc cả bài; desc từng mục 1–10 |
| `docs/templates/experiment_note.md` | Ghi chép thí nghiệm mục 7/10 |
| `labs/lab1/kich_ban_data/KICH_BAN_TOAN_v4_NGAN.md` | Khuôn kịch bản: bảng "Mở sẵn trước khi thuyết trình" (dòng 7). Dùng cho yêu cầu mở code trình bày |
| `AI_usage.md` | Mỗi người tự ghi phần dùng AI |

---

## 3. Cấu trúc thư mục

```text
labs/llm_presentation/
├── README.md                 # Hướng dẫn chạy: Kaggle, thư viện + phiên bản, môi trường, thời gian chạy (Muội)
├── project_description.md    # Define + Design: đề của cô, 10 mục, phạm vi, chữ ký hàm ở §5 (là "hợp đồng")
├── progress_status.md        # Tiến độ, bằng chứng chạy (đường dẫn file kết quả), nhật ký
├── requirements.txt          # Ghim phiên bản, lấy từ pip freeze lần chạy thật (Phúc)
│
├── src/                      # Hàm có test; mỗi file 1 người sửa
│   ├── config.py             # MODEL_NAME, SEED, N_PER_CLASS, SMOKE... — đổi tham số chỉ sửa ở đây (Phúc)
│   ├── setup.py              # Phần 0  (Phúc)
│   ├── architecture.py       # Phần 2  (Muội)
│   ├── tokenization.py       # Phần 3  (Muội)
│   ├── learning.py           # Phần 4  (Phúc)
│   ├── generation.py         # Phần 6  (Phúc)
│   ├── icl_experiment.py     # Phần 7, 10  (Toàn)
│   ├── rag.py                # Phần 8  (Muội)
│   ├── limitations.py        # Phần 9  (Phúc)
│   └── summary.py            # Phần 10: gom kết quả các phần thành 1 bảng  (Toàn)
│
├── notebook/
│   └── llm_presentation.ipynb   # 1 notebook chung, phần 0–10, chạy trên Kaggle (Toàn ghép)
│
├── data/
│   ├── raw/                  # UIT-VSFC gốc, slide môn học cho RAG — KHÔNG commit
│   ├── splits/               # test_500_ids.json, fewshot_pools.json — commit
│   └── test_sets/            # Bộ câu tự soạn — commit [cần chốt, §8 ý 2]
│       ├── tasks_prompts.json        # mục 1: 4 tác vụ            (Muội)
│       ├── base_vs_instruct.json     # mục 5                        (Phúc)
│       ├── rag_questions.csv         # mục 8: 15–20 câu + đáp án    (Muội)
│       ├── hallucination.csv         # mục 9: ~20 câu + đáp án đúng (Phúc)
│       ├── bias_prompts.json         # mục 9                        (Phúc)
│       └── long_context.json         # mục 9                        (Phúc)
│
├── outputs/                  # Code ghi kết quả ở đây; slide lấy số từ đây
│   ├── logs/                 # env.json, <ts>_<phần>_config.json
│   ├── results/              # CSV từng phần (§5.10)
│   ├── figures/              # PNG chèn vào slide
│   └── _smoke/               # Chạy thử, không dùng số để báo cáo
│
├── slides/
│   ├── llm_presentation_vN.pptx  # Bản ghép (Muội)
│   └── llm_presentation_final.pdf
│
├── tests/                    # pytest cho src/, KHÔNG tải model (dữ liệu giả)
│   ├── test_smoke.py         # Import được mọi hàm, đúng chữ ký trong project_description.md
│   ├── test_architecture.py  test_tokenization.py  test_learning.py
│   ├── test_generation.py    test_icl_experiment.py  test_rag.py
│   └── test_limitations.py   test_summary.py
│
└── docs/
    ├── 00_README_v1.md       # Mục lục + thứ tự đọc
    ├── presentation_description_v1.md   # Desc cả bài (DESC_TEMPLATE_v1)
    ├── rules/                # RULES_v1.md (copy Lab 1, sửa theo §8), FOLDER_STRUCTURE_v1.md (bản duyệt của §3)
    ├── templates/            # DESC v1, phase v3, RUN_LOG_TEMPLATE_v1, experiment_note
    ├── plans/                # ke_hoach_presentation_llms_v1.md, khung_presentation_llms_v1.md, file này
    ├── phases/               # muc01_..._v1.md … muc10_..._v1.md
    ├── experiments/          # experiment_log.md, error_analysis_v1.md (Toàn)
    ├── scripts/              # kich_ban_{muoi,phuc,toan}_v1.md
    ├── qa/                   # qa_{muoi,phuc,toan}_v1.md (~5 câu mỗi người)
    └── references.md         # Tài liệu tham khảo (Muội)
```

**Sản phẩm nộp → đường dẫn**

| Sản phẩm (theo đề) | Đường dẫn |
|---|---|
| Slide PDF hoặc PPTX | `slides/llm_presentation_final.pdf` (+ `.pptx`) |
| Notebook + hướng dẫn chạy | `notebook/llm_presentation.ipynb` (bản Kaggle đã Save & Run All), `README.md`, `requirements.txt` |
| Bộ câu thử nghiệm, kết quả, phân tích lỗi | `data/test_sets/`, `data/splits/`, `outputs/results/`, `docs/experiments/error_analysis_v1.md` |
| Tài liệu tham khảo | `docs/references.md` |

---

## 4. Vì sao dùng Hugging Face

- Model Qwen2.5 (base và instruct) được công bố trên Hugging Face Hub; tải bằng `AutoTokenizer.from_pretrained`, `AutoModelForCausalLM.from_pretrained`.
- Thư viện `transformers` cho xem bên trong model, đúng thứ các demo cần:
  - logits và trọng số attention (mục 2);
  - tokenizer của GPT-2 và Qwen (mục 3);
  - loss khi đưa nhãn vào (mục 4);
  - `generate()` có greedy, temperature, top-p (mục 6).
- Gọi API kiểu ChatGPT chỉ nhận được chữ trả về, không xem được logits, loss hay attention.
- Môn học đã dùng Hugging Face ở Practice 3; Lab 3 của nhóm cũng làm trên Hugging Face.

---

## 5. Plan code

Chữ ký hàm dưới đây là **[đề xuất]**. Sau khi duyệt, chép vào `project_description.md` và coi là hợp đồng (`test_smoke.py` kiểm). Đổi chữ ký phải báo cả nhóm (`CONTRIBUTING.md`). Tham số model và tokenizer ghi kiểu chung (`model`, `tokenizer`) để file `src/` không phải import lớp cụ thể.

### 5.1 Luồng dữ liệu

```text
config.py ──► setup.py: set_seed, load_model ×2 (instruct, base), save_env_info → outputs/logs/env.json
                 │
                 ├─► architecture / tokenization / learning / generation   (câu ví dụ "Hôm nay trời rất đẹp.")
                 │        → outputs/results/*.csv, outputs/figures/*.png
                 │
                 ├─► icl_experiment: UIT-VSFC → test 500 câu + 3 bộ few-shot → chạy zero-shot, 3-shot ×3
                 │        → outputs/results/icl_*.csv → metric, McNemar
                 │
                 ├─► rag: slide môn → chia đoạn → TF-IDF → so không RAG / có RAG → rag_results.csv
                 ├─► limitations: hallucination.csv, bias_prompts.json, long_context.json → *_results.csv
                 │
                 └─► summary: đọc outputs/results/*.csv → summary_table.csv (slide 19–20)
```

### 5.2 `config.py` (Phúc)
- `MODEL_INSTRUCT = "Qwen/Qwen2.5-1.5B-Instruct"`, `MODEL_BASE = "Qwen/Qwen2.5-1.5B"`, `TOKENIZER_EXTRA = "gpt2"`
- `SEED = 42`, `EXAMPLE_VI = "Hôm nay trời rất đẹp."`, `EXAMPLE_EN = "The weather is very nice today."`
- `N_PER_CLASS = 250`, `K_SHOT = 3`, `N_FEWSHOT_SETS = 3`
- `SMOKE = False`. Khi `True`: mỗi phần chạy vài câu, ghi vào `outputs/_smoke/`.

### 5.3 `setup.py`: phần 0 (Phúc)
```python
def load_model(name: str, device: str) -> tuple[object, object]:
    """Tải (tokenizer, model) từ Hugging Face Hub, đặt model.eval()."""
def get_env_info() -> dict:
    """Phiên bản Python, torch, transformers; CUDA; tên GPU; RAM."""
def save_env_info(path: Path) -> Path:
    """Ghi get_env_info() ra JSON — nguồn số liệu cho slide 2."""
```
- Seed dùng `shared/seed.py::set_seed`, không viết lại.

### 5.4 `architecture.py`: phần 2 (Muội)
```python
def model_config_summary(model) -> dict:            # số layer, hidden size, số head, vocab size, số tham số
def causal_mask(n: int) -> torch.Tensor:             # ma trận bool n×n, True = vị trí tương lai bị chặn
def top_k_next_tokens(model, tokenizer, prompt: str, k: int = 5) -> pd.DataFrame:  # cột: token, prob
```

### 5.5 `tokenization.py`: phần 3 (Muội)
```python
def tokenize_text(tokenizer, text: str) -> pd.DataFrame:      # cột: token, token_id
def compare_token_counts(tokenizers: dict[str, object], texts: list[str]) -> pd.DataFrame:
    """Bảng số token mỗi câu theo từng tokenizer (GPT-2 vs Qwen)."""
```

### 5.6 `learning.py`: phần 4 (Phúc)
```python
def make_context_target_pairs(token_ids: list[int]) -> list[tuple[list[int], int]]:
    """[a,b,c,d] -> [([a],b), ([a,b],c), ([a,b,c],d)]"""
def per_token_loss(model, tokenizer, text: str) -> pd.DataFrame:  # cột: context, target, prob, loss
def sequence_loss(model, tokenizer, text: str) -> float:          # cross-entropy trung bình của cả câu
```

### 5.7 `generation.py`: phần 6 (Phúc)
```python
DECODING_CONFIGS: dict[str, dict]   # greedy; temp0.7_top_p0.9; temp1.0; temp1.5; top_p0.5
def generate_step_by_step(model, tokenizer, prompt: str, max_new_tokens: int = 10) -> pd.DataFrame:
    """Mỗi bước 1 dòng: step, token chọn, prob."""
def sample_outputs(model, tokenizer, prompt: str, config: dict, n: int = 5, seed: int = 42) -> list[str]:
```

### 5.8 `icl_experiment.py`: phần 7, 10 (Toàn)
```python
LABELS = ("positive", "negative")
def load_uit_vsfc(path: Path) -> pd.DataFrame:      # cột: text, label (chỉ 2 nhãn); nguồn dữ liệu [cần chốt]
def make_test_split(df: pd.DataFrame, n_per_class: int = 250, seed: int = 42) -> list[int]:
def make_fewshot_pools(df: pd.DataFrame, exclude_ids: list[int], k: int = 3,
                       n_sets: int = 3, seed: int = 42) -> list[list[dict]]:
    """3 bộ ví dụ, KHÔNG lấy câu nằm trong test (tránh rò dữ liệu)."""
def build_prompt(text: str, examples: list[dict]) -> str:      # examples rỗng = zero-shot
def parse_label(output: str) -> str | None:                    # None = không parse được
def run_icl(model, tokenizer, df_test: pd.DataFrame, examples: list[dict],
            max_new_tokens: int = 5) -> pd.DataFrame:
    """Greedy decoding. Cột: text, true, prompt, output, pred."""
def compute_metrics(df: pd.DataFrame) -> dict:      # accuracy, unparseable_rate, confusion_matrix
def mcnemar_counts(correct_a: list[bool], correct_b: list[bool]) -> tuple[int, int]:  # (b, c)
def mcnemar_pvalue(b: int, c: int) -> float:        # statsmodels
```
- Bản instruct cần định dạng hội thoại: `build_prompt` trả chuỗi nội dung, `run_icl` bọc bằng `tokenizer.apply_chat_template`. Kiểm lại theo phiên bản `transformers` đã cài.
- Câu không parse được: tính là **sai** khi tính accuracy, và báo riêng tỉ lệ.
- Mỗi lần chạy lưu `outputs/results/icl_zero_shot.csv` và `icl_3shot_set{1,2,3}.csv`.

### 5.9 `rag.py`: phần 8 (Muội)
```python
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
def build_index(chunks: list[str]) -> tuple[object, object]:   # (TfidfVectorizer, ma trận) — scikit-learn
def retrieve(query: str, vectorizer, matrix, chunks: list[str], k: int = 3) -> list[str]:
def build_rag_prompt(question: str, contexts: list[str]) -> str:
```
- Chấm đúng/sai: so với đáp án trong `rag_questions.csv`. Cách chấm (tự động bằng từ khoá hay người chấm) **[cần chốt]**.

### 5.10 `limitations.py` (Phúc) và `summary.py` (Toàn)
```python
# limitations.py
def run_qa_set(model, tokenizer, df: pd.DataFrame) -> pd.DataFrame:   # cột: question, answer_true, output
def make_long_context(fact: str, filler: str, position: str) -> str:  # position: "start" | "middle" | "end"
# summary.py
def build_summary_table(results_dir: Path) -> pd.DataFrame:           # gom mọi CSV → summary_table.csv
```

**File kết quả** (`outputs/results/`):
- `tasks_outputs.csv`, `top5_next_tokens.csv`, `token_counts.csv`, `per_token_loss.csv`
- `base_vs_instruct.csv`, `decoding_samples.csv`
- `icl_*.csv`, `icl_metrics.json`, `rag_results.csv`
- `hallucination_results.csv`, `bias_results.csv`, `long_context_results.csv`
- `summary_table.csv`

### 5.11 Test (không tải model)
| File | Kiểm gì |
|---|---|
| `test_smoke.py` | Import được mọi hàm, chữ ký đúng §5 |
| `test_architecture.py` | `causal_mask(4)`: đường chéo và phía dưới là False, phía trên là True |
| `test_learning.py` | `make_context_target_pairs([1,2,3,4])` ra đúng 3 cặp |
| `test_icl_experiment.py` | `parse_label` (hoa/thường, thừa chữ, không có nhãn → None); `make_test_split` cân bằng 250/250, cùng seed ra cùng kết quả; `make_fewshot_pools` không trùng câu test; `mcnemar_counts` trên dữ liệu giả; `compute_metrics` khi có câu None |
| `test_rag.py` | `chunk_text` đúng độ dài và overlap; `retrieve` trả đúng đoạn trên corpus giả 3 câu |
| `test_summary.py` | Gom đúng từ thư mục tạm có 2 CSV giả |

Lệnh: `python -m pytest labs/llm_presentation/tests`

### 5.12 Notebook
- Mỗi phần một tiêu đề `## Phần X — <tên> (mục X)`. Mỗi cell code có:
  - cell markdown đứng trước: mục đích, đầu vào, đầu ra, mục lý thuyết liên quan;
  - comment tiếng Việt cho dòng không hiển nhiên;
  - cell markdown "Phân tích" sau output: 2–3 câu.
- Phần 0 tải model **một lần**; các phần sau dùng lại.
- Đầu notebook có `SMOKE`. Chạy thử với `SMOKE = True`, chạy chính thức với `SMOKE = False`.

### 5.13 Chạy thử và chạy chính thức
- **Chạy thử:** `SMOKE = True`, mỗi phần vài câu, ra `outputs/_smoke/`. Đạt khi chạy hết không lỗi và mọi CSV có đủ cột.
- **Chạy chính thức:** `SMOKE = False`, Save & Run All trên Kaggle. Lưu `env.json`, `outputs/`, bản notebook có output, và ghi vào `docs/experiments/experiment_log.md` (ngày, phiên bản, thời gian chạy).
- Sửa code sau lần chạy chính thức thì **chạy lại toàn bộ** rồi mới cập nhật số trên slide.

---

## 6. Mốc thời gian (theo kế hoạch nhóm)

| Ngày | Việc code |
|---|---|
| 1 | Chốt §8. Phúc: `config.py`, `setup.py`, chạy thử tải Qwen trên Kaggle. Toàn: tạo khung thư mục, `test_smoke.py` |
| 2 | Mọi người: desc mục mình trong `docs/phases/`. Phúc, Muội: soạn bộ câu trong `data/test_sets/` |
| 3 | Muội: `architecture`, `tokenization`. Phúc: `learning`, `generation`, `limitations`. Toàn: `icl_experiment` (hàm + test) |
| 4 | Muội: `rag`. Phúc: phần 5. Toàn: chạy thử ICL trên vài câu |
| 5 | Toàn: chạy 500 câu, metric, McNemar, `error_analysis_v1.md` |
| 6 | Toàn: `summary.py`, ghép notebook, chạy chính thức, xuất HTML dự phòng. Thư: review |
| 7 | Tập thuyết trình, nộp |

Git: mỗi người một nhánh `llm/<module>` (ví dụ `llm/icl_experiment`); commit `feat(llm/icl_experiment): ...`; PR vào `main`, Thư review.

---

## 7. Rủi ro

| Rủi ro | Cách xử lý |
|---|---|
| Qwen 1.5B chạy chậm hoặc hết RAM trên T4 | Dự phòng 0.5B; tải model một lần; giảm `max_new_tokens` |
| Model trả lời lạc đề, không ra nhãn | `parse_label` trả None, báo tỉ lệ riêng; few-shot thường giảm tỉ lệ này |
| Số trên slide lệch với notebook | Chỉ lấy số từ `outputs/results/` của lần chạy chính thức; ghi đường dẫn file dưới mỗi bảng trong slide |
| Câu few-shot trùng câu test | `make_fewshot_pools` loại câu test; có test kiểm |
| Kaggle không có `src/` | Xem §8 ý 3 |

---

## 8. Cần chốt (Thư và nhóm)

1. **Rule notebook.** `labs/lab1/docs/rules/RULES_v1.md` dòng 20 ghi "Không để kết quả trong notebook", nhưng cô yêu cầu notebook đã chạy sẵn có output. Đề xuất: làm cả hai; notebook giữ output, code vẫn ghi file vào `outputs/`.
2. **Thư mục `data/test_sets/`.** `shared/paths.py` dòng 9 chỉ cho 2 loại `("raw", "splits")`; gọi `data_dir(..., "test_sets")` sẽ báo lỗi. Hai cách:
   - (A) thêm `"test_sets"`. Đây là file dùng chung cả repo nên phải báo nhóm trưởng.
   - (B) không sửa, để bộ câu vào `data/splits/`.
3. **Đưa `src/` lên Kaggle.** Kaggle là máy khác, không có file trong repo, nên `import` sẽ lỗi `ModuleNotFoundError`. Ba cách:
   - (a) nén `src/` thành Kaggle Dataset (đề xuất, không cần token);
   - (b) `git clone` repo trong notebook (repo private thì cần token, không được để token trong notebook);
   - (c) bỏ `src/`, viết hết trong notebook (đơn giản nhất nhưng mất test).
4. **Vị trí thư mục:** trong repo (`labs/llm_presentation/`) hay ngoài repo.
5. **Ngôn ngữ tài liệu:** `AGENTS.md` dòng 8 yêu cầu Anh trước Việt sau; Lab 1 viết tiếng Việt (ngoại lệ, `RULES_v1.md` dòng 23). Đề xuất: tiếng Việt như Lab 1.
6. **Nguồn UIT-VSFC:** tải từ Hugging Face Hub hay file gốc; kiểm khi chạy thử Ngày 1.
7. **Cách chấm RAG và hallucination:** tự động bằng từ khoá hay người chấm.
8. **Slide môn học dùng cho RAG:** đề xuất không commit, chỉ commit `rag_questions.csv`.

Sau khi Thư duyệt: lưu §3 thành `docs/rules/FOLDER_STRUCTURE_v1.md`, chép §5 vào `project_description.md`, rồi tạo khung thư mục trên nhánh `chore/scaffold-llm-presentation`.
