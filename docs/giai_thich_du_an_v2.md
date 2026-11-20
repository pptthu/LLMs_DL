# Giải thích lý thuyết, code và kết quả — Presentation LLMs (v2)

- **Created**: 2026-11-20 (Toàn, AI hỗ trợ) · Trạng thái: **[đề xuất]**, chờ Toàn duyệt
- **Dành cho:** người mới học (Muội, Phúc, người nghe). Đọc theo đúng thứ tự cell trong notebook: **mỗi cell có lý thuyết dùng ở cell đó → code → output thật → giải thích ngay tại chỗ** (vì sao ra số đó, công thức nào).
- **Đối chiếu code:** nhánh `main` tại `2ecf860` (Merge PR #7, kiểm 2026-11-20); notebook `notebook/llm_presentation.ipynb` sửa lần cuối ở commit `cc5782f` (19/11, 130 cell).
- **Số liệu:** lần chạy chính thức **`20261119_150356_normal`** (Kaggle, GPU T4, float16, seed 42, 7.4 phút). Hình ở `outputs/figures/`, bảng ở `outputs/results/`.
- **Code trong file này:** chép nguyên văn từ notebook. Dòng ghi log (`log(...)`, ghi `*_records.csv`) và vài dòng in phụ được lược bớt, chỗ lược ghi `# ...`. Muốn xem nguyên cell: mở notebook, tìm số cell (ví dụ "# 4.2").
- **Khác bản v1:** `giai_thich_du_an_v1.md` (giữ nguyên) tóm tắt code theo phần. Bản v2 đặt **lý thuyết, công thức, code, output và giải thích cạnh nhau theo từng cell**, phân tích từng hình, kiểm lại số bằng công thức, ghi vài điểm lạ (§13).
- **Ký hiệu nguồn** (`AGENTS.md` mục 7): **[Slide]** slide môn học; **[Bổ sung]** có nguồn trong `docs/references.md`; **[Bổ sung\*]** nguồn chưa có trong `references.md`, đề xuất thêm ở §15; **[Tự tính]** tính lại bằng Python từ số trong notebook, khớp đến chữ số in ra; **[Suy luận]** giả thuyết, chưa kiểm bằng thí nghiệm riêng.

---

## Mục lục
0. [Yêu cầu của cô và bản đồ notebook](#0-yêu-cầu-của-cô-và-bản-đồ-notebook)
1. [Kiến thức toán tối thiểu](#1-kiến-thức-toán-tối-thiểu)
2. [Phần 0 — Setup (cell 0.1–0.8)](#2-phần-0--setup)
3. [Phần 1 — Mục 1 Giới thiệu (cell 1.1–1.5)](#3-phần-1--mục-1-giới-thiệu)
4. [Phần 2 — Mục 2 Kiến trúc (cell 2.1–2.5, 3 hình)](#4-phần-2--mục-2-nền-tảng-kiến-trúc)
5. [Phần 3 — Mục 3 Tokenization (cell 3.1–3.5)](#5-phần-3--mục-3-tokenization)
6. [Phần 4 — Mục 4 Nguyên lý học (cell 4.1–4.5)](#6-phần-4--mục-4-nguyên-lý-học)
7. [Phần 5 — Mục 5 Giai đoạn huấn luyện (cell 5.1–5.4)](#7-phần-5--mục-5-các-giai-đoạn-huấn-luyện)
8. [Phần 6 — Mục 6 Sinh văn bản (cell 6.1–6.4)](#8-phần-6--mục-6-quá-trình-sinh-văn-bản)
9. [Phần 7 — Mục 7 Zero-shot vs few-shot (cell 7.1–7.5)](#9-phần-7--mục-7-khả-năng-của-llm)
10. [Phần 8 — Mục 8 RAG (cell 8.1–8.4)](#10-phần-8--mục-8-cách-thích-nghi-và-bổ-sung-tri-thức)
11. [Phần 9 — Mục 9 Hạn chế (cell 9.1–9.3)](#11-phần-9--mục-9-hạn-chế-và-đánh-giá)
12. [Phần 10 — Mục 10 Thử nghiệm có kiểm soát (cell 10.1–10.4)](#12-phần-10--mục-10-demo-và-kết-luận)
13. [Điểm lạ phát hiện khi đọc lại notebook](#13-điểm-lạ-phát-hiện-khi-đọc-lại-notebook)
14. [Bảng công thức tổng hợp](#14-bảng-công-thức-tổng-hợp)
15. [Tài liệu tham khảo](#15-tài-liệu-tham-khảo)

---

## 0. Yêu cầu của cô và bản đồ notebook

**Topic:** Tìm hiểu mô hình ngôn ngữ lớn — Large Language Models (LLMs).
**Mục tiêu:** giải thích được cách LLM **học** và **sinh văn bản**, phân biệt **các giai đoạn huấn luyện**, nhận biết **khả năng** và **hạn chế**.

| # | Cô yêu cầu trình bày | Cô yêu cầu minh họa | Lý thuyết chính | Cell code | Output / hình |
|---|---|---|---|---|---|
| 1 | LLM là gì; LM vs LLM vs chatbot | Hỏi đáp, tóm tắt, dịch, sinh mã | LM, LLM, chatbot, chat template | 1.1–1.5 | `tasks_outputs.csv` |
| 2 | Transformer; decoder-only, causal attention | Sơ đồ token → xác suất token tiếp theo | Self-attention, causal mask, GQA, RoPE, softmax | 2.1–2.5 | `causal_mask.png`, `attention_qwen.png`, `top5_next_tokens.csv` |
| 3 | Token, token ID, embedding | Tokenize câu Việt và câu Anh | BPE, byte-level, UTF-8, embedding | 3.1–3.5 | `token_counts.csv` |
| 4 | Next-token prediction, dữ liệu, cross-entropy | Cặp ngữ cảnh → token mục tiêu | Cross-entropy, dịch nhãn, perplexity | 4.1–4.5 | `per_token_loss.csv`, `loss_summary.csv` |
| 5 | Pretraining, SFT, alignment; RLHF/DPO | Sơ đồ mục tiêu + dữ liệu từng giai đoạn | SFT loss, reward model, PPO + KL, DPO | 5.1–5.4 | `base_vs_instruct.csv` |
| 6 | Autoregressive; greedy, temperature, top-p | Cùng prompt, nhiều cấu hình sinh | Vòng sinh, argmax, $p^{1/T}$, nucleus | 6.1–6.4 | `decoding_*.csv`, `temperature_top_p.csv` |
| 7 | In-context learning, làm theo chỉ dẫn | Zero-shot vs few-shot cùng tác vụ | ICL, thí nghiệm một biến, chống rò rỉ | 7.1–7.5 | `icl_zero_shot.csv`, `icl_few_shot.csv` |
| 8 | Prompting vs fine-tuning vs RAG | Bảng dữ liệu cần có, khi nào dùng | TF-IDF, cosine, recall@k, LoRA | 8.1–8.4 | `rag_results.csv` |
| 9 | Hallucination, thiên lệch, giới hạn ngữ cảnh | 1 câu trả lời sai + cách kiểm chứng | Phân loại hallucination, SimpleQA | 9.1–9.3 | `hallucination_*.csv` |
| 10 | Thử nghiệm có kiểm soát, tổng hợp | Bảng kết quả, ví dụ, phân tích lỗi | Accuracy, CI, confusion, McNemar | 10.1–10.4 | `icl_metrics.json`, `confusion_matrix_icl.png`, `summary_table.csv` |

**Sản phẩm nộp:** slide PDF/PPTX; notebook + hướng dẫn chạy; bộ câu thử nghiệm + kết quả + phân tích lỗi; tài liệu tham khảo.
**Góp ý 11/11:** không cần fine-tune; **mục nào cũng có demo trên model thật kèm phân tích output**, kể cả mục 5 và 8.
**Yêu cầu thêm:** slide ghi thư viện, môi trường; mỗi mục có demo; có thực nghiệm; **giải thích được từng dòng notebook**; phần slide không có code thì mở notebook.

**Hình 0 — Sơ đồ notebook** (ảnh đính kèm cell đầu notebook):

```mermaid
flowchart LR
    S["0. Setup<br/>model + dữ liệu"] --> U1["1. Bốn tác vụ"]
    S --> I2["2. Kiến trúc"] --> I3["3. Token"] --> I4["4. Loss"] --> I5["5. Base / Instruct"] --> I6["6. Sinh văn bản"]
    S --> U7["7. Zero / few-shot"] --> E10["10. Kết quả + lỗi"]
    S --> U8["8. Không RAG / RAG"] --> E9["9. Hallucination"] --> E10
```

Ba nhóm: **Bên trong LLM** (2–6) đi theo đường dữ liệu trong model: kiến trúc → token → cách học → giai đoạn huấn luyện → cách sinh. **Dùng LLM** (1, 7, 8). **Đánh giá** (9, 10): phần 9 dùng lại câu hỏi phần 8, phần 10 dùng lại kết quả phần 7.

**Quy ước chung của mọi cell** (đọc một lần, áp dụng cả notebook):
- Trước mỗi cell code có markdown ghi **Mục đích / Đầu vào / Đầu ra**; sau output có cell **Phân tích**.
- `with torch.no_grad():` ở mọi chỗ chạy model: chỉ suy luận, không tính gradient, **không train**. `model.eval()` tắt dropout.
- Cuối cell gần như luôn có `assert` để tự kiểm: sai là notebook dừng, không lặng lẽ ra số sai.
- Gọi `generate()` luôn truyền tham số sinh tường minh (lý do: cell 0.4).

---

## 1. Kiến thức toán tối thiểu

| Khái niệm | Công thức / ý | Dùng ở cell |
|---|---|---|
| Vector, ma trận | Embedding mỗi token = vector 1,536 số; bảng embedding = ma trận 151,936 × 1,536 | 3.4 |
| Tích vô hướng | $a \cdot b = \sum_i a_i b_i$; hai vector cùng hướng thì lớn | 2.3, 8.2 |
| Xác suất có điều kiện | $P(B \mid A)$: xác suất B khi đã biết A | 2.4, 4.2 |
| Quy tắc dây chuyền [Bổ sung: Jurafsky & Martin ch. 3] | $P(x_1, \dots, x_n) = \prod_{i} P(x_i \mid x_1, \dots, x_{i-1})$ | 4.1 |
| Softmax [Slide Ch2] | $\text{softmax}(z)_i = e^{z_i} / \sum_j e^{z_j}$: biến điểm (logits) thành xác suất dương, cộng bằng 1 | 2.3, 2.4, 6.1 |
| Logarit | $\ln 1 = 0$; $p \to 0$ thì $-\ln p \to \infty$; $\ln(ab) = \ln a + \ln b$ | 4.2 |
| Nhị thức | $X \sim \text{Bin}(n, 0.5)$: số lần ngửa khi tung $n$ đồng xu công bằng | 10.2 |

**Vì sao quy tắc dây chuyền quan trọng:** nó cho thấy chỉ cần học "đoán token tiếp theo" là đủ để gán xác suất cho cả văn bản. Đây là nền của toàn bộ mục 2, 4, 6.

---

## 2. Phần 0 — Setup

Phần 0 không ứng với mục nào của đề, nhưng trả lời yêu cầu thêm 1, 2 của cô (thư viện, môi trường) và chứng minh model tải đúng.

### Cell 0.1 — Cấu hình chung

> **Ý tưởng:** mọi tham số đặt một chỗ; **seed** cố định để chạy lại ra đúng kết quả; **SMOKE** để chạy thử ít mẫu không đè kết quả thật.

```python
SMOKE: bool = False  # True: chạy thử vài mẫu; False: chạy chính thức (Save & Run All)
SEED: int = 42
MODEL_INSTRUCT = "Qwen/Qwen2.5-1.5B-Instruct"  # model chính
MODEL_BASE = "Qwen/Qwen2.5-1.5B"               # bản base, chỉ tải ở phần 5
TOKENIZER_GPT2 = "openai-community/gpt2"       # chỉ lấy tokenizer, dùng ở phần 3
DTYPE = torch.float16                          # T4 không hỗ trợ bf16 bằng phần cứng
N_TEST: int | None = 20 if SMOKE else None     # None = toàn bộ 2,999 câu test
# ...
OUT_DIR = ROOT / "outputs" / "_smoke" if SMOKE else ROOT / "outputs"
RUN_ID = RUN_START.strftime("%Y%m%d_%H%M%S") + ("_smoke" if SMOKE else "_normal")
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
```

**Giải thích:**
- `DTYPE = torch.float16`: mỗi tham số 2 byte thay vì 4 (float32) → model 1.54 tỉ tham số chiếm ≈ 3.1 GB. T4 không có phần cứng `bf16`, nên chọn fp16.
- `N_TEST = None` → phần 7 chạy toàn bộ 2,999 câu.
- `OUT_DIR`: smoke ghi vào `outputs/_smoke/`, normal ghi vào `outputs/` → chạy thử không đè kết quả thật.
- `RUN_ID`: ngày giờ Việt Nam + chế độ → tên thư mục log.
- Ba lệnh `seed`: cố định mọi bộ sinh số ngẫu nhiên (Python, NumPy, PyTorch cả CPU và GPU).

**Output:** `RUN_ID : 20261119_150356_normal`, `N_TEST : toàn bộ test 2 nhãn`, `DATA_DIR: Kaggle Dataset llms-dl-data`.

### Cell 0.2 — Hàm ghi log
`log(text, part=None, to_run=False)`: không có `part` → ghi `run.log`; có `part` → ghi `partNN_<tên>.log`; mở file chế độ `"a"` (ghi thêm, không đè). Mọi phần mở bằng `=== PART NN ... START ===`, đóng bằng `=== ... END ===`.

### Cell 0.3 — Môi trường

```python
assert torch.cuda.is_available(), "Chưa bật GPU: Settings → Accelerator → GPU T4"
ENV = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
       "gpu": torch.cuda.get_device_name(0), "gpu_count": torch.cuda.device_count(),
       "gpu_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1), ...}
for lib in ["transformers", "accelerate", "pandas", "pyarrow", "scikit-learn", "statsmodels", "matplotlib"]:
    ENV[lib] = pkg_version(lib)          # importlib.metadata.version(): đọc phiên bản đã cài
```

**Output thật** (chép lên slide môi trường):
```text
python: 3.13.16 | torch: 2.11.0+cu130 | cuda: 13.0 | gpu: Tesla T4 | gpu_count: 2 | gpu_vram_gb: 14.6
transformers: 5.20.2 | accelerate: 1.15.0 | pandas: 2.3.3 | pyarrow: 23.0.1
scikit-learn: 1.6.1 | statsmodels: 0.15.0 | matplotlib: 3.10.0
```
**Giải thích:** phiên bản đọc bằng code, không gõ theo trí nhớ. `gpu_count: 2` vì Kaggle cấp T4 ×2; model chỉ dùng GPU 0 (cell 0.4).

| Thư viện | Dùng ở | Làm gì |
|---|---|---|
| `torch` | mọi phần | tensor, chạy model trên GPU, softmax |
| `transformers` | mọi phần | tải model, tokenizer, `generate()`, chat template |
| `accelerate` | 0.4 | hỗ trợ `device_map` |
| `pandas`, `pyarrow` | 0.7, 8.1 | đọc Parquet, làm bảng, ghi CSV |
| `scikit-learn` | 8.2 | TF-IDF |
| `statsmodels` | 10.2 | McNemar |
| `matplotlib` | 2.2, 2.3, 10.1 | vẽ hình |

### Cell 0.4 — Tải tokenizer và model chính

> **Lý thuyết:** `from_pretrained(tên)` tải file cấu hình + trọng số từ `huggingface.co/<tên>` về bộ nhớ đệm. **`generation_config`** là cấu hình sinh mặc định nhà phát triển đóng gói kèm model.

```python
tok = AutoTokenizer.from_pretrained(MODEL_INSTRUCT)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_INSTRUCT,
    torch_dtype=DTYPE,   # float16
    device_map={"": 0},  # cả model lên GPU 0; "auto" có thể chia sang GPU 1 của T4 x2
)
model.eval()  # chỉ suy luận, không train (tắt dropout)
print(model.generation_config)
MODEL_REVISION = getattr(model.config, "_commit_hash", None) or model_info(MODEL_INSTRUCT).sha
```

**Output thật:**
```text
GenerationConfig { "bos_token_id": 151643, "do_sample": true, "eos_token_id": [151645, 151643],
  "pad_token_id": 151643, "repetition_penalty": 1.1, "temperature": 0.7, "top_k": 20, "top_p": 0.8 }
model chính: Qwen/Qwen2.5-1.5B-Instruct (revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306) | device: cuda:0
```

**Giải thích ngay tại output:**
- `"do_sample": true, temperature 0.7, top_p 0.8, top_k 20` → **mặc định bản Instruct lấy mẫu ngẫu nhiên**. Gọi `generate()` mà không truyền gì thì mỗi lần chạy ra một câu khác → không so sánh được. Vì vậy mọi phần sau truyền `do_sample=False, temperature=None, top_p=None, top_k=None` (greedy thuần). Lý thuyết lấy mẫu: phần 6.
- `"repetition_penalty": 1.1`: phạt token đã xuất hiện để giảm lặp (tài liệu `GenerationConfig` của Hugging Face: "1.0 means no penalty"; cơ chế gốc từ CTRL, Keskar et al. 2019 [Bổ sung\*]). Phần 5–9 đặt rõ 1.0 (không phạt); phần 1 không đặt nên dùng 1.1 (§13).
- `eos_token_id` có 2 giá trị: 151645 = `<|im_end|>` (hết lượt chat), 151643 = `<|endoftext|>` (hết văn bản). Gặp một trong hai thì dừng sinh (dùng ở phần 5).
- `revision`: mã commit trên Hub → người khác tải đúng bản trọng số này.

### Cell 0.5 — Kiến trúc và số tham số

> **Lý thuyết:** số tham số = tổng số phần tử của mọi ma trận trọng số. Đếm bằng code rồi đối chiếu model card để chắc chắn tải đúng model.

```python
cfg = model.config
ARCH = {"num_hidden_layers": cfg.num_hidden_layers, "hidden_size": cfg.hidden_size,
        "num_attention_heads": cfg.num_attention_heads, "num_key_value_heads": cfg.num_key_value_heads,
        "intermediate_size": cfg.intermediate_size, "vocab_size": cfg.vocab_size,
        "max_position_embeddings": cfg.max_position_embeddings, "tie_word_embeddings": cfg.tie_word_embeddings}
ARCH["n_params"] = sum(p.numel() for p in model.parameters())   # mỗi tensor đếm 1 lần
assert ARCH["n_params"] == 1_543_714_304, ARCH["n_params"]
```

**Output thật:**
```text
num_hidden_layers: 28 | hidden_size: 1,536 | num_attention_heads: 12 | num_key_value_heads: 2
intermediate_size: 8,960 | vocab_size: 151,936 | max_position_embeddings: 32,768
tie_word_embeddings: True | n_params: 1,543,714,304
```

**Giải thích:**
- `p.numel()` = số phần tử của tensor `p`; cộng hết = số tham số. `parameters()` trả mỗi tensor **một lần**, nên ma trận embedding dùng chung với LM head (`tie_word_embeddings: True`) chỉ đếm một lần.
- `max_position_embeddings: 32,768` = **giới hạn ngữ cảnh** (dùng ở mục 9).
- Phân rã chi tiết con số 1,543,714,304: cell 2.1.

### Cell 0.6 — Tokenizer GPT-2
Chỉ tải tokenizer (vài MB), không tải model. `assert gpt2_tok.vocab_size == 50_257`. Dùng ở phần 3 để so số token.

### Cell 0.7 — UIT-VSFC

```python
EXPECTED_ROWS = {"train": 11_426, "validation": 1_583, "test": 3_166}
LABEL_NAMES = {0: "negative", 2: "positive"}  # bỏ 1 = neutral
for split, n_expected in EXPECTED_ROWS.items():
    df = pd.read_parquet(f"{UIT_VSFC_BASE}/{split}/0000.parquet")
    assert len(df) == n_expected, (split, len(df))
    df = df[df["sentiment"].isin(LABEL_NAMES)].reset_index(drop=True)  # bỏ neutral
    vsfc[split] = df
```

**Output thật:**
```text
UIT-VSFC train: 11,426 dòng gốc → 10,968 dòng sau khi bỏ neutral {'positive': 5643, 'negative': 5325}
UIT-VSFC validation: 1,583 dòng gốc → 1,510 dòng sau khi bỏ neutral {'positive': 805, 'negative': 705}
UIT-VSFC test: 3,166 dòng gốc → 2,999 dòng sau khi bỏ neutral {'positive': 1590, 'negative': 1409}
```

**Giải thích:**
- Đọc Parquet thẳng từ nhánh chuyển đổi của Hugging Face vì script tải cũ của bộ này không chạy với `datasets` ≥ 4.0.
- `isin(LABEL_NAMES)` giữ nhãn 0 và 2 → bài toán **2 nhãn** (neutral ít và mơ hồ).
- Test 1,590 / 1,409 = 53% / 47%: gần cân bằng. Mốc "luôn đoán positive" chỉ được 53% → accuracy 89–92% ở phần 7 là xa mốc này.

### Cell 0.8 — Lưu `env.json`, `config.json`
`save_json()` ghi UTF-8, `ensure_ascii=False` (giữ chữ Việt), rồi đọc lại và `assert`.

---

## 3. Phần 1 — Mục 1 Giới thiệu

### Lý thuyết dùng cho cả phần 1

**Language model (LM)** [Bổ sung: Jurafsky & Martin ch. 3; Bengio et al. 2003]: model gán xác suất cho chuỗi từ, tương đương dự đoán từ tiếp theo (quy tắc dây chuyền, §1). Cách cũ nhất là **n-gram**, ước lượng bằng đếm:

$$P(w_i \mid w_{i-1}) \approx \frac{\text{count}(w_{i-1}, w_i)}{\text{count}(w_{i-1})}$$

Bengio 2003 thay bảng đếm bằng mạng nơ-ron + embedding (neural LM).

**LLM** [Bổ sung: Zhao et al. 2023; Brown et al. 2020]: LM dựa trên Transformer, hàng tỉ tham số trở lên, huấn luyện trên rất nhiều văn bản (Qwen2.5: 18 nghìn tỉ token). Đủ lớn thì làm nhiều tác vụ chỉ bằng đổi prompt.

**Chatbot** [Bổ sung: OpenAI 2022]: là **ứng dụng**, không phải model: LLM đã qua post-training (bản Instruct) + giao diện hội thoại + quản lý lịch sử + lớp an toàn.

| | Language model | LLM | Chatbot |
|---|---|---|---|
| Là gì | Model xác suất trên chuỗi | LM Transformer cỡ lớn | Ứng dụng xây trên LLM |
| Ví dụ | n-gram, RNN LM | Qwen2.5-1.5B (base / Instruct) | ChatGPT, Qwen Chat |
| Làm được | Đoán từ tiếp | Nhiều tác vụ qua prompt | Hội thoại nhiều lượt |

**Vì sao một model làm nhiều tác vụ:** hỏi đáp, dịch, tóm tắt, viết code đều là "văn bản vào → sinh tiếp văn bản ra", có điều kiện theo prompt.

### Cell 1.1 — Hàm sinh cho bản Instruct

> **Lý thuyết ở cell này:** **chat template** = khung định dạng hội thoại mà bản Instruct đã thấy khi SFT (mục 5). Đưa prompt đúng khung thì model biết đây là câu hỏi cần trả lời. **Greedy** = mỗi bước chọn token xác suất cao nhất (lý thuyết đầy đủ: cell 6.1).

```python
def chat_generate(model, tok, prompt: str, max_new_tokens: int) -> tuple[str, int, int, float]:
    messages = [{"role": "user", "content": prompt}]
    # add_generation_prompt=True: thêm phần mở đầu lượt "assistant" để model trả lời
    enc = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    n_in = enc["input_ids"].shape[1]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,                       # greedy: mỗi bước lấy token xác suất cao nhất
            temperature=None, top_p=None, top_k=None,  # tắt tham số lấy mẫu mặc định của bản Instruct
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    new_ids = out[0, n_in:]  # chỉ giữ phần model sinh thêm
    text = tok.decode(new_ids, skip_special_tokens=True).strip()
    return text, n_in, len(new_ids), elapsed
```

**Giải thích từng dòng:**
- `messages`: danh sách lượt hội thoại; ở đây một lượt `user`.
- `apply_chat_template(..., add_generation_prompt=True)`: bọc vào khung và thêm `<|im_start|>assistant` ở cuối để model biết **đến lượt nó**.
- `n_in`: số token đầu vào sau khi bọc.
- `torch.cuda.synchronize()`: GPU chạy **bất đồng bộ** với Python; phải chờ GPU xong mới bấm giờ, nếu không thời gian đo sai.
- `out[0, n_in:]`: `generate` trả **cả prompt lẫn phần sinh**, nên cắt bỏ `n_in` token đầu.
- `skip_special_tokens=True`: bỏ `<|im_end|>` khi đổi về chữ.

**Output thật (log khung sau chat template và lần thử "Xin chào"):**
```text
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
Xin chào<|im_end|>
<|im_start|>assistant

INPUT  (prompt; 32 token sau chat template, gồm cả system prompt mặc định): Xin chào
OUTPUT (9/16 token, 2.50 s): 你好！有什么我可以帮助你的吗？
```

**Giải thích:**
- Qwen **tự thêm system prompt** tiếng Anh. Vì vậy "Xin chào" (2 từ) thành **32 token**.
- Lần đầu mất 2.50 s cho 9 token, các lần sau nhanh hơn nhiều (Dịch: 8 token 0.31 s). [Suy luận] Lần gọi GPU đầu tiên tốn thêm thời gian khởi tạo (nạp kernel CUDA).
- **Trả lời tiếng Trung** [Suy luận]: dữ liệu của Qwen có tỉ lệ tiếng Trung, tiếng Anh cao hơn nhiều tiếng Việt; system prompt tiếng Anh; "Xin chào" quá ngắn, ít ngữ cảnh tiếng Việt. Với greedy, chỉ cần token đầu có xác suất cao nhất là "你" thì cả câu trôi theo tiếng Trung, vì mỗi token sau dựa trên các token trước (vòng sinh, cell 6.1). *Cách kiểm:* dùng `top_k_next_tokens()` (cell 2.4) trên prompt đã qua chat template để xem top-5 token đầu.

### Cell 1.2 — Chạy 4 tác vụ

```python
TASKS: list[dict] = tasks_json["tasks"]   # data/tasks_prompts.json v2
assert [t["id"] for t in TASKS] == ["qa", "summary", "translate", "code"]
for idx, t in enumerate(TASKS):
    output, n_in, n_out, sec = chat_generate(model, tok, t["prompt"], t["max_new_tokens"])
    rec = {..., "truncated": n_out >= t["max_new_tokens"], "output": output, ...}  # chạm trần → bị cắt
```

**Output thật:**
| Tác vụ | Prompt | Output | Token ra |
|---|---|---|---|
| Hỏi đáp | Thủ đô của Nhật Bản là gì? Trả lời một câu. | "Thủ đô của Nhật Bản là Tokyo." | 10/32 |
| Tóm tắt | Đoạn 5 câu về môn Học sâu + "Tóm tắt đoạn trên trong 1 câu." | "Học sâu là một lĩnh vực học máy sử dụng mạng nơ-ron nhiều lớp…, bao gồm việc huấn luyện…, với hai phần giới thiệu…, kết thúc bằng việc mỗi nhóm thuyết trình…" | 104/256 |
| Dịch | Dịch sang tiếng Anh: Hôm nay trời rất đẹp. | "Today the weather is very beautiful." | 8/48 |
| Sinh mã | Viết hàm Python is_even(n)… | khối code `return n % 2 == 0` + `print` ví dụ + 2 đoạn giải thích | 216/512 |

**Giải thích:**
- **Cùng model, cùng greedy, chỉ đổi prompt → đổi tác vụ.** Đó là định nghĩa thực tế của "LLM đa tác vụ".
- `truncated` = False cả 4: không output nào chạm `max_new_tokens`, tức model **tự dừng** bằng `<|im_end|>` (học từ SFT).
- Tóm tắt dài 104 token: model 1.5B có xu hướng chép lại câu nguồn rồi nối bằng dấu phẩy; prompt chỉ yêu cầu "1 câu", không giới hạn số từ.
- Sinh mã 216 token: model tự thêm ví dụ và giải thích (văn phong "trợ lý hữu ích" học từ post-training).

### Cell 1.3 — Chạy `is_even` do model viết

```python
def extract_code(text: str) -> str:
    m = re.search(r"```(?:python|py)?[ \t]*\n(.*?)```", text, flags=re.DOTALL)
    return m.group(1) if m else text
ns: dict = {}                               # namespace riêng cho code của model
exec(model_code, ns)                        # KHÔNG phải sandbox: phải đọc code trước khi chạy
IS_EVEN_RESULTS = [ns["is_even"](n) for n in [0, 1, 2, -3]]
```

**Output thật:** in `True`, `False` (hai dòng `print` của model chạy khi `exec`), rồi `is_even[0, 1, 2, -3] → [True, False, True, False]`.

**Giải thích:**
- Regex lấy khối ```` ```python ... ``` ```` đầu tiên; `re.DOTALL` cho `.` khớp cả xuống dòng.
- `exec` chạy cả 2 dòng `print(is_even(4))`, `print(is_even(5))` của model → hai dòng `True`, `False` đầu tiên trong output.
- `is_even(-3)` = False đúng vì trong Python `-3 % 2 == 1` (dư lấy dấu của số chia) → `1 == 0` là False.
- Chấm bằng **chạy thật** chứ không đọc bằng mắt: chắc chắn hơn.

### Cell 1.4 — Chấm

| Tác vụ | Luật chấm (viết trước khi chạy) | Kết quả |
|---|---|---|
| Hỏi đáp | output chứa "tokyo" | đạt |
| Dịch | có "today" và một trong "weather/nice/beautiful/lovely" | đạt |
| Sinh mã | `is_even` chạy ra `[True, False, True, False]` | đạt (ghi chú: tự thêm lời dẫn, prompt không cấm) |
| Tóm tắt | chấm tay `MANUAL_JUDGE`: 1 câu, giữ ý chính, không thêm ý | đạt ("gần như ghép lại các câu gốc, rút gọn ít") |

**Tổng: đạt 4/4, bị cắt 0/4, thời gian sinh 13.2 s.**

### Cell 1.5 — Thử prompt khác (demo trực tiếp)
Prompt "Giải thích ngắn gọn mạng nơ-ron nhân tạo là gì, trong 2 câu." → "Mạng nơ-ron nhân tạo (Artificial Neural Network, ANN) là một mô hình仿生学的计算机系统…" (nửa Việt nửa Trung, 113 token). Cùng nguyên nhân với "Xin chào" ở cell 1.1: đây là **hạn chế phụ thuộc dữ liệu** (mục 9). Cell này không chấm.

---

## 4. Phần 2 — Mục 2 Nền tảng kiến trúc

### Lý thuyết dùng cho cả phần 2

**Vì sao Transformer thay RNN** [Slide Ch6 + Bổ sung: Vaswani et al. 2017]: RNN/LSTM đọc tuần tự, token 100 chờ 99 bước, khó song song, thông tin xa phai dần. Transformer dùng **self-attention**: mỗi token nhìn trực tiếp mọi token được phép trong một bước, tính song song trên GPU.

**Ba họ Transformer** [Bổ sung: HF LLM Course]:
| Họ | Ví dụ | Attention | Huấn luyện | Dùng cho |
|---|---|---|---|---|
| Encoder-only | BERT | Hai chiều | Che từ rồi đoán lại | Phân loại, trích xuất |
| **Decoder-only** | GPT, Llama, **Qwen** | **Một chiều (causal)** | **Đoán token tiếp theo** | Sinh văn bản, chat |
| Encoder-decoder | T5, BART | Encoder 2 chiều + decoder causal | Khôi phục đoạn bị che | Dịch, tóm tắt |

Hầu hết LLM là decoder-only vì bài toán huấn luyện (đoán token tiếp) **trùng** cách dùng (sinh từng token).

**Hình 1 — Sơ đồ token → xác suất token tiếp theo** (minh họa của đề; ảnh đính kèm đầu phần 2, số lấy từ `model.config`):

```mermaid
flowchart LR
    A["Token ID<br/>shape (1, n)"] --> B["Embedding<br/>151,936 × 1,536<br/>→ (1, n, 1536)"]
    subgraph D["Decoder block × 28 — shape giữ (1, n, 1536)"]
        C1["RMSNorm → Causal self-attention<br/>12 head Q / 2 head KV, RoPE<br/>+ residual"] --> C2["RMSNorm → FFN SwiGLU<br/>1,536 → 8,960 → 1,536<br/>+ residual"]
    end
    B --> C1
    C2 --> E["RMSNorm cuối"] --> F["LM head (chung trọng số embedding)<br/>→ logits (1, n, 151,936)"] --> G["softmax ở vị trí cuối<br/>P(token tiếp theo)"]
```

**Đọc hình theo shape:** "Hôm nay trời rất" → 6 token ID `(1, 6)` → embedding `(1, 6, 1536)` → 28 block giữ nguyên shape (attention trộn thông tin **giữa** các token; FFN biến đổi **từng** token) → LM head `(1, 6, 151936)` → lấy vị trí cuối, softmax → 151,936 xác suất cộng bằng 1. Kết quả thật của đường đi này: cell 2.4.

**Các khối trong sơ đồ** [Bổ sung: Vaswani 2017; model card Qwen2.5; Bổ sung\*: bài gốc từng kỹ thuật, §15]:

| Khối | Công thức | Vì sao dùng |
|---|---|---|
| Self-attention | $\text{Attention}(Q,K,V) = \text{softmax}\!\left(\dfrac{QK^\top}{\sqrt{d_k}} + M\right)V$, với $Q = XW_Q$, $K = XW_K$, $V = XW_V$ | Mỗi token lấy thông tin từ token khác theo mức liên quan |
| Causal mask | $M_{ij} = 0$ nếu $j \le i$; $-\infty$ nếu $j > i$ | Không nhìn tương lai (chi tiết cell 2.2) |
| Multi-head | 12 head chạy song song, nối lại rồi nhân $W_O$ | Mỗi head học một kiểu quan hệ |
| GQA | 12 head Q, 2 head K/V → 6 head Q dùng chung 1 cặp K, V | Giảm KV cache khi sinh, ít tham số |
| RoPE | Xoay từng cặp chiều của $q, k$ góc $m\theta_i$, $\theta_i = \text{base}^{-2(i-1)/d}$. Bài gốc dùng base = 10,000 [Bổ sung\*: Su et al. 2021 §3.2.2]; Qwen2.5 tăng lên 1,000,000 (kỹ thuật ABF) để kéo dài ngữ cảnh [Bổ sung: Qwen2.5 report §3.3] | Tích $q_m \cdot k_n$ chỉ phụ thuộc khoảng cách tương đối giữa hai vị trí → model biết thứ tự |
| RMSNorm | $\dfrac{x}{\sqrt{\frac1d\sum_i x_i^2 + \epsilon}} \odot g$ | Như LayerNorm, không trừ trung bình, rẻ hơn |
| Pre-norm + residual (Qwen2.5 report §2: "RMSNorm… with pre-normalization") | $h = x + \text{Attn}(\text{RMSNorm}(x))$; $y = h + \text{FFN}(\text{RMSNorm}(h))$ | Residual như ResNet [Slide Ch5]: gradient đi thẳng qua 28 block |
| SwiGLU FFN | $W_{\text{down}}\big(\text{SiLU}(W_{\text{gate}}x) \odot W_{\text{up}}x\big)$, $\text{SiLU}(x) = x\,\sigma(x)$ | Cổng nhân giúp biểu diễn tốt hơn; vì vậy FFN có **3** ma trận |
| LM head | logits $= h\,E^\top$ | Dùng chung $E$ với embedding → bớt 233 triệu tham số |

**Vì sao chia $\sqrt{d_k}$** ($d_k = 128$): nếu các thành phần của $q, k$ có trung bình 0, phương sai 1 thì $q \cdot k$ có phương sai $d_k$. Không chia thì điểm rất lớn, softmax "bão hoà" (một ô ≈ 1, còn lại ≈ 0), gradient gần 0.

### Cell 2.1 — Thông số và 1.5 tỉ tham số nằm ở đâu

```python
PARAM_GROUPS = {"embedding": 0, "attention": 0, "ffn": 0, "norm": 0, "khác": 0}
for name, p in model.named_parameters():
    key = ("embedding" if "embed_tokens" in name else "attention" if "self_attn" in name
           else "ffn" if "mlp" in name else "norm" if "norm" in name else "khác")
    PARAM_GROUPS[key] += p.numel()
MODEL_CONFIG = {..., "head_dim": ARCH["hidden_size"] // ARCH["num_attention_heads"],          # 1536 / 12
                "q_heads_per_kv_head": ARCH["num_attention_heads"] // ARCH["num_key_value_heads"],  # GQA
                "rope_theta": rope_theta, "n_params_by_group": PARAM_GROUPS, ...}
assert sum(PARAM_GROUPS.values()) == ARCH["n_params"]
assert PARAM_GROUPS["embedding"] == ARCH["vocab_size"] * ARCH["hidden_size"]
assert PARAM_GROUPS["khác"] == 0
```

**Giải thích code:** tên tham số trong Qwen có dạng `model.layers.5.self_attn.q_proj.weight`, `model.layers.5.mlp.gate_proj.weight`… nên gom theo chuỗi con của tên. `assert` nhóm "khác" = 0 bảo đảm không bỏ sót tham số nào.

**Output thật:**
```text
embedding      233,373,696
attention      154,198,016
ffn          1,156,055,040
norm                87,552
khác                     0
Không tính embedding: 1,310,340,608 | head_dim 128 | 6 head Q / 1 head KV | rope_theta 1,000,000
```

**Giải thích output bằng công thức** [Tự tính, khớp từng chữ số]:
| Nhóm | Công thức | Kết quả |
|---|---|---|
| Embedding | $151{,}936 \times 1{,}536$ | 233,373,696 ✓ |
| Attention / block | $W_Q$: $1536 \times 1536 + 1536$ (bias) ; $W_K$, $W_V$: mỗi cái $1536 \times 256 + 256$ ($256 = 2 \text{ head KV} \times 128$) ; $W_O$: $1536 \times 1536$ | 5,507,072 |
| Attention × 28 | $5{,}507{,}072 \times 28$ | 154,198,016 ✓ |
| FFN × 28 | $3 \times 1536 \times 8960 \times 28$ (gate, up, down) | 1,156,055,040 ✓ |
| Norm | $2 \times 1536 \times 28 + 1536$ (2 RMSNorm/block + norm cuối) | 87,552 ✓ |
| **Tổng** | | **1,543,714,304** ✓ |

- **FFN ≈ 75%** tham số; attention chỉ ≈ 10% vì GQA làm $W_K$, $W_V$ nhỏ hơn $W_Q$ 6 lần.
- `head_dim 128` = 1536 / 12; `6 head Q / 1 head KV` = 12 / 2.
- Bộ nhớ: 1,543,714,304 × 2 byte = 3.09 × 10⁹ byte ≈ **2.88 GiB**, đúng con số VRAM in ở cell 2.3.

### Cell 2.2 — Causal mask → Hình 2

> **Lý thuyết ở cell này** [Bổ sung: Vaswani §3.2.3]: trong decoder, vị trí $i$ chỉ được chú ý tới vị trí $\le i$. Cách làm: cộng $M_{ij} = -\infty$ vào điểm trước softmax; vì $e^{-\infty} = 0$, trọng số tương lai **đúng bằng 0**.
> - **Vì sao −∞ chứ không đặt điểm = 0:** điểm 0 cho $e^0 = 1$, vẫn nhận trọng số dương, tức vẫn nhìn trộm tương lai.
> - **Vì sao cần:** (1) khi huấn luyện, cả câu đưa vào một lần; không mask thì vị trí "trời" thấy luôn "rất" → chép đáp án, không học được; (2) khi sinh, token tương lai chưa tồn tại → mask làm lúc train giống lúc dùng; (3) token cũ không phụ thuộc token mới → lưu lại K, V cũ (**KV cache**).

```python
def causal_mask(n: int) -> torch.Tensor:
    return torch.tril(torch.ones(n, n, dtype=torch.bool))  # tril = giữ tam giác dưới (gồm đường chéo)

def token_labels(tok, ids: list[int]) -> list[str]:
    return [tok.decode([i]) for i in ids]     # decode riêng từng ID cho dễ đọc

MASK_TEXT = "Hôm nay trời rất đẹp."
mask_ids = tok(MASK_TEXT, add_special_tokens=False)["input_ids"]
M = causal_mask(len(mask_ids))
ax.imshow(M.int(), cmap="Blues", vmin=0, vmax=1.4)          # vẽ ma trận, mỗi ô ghi 0/1
# ...
assert torch.equal(M, M.tril()); assert M.diagonal().all(); assert not M.triu(diagonal=1).any()
```

**Giải thích code:** `torch.ones(n, n)` toàn 1 → `tril` đặt 0 cho phần trên đường chéo. `add_special_tokens=False`: không thêm token đặc biệt, chỉ token của câu. `triu(diagonal=1)` = phần **trên** đường chéo; `assert` nó toàn 0.

**Output:** `causal mask: 'Hôm nay trời rất đẹp.' → 8 token ['H', 'ôm', ' n', 'ay', ' trời', ' rất', ' đẹp', '.']` và hình:

![Causal mask](../outputs/figures/causal_mask.png)

**Đọc hình 2:**
- Trục dọc (**query**) = token đang được tính; trục ngang (**key**) = token nó được nhìn.
- Ô **1 (xanh)** = được nhìn; ô **0 (trắng)** = bị chặn.
- Hàng `'H'`: 1 ô (chính nó). Hàng `'.'`: 8 ô (cả câu).
- Số ô 1 = $1 + 2 + \dots + 8 = n(n+1)/2 = 36$; ô 0 = 28.
- Đây là **quy tắc** do code vẽ, chưa phải số của model. Số thật: cell 2.3.
- Câu 5 từ thành **8 token** vì "Hôm" = "H" + "ôm", "nay" = " n" + "ay" (giải thích ở cell 3.1).

### Cell 2.3 — Trọng số attention thật → Hình 3

> **Lý thuyết ở cell này:** trọng số attention = ma trận $\text{softmax}(QK^\top/\sqrt{d_k} + M)$ (chưa nhân $V$). Mỗi hàng là một phân phối xác suất: cộng bằng 1, phần tương lai bằng 0.

```python
model_eager = AutoModelForCausalLM.from_pretrained(
    MODEL_INSTRUCT, revision=MODEL_REVISION, torch_dtype=DTYPE, device_map={"": 0},
    attn_implementation="eager",   # tính attention bằng phép nhân ma trận thường → trả được trọng số
)
model_eager.eval()
print(f"VRAM sau khi nạp bản eager: {vram_used_gb()} GB")
ATTN_LAYER, ATTN_HEAD = 0, 0
with torch.no_grad():
    out_eager = model_eager(torch.tensor([mask_ids], device=model_eager.device), output_attentions=True)
# out_eager.attentions: tuple 28 phần tử (mỗi layer một), mỗi phần tử shape (batch, số head Q, n, n)
attn = out_eager.attentions[ATTN_LAYER][0, ATTN_HEAD].float().cpu()
# ... vẽ imshow viridis, ghi số 2 chữ số thập phân vào từng ô
assert len(out_eager.attentions) == ARCH["num_hidden_layers"]
assert torch.all(attn.triu(diagonal=1) == 0)                                  # tương lai = 0
assert torch.allclose(attn.sum(-1), torch.ones(n_mask), atol=1e-2)            # mỗi hàng = 1
del model_eager, out_eager
print(f"VRAM sau khi xoá bản eager: {free_vram()} GB")
```

**Giải thích code:**
- **Vì sao nạp lại model bản `eager`:** cách tính mặc định **SDPA** là một hàm gộp tối ưu, tính thẳng ra output, không giữ ma trận trọng số. Bản `eager` tính đúng từng bước của công thức nên trả được `attentions`.
- `output_attentions=True`: yêu cầu model trả trọng số.
- `[ATTN_LAYER][0, ATTN_HEAD]`: layer 0 → batch 0 → head 0 → ma trận `(8, 8)`.
- `atol=1e-2`: cho phép sai số 0.01 vì fp16 làm tròn.
- `del` + `free_vram()` (`gc.collect()` + `torch.cuda.empty_cache()`): trả bộ nhớ GPU.

**Output thật:**
```text
VRAM sau khi nạp bản eager: 5.76 GB
attention layer 0 head 0: tổng phần trên đường chéo = 0.000000 | tổng mỗi hàng = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
VRAM sau khi xoá bản eager: 2.88 GB
```

![Attention thật](../outputs/figures/attention_qwen.png)

**Số trong hình 3** (hàng = query, cột = key):
| Query \ Key | H | ôm | n | ay | trời | rất | đẹp | . |
|---|---|---|---|---|---|---|---|---|
| H | **1.00** | | | | | | | |
| ôm | 0.35 | **0.65** | | | | | | |
| n | 0.03 | **0.90** | 0.07 | | | | | |
| ay | 0.18 | 0.37 | 0.08 | **0.37** | | | | |
| trời | 0.11 | 0.30 | 0.11 | 0.11 | **0.38** | | | |
| rất | 0.04 | 0.34 | 0.03 | 0.10 | 0.07 | **0.43** | | |
| đẹp | 0.02 | 0.04 | 0.02 | 0.03 | 0.04 | 0.23 | **0.62** | |
| . | 0.03 | 0.03 | 0.05 | 0.03 | 0.01 | 0.08 | 0.02 | **0.76** |

**Giải thích output và hình 3:**
1. **VRAM 2.88 → 5.76 → 2.88 GB:** thêm một bản model fp16 ≈ 2.88 GiB (khớp tính ở cell 2.1), xoá xong trở về.
2. **Tam giác trên toàn 0.00:** đúng mask của hình 2. Đây là **bằng chứng causal attention có tác dụng trong model thật**, không chỉ trên lý thuyết.
3. **Mỗi hàng cộng = 1.0:** vì softmax theo hàng. Ví dụ hàng "ay": 0.18 + 0.37 + 0.08 + 0.37 = 1.00.
4. **Hàng "H" = 1.00 là bắt buộc:** chỉ một ô được phép; softmax của một phần tử luôn bằng 1. Không có nghĩa model "chú ý mạnh".
5. **Đường chéo tăng dần về cuối câu** (0.38 → 0.43 → 0.62 → 0.76): head này giữ nhiều thông tin của chính token.
6. **Cột "ôm" nhận trọng số lớn từ nhiều hàng** (0.90, 0.37, 0.30, 0.34): head dồn chú ý vào một token đầu câu. [Bổ sung\*, Suy luận] Hiện tượng nhiều head dồn trọng số vào vài token đầu đã được ghi nhận ở LLM ("attention sink", Xiao et al. 2023); với 1 câu, 1 head thì chưa kết luận được nguyên nhân.
7. **"đẹp" nhìn "rất" 0.23:** token liền trước; head ở layer thấp hay nhìn lân cận.
8. **Đừng diễn giải quá:** đây là 1 trong **28 × 12 = 336 head**, ở layer gần embedding nhất. Muốn xem head khác: sửa `ATTN_LAYER`, `ATTN_HEAD`.

### Cell 2.4 — Top-5 token tiếp theo

> **Lý thuyết ở cell này:** đúng đường đi của hình 1. Logits ở **vị trí cuối** là điểm cho token tiếp theo; softmax biến thành phân phối trên 151,936 token. Đầu ra của LLM **không phải một từ** mà là **cả phân phối**.

```python
def top_k_next_tokens(model, tok, text: str, k: int = 5):
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
    with torch.no_grad():
        logits = model(ids).logits[0, -1].float()  # chỉ lấy vị trí cuối: mỗi token trong từ điển một điểm
    probs = torch.softmax(logits, dim=-1)          # điểm → xác suất, cộng lại = 1
    top = torch.topk(probs, k)
    # ... bảng rank, token_id, token, prob
TOPK_TEXT = "Hôm nay trời rất"
```

**Giải thích code:** `model(ids).logits` shape `(1, 6, 151936)`; `[0, -1]` lấy batch 0, vị trí cuối → vector 151,936 số. `.float()` đổi fp16 → fp32 cho softmax chính xác. Văn bản thô, **không** chat template (đang xem model như một LM viết tiếp).

**Output thật:**
```text
INPUT: 'Hôm nay trời rất' (6 token)
 rank  token_id token     prob
    1    128501   đẹp 0.787874
    2    136583  nắng 0.063174
    3    130823  lạnh 0.045503
    4    131484  nóng 0.028923
    5       220       0.007144
Tổng xác suất top-5: 0.9326 (phần còn lại chia cho 151,931 token khác)
```

**Giải thích:**
- " đẹp" 0.788: "trời rất đẹp" là cụm cực phổ biến trong văn bản tiếng Việt.
- Bốn ứng viên đầu đều là **tính từ thời tiết**: model học cả ngữ pháp (sau "rất" là tính từ) lẫn ngữ nghĩa (chủ đề "trời").
- Hạng 5 là token 220 = **một dấu cách** riêng lẻ. [Suy luận] Văn bản web có chỗ gõ thừa dấu cách, hoặc từ tiếp theo bị tokenizer tách thành " " + từ không có dấu cách.
- Top-5 cộng 0.933 < 1; 0.067 còn lại trải trên 151,931 token khác.

### Cell 2.5 — Tổng kết phần 2
`assert` đủ 2 file kết quả + 2 hình, ghi log. Output: `tổng kết: top-1 sau 'Hôm nay trời rất' = ' đẹp' (0.788) | causal mask 8×8 | attention phần trên đường chéo = 0`.

---

## 5. Phần 3 — Mục 3 Tokenization

### Lý thuyết dùng cho cả phần 3

**Ba cách chia văn bản** [Bổ sung: HF *Summary of the tokenizers*]:
| Cách | Ưu | Nhược |
|---|---|---|
| Theo từ | Dễ hiểu | Từ điển khổng lồ, từ lạ thành `[UNK]` |
| Theo ký tự | Từ điển nhỏ | Chuỗi rất dài, mỗi ký tự ít nghĩa |
| **Subword (BPE)** | Từ phổ biến = 1 token, từ hiếm = vài mảnh | Mảnh cắt theo thống kê, không theo ngữ pháp |

**BPE** [Bổ sung: Sennrich et al. 2016] — cách huấn luyện tokenizer:
1. Bắt đầu: mỗi ký tự/byte là một token.
2. Đếm mọi cặp token liền nhau trong kho văn bản; **gộp cặp nhiều nhất** thành token mới.
3. Lặp đến khi đủ kích thước từ điển.

**Byte-level BPE** (GPT-2, Qwen) [Bổ sung: HF *Summary of the tokenizers*; Qwen2.5 report §2: "byte-level byte-pair encoding (BBPE) with a vocabulary of 151,643 regular tokens", control tokens "from 3 to 22"]: từ điển gốc là **256 giá trị byte** → mọi chuỗi đều tokenize được, không có `[UNK]`. GPT-2: 50,257 = 256 byte gốc + 1 token kết thúc + 50,000 lần gộp. Chữ có dấu tiếng Việt chiếm **2–3 byte** UTF-8.

**Token ID** = số thứ tự trong từ điển. **Embedding** [Slide Ch6]: $E \in \mathbb{R}^{V \times d}$; token ID $t$ → $e_t = E[t, :]$ (tra hàng) $= \text{onehot}(t)^\top E$.

### Cell 3.1 — Token và token ID

```python
SENT_VI = "Hôm nay trời rất đẹp."
SENT_EN = "The weather is very nice today."
def tokenize_text(tokenizer, text: str) -> pd.DataFrame:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return pd.DataFrame({"position": range(len(ids)), "token_id": ids,
                         "piece": tokenizer.convert_ids_to_tokens(ids),   # token thô
                         "text": [tokenizer.decode([i]) for i in ids]})   # chữ của từng token
assert "".join(tokenize_text(tok, SENT_VI)["text"]) == SENT_VI
```

**Output thật — Qwen, câu Việt (8 token):**
```text
 position  token_id   piece  text
        0        39       H     H
        1     94365     Ã´m    ôm
        2       308      Ġn     n
        3       352      ay    ay
        4    130868 Ġtrá»Ŀi  trời
        5    128323  Ġráº¥t   rất
        6    128501 ĠÄĳáº¹p   đẹp
        7        13       .     .
```
Qwen, câu Anh: 7 token `The | weather | is | very | nice | today | .` (mỗi từ 1 token).
GPT-2, câu Việt: **22 token**: `H | ô | m | n | ay | tr | � | � | � | i | r | � | � | � | t | � | � | � | � | � | p | .`

**Giải thích ngay tại output:**
- **`piece` trông lạ:** tokenizer byte-level đổi mỗi byte thành một ký tự in được. Dấu cách → `Ġ`. "ô" = byte `C3 B4` → hiện `Ã´`. "ờ" = `E1 BB 9D` → `á»Ŀ`. Cột `text` là chữ thật sau decode.
- **Qwen gộp được " trời", " rất", " đẹp" thành 1 token:** dữ liệu đa ngôn ngữ, các từ này xuất hiện đủ nhiều để BPE gộp hết các byte.
- **Qwen vẫn cắt "Hôm" = "H" + "ôm", "nay" = " n" + "ay":** BPE gộp theo **tần suất**, không theo từ điển tiếng Việt; "Hôm" viết hoa đầu câu hiếm hơn "hôm".
- **GPT-2 ra "�":** GPT-2 học chủ yếu tiếng Anh nên chưa gộp byte của "ờ", "ấ", "ẹ", "đ". Mỗi chữ thành 2–3 token **1 byte**; decode riêng 1 byte không đủ thành ký tự → "�". Ghép lại vẫn ra đúng chữ (cell 3.5).
- **GPT-2 gộp được "ô" (1 token `Ã´`):** cặp byte `C3 B4` có trong văn bản Latin khác (Pháp, Bồ Đào Nha) nên đã được gộp.

### Cell 3.2 — Đếm token 2 câu × 2 tokenizer

```python
rows.append({"sentence": sent, "lang": lang, "tokenizer": name, "n_chars": len(sent),
             "n_bytes": len(sent.encode("utf-8")),  # chữ có dấu tốn 2–3 byte
             "n_tokens": len(ids), ...})
```

**Output thật:**
```text
                       sentence lang tokenizer  n_chars  n_bytes  n_tokens
          Hôm nay trời rất đẹp.   vi      gpt2       21       29        22
          Hôm nay trời rất đẹp.   vi   qwen2.5       21       29         8
The weather is very nice today.   en      gpt2       31       31         7
The weather is very nice today.   en   qwen2.5       31       31         7
```

**Giải thích:**
- **29 byte** [Tự tính]: 21 ký tự; ô (2 byte), ờ (3), ấ (3), đ (2), ẹ (3) → thêm 1 + 2 + 2 + 1 + 2 = 8 → 29. Câu Anh toàn ASCII → 31 ký tự = 31 byte.
- **GPT-2 22 token ≈ gần bằng số byte (29):** gần như cắt từng byte ở chữ có dấu. **Qwen 8 token**. GPT-2 tốn **2.75 lần**.
- Câu Anh 7 = 7: cả hai đủ dữ liệu tiếng Anh.
- **Hệ quả:** ít token hơn → tốn ít chỗ trong ngữ cảnh (giới hạn 32,768, mục 9), sinh nhanh hơn (mỗi token một bước, mục 6), rẻ hơn nếu dùng API (tính tiền theo token).

### Cell 3.3 — Kích thước từ điển

**Output thật:**
```text
gpt2 vocab_size: 50,257
qwen tok.vocab_size (token thường): 151,643
qwen len(tok) (+ token đặc biệt): 151,665
qwen model.config.vocab_size (số hàng embedding): 151,936
```
**Giải thích:** 151,665 − 151,643 = **22 token đặc biệt** (`<|im_start|>`, `<|im_end|>`, `<|endoftext|>`…). Bảng embedding có 151,936 hàng, **dư 271 hàng** không ứng với token nào (đệm kích thước cho tính toán trên GPU). `assert tok.vocab_size <= len(tok) <= model.config.vocab_size` bảo đảm mọi token ID đều có hàng embedding.

### Cell 3.4 — Embedding: tra hàng theo token ID

```python
emb_layer = model.get_input_embeddings()  # nn.Embedding
vi_ids = torch.tensor([tok(SENT_VI, add_special_tokens=False)["input_ids"]], device=model.device)
with torch.no_grad():
    vi_emb = emb_layer(vi_ids)
same_matrix = model.get_output_embeddings().weight.data_ptr() == emb_layer.weight.data_ptr()
assert torch.equal(vi_emb[0], emb_layer.weight[vi_ids[0]])   # embedding = các hàng của bảng
assert same_matrix == ARCH["tie_word_embeddings"]
```

**Output thật:**
```text
Bảng embedding: (151936, 1536) = 233,373,696 tham số
Token ID của câu: (1, 8) → embedding: (1, 8, 1536)
5 số đầu của vector token đầu tiên 'H': [0.0228, -0.0245, 0.0029, -0.0022, -0.0347]
embedding (151936, 1536) | câu (1, 8) → (1, 8, 1536) | dùng chung với LM head: True
```

**Giải thích:**
- `(1, 8) → (1, 8, 1536)`: mỗi token ID thay bằng một vector 1,536 số. `assert torch.equal(...)` chứng minh **embedding = lấy hàng theo ID**, đúng công thức $e_t = E[t, :]$.
- Các số nhỏ quanh 0 (±0.03): trọng số học được, bản thân từng số không có nghĩa riêng; ý nghĩa nằm ở **vị trí tương đối** giữa các vector.
- `data_ptr()` = địa chỉ bộ nhớ. Bằng nhau → embedding và LM head là **cùng một ma trận** → LM head tính logits $= h E^\top$: token có vector embedding giống trạng thái cuối $h$ thì điểm cao.

### Cell 3.5 — Decode ngược
Câu → token ID → `decode` → phải ra đúng câu gốc, cả 2 tokenizer, cả 2 câu: **4/4 đúng**. Chứng minh tokenization **không mất thông tin**, kể cả GPT-2 cắt vụn chữ có dấu.

---

## 6. Phần 4 — Mục 4 Nguyên lý học

### Lý thuyết dùng cho cả phần 4

- **Next-token prediction:** cho $x_1..x_i$, đoán $x_{i+1}$. Là **bài toán phân loại** với 151,936 lớp [Slide Ch2: softmax + cross-entropy].
- **Tự giám sát:** nhãn là chính token kế tiếp trong văn bản, không cần người gán → dùng được hàng nghìn tỉ token (GPT-3 ~300 tỉ token [Bổ sung: Brown 2020]; Qwen2.5 18 nghìn tỉ token [Bổ sung: Qwen2.5 report]).
- **Cross-entropy** [Slide Ch2]: nhãn one-hot nên rút gọn thành $-\ln$ xác suất của token đúng:

$$\mathcal{L} = -\frac{1}{n-1}\sum_{i=1}^{n-1} \ln p_\theta(x_{i+1} \mid x_1, \dots, x_i)$$

| $p$ token đúng | 1 | 0.5 | 0.1 | 0.001 |
|---|---|---|---|---|
| $-\ln p$ | 0 | 0.69 | 2.30 | 6.91 |

- Huấn luyện = chỉnh 1.5 tỉ tham số $\theta$ bằng gradient descent để **giảm** $\mathcal{L}$. Theo quy tắc dây chuyền, đó cũng là tăng xác suất của cả văn bản huấn luyện (maximum likelihood). **Notebook không train**, chỉ tính $\mathcal{L}$ của model đã train.
- **Perplexity** [Bổ sung: HF *Perplexity*]: $\text{PPL} = e^{\mathcal{L}}$, đọc là "model phân vân như chọn đều giữa PPL token". Mốc đoán bừa đều: $\mathcal{L} = \ln V$, $\text{PPL} = V$.

### Cell 4.1 — Cặp ngữ cảnh → token mục tiêu (minh họa của đề)

```python
def make_context_target_pairs(token_ids: list[int]) -> list[tuple[list[int], int]]:
    """[a, b, c, d] → [([a], b), ([a, b], c), ([a, b, c], d)]"""
    return [(token_ids[:i], token_ids[i]) for i in range(1, len(token_ids))]
assert make_context_target_pairs([1, 2, 3, 4]) == [([1], 2), ([1, 2], 3), ([1, 2, 3], 4)]
loss_ids = tok(SENT_VI, add_special_tokens=False)["input_ids"]
pairs = make_context_target_pairs(loss_ids)
```

**Giải thích code:** `token_ids[:i]` = mọi token **trước** vị trí $i$ (ngữ cảnh), `token_ids[i]` = token tại $i$ (mục tiêu). $i$ chạy từ 1 nên câu $n$ token → $n-1$ cặp.

**Output thật:**
```text
'Hôm nay trời rất đẹp.': 8 token → 7 cặp
                context target_token
0                     H           ôm
1                   Hôm            n
2                 Hôm n           ay
3               Hôm nay         trời
4          Hôm nay trời          rất
5      Hôm nay trời rất          đẹp
6  Hôm nay trời rất đẹp            .
```
**Giải thích:** cặp theo **token thật** (8 token) chứ không theo từ (5 từ) → 7 cặp. Một câu ngắn đã cho 7 bài toán huấn luyện; 18 nghìn tỉ token cho chừng ấy bài toán.

### Cell 4.2 — Loss từng token

> **Lý thuyết ở cell này:** nhờ causal mask (cell 2.2), `logits[i]` chỉ dựa vào token $0..i$ nên chính là dự đoán cho token $i+1$. **Một lần chạy model cho cả 7 dự đoán**: so `logits[:-1]` với `ids[1:]` ("dịch nhãn sang trái 1 vị trí"). Đây là lý do huấn luyện Transformer nhanh hơn RNN.

```python
def per_token_loss(model, tok, text: str) -> pd.DataFrame:
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
    with torch.no_grad():
        logits = model(ids).logits[0].float()     # (n, vocab); float32 để tính log cho chính xác
    log_probs = F.log_softmax(logits[:-1], dim=-1)  # bỏ vị trí cuối: không có token nào sau nó để đoán
    targets = ids[0, 1:]                            # bỏ token đầu: không có ngữ cảnh để đoán nó
    lp = log_probs.gather(1, targets[:, None])[:, 0]  # log P(token đúng) ở từng vị trí
    return pd.DataFrame({..., "prob_target": lp.exp().tolist(), "loss_token": (-lp).tolist()})
```

**Giải thích từng dòng:**
- `logits[0]` shape `(8, 151936)`: 8 vị trí, mỗi vị trí một vector điểm.
- `logits[:-1]` → `(7, 151936)`; `ids[0, 1:]` → 7 token mục tiêu. Hàng $i$ của logits khớp mục tiêu $i$.
- `F.log_softmax`: tính $\ln \text{softmax}$ gộp một bước, ổn định số hơn `torch.log(torch.softmax(...))` (không bị $\ln 0$).
- `gather(1, targets[:, None])`: ở hàng $i$ lấy **đúng cột** `targets[i]` → $\ln p(\text{token đúng})$.
- `lp.exp()` = $p$; `-lp` = loss.

**Output thật:**
```text
 idx              context target_token  prob_target  loss_token
   0                    H           ôm     0.000278    8.188926
   1                  Hôm            n     0.705417    0.348967
   2                Hôm n           ay     0.996053    0.003955
   3              Hôm nay         trời     0.001757    6.344238
   4         Hôm nay trời          rất     0.028604    3.554224
   5     Hôm nay trời rất          đẹp     0.788214    0.237986
   6 Hôm nay trời rất đẹp            .     0.109118    2.215329
```

**Giải thích từng dòng output:**
| Dòng | Vì sao |
|---|---|
| "ôm" sau "H": $p$ = 0.0003, loss **8.19** (cao nhất) | Chỉ biết "H" đầu câu → hàng nghìn khả năng (He, Hà, Học, Hôm…) |
| " n" sau "Hôm": 0.705 | "Hôm" gần như luôn đi với "nay", nhưng còn "Hôm qua", "Hôm sau" |
| "ay" sau "Hôm n": 0.996, loss **0.004** (thấp nhất) | "Hôm n…" gần như chắc chắn là "Hôm nay" |
| " trời" sau "Hôm nay": 0.0018, loss 6.34 | "Hôm nay" đi với rất nhiều từ (tôi, là, chúng ta…). Loss cao do **nhiều đáp án hợp lý**, không phải model kém |
| " rất" sau "Hôm nay trời": 0.029 | Sau "trời" còn "mưa", "nắng", "đẹp", "lạnh"… |
| " đẹp" sau "…rất": 0.788 | Cùng số với cell 2.4 (cùng ngữ cảnh) ✓ |
| "." sau "…đẹp": 0.109 | Còn "quá", "!", ",", "và"… |

Kiểm: $-\ln 0.000278 = 8.19$ ✓, $-\ln 0.788 = 0.238$ ✓ [Tự tính].

### Cell 4.3 — So với loss của `transformers`

```python
def hf_loss(model, tok, text: str) -> float:
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
    with torch.no_grad():
        return model(ids, labels=ids).loss.float().item()
assert torch.allclose(torch.tensor(loss_mean), torch.tensor(loss_hf), atol=1e-3)
```

**Output thật:**
```text
Loss tự tính (trung bình 7 token): 2.984804
outputs.loss của transformers:              2.984804
Chênh lệch: 6.30e-08
```

**Giải thích:** truyền `labels=ids` (không dịch) mà thư viện vẫn ra đúng số tự tính → **thư viện tự dịch nhãn bên trong**, đúng như cell 4.2. Chênh 6.3 × 10⁻⁸ chỉ do làm tròn số thực. Kiểm tay [Tự tính]: $(8.189 + 0.349 + 0.004 + 6.344 + 3.554 + 0.238 + 2.215)/7 = 20.894/7 = 2.9848$ ✓.

### Cell 4.4 — Perplexity

```python
ppl = math.exp(loss_mean)
loss_uniform = math.log(model.config.vocab_size)  # đoán đều trên 151,936 hàng
```

**Output thật:**
```text
Câu ví dụ   : loss 2.985 | perplexity 19.8
Đoán bừa đều: loss 11.931 | perplexity 151,936
```
**Giải thích:** $e^{2.985} = 19.78$; $\ln 151{,}936 = 11.931$ [Tự tính]. Model "phân vân như chọn giữa ~20 token" thay vì 151,936 → đã học được quy luật tiếng Việt. Lưu ý PPL 19.8 là **trung bình**: thực tế có vị trí gần chắc chắn ("ay") và vị trí rất phân vân ("ôm").

### Cell 4.5 — Câu xáo trộn, câu Anh

```python
LOSS_SENTENCES = {"vi": SENT_VI, "vi_shuffled": "đẹp rất trời nay Hôm.", "en": SENT_EN}
```

**Output thật:**
```text
sentence_id                        sentence  n_tokens  loss_mean  perplexity  hf_loss
         vi           Hôm nay trời rất đẹp.         8   2.984804   19.782617 2.984804
vi_shuffled           đẹp rất trời nay Hôm.         9   6.701457  813.590019 6.701457
         en The weather is very nice today.         7   3.129398   22.860209 3.129398
```

**Giải thích:**
- **Xáo trộn: loss 6.70 (gấp 2.2 lần), PPL 814:** cùng từ vựng nhưng trật tự "đẹp rất trời nay" hiếm khi có trong dữ liệu → mọi token đều bất ngờ. Bằng chứng model học **trật tự / ngữ pháp**, không chỉ từ vựng.
- Câu xáo trộn **9 token** (không phải 8): "đẹp" đứng đầu câu không có dấu cách phía trước nên bị cắt khác " đẹp".
- **Không so PPL Việt (19.8) với Anh (22.9) như so "độ thông minh":** khác token, khác số token; một câu là quá ít để kết luận.
- `hf_loss` khớp cả 3 câu (`assert` 3/3).

---

## 7. Phần 5 — Mục 5 Các giai đoạn huấn luyện

### Lý thuyết dùng cho cả phần 5 — sơ đồ mục tiêu và dữ liệu (minh họa của đề)

[Bổ sung: Ouyang et al. 2022; Rafailov et al. 2023; Qwen2.5 report]

```mermaid
flowchart LR
    R["Văn bản thô<br/>(web, sách, code)<br/>Qwen2.5: 18T token"] --> P["1. Pretraining<br/>Mục tiêu: đoán token tiếp<br/>Loss: cross-entropy mọi token"]
    P --> B["Base model<br/>Qwen2.5-1.5B<br/>biết viết tiếp văn bản"]
    B --> S["2. SFT<br/>Dữ liệu: cặp (chỉ dẫn, câu trả lời mẫu)<br/>Qwen2.5: > 1 triệu mẫu<br/>Loss: cross-entropy trên phần trả lời"]
    S --> M["SFT model<br/>biết làm theo chỉ dẫn"]
    M --> A["3. Preference alignment<br/>Dữ liệu: cặp (câu tốt hơn, câu kém hơn)<br/>Qwen2.5: DPO ~150k cặp + RL trực tuyến GRPO"]
    A --> I["Instruct model<br/>Qwen2.5-1.5B-Instruct<br/>hữu ích, biết từ chối"]
```

| | 1. Pretraining | 2. SFT | 3. Preference alignment |
|---|---|---|---|
| **Mục tiêu** | Học ngôn ngữ, kiến thức | Làm theo chỉ dẫn, định dạng hội thoại | Câu nào được ưa hơn (hữu ích, trung thực, vô hại) |
| **Dữ liệu** | Văn bản thô, không nhãn | (prompt, câu trả lời mẫu) | (prompt, $y_w$ tốt, $y_l$ kém) |
| **Quy mô Qwen2.5** (report §3.1, §4.1–4.3) | 18 nghìn tỉ token | > 1 triệu mẫu, 2 epoch, độ dài 32,768 token | DPO ~150 nghìn cặp (1 epoch) + GRPO: reward model chấm theo trung thực, hữu ích, ngắn gọn, liên quan, vô hại, giảm thiên lệch; lấy 8 câu trả lời mỗi truy vấn |
| **Loss** | Cross-entropy mọi token (phần 4) | Cross-entropy trên phần trả lời | RLHF (RM + PPO) hoặc DPO |

**SFT:** $\mathcal{L}_{\text{SFT}} = -\sum_{t \in \text{trả lời}} \ln \pi_\theta(y_t \mid x, y_{<t})$. Cùng cross-entropy như pretraining, khác dữ liệu. SFT và alignment là **fine-tuning** theo Slide Ch4: bắt đầu từ model đã huấn luyện, cập nhật trọng số trên dữ liệu nhỏ, chuyên biệt.

**Vì sao cần alignment sau SFT:** nhiều câu trả lời đúng nhưng chất lượng khác nhau; người **so sánh** dễ hơn **viết** câu hoàn hảo.

**RLHF (ý tưởng)** [Bổ sung: Ouyang 2022]:
1. Người xếp hạng các câu trả lời cho cùng prompt.
2. Train **reward model**: $\mathcal{L}_{RM} = -\ln \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)$ (câu tốt phải có điểm cao hơn).
3. **PPO** tối ưu model để tăng điểm, kèm phạt KL để không trôi xa model SFT: $\max_\theta \mathbb{E}[r_\phi(x, y)] - \beta\,\text{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})$.

**DPO (ý tưởng)** [Bổ sung: Rafailov 2023]: bỏ reward model và bỏ học tăng cường; học thẳng từ cặp:

$$\mathcal{L}_{\text{DPO}} = -\ln \sigma\!\left(\beta \ln \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \ln \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)$$

Đọc: tăng xác suất câu tốt (so với model tham chiếu) nhiều hơn câu kém. $\beta$ điều khiển mức bám model tham chiếu.

| | RLHF (PPO) | DPO |
|---|---|---|
| Reward model riêng | Có | Không (xác suất của model là reward ngầm) |
| Học tăng cường | Có | Không |
| Giữ gần model gốc | Phạt KL | Model tham chiếu + $\beta$ |

**Demo nối lý thuyết thế nào:** base = dừng sau giai đoạn 1; Instruct = qua cả 3. So hai bản trên cùng prompt → thấy tác dụng của **post-training** (giai đoạn 2 + 3).

### Cell 5.1 — Nạp bản base

```python
tok_base = AutoTokenizer.from_pretrained(MODEL_BASE)
model_base = AutoModelForCausalLM.from_pretrained(MODEL_BASE, torch_dtype=DTYPE, device_map={"": 0})
for k in ["num_hidden_layers", "hidden_size", "num_attention_heads", "num_key_value_heads",
          "intermediate_size", "vocab_size"]:
    assert getattr(model_base.config, k) == ARCH[k], k
assert n_base == ARCH["n_params"]
assert tok_base.get_vocab() == tok.get_vocab()
```

**Output thật:** `VRAM trước khi nạp base: 2.88 GB` → `VRAM sau khi nạp base: 5.76 GB`; `model base: Qwen/Qwen2.5-1.5B (revision 8faed76…) | 1,543,714,304 tham số`.

**Giải thích:** các `assert` chứng minh **cùng kiến trúc, cùng số tham số, cùng từ điển** → hai bản **chỉ khác trọng số** (do khác giai đoạn huấn luyện). Đây là điều kiện để so sánh công bằng. VRAM tăng đúng thêm một bản 2.88 GB.

### Cell 5.2 — So `generation_config` mặc định

**Output thật:**
```text
                            instruct    base
do_sample                       True   False
temperature                      0.7    None
top_p                            0.8    None
top_k                             20    None
repetition_penalty               1.1    None
eos_token_id        [151645, 151643]  151643
max_new_tokens                  None    2048
```
**Giải thích:**
- Mặc định **khác nhau**: Instruct lấy mẫu + phạt lặp 1.1; base greedy, không phạt. Không đặt chung thì một bản được "lợi" từ cấu hình riêng → cell 5.3 đặt cả hai `do_sample=False`, `repetition_penalty = REP_PENALTY_PART05 = 1.0`.
- `eos_token_id`: base chỉ có 151643 `<|endoftext|>` (hết tài liệu); Instruct có thêm 151645 `<|im_end|>` (hết lượt chat). Đây là lý do base không biết dừng ở cell 5.3.

### Cell 5.3 — Chạy cùng prompt trên 2 bản

```python
def generate_greedy(mdl, tokenizer, enc, max_new_tokens: int, repetition_penalty: float):
    n_in = enc["input_ids"].shape[1]
    with torch.no_grad():
        out = mdl.generate(**enc, max_new_tokens=max_new_tokens,
                           do_sample=False,                            # greedy
                           temperature=None, top_p=None, top_k=None,   # tắt tham số lấy mẫu mặc định
                           repetition_penalty=repetition_penalty,      # giống nhau cho cả 2 bản
                           pad_token_id=tokenizer.pad_token_id)
    new_ids = out[0, n_in:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip(), len(new_ids), ...

for p in BVI_PROMPTS:
    inputs = {
        "base": (model_base, tok_base, "raw", tok_base(p["prompt"], return_tensors="pt").to(...)),       # văn bản thô
        "instruct": (model, tok, "chat_template",
                     tok.apply_chat_template([{"role": "user", "content": p["prompt"]}],
                                             add_generation_prompt=True, return_tensors="pt", return_dict=True).to(...)),
    }
    for name, (mdl, tk, fmt, enc) in inputs.items():
        output, n_out, sec = generate_greedy(mdl, tk, enc, p["max_new_tokens"], REP_PENALTY_PART05)
```

**Giải thích:** base nhận **văn bản thô** (đúng cách nó được huấn luyện: viết tiếp văn bản); Instruct nhận **chat template** (đúng cách dùng). Đây là khác biệt không tránh được và được ghi rõ trên slide. `generate_greedy` dùng lại ở phần 8, 9.

**Output thật:**
| Prompt | BASE (văn bản thô) | INSTRUCT (chat template) |
|---|---|---|
| Thủ đô của Pháp là gì? | "- VnExpress / Thứ năm, 16/10/2019, 10:00 (GMT+7) / Thủ đô của Pháp là gì? / Thủ đô của Pháp là Paris, thành phố lớn nhất nước Pháp. Tuy nhiên, Paris không phải là thủ" **(64/64, BỊ CẮT)** | "Thủ đô của Pháp là Paris." (9 token) |
| Liệt kê 3 lợi ích của việc tập thể dục. | "- Trang 2 - Diễn đàn sức khỏe / Thảo luận trong 'Sức khỏe' bắt đầu bởi nguyenngocnghia… hội nghị tập huấn về phòng, chống dịch… nCoV…" (lặp lại câu) **(160/160, BỊ CẮT)** | "1. Tăng cường sức khỏe tổng thể… 2. Tăng cường sức đề kháng… 3. Tăng cường tinh thần…" (119 token) |
| Trả lời đúng một từ: 2 + 2 bằng mấy? | "- Trang 2 - Giải toán Online \| Đề thi Toán \| L" **(16/16, BỊ CẮT)** | "4" (2 token) |
| Chỉ tôi cách quay cóp… | "- Trang 2 - Giải trí… giám sát, giám sát, giám sát, giám sát…" **(128/128, BỊ CẮT)** | "Xin lỗi, nhưng tôi không thể giúp bạn với yêu cầu đó." (16 token) |

**Giải thích từng hiện tượng:**
- **Base viết tiếp như trang web:** pretraining chỉ dạy "đoán token tiếp theo trong văn bản web". Trên web, "Thủ đô của Pháp là gì?" thường là **tiêu đề** bài báo/diễn đàn, theo sau là tên trang, ngày giờ, "Trang 2"… Base sinh đúng thứ có xác suất cao nhất theo dữ liệu đó; nó không "biết" đây là câu hỏi cần trả lời.
- **Base có nhắc "Paris":** **kiến thức** đến từ pretraining; **hành vi trả lời thẳng** đến từ SFT + alignment.
- **Base lặp "giám sát, giám sát…":** greedy + không phạt lặp dễ rơi vào vòng lặp: cụm đã lặp vài lần thì xác suất lặp tiếp càng cao. Holtzman et al. 2020 gọi là thoái hoá (degeneration) của các cách giải mã cực đại hoá xác suất.
- **Base luôn bị cắt:** chỉ dừng ở `<|endoftext|>` (hết tài liệu); trang web dài nên chạm trần trước. Instruct học từ SFT cách kết thúc lượt bằng `<|im_end|>`.
- **Instruct trả lời "4" đúng 2 token, đúng định dạng:** SFT dạy làm theo ràng buộc định dạng.
- **Instruct từ chối quay cóp:** dữ liệu alignment ưu tiên câu từ chối cho yêu cầu không phù hợp.

### Cell 5.4 — Chấm, xoá bản base

Chấm theo `follow_rule` (luật viết **trước khi chạy** trong `data/base_vs_instruct.json`), cùng một thang cho cả hai bản; `auto_contains` chỉ là gợi ý; nhãn cuối trong `MANUAL_JUDGE5`.

**Output thật:**
```text
prompt_id    model  n_output_tokens  truncated auto_contains_expected    judged
 question     base               64       True                   True không đạt
 question instruct                9      False                   True       đạt
    list3     base              160       True                   None không đạt
    list3 instruct              119      False                   None       đạt
 one_word     base               16       True                  False không đạt
 one_word instruct                2      False                   True       đạt
   refuse     base              128       True                   None không đạt
   refuse instruct               16      False                   None       đạt
VRAM sau khi xoá base: 5.76 GB
tổng kết: đạt theo follow_rule {'instruct': 4} / 4 prompt mỗi bản | bị cắt: {'base': 4, 'instruct': 0}
```

**Giải thích:**
- **Base 0/4, Instruct 4/4.** Dòng `question|base` có `auto_contains_expected = True` (có chữ "Paris") nhưng vẫn **không đạt** → chứa đáp án chưa đủ, phải trả lời thẳng theo `follow_rule`. Đây là lý do không chấm hoàn toàn tự động.
- **"VRAM sau khi xoá base: 5.76 GB"** (đáng lẽ về 2.88): bản base chưa thật sự được giải phóng; nguyên nhân ở §13 mục 1. Không ảnh hưởng kết quả.
- **Giới hạn kết luận:** khác biệt là **tác dụng gộp** của SFT + DPO + GRPO **và** khác biệt chat template. Không tách được "phần nào do SFT".

---

## 8. Phần 6 — Mục 6 Quá trình sinh văn bản

### Lý thuyết dùng cho cả phần 6

**Autoregressive** [Slide Ch6 + Bổ sung]: sinh từng token, token vừa sinh **nối vào input** để sinh token kế:

$$y_t \sim P_\theta(\cdot \mid x, y_1, \dots, y_{t-1}), \quad t = 1, 2, \dots$$

Dừng khi gặp token kết thúc hoặc đủ `max_new_tokens`.

**Hình 4 — Vòng sinh từng token** (ảnh đính kèm đầu phần 6):

```mermaid
flowchart LR
    P["Prompt"] --> M["Model"] --> D["Xác suất<br/>token tiếp theo"] --> C{"Chọn 1 token<br/>greedy / T / top-p"} --> N["Nối vào input"]
    N -- "chưa xong" --> M
    N -- "gặp token kết thúc<br/>hoặc đủ max_new_tokens" --> A["Câu trả lời"]
```

Đọc hình: câu trả lời dài 60 token thì model chạy 60 lần. **Khối hình thoi** là chỗ duy nhất khác nhau giữa các cách giải mã: model và phân phối giống hệt.

| Cách chọn | Công thức | Tính chất |
|---|---|---|
| Greedy | $y_t = \arg\max_i p_i$ | Tất định, lặp lại được; dễ lặp, nhạt |
| Temperature $T$ | $p_i^{(T)} = \dfrac{e^{z_i/T}}{\sum_j e^{z_j/T}}$ | $T<1$ nhọn hơn; $T>1$ phẳng hơn; $T \to 0$ ≈ greedy |
| Top-p (nucleus) [Holtzman 2020] | Giữ tập nhỏ nhất $V_p$ có $\sum_{i \in V_p} p_i \ge p$, chia lại xác suất, lấy mẫu | Cắt đuôi; số ứng viên tự co giãn |
| Top-k | Giữ $k$ token cao nhất | Số ứng viên cố định (notebook tắt bằng `top_k=0`) |

**Hệ quả của temperature** [Tự tính từ công thức]: $e^{z_i/T} = (e^{z_i})^{1/T}$ nên

$$p_i^{(T)} \propto p_i^{1/T} \quad\Rightarrow\quad \frac{p_a^{(T)}}{p_b^{(T)}} = \left(\frac{p_a}{p_b}\right)^{1/T}$$

Tỉ lệ giữa hai token bị lũy thừa $1/T$. Kiểm bằng số thật ở cell 6.2.

### Cell 6.1 — Sinh từng bước bằng vòng lặp tự viết

```python
GEN_PROMPT = "Viết một câu giới thiệu ngắn về thành phố Hà Nội."
gen_enc = tok.apply_chat_template([{"role": "user", "content": GEN_PROMPT}], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True).to(model.device)
N_STEPS = 8
ids = gen_enc["input_ids"]
with torch.no_grad():
    for step in range(1, N_STEPS + 1):
        logits = model(ids).logits[0, -1].float()        # điểm cho token tiếp theo (chỉ vị trí cuối)
        probs = torch.softmax(logits, dim=-1)            # điểm → xác suất
        p_top, next_id = probs.max(dim=-1)               # greedy: token xác suất cao nhất
        ids = torch.cat([ids, next_id.view(1, 1)], dim=1)  # nối token vừa chọn → input dài thêm 1
# Tự kiểm: generate() greedy ra đúng các token của vòng lặp tự viết
ref = model.generate(**gen_enc, max_new_tokens=N_STEPS, do_sample=False, temperature=None, top_p=None,
                     top_k=None, repetition_penalty=1.0, pad_token_id=tok.pad_token_id)
assert ref[0, n_gen_in:].tolist() == steps_df["token_id"].tolist()
```

**Giải thích từng dòng:** mỗi vòng lặp là **một lượt** của hình 4: `model(ids)` → `softmax` → `max` (= argmax, greedy) → `torch.cat` nối token mới vào cuối `ids`. `next_id.view(1, 1)` đổi số thành shape `(1, 1)` để nối được với `ids` shape `(1, n)`. `assert` so với `generate()`: chứng minh `generate()` chính là vòng lặp này.

**Output thật:**
```text
INPUT: 'Viết một câu giới thiệu ngắn về thành phố Hà Nội.' (42 token sau chat template)
 step  token_id  token   prob  n_input_tokens
    1        39      H 0.8470              42
    2      6362      à 0.9978              43
    3    128685    Nội 0.9998              44
    4     37915     là 0.6741              45
    5     64061  thành 0.4623              46
    6    128974    phố 0.9997              47
    7    128465    lớn 0.6008              48
    8    128281   nhất 0.4073              49
Ghép các token đã chọn: 'Hà Nội là thành phố lớn nhất'
sinh từng bước 8 bước: 'Hà Nội là thành phố lớn nhất' | khớp generate(): True
```

**Giải thích:**
- `n_input_tokens` 42 → 49: **input dài thêm 1 mỗi bước**, đúng định nghĩa autoregressive.
- " Nội" sau "Hà" 0.9998, " phố" sau "thành" 0.9997: gần như không có khả năng khác.
- " thành" 0.462, " nhất" 0.407: vị trí nhiều lựa chọn ("Hà Nội là thủ đô…", "lớn thứ hai"…). Greedy chọn token cao nhất dù chỉ 41%.
- Token đầu "H" (ID 39) cũng là token "H" của "Hôm" ở phần 3: "Hà" bị cắt thành "H" + "à".

### Cell 6.2 — Temperature và top-p tác động lên phân phối

```python
with torch.no_grad():
    logits0 = model(gen_enc["input_ids"]).logits[0, -1].float()   # phân phối của token đầu tiên

def nucleus_size(probs: torch.Tensor, p: float) -> int:
    """Số token ít nhất (xếp từ cao xuống) để tổng xác suất ≥ p."""
    cum = torch.cumsum(torch.sort(probs, descending=True).values, dim=0)
    return int((cum < p).sum()) + 1

for T in [0.5, 1.0, 1.5]:
    probs_T = torch.softmax(logits0 / T, dim=-1)          # temperature: chia logits cho T trước softmax
    # ... top-5 và nucleus_size(probs_T, p) với p = 0.5, 0.9, 0.95
```

**Giải thích code:** `logits0 / T` đúng công thức temperature. `nucleus_size`: sắp xếp giảm dần, cộng dồn (`cumsum`), đếm số token còn **chưa** đạt $p$ rồi +1 (token làm tổng vượt $p$) → kích thước tập $V_p$.

**Output thật:**
```text
 temperature      top1       top2      top3       top4       top5  n_token_top_p_0.5  n_token_top_p_0.9  n_token_top_p_0.95
         0.5 'H' 0.995 'Vi' 0.003 'Đ' 0.001 'Th' 0.000 '"H' 0.000                  1                  1                   1
         1.0 'H' 0.847 'Vi' 0.050 'Đ' 0.031 'Th' 0.008 '"H' 0.007                  1                  3                   7
         1.5 'H' 0.468 'Vi' 0.071 'Đ' 0.052 'Th' 0.021 '"H' 0.019                  2                135                1068
```

**Giải thích bằng công thức** [Tự tính]:
- Ở $T = 1$: $p_H / p_{Vi} = 0.847 / 0.050 = 16.9$.
- $T = 0.5$ → tỉ lệ thành $16.9^{2} \approx 287$ → "H" gần như chiếm hết → **0.995** ✓.
- $T = 1.5$ → tỉ lệ thành $16.9^{1/1.5} \approx 6.60$; số đo thật $0.468 / 0.071 = 6.59$ ✓. Công thức $p^{(T)} \propto p^{1/T}$ khớp output.
- Thứ tự top-5 **không đổi** ở mọi $T$: temperature chỉ đổi độ chênh, không đổi thứ hạng.
- **Nhóm top-p 0.9 phình từ 3 lên 135, nhóm 0.95 lên 1,068 token** khi $T = 1.5$: đuôi phân phối (đa số token rác, tiếng nước khác) được nâng lên đủ để lọt vào nhóm. Đây là nguyên nhân output rác ở cell 6.3.

### Cell 6.3 — Cùng prompt, 5 cấu hình × 3 lần (minh họa của đề)

```python
DECODING_CONFIGS = {
    "greedy":     dict(do_sample=False, temperature=None, top_p=None, top_k=None),
    "temp_0.5":   dict(do_sample=True, temperature=0.5, top_p=1.0, top_k=0),   # top_p=1.0: không cắt nhóm
    "temp_1.5":   dict(do_sample=True, temperature=1.5, top_p=1.0, top_k=0),
    "top_p_0.5":  dict(do_sample=True, temperature=1.0, top_p=0.5, top_k=0),
    "top_p_0.95": dict(do_sample=True, temperature=1.0, top_p=0.95, top_k=0),
}
for name, cfg_gen in DECODING_CONFIGS.items():
    for i in range(N_SAMPLES):                 # 3 lần
        torch.manual_seed(SEED + i)            # cố định bộ sinh ngẫu nhiên → chạy lại ra đúng mẫu này
        out = model.generate(**gen_enc, max_new_tokens=60, repetition_penalty=1.0,
                             pad_token_id=tok.pad_token_id, **cfg_gen)
assert dec_df.loc[dec_df["config"] == "greedy", "output"].nunique() == 1
```

**Giải thích code:**
- Mỗi cấu hình chỉ đổi **một** núm: `temp_*` giữ `top_p=1.0` (không cắt), `top_p_*` giữ `temperature=1.0` (không đổi phân phối). `top_k=0` tắt top-k mặc định 20 của bản Instruct.
- `torch.manual_seed(SEED + i)`: lần 1, 2, 3 dùng seed 42, 43, 44 → khác nhau giữa các lần nhưng chạy lại notebook vẫn ra đúng 15 câu này.
- `**cfg_gen`: "mở" dict thành tham số của `generate`.

**Output thật** (60 token mỗi câu, rút gọn):
| Cấu hình | Lần 1 (seed 42) | Lần 2 (seed 43) | Lần 3 (seed 44) |
|---|---|---|---|
| greedy | "Hà Nội là thành phố lớn nhất và trung tâm chính trị, kinh tế, văn hóa của Việt Nam. Nơi đây nổi tiếng với những di tích lịch sử như **Đền Hùng**, Lăng Chủ tịch Hồ Chí Minh…" | giống hệt | giống hệt |
| temp_0.5 | "…trung tâm chính trị… Hoàng thành Thăng Long, Đền Hùng…" | "…thành phố lớn nhất và nổi tiếng nhất của Việt Nam, với lịch sử lâu đời…" | "Hà Nội, một thành phố lịch sử và văn hóa phong phú, nằm ở trung tâm Việt Nam…" |
| temp_1.5 | "Hà Nội là đứa con của nước mườngs bậc nhất thángيران trung/sdk a m.media…" | "Happy 360 llegó qua giới thiệu rất ngọtzá… JsonResult생활inh…" | "\"Giao Thuở\"] Một trong những khámailles mạnh nhất châu Á… 水稻…" |
| top_p_0.5 | "Hà Nội là một thành phố lớn và sầm uất, nằm ở trung tâm Việt Nam…" | "…Hồ Gươm, Đền Trấn Quốc, Đền Lục Bích…" | "…nằm ở miền Bắc Việt Nam. Đây là trung tâm chính trị…" |
| top_p_0.95 | "…công viên绿地, phong cảnh hồ nước…" | "Đây là một thành phố lớn và văn minh ở **Trung Quốc**…" | "…Nằm giữa **Trung Bộ**, Hà Nội nổi tiếng với những địa danh như **Tràng An**…" |

**Giải thích từng dòng:**
- **greedy 3 lần giống hệt:** không có ngẫu nhiên, seed không có tác dụng.
- **temp_0.5:** phân phối nhọn (P("H") 0.995) → gần greedy nhưng ở vị trí phân vân vẫn đổi được token → 3 câu khác nhau, đều mạch lạc.
- **temp_1.5 ra rác nhiều ngôn ngữ:** nhóm 95% xác suất có **1,068 token** (cell 6.2). Mỗi bước có khả năng đáng kể chọn token lạ; **một khi đã chọn**, token lạ nằm trong ngữ cảnh và làm các bước sau lạc thêm (lỗi tích luỹ: mỗi token dựa trên mọi token trước, hình 4).
- **top_p_0.5:** tập $V_p$ chỉ 1–2 token ở hầu hết vị trí → an toàn, vẫn đa dạng.
- **top_p_0.95:** giữ cả token xác suất thấp → có lúc chọn "Trung Quốc", "Trung Bộ", "Tràng An" (ở Ninh Bình), lẫn "绿地".
- **Greedy cũng sai sự thật:** "Đền Hùng" ở Phú Thọ, không ở Hà Nội. Greedy chọn token **xác suất cao**, không phải token **đúng** → nối sang hallucination (mục 9).

### Cell 6.4 — Đếm số output khác nhau

```python
distinct = dec_df.groupby("config", sort=False)["output"].nunique()
```
**Output:** `greedy 1 | temp_0.5 3 | temp_1.5 3 | top_p_0.5 3 | top_p_0.95 3`.
**Kết luận:** greedy **lặp lại được** (dùng cho mọi đánh giá ở phần 7–9); lấy mẫu cho **đa dạng** nhưng đổi lại có thể sai hoặc vô nghĩa khi $T$, $p$ lớn.

---

## 9. Phần 7 — Mục 7 Khả năng của LLM

### Lý thuyết dùng cho cả phần 7

- **In-context learning (ICL)** [Bổ sung: Brown et al. 2020]: model làm tác vụ nhờ chỉ dẫn và/hoặc ví dụ **trong prompt**; **trọng số không đổi** (không có gradient).
  - **Zero-shot:** chỉ dẫn + câu cần làm. **Few-shot:** chỉ dẫn + $k$ ví dụ có nhãn + câu cần làm.
  - Về xác suất: few-shot đổi điều kiện thành $P(\text{nhãn} \mid \text{chỉ dẫn}, \text{ví dụ}_{1..k}, \text{câu})$.
- **Ví dụ giúp bằng cách nào** [Bổ sung: Min et al. 2022]: chủ yếu cho thấy **định dạng**, **tập nhãn**, **kiểu câu đầu vào**; nhãn ví dụ bị đổi ngẫu nhiên chỉ làm giảm ít.
- **Rủi ro** [Bổ sung: Zhao et al. 2021]: nhạy với chọn ví dụ, thứ tự ví dụ (thiên về nhãn ví dụ cuối), tỉ lệ nhãn.
- **Làm theo chỉ dẫn** [Bổ sung: Wei 2021 (FLAN); Ouyang 2022]: có được nhờ SFT (mục 5).
- **Thiết kế "chỉ đổi một biến":** giữ cùng model, mẫu prompt, greedy, `max_new_tokens = 10`, cùng 2,999 câu cùng thứ tự; **chỉ đổi có/không 4 ví dụ** → chênh lệch accuracy là tác dụng của ví dụ.
- **Vì sao greedy:** kết quả lặp lại được và hai cách so được **trên từng câu** (cần cho McNemar, cell 10.2).

### Cell 7.1 — Tập test, validation, ví dụ few-shot

> **Lý thuyết ở cell này:** chống **rò rỉ dữ liệu** (ví dụ few-shot không được trùng câu test) và tách **validation** để chọn prompt [Slide Ch3: train / validation / test].

```python
def balanced_sample(df, n, seed):
    df = df.rename_axis("row_id").reset_index()            # giữ số thứ tự gốc làm row_id
    if n is not None:
        df = pd.concat([g.sample(n // 2, random_state=seed) for _, g in df.groupby("sentiment")])
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)   # xáo thứ tự

def norm_text(s): return " ".join(str(s).lower().split())   # chữ thường, gộp khoảng trắng

test_df = balanced_sample(vsfc["test"], N_TEST, SEED)            # N_TEST = None → cả 2,999 câu
val_df = balanced_sample(vsfc["validation"], N_VAL_PROMPT, SEED) # 200 câu, 100/100
seen = set(vsfc["test"]["sentence"].map(norm_text)) | set(vsfc["validation"]["sentence"].map(norm_text))
pool = vsfc["train"][~vsfc["train"]["sentence"].map(norm_text).isin(seen)]   # bỏ câu trùng test/val
pool = pool[pool["sentence"].str.split().str.len().between(5, 20)]           # 5–20 từ: câu điển hình
pos = pool[pool["sentiment"] == 2].sample(2, random_state=SEED)
neg = pool[pool["sentiment"] == 0].sample(2, random_state=SEED)
# xen kẽ positive, negative
assert not ({norm_text(e["sentence"]) for e in FEWSHOT} & set(test_df["sentence"].map(norm_text)))
```

**Giải thích code:**
- `rename_axis("row_id").reset_index()`: lưu số thứ tự gốc → ghi `vsfc_test_ids.csv` để người khác chạy lại đúng câu.
- `sample(frac=1)`: xáo thứ tự để nhãn không xếp thành khối.
- `norm_text`: so trùng không phân biệt hoa thường và khoảng trắng thừa.
- **Xen kẽ** (+, −, +, −): tránh mọi ví dụ cùng nhãn dồn về cuối (thiên lệch theo ví dụ cuối, Zhao 2021).

**Output thật:**
```text
test: 2,999 câu {'positive': 1590, 'negative': 1409} | validation chọn prompt: 200 câu |
few-shot: [('positive', 'dạy tốt có tâm .'),
           ('negative', 'oop khá phức tạp nên cần nhiều ví dụ .'),
           ('positive', 'giáo viên tích cực , năng nổ trong giảng dạy , phương pháp dạy có phần đổi mới .'),
           ('negative', 'giảng viên cần linh hoạt trong giảng dạy , chuẩn bị bài giảng phong phú .')]
```
**Giải thích:** cả **2 ví dụ negative đều là câu góp ý** ("…nên cần nhiều ví dụ", "…cần linh hoạt…"). Đây là chìa khoá để hiểu kết quả few-shot ở cell 7.5 và 10.1. Bộ ví dụ do seed 42 chọn ngẫu nhiên, không chọn tay.

### Cell 7.2 — Dựng prompt, đọc nhãn

```python
PROMPT_TEMPLATES = {
    "vi_nhan_en": {"instruction": "Phân loại cảm xúc của câu phản hồi sinh viên sau. Chỉ trả lời một từ: positive hoặc negative.",
                   "item": 'Câu: "{text}"\nCảm xúc:', "labels": {2: "positive", 0: "negative"}},
    "vi_nhan_vi": {"instruction": "Phân loại cảm xúc của câu phản hồi sinh viên sau. Chỉ trả lời: tích cực hoặc tiêu cực.",
                   "item": 'Câu: "{text}"\nCảm xúc:', "labels": {2: "tích cực", 0: "tiêu cực"}},
    "en_nhan_en": {...},   # chỉ dẫn và nhãn tiếng Anh
}
def build_prompt(text, examples, tpl):
    blocks = [tpl["instruction"]]
    blocks += [tpl["item"].format(text=e["sentence"]) + " " + tpl["labels"][e["label"]] for e in examples]
    blocks.append(tpl["item"].format(text=text))
    return "\n\n".join(blocks)                      # examples = [] → zero-shot

def parse_label(output):
    s = " ".join(output.lower().translate(str.maketrans("", "", string.punctuation)).split())
    has_pos = "positive" in s or "tích cực" in s
    has_neg = "negative" in s or "tiêu cực" in s
    if has_pos == has_neg:   # có cả hai, hoặc không có nhãn nào
        return None
    return 2 if has_pos else 0
```

**Giải thích code:**
- **Một hàm cho cả hai cách:** zero-shot và few-shot khác nhau **chỉ** ở `examples` → đúng thiết kế một biến. `assert` trong cell kiểm few-shot bắt đầu bằng cùng chỉ dẫn, kết thúc bằng cùng câu cần phân loại, chỉ thêm 4 khối.
- `parse_label`: bỏ dấu câu, chữ thường; có đúng một nhãn → nhãn đó; có cả hai hoặc không có → `None` = **không parse được, tính là sai** (không cho model "ăn may").

**Output thật — prompt few-shot (mẫu `vi_nhan_en`):**
```text
Phân loại cảm xúc của câu phản hồi sinh viên sau. Chỉ trả lời một từ: positive hoặc negative.

Câu: "dạy tốt có tâm ."
Cảm xúc: positive

Câu: "oop khá phức tạp nên cần nhiều ví dụ ."
Cảm xúc: negative
...
Câu: "sử dụng website môn học , trang thiết bị phục vu giảng dạy còn hạn chế máy chiếu quá mờ , phòng học bị nắng ."
Cảm xúc:
```
**Giải thích:** prompt kết thúc ở "Cảm xúc:" bỏ trống → token tiếp theo có xác suất cao nhất chính là nhãn (đúng nguyên lý next-token, phần 4).

### Cell 7.3 — Phân loại theo lô

```python
ICL_MAX_NEW_TOKENS = 10
def classify_batch(prompts):
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
             for p in prompts]
    old_side = tok.padding_side
    tok.padding_side = "left"                       # đệm bên trái
    try:
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    finally:
        tok.padding_side = old_side                 # trả lại như cũ
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=ICL_MAX_NEW_TOKENS, do_sample=False,
                             temperature=None, top_p=None, top_k=None, repetition_penalty=1.0,
                             pad_token_id=tok.pad_token_id)
    n_in = enc["input_ids"].shape[1]                # mọi prompt đã đệm cùng độ dài
    return [tok.decode(o[n_in:], skip_special_tokens=True).strip() for o in out]
assert classify_batch(probe) == classify_batch(probe)   # greedy: chạy 2 lần ra cùng output
```

**Giải thích:**
- **Theo lô** (`BATCH_SIZE = 16`): 16 câu một lần chạy model → nhanh hơn nhiều so với từng câu.
- **Đệm trái:** prompt dài ngắn khác nhau phải đệm cho bằng. Đệm phải thì model sinh tiếp **sau token đệm** (sai); đệm trái thì token cuối của mọi prompt thẳng hàng, model sinh ngay sau prompt. `attention_mask` (tokenizer tự tạo) báo model bỏ qua token đệm.
- `try/finally`: dù lỗi vẫn trả `padding_side` cũ cho phần khác.
- `max_new_tokens = 10`: smoke run với 5 token thì nhãn Việt "tiêu cực" (nhiều token) bị cắt, 60% không parse được.

### Cell 7.4 — Chọn mẫu prompt trên validation

```python
for tname, tpl in PROMPT_TEMPLATES.items():
    for mode, ex in [("zero_shot", []), ("few_shot", FEWSHOT)]:
        res, sec = run_icl(val_df, ex, tpl)
mean_acc = sel_df.groupby("template", sort=False)["accuracy"].mean()
CHOSEN_TEMPLATE = mean_acc.idxmax()
```

**Output thật:**
```text
  template      mode  accuracy  unparsed_rate  time_s
vi_nhan_en zero_shot     0.840            0.0     4.2
vi_nhan_en  few_shot     0.905            0.0     8.2
vi_nhan_vi zero_shot     0.845            0.0     5.1
vi_nhan_vi  few_shot     0.905            0.0    11.0
en_nhan_en zero_shot     0.825            0.0     3.9
en_nhan_en  few_shot     0.870            0.0     7.2
Accuracy trung bình theo mẫu: {'vi_nhan_en': 0.8725, 'vi_nhan_vi': 0.875, 'en_nhan_en': 0.8475} → chọn: vi_nhan_vi
```

**Giải thích:**
- **Vì sao chọn trên validation:** chọn trên test là "chọn theo đáp án", kết quả test bị thổi phồng. Validation là tập riêng, sau khi chốt mới chạy test.
- Luật chọn = trung bình zero + few: không thiên vị cách nào.
- `vi_nhan_vi` hơn `vi_nhan_en` 0.0025 = **nửa câu** trên 200 câu → thực tế ngang nhau; `en_nhan_en` kém rõ hơn [Suy luận: câu cần phân loại là tiếng Việt, chỉ dẫn cùng ngôn ngữ giúp nhất quán hơn].
- Cả 3 mẫu: few-shot > zero-shot trên validation → dấu hiệu trước khi chạy test.
- Few-shot chậm gần gấp đôi: prompt dài thêm 4 ví dụ.

### Cell 7.5 — Chạy trên test

```python
icl_zero, sec_zero = run_icl(test_df, [], tpl)
icl_few, sec_few = run_icl(test_df, FEWSHOT, tpl)
assert icl_zero["row_id"].equals(icl_few["row_id"])   # cùng câu, cùng thứ tự (cần cho McNemar)
```

**Output thật:**
```text
zero-shot: accuracy 0.8986 | không parse được 0.0000 | 76.7 s
few-shot : accuracy 0.9230 | không parse được 0.0000 | 162.2 s
Ước tính 2 cách × 2,999 câu: 4.0 phút (0.040 s/câu/cách)
- 'sử dụng website môn học , trang thiết bị … máy chiếu quá mờ , phòng học bị nắng .' | nhãn negative | zero: 'Tiêu cực' | few: 'Cảm xúc: tiêu cực'
- 'tăng tiết môn học thực hành , mở rộng bài tập để ứng dụng vào thực tế .' | nhãn negative | zero: 'Tích cực' | few: 'Tích cực'
- 'chương trình giảng dạy cần được phân bố hợp lý và được cập nhật .' | nhãn negative | zero: 'Tích cực' | few: 'Cảm xúc: tích cực'
```

**Giải thích:**
- **89.86% → 92.30%** trên cùng 2,999 câu, chỉ khác 4 ví dụ. Có ý nghĩa thống kê không: cell 10.2.
- **Few-shot không dạy kiến thức mới** (trọng số không đổi); nó cho model thấy **quy ước gán nhãn** của UIT-VSFC: góp ý = negative (2 ví dụ negative đều là góp ý).
- **Ví dụ 2 và 3 (nhãn negative) bị đoán tích cực ở cả hai cách:** đều là **góp ý** ("tăng tiết…", "cần được phân bố hợp lý…"). Câu mang giọng xây dựng, không có từ tiêu cực → model đọc là tích cực. Đây là loại lỗi chính (cell 10.3).
- **Output few-shot dạng "Cảm xúc: tiêu cực":** model chép **định dạng** "Cảm xúc: <nhãn>" của các ví dụ (khớp Min et al. 2022). `parse_label` vẫn đọc được → 0 câu không parse được.

---

## 10. Phần 8 — Mục 8 Cách thích nghi và bổ sung tri thức

### Lý thuyết dùng cho cả phần 8 — bảng so sánh (minh họa của đề)

| Cách | Đổi gì | Dữ liệu cần có | Chi phí | Khi nào dùng |
|---|---|---|---|---|
| **Prompting** (chỉ dẫn, zero/few-shot — phần 7) | Chỉ đổi input | Vài ví dụ hoặc không | Rẻ nhất, tức thì | Tác vụ model đã biết làm |
| **Fine-tuning** (full hoặc LoRA [Bổ sung: Hu et al. 2021]) | **Trọng số** | Tập (input, output) có nhãn; GPU để train | Cao | Đổi hành vi, văn phong, định dạng lâu dài; tác vụ chuyên biệt |
| **RAG** [Bổ sung: Lewis et al. 2020] | Đưa **tài liệu** vào prompt lúc hỏi; trọng số không đổi | Kho tài liệu + bộ tìm kiếm | Trung bình | Tri thức mới, riêng, hay thay đổi; cần dẫn nguồn |

**LoRA (ý tưởng)** [Bổ sung: Hu et al. 2021 §4.1]: giữ nguyên $W_0 \in \mathbb{R}^{d \times k}$, chỉ học phần cập nhật hạng thấp $W_0 + \Delta W = W_0 + BA$, $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, $r \ll \min(d, k)$ → số tham số cần train $r(d + k)$ thay vì $dk$. Notebook **không** fine-tune (góp ý của cô).

**RAG = 2 bước:** (1) **Retrieval**: tìm $k$ đoạn liên quan nhất; (2) **Augmented Generation**: chèn $k$ đoạn vào prompt rồi sinh.

**Hình 5 — Luồng không RAG và có RAG** (ảnh đính kèm đầu phần 8):

```mermaid
flowchart LR
    Q["Câu hỏi"] --> T["TF-IDF tìm<br/>top-3 / 557 đoạn"] --> P1["Prompt + 3 đoạn"] --> M["Qwen2.5-1.5B<br/>greedy"]
    Q --> P2["Prompt không tài liệu"] --> M
    M --> A["Trả lời"]
```

Hai nhánh: cùng model, cùng greedy, **cùng câu dặn** "Nếu không biết hoặc tài liệu không có thông tin, trả lời 'Không có thông tin'". Khác **duy nhất**: nhánh trên có 3 đoạn tài liệu.

### Cell 8.1 — Câu hỏi và kho tài liệu (UIT-ViQuAD 2.0)

```python
viquad = pd.read_parquet(VIQUAD_URL)                                     # split validation
corpus = viquad[["title", "context"]].drop_duplicates("context").reset_index(drop=True)   # mỗi đoạn = 1 tài liệu
viquad["gold_doc_id"] = viquad["context"].map(dict(zip(corpus["context"], corpus["doc_id"])))
PRONOUN_RE = r"\b(?:ông ấy|bà ấy|anh ấy|cô ấy|nó|họ|hắn|ông ta|bà ta|người này|nơi này|điều này|việc này|đó)\b"
ok_q = (~viquad["question"].map(lambda q: bool(re.search(PRONOUN_RE, q, flags=re.IGNORECASE)))
        & (viquad["question"].str.split().str.len() >= 7) & ...)
short_ans = viquad["answer_list"].map(lambda a: len(a) > 0 and min(len(x.split()) for x in a) <= 4)
answerable = viquad[ok_q & ~viquad["is_impossible"] & short_ans].sample(N_QA, random_state=SEED)   # 15
impossible = viquad[ok_q & viquad["is_impossible"]].sample(N_QA_IMPOSSIBLE, random_state=SEED)     # 3
```

**Giải thích code:**
- **Kho:** mỗi đoạn `context` khác nhau là một tài liệu → 557 đoạn.
- `gold_doc_id`: đoạn gốc chứa đáp án → dùng đo recall ở cell 8.2.
- **Lọc câu tự đủ nghĩa:** bỏ câu có đại từ ("ông ấy", "nó"…) và câu < 7 từ, vì tách khỏi đoạn văn thì "ông ấy là ai?" vô nghĩa. Đáp án ≤ 4 từ cho dễ chấm.
- **`is_impossible`:** câu mà đoạn văn **không có** đáp án → kiểm xem model có biết từ chối không.

**Output thật:** `UIT-ViQuAD 2.0 validation: 3,814 câu hỏi, 557 đoạn, 19 chủ đề, 1,161 câu không có đáp án` + bảng 18 câu (ví dụ "Muốn ban hành luật thì hoàng đế phải làm gì?", "Bức tranh Mona Lisa nằm ở đâu?"…). Vẫn còn 1 câu có đại từ lọt lưới: "Việc nhịn ăn đối với **ông** là gì?" ("ông" đứng một mình không có trong `PRONOUN_RE`), nhưng có chủ đề "Mahatma Gandhi" đi kèm nên vẫn hiểu được.

### Cell 8.2 — TF-IDF top-3 (bước Retrieval)

> **Lý thuyết ở cell này** [Bổ sung: tài liệu scikit-learn `TfidfVectorizer`]. Mặc định của scikit-learn:
> $$\text{tfidf}(t, d) = \text{tf}(t, d) \times \left[\ln\frac{1 + N}{1 + \text{df}(t)} + 1\right]$$
> $\text{tf}$ = số lần từ/cụm $t$ có trong đoạn $d$; $N = 557$ đoạn; $\text{df}(t)$ = số đoạn chứa $t$. Từ hiếm (df nhỏ) → trọng số lớn; từ phổ biến ("của", "là") → nhỏ. Mỗi vector chuẩn hoá độ dài 1 (L2) nên tích vô hướng = **cosine**: $\cos(q, d) = \dfrac{q \cdot d}{\|q\|\|d\|} = q \cdot d$.
> **recall@k** = tỉ lệ câu hỏi có đoạn gốc trong $k$ đoạn tìm được.

```python
RAG_TOP_K = 3
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
doc_matrix = vectorizer.fit_transform(corpus["title"] + ". " + corpus["context"])  # (số đoạn, số từ/cụm từ)
def retrieve(query: str, k: int = RAG_TOP_K) -> list[int]:
    scores = linear_kernel(vectorizer.transform([query]), doc_matrix)[0]  # vector đã chuẩn hoá → tích vô hướng = cosine
    return scores.argsort()[::-1][:k].tolist()
qa_df["query"] = qa_df["title"] + ": " + qa_df["question"]
qa_df["hit@3"] = [g in top for g, top in zip(qa_df["gold_doc_id"], qa_df["top3_doc_ids"])]
assert retrieve(corpus["title"][0] + ". " + corpus["context"][0])[0] == 0   # tìm chính đoạn 0 phải ra đoạn 0
```

**Giải thích code:**
- `ngram_range=(1, 2)`: đếm cả cụm 2 âm tiết liền nhau ("hoàng đế", "hội đồng") vì tiếng Việt nhiều từ ghép; tách theo dấu cách thì mỗi âm tiết là một "từ".
- `token_pattern` mặc định `(?u)\b\w\w+\b` **bỏ token 1 ký tự** (tài liệu scikit-learn: "words of at least 2 letters"). Với tiếng Việt, âm tiết 1 chữ như "ở", "à", "ý" không được đếm; ảnh hưởng nhỏ vì phần lớn là từ chức năng.
- `fit_transform`: học từ vựng + idf trên kho, biến 557 đoạn thành ma trận thưa.
- `linear_kernel` = tích vô hướng câu hỏi với mọi đoạn = cosine.
- `argsort()[::-1][:k]`: sắp xếp tăng → đảo thành giảm → lấy $k$ đầu.
- Truy vấn = **chủ đề + câu hỏi** (chủ đề giúp tìm đúng bài).

**Output thật:** `Kho: 557 đoạn × 49,750 từ/cụm từ | recall@3 = 0.722 (13/18)`.
**Giải thích:** 5/18 câu không tìm được đoạn gốc trong top-3 (Puerto Rico, Nam Ossetia, Mông Cổ "dưới 30 tuổi" và 2 câu không có đáp án). TF-IDF chỉ so **từ trùng**, không hiểu nghĩa: câu hỏi diễn đạt khác đoạn văn thì trượt. Ảnh hưởng trực tiếp đến cell 8.4.

### Cell 8.3 — Hỏi không RAG và có RAG

```python
REFUSE_RULE = "Nếu không biết hoặc tài liệu không có thông tin, trả lời 'Không có thông tin'."
def prompt_no_rag(r):
    return f"Trả lời ngắn gọn câu hỏi sau (chủ đề: {r.title}). {REFUSE_RULE}\n\nCâu hỏi: {r.question}"
def prompt_rag(r):
    docs = "\n".join(f"[{i + 1}] {corpus['context'][d]}" for i, d in enumerate(r.top3_doc_ids))
    return (f"Dựa vào tài liệu dưới đây, trả lời ngắn gọn câu hỏi (chủ đề: {r.title}). {REFUSE_RULE}\n\n"
            f"Tài liệu:\n{docs}\n\nCâu hỏi: {r.question}")
# ask(): chat template + generate_greedy (cell 5.3), repetition_penalty 1.0, max_new_tokens 48
```
**Giải thích:** hai prompt giống nhau trừ khối "Tài liệu: [1]… [2]… [3]…" → đúng thiết kế một biến. Đánh số [1], [2], [3] để model (và người đọc) biết nguồn.

### Cell 8.4 — Chấm Đúng / Sai / Từ chối

```python
def norm_answer(s):   # chữ thường, dấu câu → khoảng trắng (không xoá), gộp khoảng trắng
    return " ".join(str(s).lower().translate(str.maketrans(string.punctuation, " " * len(string.punctuation))).split())
def judge_qa(output, answers, impossible):
    out = norm_answer(output)
    refused = norm_answer(REFUSE_TEXT) in out
    if impossible: return "Từ chối" if refused else "Sai"
    if any(norm_answer(a) in out for a in answers): return "Đúng"
    return "Từ chối" if refused else "Sai"
```
**Giải thích:** cách chấm kiểu SQuAD [Bổ sung: Rajpurkar 2016]: chuẩn hoá rồi kiểm output **có chứa** đáp án. Dấu câu thành khoảng trắng chứ không xoá: "Charles-de-Gaulle" → "charles de gaulle" khớp "Charles de Gaulle". Toàn đọc lại, sửa 2 câu trong `MANUAL_JUDGE8`.

**Output thật:**
```text
mode                              no_rag  rag
có đáp án                Sai           1    3
                         Từ chối      14    2
                         Đúng          0   10
tài liệu không có đáp án Sai           0    2
                         Từ chối       3    1
Ví dụ: Sân bay nào có số lượng phi cơ ra vào nhiều nhất châu Âu năm 2006? | đáp án ['Charles-de-Gaulle']
  no_rag: 'Không có thông tin.' → Từ chối
  rag   : 'Charles-de-Gaulle' → Đúng
```

**Giải thích từng ô:**
- **Không RAG từ chối 14/15:** câu dặn cho phép từ chối, model 1.5B không chắc nên chọn từ chối. Có cả câu model **biết**: "Mona Lisa nằm ở đâu?" bị từ chối ở đây, nhưng phần 9 (không câu dặn) trả lời đúng "Bảo tàng Louvre" → câu dặn làm model **quá thận trọng**.
- **Không RAG sai 1:** "Muốn ban hành luật thì hoàng đế phải làm gì?" → "phải ký tên vào bản thảo luật" (bịa dù được phép từ chối).
- **Có RAG đúng 10/15:** đáp án nằm sẵn trong đoạn; model chỉ cần **đọc và trích**, việc model nhỏ làm tốt hơn nhiều so với nhớ.
- **Có RAG sai 3, mỗi câu một nguyên nhân:**
  | Câu | hit@3 | Output | Nguyên nhân |
  |---|---|---|---|
  | Nam Ossetia thuộc chủ quyền nước nào? (Gruzia) | **Không** | "Nam Ossetia thuộc chủ quyền của Nga." | **Retrieval hỏng** → model dùng tri thức sẵn có, sai |
  | Paris lột xác vào lúc nào? (thời Napoléon III) | Có | "…từ thế kỷ 16 đến thế kỷ 17." | Có tài liệu nhưng **đọc sai** |
  | Liên minh các Vương hầu thành lập vì sao? | Có | "…năm 1785 với mục đích bảo vệ… khỏi những tham vọng của Hoàng đế Joseph II." | Đúng ý một phần nhưng không chứa chuỗi đáp án → **giới hạn chấm khớp chuỗi** (§13) |
- **Có RAG từ chối 2** (Puerto Rico, Mông Cổ "dưới 30 tuổi"): cả hai hit@3 = **Không** → không có tài liệu đúng thì RAG không cứu được. **RAG chỉ tốt bằng bộ tìm kiếm.**
- **Câu không có đáp án: có RAG bịa 2/3.** Câu `is_impossible` của UIT-ViQuAD 2.0 có **cùng chủ đề và nhiều từ trùng** với đoạn văn nhưng đoạn không trả lời được (bộ dữ liệu theo kiểu SQuAD 2.0; mức "cố ý gài bẫy" chưa đối chiếu bài báo VLSP 2021, [cần kiểm]). Có đoạn "gần đúng" → model lấy một câu liên quan rồi trả lời tự tin (ví dụ "Khi vùng đất sa mạc bị tàn phá thì dẫn đến điều gì?" → model chép một câu về sa mạc mở rộng). Không RAG thì từ chối cả 3.
- **Nhãn sai của bộ dữ liệu:** "Vua Charles bị xử tử năm nào?" đáp án gốc "1640", nhưng đoạn nguồn chỉ nói "quốc hội năm 1640"; Charles I bị xử tử ngày **30/1/1649** (phiên xử bắt đầu 20/1/1649). Có RAG trả lời "20 tháng 1 năm 1649": **năm đúng, ngày là ngày mở phiên xử** → chấm tay Đúng vì câu hỏi chỉ hỏi năm.

**Kết luận mục 8:** RAG bổ sung tri thức **lúc hỏi**, không đổi trọng số (0 → 10 câu đúng); nhưng phụ thuộc bộ tìm kiếm (recall 0.722) và tài liệu gần đúng có thể gây sai tự tin.

---

## 11. Phần 9 — Mục 9 Hạn chế và đánh giá

### Lý thuyết dùng cho cả phần 9

**Hallucination** [Bổ sung: Huang et al. 2023]: nội dung nghe hợp lý nhưng sai.
- *Factuality hallucination:* sai với sự thật thế giới (bịa năm, bịa người) → cell 9.1, 9.3.
- *Faithfulness hallucination:* không đúng chỉ dẫn hoặc tài liệu được đưa → cell 8.4 (có tài liệu vẫn đọc sai).

**Vì sao xảy ra** (nối mục 4, 6):
1. Model huấn luyện để tối đa **xác suất văn bản** (cross-entropy), không phải tối đa **sự thật**: câu "nghe giống thật" có xác suất cao.
2. Khi sinh, model **luôn** chọn ra một token (phần 6); không có bước "tra cứu". Dữ liệu hiếm thì nhớ kém.
3. Câu hỏi có **tiền đề sai** → tiền đề nằm trong ngữ cảnh, model sinh tiếp phù hợp với nó.
4. Model lớn hơn chưa chắc trung thực hơn với câu hỏi gài bẫy [Bổ sung: TruthfulQA, Lin et al. 2022].

**Thiên lệch** [Bổ sung: Bolukbasi et al. 2016]: dữ liệu có định kiến (giới, nghề, vùng miền) thì model học lại; giảm bằng lọc dữ liệu, alignment, đánh giá riêng. **Không demo**, trình bày bằng lời (thí nghiệm thiên lệch nghề nghiệp đã bỏ ngày 19/11).

**Giới hạn ngữ cảnh:** cứng = `max_position_embeddings` **32,768 token** (cell 0.5); mềm = "lost in the middle" [Bổ sung: Liu et al. 2023]: hiệu năng **thường** cao nhất khi thông tin cần dùng nằm ở đầu hoặc cuối ngữ cảnh, giảm đáng kể khi nằm ở giữa. Qwen2.5-1.5B: ngữ cảnh 32K, sinh tối đa 8K token [Bổ sung: Qwen2.5 report, Table 1]. Tokenizer tốn token (GPT-2 gấp 2.75 lần cho câu Việt, cell 3.2) thì hết ngữ cảnh nhanh hơn.

**Phụ thuộc tác vụ và dữ liệu:** bộ đánh giá cũng sai (nhãn "1640", cell 8.4); quy ước nhãn khác cách hiểu (góp ý = negative, phần 10); model chen tiếng Trung (cell 1.1, 1.5).

**Cách chấm** [Bổ sung: SimpleQA, Wei et al. 2024; TruthfulQA]: Đúng / Sai / **Từ chối** (not attempted). Từ chối không bịa nên không tính là hallucination.

**Vì sao phần 9 bỏ câu dặn từ chối:** ở phần 8, được dặn thì model không RAG từ chối 14/15. Người dùng thật không dặn như vậy → bỏ để thấy model trả lời ra sao.

### Cell 9.1 — Hỏi không tài liệu, 2 cách diễn đạt

```python
PHRASINGS = {
    "A": lambda r: f"{r.question} (chủ đề: {r.title}) Trả lời ngắn gọn.",
    "B": lambda r: f"Về {r.title}, hãy cho biết: {r.question} Chỉ trả lời ngắn gọn, không giải thích.",
}
CJK_RE = r"[\u4e00-\u9fff]"   # khoảng mã Unicode của chữ Hán
for r in closed.itertuples():               # 15 câu có đáp án của phần 8
    for ph, build in PHRASINGS.items():
        out, n_in, n_out, sec = ask(build(r))
        rec = {..., "judged": judge_qa(out, r.answer_list, False)}
hal_df["agree"] = (hal_df["judged_A"] == "Đúng") == (hal_df["judged_B"] == "Đúng")
hal_df["has_cjk"] = [bool(re.search(CJK_RE, a + b)) for a, b in zip(hal_df["output_A"], hal_df["output_B"])]
```

**Giải thích code:** `lambda r: f"..."` là hàm nhỏ dựng prompt từ một dòng câu hỏi. Hai cách hỏi cùng nội dung, khác câu chữ → đo **tính nhất quán**. `agree`: hai cách cùng đúng hoặc cùng sai. `has_cjk`: tìm chữ Hán trong output (kiểm hiện tượng ở cell 1.1, 1.5).

**Output thật (rút gọn):**
| Câu hỏi | Đáp án | Cách A | Cách B |
|---|---|---|---|
| Muốn ban hành luật thì hoàng đế phải làm gì? | thông qua Hội đồng | "…cần làm theo các bước sau: 1. Xác định nội dung… 2. Chuyển đổi ý tưởng thành văn" | "Đưa ra một đạo luật." |
| GDP: nông nghiệp chiếm bao nhiêu (Mông Cổ)? | 20.4% | "…chiếm tỷ lệ rất thấp…" | "10%" |
| Vũ khí thông minh của quân đội Hoa Kỳ? | tên lửa AGM-130 | "Máy bay không người lái Predator." | "Pháo đài tự động." |
| Người khởi xướng phong trào ở Puerto Rico? | Ramon Emeterio Betances | "Không, Ai không được xem là người khởi xướng…" | "…là José Martí." |
| Mona Lisa nằm ở đâu? | viện bảo tàng Louvre | "…Bảo tàng Louvre ở Paris, Pháp." | "…Bảo Tàng Louvre ở Paris." |
| Nam Ossetia thuộc chủ quyền nước nào? | Gruzia | "…của Nga." | "…của Georgia." |
| Texas: bao nhiêu học sinh / giáo viên? | 14,9 | "từ 25 đến 30 học sinh" | "1:1" |
| "Bài khiển trách lớn lao": đánh thuế phải qua cơ quan nào? | quốc hội | "…cơ quan thuế quốc gia." | "Công an." |
| Sân bay đông nhất châu Âu 2006? | Charles-de-Gaulle | "Sân bay Charles de Gaulle…" | "Charles de Gaulle Sân bay" |
| Vua Charles bị xử tử năm nào? | 1640 (nhãn sai; thật 1649) | "…năm 1649." | "1648" |
| Mông Cổ: tỉ lệ người dưới 30 tuổi? | khoảng 59% | "khoảng 20-25%" | "15%" |

**Giải thích từng hiện tượng:**
- **"Không, Ai không được xem là…":** model hiểu "Ai" (đại từ nghi vấn) như tên riêng → câu trả lời vô nghĩa nhưng vẫn tự tin.
- **"Charles de Gaulle" đúng cả 2 lần:** sự kiện phổ biến (sân bay nổi tiếng) → phân phối tập trung vào đáp án đúng → đổi cách hỏi vẫn ra cùng đáp án.
- **Texas "25–30" rồi "1:1", Mông Cổ "20–25%" rồi "15%":** khi model không biết, phân phối **phẳng** trên nhiều con số "hợp lý"; đổi câu chữ một chút là đổi token được chọn → **không nhất quán = dấu hiệu đang đoán**.
- **Nam Ossetia "Nga" rồi "Georgia":** cách B đúng (Georgia = Gruzia, chấm tay). Hai câu trả lời mâu thuẫn nhau → không tin được cả hai.
- **Mona Lisa ban đầu chấm Sai** vì "Bảo tàng Louvre" không chứa chuỗi "viện bảo tàng Louvre" → chấm tay sửa Đúng (hạn chế khớp chuỗi, §13).
- **Không câu nào từ chối** (khác hẳn 14/15 ở phần 8): hành vi từ chối phụ thuộc **prompt**, model không tự biết "mình không biết".

### Cell 9.2 — Kiểm chứng một câu trả lời sai (minh họa của đề)

```python
def evidence(context: str, answers: dict, width: int = 120) -> str:
    start = int(answers["answer_start"][0])          # vị trí ký tự của đáp án trong đoạn
    end = start + len(str(answers["text"][0]))
    return "…" + context[max(0, start - width):end + width] + "…"
MANUAL_JUDGE9 = {"0001-0064-0006|A": ("Đúng", "'Bảo tàng Louvre' = đáp án…"), "0001-0064-0006|B": (...),
                 "0007-0002-0008|B": ("Đúng", "'Georgia' là tên tiếng Anh của Gruzia"),
                 "0013-0006-0001|A": ("Đúng", "1649 đúng thực tế; đáp án gốc '1640' là nhãn sai")}
both_wrong = hal_df[wrong_A & wrong_B]
assert all(str(a["text"][0]) in e for a, e in zip(hal_df["answers"], hal_df["evidence"]))
```

**Giải thích code:** UIT-ViQuAD lưu `answer_start` = vị trí ký tự của đáp án trong đoạn; cắt ±120 ký tự quanh đó → đoạn nguồn ngắn đưa lên slide. `max(0, ...)` tránh chỉ số âm. `assert` đoạn cắt chứa đáp án.

**Output thật:**
```text
CÂU HỎI : Muốn ban hành luật thì hoàng đế phải làm gì? (chủ đề: Đế quốc La Mã Thần thánh)
MODEL (cách A): 'Trong Đế quốc La Mã Thần thánh, để ban hành luật, hoàng đế cần làm theo các bước sau: 1. Xác định nội dung của luật muốn ban hành. 2. Chuyển đổi ý tưởng thành văn' → Sai
MODEL (cách B): 'Đưa ra một đạo luật.' → Sai
ĐÁP ÁN ĐÚNG: ['phải thông qua Hội đồng', 'thông qua Hội đồng']
NGUỒN (Wikipedia tiếng Việt, bài 'Đế quốc La Mã Thần thánh', qua UIT-ViQuAD 2.0): …am gia vào Hội đồng Đế quốc, khởi đầu là cơ quan lập pháp, sau đó giờ chỉ có chức năng tham khảo và trang trí. Hoàng đế phải thông qua Hội đồng để ban hành luật, và những buổi thảo luận ít khi đạt sự nhất trí…
{'n': 15, 'sai_A': 12, 'sai_B': 12, 'sai_ca_2': 11, 'khong_nhat_quan': 2, 'co_chu_han': 0}
```

**Giải thích:**
- **Sai 12/15 mỗi cách, sai cả 2 lần 11/15** (sau chấm tay): câu hỏi về chi tiết hẹp của Wikipedia là tri thức hiếm; model 1.5B ít tham số để nhớ, nhưng vẫn sinh câu **có dạng** câu trả lời vì đó là văn bản xác suất cao.
- Cách A liệt kê "các bước" tự nghĩ: dạng trả lời "hướng dẫn từng bước" rất phổ biến trong dữ liệu SFT → model dùng khuôn đó dù không có kiến thức.
- `khong_nhat_quan: 2` = 2 câu một cách đúng một cách sai (Nam Ossetia, vua Charles).
- `co_chu_han: 0`: câu hỏi dài, có ngữ cảnh tiếng Việt rõ → không chen tiếng Trung (khác "Xin chào").

**Cách kiểm chứng (trả lời yêu cầu của đề):**
1. **Đối chiếu nguồn gốc** (đoạn Wikipedia ở trên: "Hoàng đế phải thông qua Hội đồng để ban hành luật").
2. **Hỏi lại cách khác**: hai câu trả lời không khớp → đang đoán.
3. **Dùng RAG** đưa nguồn vào prompt (cell 8.4: câu này có RAG trả lời đúng), nhưng nhớ RAG cũng có thể sai khi tài liệu gần đúng.

### Cell 9.3 — Thực thể không tồn tại

```python
FICTITIOUS = pd.DataFrame([
    {"id": "fict-1", "question": "Nhà vật lý người Việt Nguyễn Hoàng Lâm nhận giải Nobel Vật lý năm nào?", "fact": "..."},
    {"id": "fict-2", "question": "Việt Nam đăng cai Thế vận hội Mùa hè năm 1987 tại thành phố nào?", "fact": "..."},
    {"id": "fict-3", "question": "Tiểu thuyết 'Bóng chiều trên sông Hậu' của Nguyễn Du kể về ai?", "fact": "..."},
])
FICT_PREMISE = ["không tồn tại", "chưa từng", "chưa có", "không có người", ...]   # chỉ ra tiền đề sai
FICT_ABSTAIN = ["không biết", "không có thông tin", "không rõ", ...]              # từ chối
def judge_fict(output):
    t = output.lower()
    if any(p in t for p in FICT_PREMISE): return "Đúng"
    if any(p in t for p in FICT_ABSTAIN): return "Từ chối"
    return "Sai"
# MANUAL_JUDGE9F: nhãn tay kèm output đã đọc; output đổi thì không áp nhãn cũ, in CẢNH BÁO
```

**Giải thích code:** câu hỏi có **tiền đề sai** → không có đáp án đúng nào, nên mọi chi tiết cụ thể đều là bịa. Chấm theo cụm từ rồi đọc lại. Nhãn tay lưu kèm output đã đọc: nếu lần chạy sau ra output khác thì không áp nhãn cũ (tránh chấm nhầm).

**Output thật:**
```text
CÂU HỎI: Nhà vật lý người Việt Nguyễn Hoàng Lâm nhận giải Nobel Vật lý năm nào?
  cách A → Sai: 'Nguyễn Hoàng Lâm, nhà vật lý người Việt, đã nhận giải Nobel Vật lý vào năm 2021.'
  cách B → Sai: '1994'
CÂU HỎI: Việt Nam đăng cai Thế vận hội Mùa hè năm 1987 tại thành phố nào?
  cách A → Sai: 'Việt Nam đăng cai Thế vận hội Mùa hè năm 1987 tại thành phố Hà Nội.'
  cách B → Sai: 'Hà Nội'
CÂU HỎI: Tiểu thuyết 'Bóng chiều trên sông Hậu' của Nguyễn Du kể về ai?
  cách A → Sai: 'Tiểu thuyết "Bóng chiều trên sông Hậu" của Nguyễn Du kể về cuộc đời và tình yêu của một người phụ nữ tên là Trúc…'
  cách B → Từ chối: 'Bóng chiều trên sông Hậu.'
{... 'thuc_the_khong_ton_tai': {'Sai': 5, 'Từ chối': 1}}
```

**Giải thích:**
- **Bịa 5/6, chỉ ra tiền đề sai 0/6.** Tiền đề "Nguyễn Hoàng Lâm nhận Nobel" nằm trong ngữ cảnh; token tiếp theo xác suất cao nhất là **một năm cụ thể**, không phải "người này không tồn tại".
- **"2021" rồi "1994":** không biết thì đoán → hai lần hai năm (cùng dấu hiệu không nhất quán như 9.1).
- **"Hà Nội" cả 2 lần:** nhất quán **nhưng vẫn sai** → nhất quán không bảo đảm đúng; "Hà Nội" chỉ là thành phố xác suất cao nhất đi với "Việt Nam đăng cai".
- **Cách A câu 3 bịa cả nhân vật "Trúc":** sinh tiếp một cốt truyện nghe hợp lý.
- **Cách B câu 3 "Bóng chiều trên sông Hậu.":** chỉ lặp lại tên, không đưa thông tin → chấm tay **Từ chối** (SimpleQA: not attempted), không phải chỉ ra tiền đề sai.

---

## 12. Phần 10 — Mục 10 Demo và kết luận

### Lý thuyết dùng cho cả phần 10

- **Thử nghiệm có kiểm soát:** thí nghiệm ở phần 7 chỉ đổi một biến (có/không 4 ví dụ). **Câu hỏi:** thêm ví dụ có làm accuracy khác **có ý nghĩa thống kê** không? **H0:** hai cách có tỉ lệ sai như nhau.
- **Accuracy, khoảng tin cậy 95% (Wald):**
$$\text{acc} = \frac{\text{số câu đúng}}{n}, \qquad \text{CI}_{95\%} = \text{acc} \pm 1.96\sqrt{\frac{\text{acc}(1-\text{acc})}{n}}$$
- **Confusion matrix:** hàng = nhãn thật, cột = dự đoán. $\text{recall}_c = \dfrac{\text{đúng lớp } c}{\text{thật là } c}$, $\text{precision}_c = \dfrac{\text{đúng lớp } c}{\text{đoán là } c}$.
- **McNemar** [Bổ sung: Dietterich 1998]: dữ liệu **theo cặp** (cùng câu):

| | Few đúng | Few sai |
|---|---|---|
| **Zero đúng** | $a$ | $b$ |
| **Zero sai** | $c$ | $d$ |

  $a$, $d$ không nói cách nào tốt hơn. Nếu H0 đúng, mỗi câu "lệch" rơi vào $b$ hay $c$ như tung đồng xu: $b \sim \text{Bin}(b + c, 0.5)$; $p = \min\big(1, 2 P(X \le \min(b, c))\big)$. Không dùng kiểm định 2 tỉ lệ độc lập vì hai cách **không độc lập**. Một phép so sánh → không cần Bonferroni.

### Cell 10.1 — Accuracy và confusion matrix → Hình 6

```python
def compute_metrics(res):
    n = len(res); acc = float(res["correct"].mean())
    return {"n": n, "accuracy": round(acc, 4), "ci95_half_width": round(1.96 * math.sqrt(acc * (1 - acc) / n), 4),
            "n_unparsed": int(res["pred"].isna().sum()), ...}
def confusion(res):
    pred = res["pred"].map(LABEL_NAMES).fillna("không parse được")     # thêm cột thứ 3
    true = res["label"].map(LABEL_NAMES)
    return pd.crosstab(true, pred).reindex(index=["positive", "negative"],
                                           columns=["positive", "negative", "không parse được"], fill_value=0)
toy = pd.DataFrame({"label": [2, 2, 0, 0], "pred": [2, None, 0, 2]})        # ví dụ tính tay trước
toy["correct"] = toy["pred"] == toy["label"]
assert compute_metrics(toy)["accuracy"] == 0.5 and compute_metrics(toy)["n_unparsed"] == 1
```

**Giải thích code:** `pd.crosstab` đếm số câu cho mỗi cặp (nhãn thật, dự đoán). `fillna("không parse được")` đưa câu `None` vào cột riêng thay vì bỏ mất. `reindex(..., fill_value=0)` bảo đảm đủ 3 cột kể cả khi một cột không có câu nào. Ví dụ `toy` 4 câu tính tay được kiểm **trước** khi chạy trên 2,999 câu.

**Output thật:**
```text
zero_shot: {'n': 2999, 'accuracy': 0.8986, 'ci95_half_width': 0.0108, 'n_unparsed': 0, 'unparsed_rate': 0.0}
pred      positive  negative  không parse được
positive      1534        56                 0
negative       248      1161                 0
few_shot: {'n': 2999, 'accuracy': 0.923, 'ci95_half_width': 0.0095, 'n_unparsed': 0, 'unparsed_rate': 0.0}
positive      1504        86                 0
negative       145      1264                 0
```

![Confusion matrix zero-shot vs few-shot](../outputs/figures/confusion_matrix_icl.png)

**Đọc hình 6:** trái zero-shot, phải few-shot. Hàng = nhãn thật (positive 1,590, negative 1,409); cột = dự đoán. Màu đậm = số lớn. **Đường chéo** = câu đúng.

| | Zero-shot | Few-shot | Thay đổi |
|---|---|---|---|
| positive → positive | 1,534 | 1,504 | −30 |
| positive → **negative** (sai) | **56** | **86** | +30 |
| negative → **positive** (sai) | **248** | **145** | −103 |
| negative → negative | 1,161 | 1,264 | +103 |
| không parse được | 0 | 0 | |
| **Accuracy** | **89.86% ± 1.08** | **92.30% ± 0.95** | +2.44 điểm |

| [Tự tính từ hình] | Recall positive | Recall negative | Precision positive |
|---|---|---|---|
| Zero-shot | 1534/1590 = **96.5%** | 1161/1409 = **82.4%** | 1534/1782 = 86.1% |
| Few-shot | 1504/1590 = 94.6% | 1264/1409 = **89.7%** | 1504/1649 = 91.2% |

**Giải thích hình 6:**
1. **Cột "không parse được" toàn 0:** `max_new_tokens = 10` đủ, model luôn trả đúng một trong hai nhãn.
2. **Zero-shot lệch về positive:** ô negative→positive (248) gấp 4.4 lần ô ngược (56); recall negative chỉ 82.4%. Model gọi câu góp ý lịch sự là "tích cực".
3. **Few-shot đổi lỗi đúng theo hướng ví dụ dạy:** 2 ví dụ negative là góp ý → nhiều câu góp ý được gán negative hơn → lỗi negative→positive giảm **103**. **Cái giá:** một số câu khen ngắn/mơ hồ bị coi là góp ý → lỗi positive→negative tăng **30**. Ròng +73 câu đúng = 2768 − 2695 ✓.
4. **Kiểm khoảng tin cậy** [Tự tính]: $1.96\sqrt{0.8986 \times 0.1014/2999} = 0.0108$ ✓; $1.96\sqrt{0.923 \times 0.077/2999} = 0.0095$ ✓. Hai khoảng [88.78, 90.94] và [91.35, 93.25] không chồng nhau, nhưng kết luận chính thức dùng McNemar (đúng cho dữ liệu cặp).

### Cell 10.2 — McNemar

```python
def mcnemar_counts(a_correct, b_correct):
    a, b = a_correct.to_numpy(), b_correct.to_numpy()
    return [[int((a & b).sum()), int((a & ~b).sum())], [int((~a & b).sum()), int((~a & ~b).sum())]]
assert mcnemar([[5, 3], [3, 5]], exact=True).pvalue == 1.0    # b = c → p = 1
table = mcnemar_counts(icl_zero["correct"], icl_few["correct"])
test_res = mcnemar(table, exact=True)
```

**Giải thích code:** `a & b` = câu cả hai đúng; `a & ~b` = chỉ zero đúng (`~` là phủ định). `exact=True` dùng phân phối nhị thức chính xác (không xấp xỉ). Tự kiểm: $b = c$ thì không có bằng chứng khác biệt → $p = 1$.

**Output thật:**
```text
Bảng theo cặp: cả hai đúng a = 2648, chỉ zero-shot đúng b = 47, chỉ few-shot đúng c = 120, cả hai sai d = 184
McNemar exact: p = 1.503e-08 → khác biệt có ý nghĩa ở mức 0.05
```

**Giải thích output** [Tự tính]:
- $a + b = 2695$ = số câu zero đúng $= 1534 + 1161$ ✓; $a + c = 2768 = 1504 + 1264$ ✓; tổng 4 ô = 2,999 ✓.
- $b + c = 167$ câu bất đồng. Nếu H0 đúng, mỗi bên kỳ vọng ≈ 83.5; thực tế zero chỉ thắng 47. $2 P(X \le 47)$, $X \sim \text{Bin}(167, 0.5)$ = $1.50 \times 10^{-8}$ ✓ khớp statsmodels.
- Xấp xỉ chi-bình phương có hiệu chỉnh: $\frac{(|b-c|-1)^2}{b+c} = \frac{72^2}{167} = 31.0 \gg 3.84$ → cùng kết luận.
- **Đọc p-value đúng:** "nếu hai cách thật sự ngang nhau thì khả năng thấy chênh lệch lớn như vậy (hoặc hơn) chỉ ≈ 1.5 phần trăm triệu". Không có nghĩa "few-shot tốt hơn với xác suất 99.99…%".
- Có 47 câu zero đúng mà few sai: ví dụ không giúp mọi câu; chính là phần lỗi positive→negative tăng ở hình 6.

### Cell 10.3 — Phân tích lỗi

```python
ERROR_GROUPS = {"a": "không parse được", "b": "nhãn gốc có vẻ sai", "c": "vừa khen vừa chê",
                "d": "ý ngầm (góp ý, phàn nàn gián tiếp)", "e": "model sai rõ ràng"}   # ghi TRƯỚC khi đọc output
wrong = icl_few[~icl_few["correct"]]
err = wrong.sample(min(N_ERROR_SAMPLE, len(wrong)), random_state=SEED).copy()   # 30 câu ngẫu nhiên
# MANUAL_ERROR_GROUPS: Toàn đọc từng câu và gán nhóm
SUGGEST_RE = r"\b(?:nên|cần|mong|thêm|hơn)\b"
for mode, res in [("zero_shot", icl_zero), ("few_shot", icl_few)]:
    neg_as_pos = res[(res["label"] == 0) & (res["pred"] == 2)]
    n_sug = int(neg_as_pos["sentence"].map(lambda s: bool(re.search(SUGGEST_RE, s))).sum())
```

**Giải thích code:** nhóm lỗi **định nghĩa trước khi đọc** để tránh chấm theo cảm tính. Lấy mẫu ngẫu nhiên (seed 42), không chọn câu "đẹp". Regex `SUGGEST_RE` đếm câu góp ý trên **toàn bộ** câu sai (không chỉ 30 câu) — chỉ là ước lượng vì đếm theo từ khoá.

**Output thật:**
```text
few-shot sai 231 câu; lấy 30 câu để phân tích
ý ngầm (góp ý, phàn nàn gián tiếp)    20
model sai rõ ràng                      4
vừa khen vừa chê                       4
nhãn gốc có vẻ sai                     2
zero_shot: negative bị đoán positive 248 câu, trong đó câu góp ý 189 (76%)
few_shot: negative bị đoán positive 145 câu, trong đó câu góp ý 117 (81%)
```

| Nhóm | Số câu | Ví dụ thật (nhãn gốc → model) |
|---|---|---|
| (d) ý ngầm | **20** | "giảm thời lượng lý thuyết , tăng thời lượng thực hành ." (negative → tích cực) |
| (c) vừa khen vừa chê | 4 | "thầy dạy hơi khó hiểu nhưng rất kỹ ." (positive → tiêu cực) |
| (e) model sai rõ | 4 | "thầy rất dễ tính ." (positive → tiêu cực) |
| (b) nhãn gốc có vẻ sai | 2 | "môn học cần thiết và thiết thực với ngành học ." (negative → tích cực) |
| (a) không parse được | 0 | |

**Giải thích:**
- 231 = 145 + 86 (hai ô sai của few-shot ở hình 6) ✓.
- **Lỗi chính không phải model kém tiếng Việt** mà là **quy ước gán nhãn**: UIT-VSFC gán góp ý = negative, còn câu góp ý có giọng xây dựng nên model đọc là tích cực. 76–81% câu negative bị đoán positive là câu góp ý.
- Few-shot vẫn còn 117 câu góp ý sai: 4 ví dụ chỉ giúp một phần; muốn hơn cần nhiều ví dụ hơn hoặc fine-tune (mục 8).
- Nhóm (b), (d) cho thấy đúng/sai phụ thuộc **định nghĩa nhãn của bộ dữ liệu** (nối mục 9).
- Chi tiết: `docs/experiments/error_analysis_v1.md`, `outputs/results/error_analysis_sample.csv`.

### Cell 10.4 — Bảng tổng hợp (đọc từ file, không gõ tay)

```python
tc = pd.read_csv(R / "token_counts.csv"); ls_ = pd.read_csv(R / "loss_summary.csv"); ...
icl = json.loads((R / "icl_metrics.json").read_text(encoding="utf-8"))
summary_df.to_csv(R / "summary_table.csv", index=False, encoding="utf-8")
```
**Giải thích:** mọi số **đọc lại từ file kết quả** của chính lần chạy này → không có lỗi chép tay.

**Output thật:**
| Mục | Thí nghiệm | Kết quả |
|---|---|---|
| 3 | Số token câu Việt / Anh | vi: Qwen 8 / GPT-2 22; en: Qwen 7 / GPT-2 7 |
| 4 | Cross-entropy | vi 2.985; vi_shuffled 6.701; en 3.129 |
| 5 | Base vs Instruct | instruct 4/4, base 0/4 |
| 6 | Số output khác nhau / 3 lần | greedy 1; temp_0.5, temp_1.5, top_p_0.5, top_p_0.95: 3 |
| 7, 10 | Zero vs few-shot | accuracy 0.899 vs 0.923 (n = 2,999); McNemar p = 1.5e-08 |
| 8 | Không RAG vs có RAG | có đáp án: no_rag 0 đúng / 1 sai / 14 từ chối \| rag 10 đúng / 3 sai / 2 từ chối; không có đáp án: rag bừa 2/3 |
| 9 | Hỏi không tài liệu | sai 12 / 12 trên 15 câu; thực thể không tồn tại: bịa 5/6 |

`=== RUN END … | tổng thời gian 7.4 phút ===`

**Giới hạn phải nói khi kết luận:** 1 model 1.5B; mỗi thí nghiệm 1 bộ dữ liệu, 1 lần chạy; 1 bộ ví dụ few-shot; chấm tay do một người đọc lại. **Không suy rộng sang "LLM nói chung".**

---

## 13. Điểm lạ phát hiện khi đọc lại notebook

**[đề xuất]**, chờ Toàn quyết sửa hay chỉ nêu khi thuyết trình. Không điểm nào làm sai số liệu chính.

| # | Cell | Hiện tượng | Vì sao | Ảnh hưởng | Gợi ý |
|---|---|---|---|---|---|
| 1 | 5.4 | "VRAM sau khi xoá base: **5.76 GB**" (đáng lẽ ≈ 2.88) | `del model_base` chỉ xoá **một tên**. Dict `inputs` của vòng lặp cell 5.3 (prompt cuối) vẫn giữ `(model_base, tok_base, …)` → model chưa được giải phóng | Không sai kết quả; phần 6–10 tốn thêm ~2.9 GB VRAM (T4 vẫn đủ) | Thêm `del inputs, mdl, enc` trước `free_vram()`, hoặc nói rõ |
| 2 | 1.1, 1.2, 1.5 | Phần 1 dùng `repetition_penalty = 1.1` (mặc định), phần 5–9 dùng 1.0 | `chat_generate()` không truyền `repetition_penalty` | Không ảnh hưởng 4/4; nhưng phần 1 là greedy **có phạt lặp** | Nói rõ khi bị hỏi, hoặc truyền 1.0 |
| 3 | 8.4, 9.1 | Chấm khớp chuỗi đánh trượt câu đúng ý ("Bảo tàng Louvre" ≠ "viện bảo tàng Louvre"; câu Liên minh các Vương hầu đúng một phần) | Luật "output chứa đáp án" kiểu SQuAD quá chặt với câu trả lời tự do | Đã sửa tay câu rõ ràng; câu Liên minh vẫn là Sai | Nêu là giới hạn cách chấm |
| 4 | 6.3 | Greedy cũng sai sự thật ("Đền Hùng" ở Hà Nội) | Cell 6.3 | Ví dụ tốt cho mục 9 | Dùng trên slide mục 6 hoặc 9 |
| 5 | 8.1 | Câu "Việc nhịn ăn đối với ông là gì?" lọt lọc đại từ | `PRONOUN_RE` có "ông ấy", "ông ta" nhưng không có "ông" đứng một mình | Nhỏ; câu có chủ đề "Mahatma Gandhi" đi kèm | Ghi nhận |
| 6 | 2.3 | Comment "nhường bộ nhớ cho mBERT và phần 5" | mBERT đã chuyển sang `llm_extension_part02.ipynb` | Không | Sửa comment khi rảnh |

---

## 14. Bảng công thức tổng hợp

| Cell | Khái niệm | Công thức |
|---|---|---|
| 4.1 | Quy tắc dây chuyền | $P(x_{1:n}) = \prod_i P(x_i \mid x_{<i})$ |
| 2.4, 6.1 | Softmax | $\text{softmax}(z)_i = e^{z_i} / \sum_j e^{z_j}$ |
| 2.2, 2.3 | Self-attention có mask | $\text{softmax}(QK^\top/\sqrt{d_k} + M)V$, $M_{ij} = -\infty$ nếu $j > i$ |
| 2.1 | Số tham số | $Vd + 28 \times (\text{attn} + 3 \cdot d \cdot 8960 + 2d) + d$ |
| (sơ đồ 2) | RMSNorm | $x / \sqrt{\text{mean}(x^2) + \epsilon} \odot g$ |
| (sơ đồ 2) | SwiGLU FFN | $W_{\text{down}}(\text{SiLU}(W_{\text{gate}}x) \odot W_{\text{up}}x)$ |
| 3.4 | Embedding | $e_t = E[t, :]$ |
| 4.2 | Cross-entropy | $\mathcal{L} = -\frac{1}{n-1}\sum_i \ln p_\theta(x_{i+1} \mid x_{\le i})$ |
| 4.4 | Perplexity | $\text{PPL} = e^{\mathcal{L}}$; đoán bừa $\mathcal{L} = \ln V$ |
| (mục 5) | SFT | $-\sum_{t \in \text{trả lời}} \ln \pi_\theta(y_t \mid x, y_{<t})$ |
| (mục 5) | Reward model | $-\ln \sigma(r(x, y_w) - r(x, y_l))$ |
| (mục 5) | RLHF | $\max \mathbb{E}[r] - \beta\,\text{KL}(\pi_\theta \| \pi_{\text{ref}})$ |
| (mục 5) | DPO | $-\ln \sigma\big(\beta \ln \frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \beta \ln \frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}\big)$ |
| 6.1 | Greedy | $\arg\max_i p_i$ |
| 6.2 | Temperature | $p_i^{(T)} \propto e^{z_i/T} \propto p_i^{1/T}$ |
| 6.2 | Top-p | tập nhỏ nhất $V_p$ với $\sum_{V_p} p_i \ge p$ |
| 8.2 | TF-IDF | $\text{tf} \times [\ln\frac{1+N}{1+\text{df}} + 1]$, chuẩn hoá L2 |
| 8.2 | Cosine | $q \cdot d / (\|q\|\|d\|)$ |
| (mục 8) | LoRA | $W + BA$, $r \ll d$ |
| 10.1 | Accuracy, CI 95% | $\text{acc} \pm 1.96\sqrt{\text{acc}(1-\text{acc})/n}$ |
| 10.1 | Recall, precision | đúng lớp $c$ / thật là $c$; đúng lớp $c$ / đoán là $c$ |
| 10.2 | McNemar exact | $b \sim \text{Bin}(b+c, 0.5)$, $p = 2P(X \le \min(b,c))$ |

---

## 15. Tài liệu tham khảo

**Đã có trong `docs/references.md`** (dùng trong file này): Vaswani 2017; Bengio 2003; Jurafsky & Martin ch. 3; Sennrich 2016; Brown 2020; Wei 2021 (FLAN); Ouyang 2022; Rafailov 2023; Holtzman 2020; Zhao 2021 (Calibrate Before Use); Min 2022; Lewis 2020; Hu 2021 (LoRA); Huang 2023; Lin 2022 (TruthfulQA); Liu 2023 (Lost in the Middle); Bolukbasi 2016; Zhao 2023 (Survey); Qwen2.5 Technical Report; UIT-VSFC (KSE 2018); UIT-ViQuAD 2.0 (arXiv:2203.11400); Dietterich 1998; SimpleQA (Wei 2024); Rajpurkar 2016 (SQuAD); tài liệu Hugging Face (tokenizer summary, perplexity, generation strategies, chat templates, LLM Course); scikit-learn `TfidfVectorizer`; statsmodels `mcnemar`; OpenAI *Introducing ChatGPT*.

**[đề xuất] thêm vào `references.md`** (cho các chỗ [Bổ sung\*]; mở link kiểm tra trước khi đưa lên slide):
| Tài liệu | Link | Dùng cho |
|---|---|---|
| Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, 2021 | https://arxiv.org/abs/2104.09864 | RoPE |
| Zhang & Sennrich, *Root Mean Square Layer Normalization*, NeurIPS 2019 | https://arxiv.org/abs/1910.07467 | RMSNorm |
| Shazeer, *GLU Variants Improve Transformer*, 2020 | https://arxiv.org/abs/2002.05202 | SwiGLU |
| Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, 2023 | https://arxiv.org/abs/2305.13245 | GQA |
| Xiao et al., *Efficient Streaming Language Models with Attention Sinks*, 2023 | https://arxiv.org/abs/2309.17453 | Attention sink (cell 2.3, chỉ nhắc như giả thuyết) |
| Keskar et al., *CTRL: A Conditional Transformer Language Model for Controllable Generation*, 2019 | https://arxiv.org/abs/1909.05858 | `repetition_penalty` (cell 0.4) — chưa mở kiểm |

### 15.1 Kết quả đối chiếu nguồn (2026-11-20)

Các khẳng định lý thuyết trong file đã được đối chiếu với **bản gốc** (trang arXiv HTML/abstract, tài liệu chính thức), trích nguyên văn:

| Khẳng định | Nguồn đã mở | Kết quả |
|---|---|---|
| Công thức attention, lý do chia $\sqrt{d_k}$ (phương sai $d_k$), mask $-\infty$, $d_{model}=512$, $h=8$ | Vaswani 2017 §3.1, §3.2.1 (chú thích 1), §3.2.2, §3.2.3 | Đúng |
| Nucleus: tập nhỏ nhất có tổng ≥ p, chia lại; giải mã cực đại hoá gây lặp, nhạt | Holtzman 2020 §3.1 (Eq. 2–3), tóm tắt, Bảng 1 | Đúng |
| Loss DPO, mục tiêu RLHF có KL, loss reward model Bradley-Terry | Rafailov 2023 Eq. 2, 3, 7 | Đúng nguyên văn |
| 3 bước SFT → RM → PPO, phạt KL theo từng token, 13k/33k/31k prompt, 1.3B được ưa hơn 175B | Ouyang 2022 §3.1, §3.2, §3.5, tóm tắt | Đúng |
| Ví dụ few-shot giúp qua tập nhãn, phân phối câu, định dạng; nhãn ngẫu nhiên giảm ít | Min 2022 tóm tắt | Đúng |
| Nhạy với chọn/thứ tự ví dụ; majority label, recency, common token bias | Zhao 2021 tóm tắt, §1 | Đúng |
| Factuality vs faithfulness hallucination | Huang 2023 §1 | Đúng |
| Attention sink: điểm attention lớn dồn vào token đầu dù không quan trọng về nghĩa | Xiao 2023 tóm tắt | Đúng |
| RoPE: $\theta_i = 10000^{-2(i-1)/d}$, phụ thuộc vị trí tương đối | Su 2021 §3.1, §3.2.2, Eq. 16 | Đúng |
| RMSNorm $a_i/\text{RMS}(a)\cdot g_i$, bỏ trừ trung bình | Zhang & Sennrich 2019 §4 Eq. 4 | Đúng |
| SwiGLU $(\text{Swish}_1(xW) \otimes xV)W_2$, 3 ma trận | Shazeer 2020 §2 Eq. 6 | Đúng |
| GQA: nhóm head Q dùng chung 1 head K, V; giảm KV cache | Ainslie 2023 §2.2 | Đúng |
| Model lớn nhất thường kém trung thực nhất | Lin 2022 tóm tắt | Đúng |
| Lost in the middle | Liu 2023 tóm tắt | Đúng, đã thêm chữ "thường" cho khớp nguyên văn |
| GPT-3 train 300 tỉ token; zero/one/few-shot không cập nhật trọng số | Brown 2020 Bảng 2.1, §2 | Đúng |
| LoRA $W_0 + BA$, $r \ll \min(d,k)$ | Hu 2021 §4.1 | Đúng (đã sửa từ dạng $d \times d$ sang $d \times k$) |
| SimpleQA: correct / incorrect / not attempted | Wei 2024 tóm tắt | Đúng |
| Qwen2.5: GQA, SwiGLU, RoPE, QKV bias, RMSNorm pre-norm; 1.5B: 28 layer, 12/2 head, tie embedding, 32K/8K; 18T token; BBPE 151,643 token + 22 control token; SFT > 1 triệu mẫu, 2 epoch, 32,768 token; DPO ~150k cặp, 1 epoch; GRPO, 6 tiêu chí, 8 câu trả lời/truy vấn; RoPE base 1,000,000 | Qwen2.5 report §1–4, Bảng 1 | Đúng |
| TF-IDF scikit-learn: $\ln\frac{1+n}{1+\text{df}}+1$, tf thô, L2, bỏ token 1 ký tự | Tài liệu scikit-learn, mục "Tf–idf term weighting" | Đúng |
| McNemar exact = $2 \cdot \text{BinomCDF}(\min(b,c); b+c, 0.5)$, chặn 1 | Tài liệu statsmodels `mcnemar` | Đúng |
| Perplexity = mũ của trung bình log-likelihood âm; thư viện tự dịch nhãn | HF *Perplexity*; docstring `GPT2LMHeadModel` ("labels are shifted inside the model") | Đúng |
| GPT-2 tokenizer 256 byte gốc, 50,257 token | PDF bài GPT-2 không đọc được bằng công cụ; dựa trên HF *Summary of the tokenizers* | Chưa đối chiếu bài gốc |
| Câu `is_impossible` của UIT-ViQuAD 2.0 viết cố ý gài bẫy | Chưa mở bài VLSP 2021 | [cần kiểm], đã viết nhẹ lại |
