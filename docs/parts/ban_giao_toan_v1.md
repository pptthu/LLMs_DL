# Bàn giao code — Toàn: phần 0 (Setup), 7 (Zero/few-shot), 10 (Thử nghiệm có kiểm soát)

- **Created**: 2026-11-19 (Toàn, AI hỗ trợ)
- Dành cho: Toàn (trình bày slide 2, 9, 12, 13) và cả nhóm (phần 0 ai cũng phải hiểu: cô hỏi "thư viện gì, chạy ở đâu, model lấy ở đâu").
- Đọc xong phải giải thích được **từng cell** của phần 0, 7, 10 trong `notebook/llm_presentation.ipynb`.
- Mọi số lấy từ lần chạy chính thức **`20261119_150356_normal`** (Kaggle, GPU T4, 7.4 phút). Kết quả: `outputs/results/`; hình: `outputs/figures/`; log: `outputs/logs/20261119_150356_normal/`.

---

## 0. Cách chạy notebook (ai cũng phải biết)

1. Vào Kaggle → **File → Import Notebook** → chọn `notebook/llm_presentation.ipynb`.
2. Settings: **Accelerator: GPU T4**, **Internet: On**. Gắn Dataset `llms-dl-data` (chứa `data/tasks_prompts.json`, `data/base_vs_instruct.json`).
3. Cell 0.1: `SMOKE = True` để chạy thử ít câu (khoảng 2 phút); `SMOKE = False` để chạy chính thức (khoảng 7 phút).
4. **Save Version → Save & Run All**. Xong thì tải notebook đã chạy và `outputs/` về repo.

Dữ liệu tải thẳng từ Hugging Face, không cần upload: UIT-VSFC (phần 7, 10), UIT-ViQuAD 2.0 (phần 8, 9).

Từ khoá chung:
| Từ | Nghĩa đơn giản |
|---|---|
| **`assert`** | Câu lệnh tự kiểm; điều kiện sai thì notebook dừng |
| **Smoke run** | Chạy thử ít mẫu để bắt lỗi code; số không dùng cho slide |
| **Normal run** | Chạy chính thức đủ dữ liệu; số trên slide lấy từ đây |
| **Greedy** | Mỗi bước chọn token xác suất cao nhất; chạy lại ra y hệt |
| **Seed** | Hạt giống ngẫu nhiên; cùng seed → cùng câu được chọn |

---

## 1. Phần 0 — Setup (slide 2: thư viện, môi trường)

### Cell 0.1 — Cấu hình chung
- **Mọi tham số đặt ở một chỗ**, các phần sau chỉ đọc lại: `SMOKE`, `SEED = 42`, tên 2 model, số câu từng phần, đường dẫn.
- Số câu theo chế độ:

| Tham số | Smoke | Normal | Dùng ở |
|---|---|---|---|
| `N_TEST` | 20 | `None` = toàn bộ 2,999 câu | phần 7, 10 |
| `N_VAL_PROMPT` | 20 | 200 | chọn mẫu prompt (7.4) |
| `N_FEWSHOT` | 4 | 4 | ví dụ few-shot |
| `BATCH_SIZE` | 16 | 16 | sinh theo lô |
| `N_QA`, `N_QA_IMPOSSIBLE` | 5, 1 | 15, 3 | phần 8, 9 |

- `RUN_ID` = giờ bắt đầu (giờ Việt Nam) + chế độ, ví dụ `20261119_150356_normal`. Smoke ghi vào `outputs/_smoke/` để không đè kết quả thật.
- `random.seed`, `np.random.seed`, `torch.manual_seed` cố định ngẫu nhiên.

### Cell 0.2 — Hàm ghi log
- `log(text, part=...)` ghi thêm một dòng (không ghi đè) vào `run.log` (cả lần chạy) và `partNN_<tên>.log` (từng phần).

### Cell 0.3 — Môi trường (số cho slide 2)
| Thành phần | Phiên bản |
|---|---|
| Python | 3.13.16 |
| torch | 2.11.0+cu130 (CUDA 13.0) |
| transformers | 5.20.2 |
| accelerate | 1.15.0 |
| pandas / pyarrow | 2.3.3 / 23.0.1 |
| scikit-learn | 1.6.1 (TF-IDF ở phần 8) |
| statsmodels | 0.15.0 (McNemar ở phần 10) |
| matplotlib | 3.10.0 (vẽ hình) |
| GPU | Tesla T4, 14.6 GB VRAM (Kaggle cấp 2 GPU, dùng GPU 0), float16 |

- Phiên bản đọc bằng `importlib.metadata`, **không ghi theo trí nhớ**. File: `outputs/results/env.json`.

### Cell 0.4 — Tải model chính
```python
tok = AutoTokenizer.from_pretrained(MODEL_INSTRUCT)
model = AutoModelForCausalLM.from_pretrained(MODEL_INSTRUCT, torch_dtype=DTYPE, device_map={"": 0})
model.eval()
```
- `from_pretrained(tên)` tải file từ `huggingface.co/Qwen/Qwen2.5-1.5B-Instruct` (khoảng 3.1 GB) về bộ nhớ đệm. Model công khai, không cần token.
- `torch_dtype=float16`: mỗi số 2 byte; T4 không hỗ trợ bf16 bằng phần cứng. `device_map={"": 0}`: cả model lên GPU 0. `eval()`: chế độ suy luận.
- In `generation_config` mặc định: `do_sample=True`, temperature 0.7, top_p 0.8, top_k 20, repetition_penalty 1.1 → bản Instruct **mặc định lấy mẫu ngẫu nhiên**, nên mọi lần gọi `generate()` trong notebook đều truyền tham số tường minh.
- Ghi revision (mã phiên bản trên Hub) `989aa79…` để người khác tải đúng bản.

### Cell 0.5 — Kiến trúc và số tham số
- Đọc từ `model.config`: 28 block, hidden 1,536, 12 head Q / 2 head KV, FFN 8,960, vocab 151,936, context tối đa 32,768.
- Đếm `sum(p.numel() for p in model.parameters())` = **1,543,714,304**, `assert` khớp model card.

### Cell 0.6 — Tokenizer GPT-2
- Chỉ tải tokenizer (vài MB), không tải model; dùng ở phần 3 để so số token.

### Cell 0.7 — UIT-VSFC
- Đọc 3 file Parquet (định dạng bảng nén) thẳng từ Hugging Face bằng `pd.read_parquet`.
- Nhãn gốc: 0 negative, 1 neutral, 2 positive. **Bỏ neutral** → bài toán 2 nhãn.
- Kiểm số dòng: train 11,426 → 10,968; validation 1,583 → 1,510; test 3,166 → **2,999** (1,590 positive, 1,409 negative).

### Cell 0.8 — Lưu `env.json`, `config.json`
- Lưu môi trường và toàn bộ tham số để chép lên slide và để người khác chạy lại.

---

## 2. Phần 7 — Zero-shot và few-shot (slide 9)

**Đề:** in-context learning, làm theo chỉ dẫn, xử lý tác vụ ngôn ngữ. Minh họa: **so sánh zero-shot và few-shot trên cùng tác vụ**.

- **Zero-shot**: prompt chỉ có chỉ dẫn. **Few-shot**: chỉ dẫn + vài ví dụ có nhãn. Model **không đổi trọng số** — chỉ đọc ví dụ trong prompt (in-context learning).
- Tác vụ: phân loại cảm xúc câu phản hồi sinh viên (UIT-VSFC) thành tích cực / tiêu cực.
- **Chỉ đổi một biến**: có ví dụ hay không. Cùng model, cùng mẫu prompt, cùng greedy, cùng tập câu test.

### Cell 7.1 — Tập test, validation, ví dụ few-shot
```python
def balanced_sample(df, n, seed):
    df = df.rename_axis("row_id").reset_index()
    if n is not None:
        df = pd.concat([g.sample(n // 2, random_state=seed) for _, g in df.groupby("sentiment")])
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)
```
- Giữ số thứ tự gốc làm `row_id`; `n = None` → lấy hết; có `n` thì mỗi nhãn n/2 câu; cuối cùng xáo thứ tự (cùng seed → cùng thứ tự).
- Test: **toàn bộ 2,999 câu**. Validation để chọn prompt: 200 câu (100/100).
- **4 ví dụ few-shot** lấy từ split **train**, bỏ câu trùng với test/validation, dài 5–20 từ, xếp xen kẽ:
  1. "dạy tốt có tâm ." → tích cực
  2. "oop khá phức tạp nên cần nhiều ví dụ ." → tiêu cực
  3. "giáo viên tích cực , năng nổ trong giảng dạy , phương pháp dạy có phần đổi mới ." → tích cực
  4. "giảng viên cần linh hoạt trong giảng dạy , chuẩn bị bài giảng phong phú ." → tiêu cực
- Lưu `vsfc_test_ids.csv`, `fewshot_examples.json` (sản phẩm nộp "bộ câu thử nghiệm"). `assert`: ví dụ không trùng câu test; cùng seed ra cùng câu.

### Cell 7.2 — Mẫu prompt, `build_prompt`, `parse_label`
- 3 mẫu: `vi_nhan_en` (chỉ dẫn Việt, nhãn positive/negative), `vi_nhan_vi` (chỉ dẫn Việt, nhãn tích cực/tiêu cực), `en_nhan_en` (chỉ dẫn Anh).
- `build_prompt(text, examples, tpl)`: chỉ dẫn + (ví dụ có nhãn) + câu cần phân loại. `examples = []` → zero-shot. Một hàm cho cả hai cách → chỉ khác phần ví dụ.
- Prompt few-shot thật (mẫu `vi_nhan_vi`):
```
Phân loại cảm xúc của câu phản hồi sinh viên sau. Chỉ trả lời: tích cực hoặc tiêu cực.

Câu: "dạy tốt có tâm ."
Cảm xúc: tích cực
...
Câu: "<câu cần phân loại>"
Cảm xúc:
```
- `parse_label(output)`: chữ thường, bỏ dấu câu; chỉ có "positive"/"tích cực" → positive; chỉ có "negative"/"tiêu cực" → negative; có cả hai hoặc không có → `None` (**không parse được, tính là sai**).

### Cell 7.3 — Phân loại theo lô
- Sinh 16 câu một lần cho nhanh. Prompt dài ngắn khác nhau nên phải **đệm cho bằng nhau**; đệm **bên trái** (`padding_side="left"`) để model sinh tiếp ngay sau token cuối của mỗi prompt; `attention_mask` báo bỏ qua phần đệm.
- Greedy, `repetition_penalty=1.0`, `max_new_tokens = 10`. (Smoke run với 5 token cắt mất nhãn "tiêu cực", mẫu `vi_nhan_vi` không parse được 60% → tăng lên 10.)
- `assert`: chạy 2 lần cùng lô ra cùng output.

### Cell 7.4 — Chọn mẫu prompt trên validation
- Chọn **trước** khi nhìn test, để kết quả test không bị "chọn theo đáp án".
- Output thật (200 câu validation):

| Mẫu | Zero-shot | Few-shot | Trung bình |
|---|---|---|---|
| `vi_nhan_en` | 0.840 | 0.905 | 0.8725 |
| **`vi_nhan_vi`** | 0.845 | 0.905 | **0.875** → chọn |
| `en_nhan_en` | 0.825 | 0.870 | 0.8475 |

### Cell 7.5 — Chạy trên tập test
- 2,999 câu × 2 cách, khoảng 5 phút trên T4. Kết quả: zero-shot **89.86%**, few-shot **92.30%**; không parse được 0. File: `icl_zero_shot.csv`, `icl_few_shot.csv` (mỗi câu: câu, nhãn, output, dự đoán, đúng/sai).

### Ý nói trên slide 9
- Ba khả năng: in-context learning (học từ ví dụ trong prompt, không đổi trọng số); làm theo chỉ dẫn (nhờ SFT, mục 5); một model làm nhiều tác vụ (phân loại cũng chỉ là sinh ra chữ "tích cực").
- Hai prompt cạnh nhau + output thật của vài câu; con số chi tiết ở slide 12.

---

## 3. Phần 10 — Thử nghiệm có kiểm soát (slide 12, 13)

**Đề:** thử nghiệm có kiểm soát, tổng hợp kết quả và nhận xét. Minh họa: bảng kết quả, ví dụ đầu ra và phân tích lỗi. Yêu cầu thêm 4: slide phải có **thiết lập, kết quả, phân tích**.

**Câu hỏi thí nghiệm:** thêm 4 ví dụ vào prompt có làm accuracy khác zero-shot **một cách có ý nghĩa thống kê** không? H0: hai cách có tỉ lệ sai như nhau.

### Cell 10.1 — Accuracy và confusion matrix
- `accuracy = số câu đúng / tổng số câu` (câu không parse được tính sai). Khoảng tin cậy 95% ≈ ±1.96·√(p(1−p)/n).
- `assert` trên ví dụ nhỏ tính tay (4 câu, đúng 2 → accuracy 0.5).

| | Zero-shot | Few-shot |
|---|---|---|
| Accuracy | **89.86%** (±1.08) | **92.30%** (±0.95) |
| Negative bị đoán positive | 248 | 145 |
| Positive bị đoán negative | 56 | 86 |
| Không parse được | 0 | 0 |

- Hình: `outputs/figures/confusion_matrix_icl.png`. Few-shot sửa chủ yếu lỗi "negative → positive", đổi lại tăng nhẹ lỗi ngược.

### Cell 10.2 — McNemar
```python
[[a, b], [c, d]] = mcnemar_counts(icl_zero["correct"], icl_few["correct"])
mcnemar(table, exact=True)
```
- Hai cách chạy trên **cùng câu**, nên chỉ câu hai cách cho kết quả **khác nhau** mới nói lên cách nào tốt hơn:

| | Few-shot đúng | Few-shot sai |
|---|---|---|
| **Zero-shot đúng** | a = 2,648 | **b = 47** |
| **Zero-shot sai** | **c = 120** | d = 184 |

- Nếu H0 đúng thì b ≈ c. Ở đây c (120) lớn hơn b (47) nhiều; `exact=True` dùng phân phối nhị thức trên b + c = 167 câu → **p = 1.5e-08** < 0.05 → **few-shot tốt hơn, khác biệt có ý nghĩa thống kê**.
- Vì sao McNemar: Dietterich (1998) — với thuật toán chỉ chạy được một lần trên một tập test, McNemar có sai lầm loại I chấp nhận được. Chỉ 1 phép so sánh nên không cần hiệu chỉnh Bonferroni.
- `assert`: b = c thì p = 1; a + b + c + d = 2,999. File: `icl_metrics.json`.

### Cell 10.3 — Phân tích lỗi
- Lấy ngẫu nhiên **30 câu few-shot sai** (seed 42). Nhóm lỗi **ghi sẵn trước khi đọc output**: (a) không parse được, (b) nhãn gốc có vẻ sai, (c) vừa khen vừa chê, (d) ý ngầm (góp ý, phàn nàn gián tiếp), (e) model sai rõ ràng.

| Nhóm | Số câu | Ví dụ thật (nhãn gốc → model) |
|---|---|---|
| (d) ý ngầm: góp ý "nên/cần…" | **20** | "giảm thời lượng lý thuyết , tăng thời lượng thực hành ." (negative → tích cực) |
| (c) vừa khen vừa chê | 4 | "thầy dạy hơi khó hiểu nhưng rất kỹ ." (positive → tiêu cực) |
| (e) model sai rõ ràng | 4 | "thầy rất dễ tính ." (positive → tiêu cực) |
| (b) nhãn gốc có vẻ sai | 2 | "môn học cần thiết và thiết thực với ngành học ." (negative → tích cực) |
| (a) không parse được | 0 | — |

- Đếm trên **toàn bộ** câu sai: câu negative bị đoán positive là câu góp ý (có "nên", "cần", "mong", "thêm", "hơn") chiếm **189/248 (76%)** ở zero-shot và **117/145 (81%)** ở few-shot.
- **Giải thích:** UIT-VSFC gán negative cho câu góp ý (góp ý ngầm chỉ ra điều chưa tốt), còn model đọc giọng lịch sự nên đoán tích cực. Hai ví dụ negative trong prompt few-shot đều là góp ý → model thấy được quy ước này → lỗi negative → positive giảm từ 248 xuống 145.
- File: `error_analysis_sample.csv`; viết đầy đủ ở `docs/experiments/error_analysis_v1.md`.

### Cell 10.4 — Bảng tổng hợp
- Đọc **từ file kết quả** của lần chạy (không chép số tay) → `summary_table.csv`:

| Mục | Kết quả |
|---|---|
| 3 | Câu Việt: Qwen 8 / GPT-2 22 token; câu Anh: 7 / 7 |
| 4 | Loss câu đúng 2.985; câu xáo trộn 6.701; câu Anh 3.129 |
| 5 | Base vs Instruct đạt: 0/4 vs 4/4 |
| 6 | Output khác nhau / 3 lần: greedy 1, mọi cấu hình lấy mẫu 3 |
| 7, 10 | 89.86% vs 92.30% (2,999 câu); McNemar p = 1.5e-08 |
| 8 | Câu có đáp án: không RAG 0 đúng / 1 sai / 14 từ chối, có RAG 10 / 3 / 2; câu không có đáp án: RAG trả lời bừa 2/3 |
| 9 | Hỏi không tài liệu: sai 12/15 mỗi cách hỏi; thực thể không tồn tại: bịa 5/6 lần hỏi |

### Slide 12 (thiết lập + kết quả) và 13 (phân tích)
- **Thiết lập:** Qwen2.5-1.5B-Instruct, float16, Kaggle T4; UIT-VSFC test 2,999 câu; mẫu `vi_nhan_vi` chọn trên validation; greedy, `max_new_tokens` 10; 4 ví dụ từ train; seed 42.
- **Kết quả:** bảng accuracy + McNemar ở trên.
- **Phân tích:** confusion matrix, nhóm lỗi, câu góp ý.
- **Giới hạn phải nói:** 1 model 1.5B; 1 bộ dữ liệu, 1 tác vụ; 1 bộ ví dụ few-shot (đổi ví dụ hoặc thứ tự có thể cho kết quả khác — Zhao et al., 2021); mỗi cấu hình chạy 1 lần; nhóm lỗi do 1 người gán. Không suy rộng sang "LLM nói chung".

---

## 4. Câu cô có thể hỏi
| Câu hỏi | Ý trả lời |
|---|---|
| Sao cùng seed luôn ra cùng câu? | Bộ sinh ngẫu nhiên khởi tạo bằng seed cố định |
| Làm sao chắc ví dụ few-shot không trùng test? | Lấy từ train, loại câu trùng nội dung với test/validation, có `assert` |
| Zero-shot khác few-shot ở đâu trong code? | Cùng hàm `build_prompt`; `examples = []` là zero-shot |
| Sao chắc là greedy? | `do_sample=False`; chạy 2 lần cùng lô ra y hệt (`assert` ở 7.3) |
| Output "Cảm xúc: tích cực" thì sao? | Chữ thường, bỏ dấu câu, tìm từ nhãn → positive |
| Câu không parse được tính thế nào? | Tính là sai; báo riêng tỉ lệ (ở đây 0) |
| b, c là gì? | b: chỉ zero-shot đúng; c: chỉ few-shot đúng |
| Vì sao `exact=True`? | Dùng nhị thức chính xác, không cần xấp xỉ khi b + c nhỏ |
| Sao chọn prompt trên validation? | Để không chọn prompt theo kết quả test |
| Sao đệm bên trái? | Để model sinh tiếp ngay sau token cuối của prompt |
| Sao không dùng 500 câu như kế hoạch cũ? | Sinh theo lô đủ nhanh (khoảng 5 phút) nên chạy hết 2,999 câu, khỏi giải thích cách lấy mẫu; sai số ±1 điểm thay vì ±3.5 |
