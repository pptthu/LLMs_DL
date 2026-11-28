# Mục 7 — Khả năng của LLM: in-context learning, zero-shot vs few-shot

- **Created**: 2026-11-28 (Toàn, AI hỗ trợ, Claude Code). Trạng thái **[đề xuất]**, chờ Toàn duyệt.
- **Nội dung:** phân loại cảm xúc UIT-VSFC, chọn prompt trên validation, zero-shot và few-shot trên 2,999 câu test (cell 7.1–7.5).
- **Số liệu:** run chính thức `20261128_041850_normal` (Kaggle, GPU T4 x2, float16, seed 42, 22.0 phút, 188 cell; G-Eval chấm 20 lần lấy trung bình). Phần 1–9 và 10.1–10.3b trùng từng file kết quả với run `20261127_234527_normal`, `20261125_214650_normal`, `20261121_010447_normal` (chỉ khác cột thời gian, bộ nhớ); riêng 10.3c, 10.3d đổi theo giám khảo (xem mục 10). **VRAM 6.6, 9.5 lấy run `20261127_234527_normal`**: run chính thức đo lệch +2.97 GiB vì model của lần chạy dừng giữa chừng còn trên GPU. Bảng ở `outputs/results/`, hình ở `outputs/figures/`.
- **Ký hiệu nguồn:** **[Slide ChX]** slide môn học; **[Bổ sung]** nguồn có trong `docs/references.md`; **[Tự tính]** tính lại từ số trong notebook; **[Suy luận]** giải thích của nhóm, chưa đo.
- **Code** chép từ `notebook/llm_presentation.ipynb`; dòng ghi log, gọi `show_input()` và vài dòng in phụ được lược (`# ...`). Mở notebook, tìm số cell (ví dụ "# 4.2") để xem nguyên cell.
- **Bộ đánh giá** (thẻ ở notebook cell 10.5, `figures/suite_*.png`): E6.

---

> **Cô yêu cầu → notebook đáp ứng**
> - Nội dung cần trình bày: *In-context learning, làm theo chỉ dẫn, xử lý các tác vụ ngôn ngữ*.
> - Yêu cầu minh họa: *So sánh zero-shot và few-shot trên cùng tác vụ*.
> - Đáp ứng: Cell 7.1–7.5: cùng 2,999 câu UIT-VSFC, cùng mẫu prompt (chọn trên validation), chỉ khác 4 ví dụ.
> - Góp ý 11/11 và yêu cầu thêm 3: demo trên model thật, output thật, có phân tích.

## Note lý thuyết để hiểu (dùng cho cả phần 7)

- **In-context learning (ICL)** [Bổ sung: Brown et al. 2020]: model làm tác vụ nhờ chỉ dẫn và/hoặc ví dụ **trong prompt**; **trọng số không đổi** (không có gradient).
  - **Zero-shot:** chỉ dẫn + câu cần làm. **Few-shot:** chỉ dẫn + $k$ ví dụ có nhãn + câu cần làm.
  - Về xác suất: few-shot đổi điều kiện thành $P(\text{nhãn} \mid \text{chỉ dẫn}, \text{ví dụ}_{1..k}, \text{câu})$.
- **Ví dụ giúp bằng cách nào** [Bổ sung: Min et al. 2022]: chủ yếu cho thấy **định dạng**, **tập nhãn**, **kiểu câu đầu vào**; nhãn ví dụ bị đổi ngẫu nhiên chỉ làm giảm ít.
- **Rủi ro** [Bổ sung: Zhao et al. 2021]: nhạy với chọn ví dụ, thứ tự ví dụ (thiên về nhãn ví dụ cuối), tỉ lệ nhãn.
- **Làm theo chỉ dẫn** [Bổ sung: Wei 2021 (FLAN); Ouyang 2022]: có được nhờ SFT (mục 5).
- **Thiết kế "chỉ đổi một biến":** giữ cùng model, mẫu prompt, greedy, `max_new_tokens = 10`, cùng 2,999 câu cùng thứ tự; **chỉ đổi có/không 4 ví dụ** → chênh lệch accuracy là tác dụng của ví dụ.
- **Vì sao greedy:** kết quả lặp lại được và hai cách so được **trên từng câu** (cần cho McNemar, cell 10.2).

## Cell 7.1 — Tập test, validation, ví dụ few-shot

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
- **Xen kẽ** (+, −, +, −): tránh mọi ví dụ cùng nhãn dồn về cuối (thiên lệch theo ví dụ cuối, *recency bias*, Zhao 2021). **Lưu ý:** ví dụ cuối vẫn là negative, nên recency bias (nếu có) đẩy model về phía negative. Đây là một cách giải thích khác cho kết quả few-shot ở cell 10.1; notebook không có thí nghiệm đảo thứ tự để tách hai cách giải thích.

**Output thật:**
```text
test: 2,999 câu {'positive': 1590, 'negative': 1409} | validation chọn prompt: 200 câu |
few-shot: [('positive', 'dạy tốt có tâm .'),
           ('negative', 'oop khá phức tạp nên cần nhiều ví dụ .'),
           ('positive', 'giáo viên tích cực , năng nổ trong giảng dạy , phương pháp dạy có phần đổi mới .'),
           ('negative', 'giảng viên cần linh hoạt trong giảng dạy , chuẩn bị bài giảng phong phú .')]
```
**Giải thích:** cả **2 ví dụ negative đều là câu góp ý** ("…nên cần nhiều ví dụ", "…cần linh hoạt…"). Đây là chìa khoá để hiểu kết quả few-shot ở cell 7.5 và 10.1. Bộ ví dụ do seed 42 chọn ngẫu nhiên, không chọn tay.

## Cell 7.2 — Dựng prompt, đọc nhãn

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
**Giải thích:** prompt kết thúc ở "Cảm xúc:" bỏ trống → điền nhãn vào chỗ trống là phần tiếp theo **hợp lý nhất** (nguyên lý next-token, phần 4). Không bảo đảm token đầu tiên là nhãn: prompt được bọc trong chat template, model có thể viết thêm chữ trước nhãn (cell 7.5: "Cảm xúc: tiêu cực"). Vì vậy cho `max_new_tokens = 10` và đọc nhãn bằng `parse_label`.

## Cell 7.3 — Phân loại theo lô

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

## Cell 7.4 — Chọn mẫu prompt trên validation

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
- `vi_nhan_vi` hơn `vi_nhan_en` 0.0025 = **1 câu đúng** trên 400 lượt (200 câu × 2 cách) → thực tế ngang nhau; `en_nhan_en` kém rõ hơn [Suy luận: câu cần phân loại là tiếng Việt, chỉ dẫn cùng ngôn ngữ giúp nhất quán hơn].
- Cả 3 mẫu: few-shot > zero-shot trên validation → dấu hiệu trước khi chạy test.
- Few-shot chậm gần gấp đôi: prompt dài thêm 4 ví dụ.

## Cell 7.5 — Chạy trên test

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
- **Few-shot không đổi trọng số** (không có gradient). [Suy luận] Nó cho model thấy **quy ước gán nhãn** của UIT-VSFC: góp ý = negative (2 ví dụ negative đều là góp ý). Cách giải thích khác: recency bias do ví dụ cuối là negative (cell 7.1). Hai giả thuyết chưa tách được.
- **Ví dụ 2 và 3 (nhãn negative) bị đoán tích cực ở cả hai cách:** đều là **góp ý** ("tăng tiết…", "cần được phân bố hợp lý…"). Câu mang giọng xây dựng, không có từ tiêu cực → model đọc là tích cực. Đây là loại lỗi chính (có số liệu kiểm ở cell 10.3).
- **Output few-shot dạng "Cảm xúc: tiêu cực":** model chép **định dạng** "Cảm xúc: <nhãn>" của các ví dụ (khớp Min et al. 2022). `parse_label` vẫn đọc được → 0 câu không parse được.

## Câu hỏi tự kiểm

| Câu hỏi | Ý trả lời |
|---|---|
| ICL có phải là train không? | Không; trọng số không đổi, ví dụ chỉ nằm trong prompt |
| Sao chọn prompt trên validation? | Để không chọn prompt theo kết quả test |
| Sao chắc ví dụ không trùng test? | Lấy từ train, loại câu trùng nội dung, có `assert` |
| Sao đệm bên trái? | Để model sinh tiếp ngay sau token cuối của prompt |
| Đổi ví dụ có ra khác không? | Có thể [10]; đây là giới hạn đã ghi (1 bộ ví dụ) |

---
