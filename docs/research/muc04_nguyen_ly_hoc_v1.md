# Mục 4 — Nguyên lý học: next-token prediction, cross-entropy

- **Created**: 2026-11-28 (Toàn, AI hỗ trợ, Claude Code). Trạng thái **[đề xuất]**, chờ Toàn duyệt.
- **Nội dung:** cặp ngữ cảnh → token mục tiêu, cross-entropy, perplexity, top-k accuracy (cell 4.1–4.6).
- **Số liệu:** run chính thức `20261128_041850_normal` (Kaggle, GPU T4 x2, float16, seed 42, 22.0 phút, 188 cell; G-Eval chấm 20 lần lấy trung bình). Phần 1–9 và 10.1–10.3b trùng từng file kết quả với run `20261127_234527_normal`, `20261125_214650_normal`, `20261121_010447_normal` (chỉ khác cột thời gian, bộ nhớ); riêng 10.3c, 10.3d đổi theo giám khảo (xem mục 10). **VRAM 6.6, 9.5 lấy run `20261127_234527_normal`**: run chính thức đo lệch +2.97 GiB vì model của lần chạy dừng giữa chừng còn trên GPU. Bảng ở `outputs/results/`, hình ở `outputs/figures/`.
- **Ký hiệu nguồn:** **[Slide ChX]** slide môn học; **[Bổ sung]** nguồn có trong `docs/references.md`; **[Tự tính]** tính lại từ số trong notebook; **[Suy luận]** giải thích của nhóm, chưa đo.
- **Code** chép từ `notebook/llm_presentation.ipynb`; dòng ghi log, gọi `show_input()` và vài dòng in phụ được lược (`# ...`). Mở notebook, tìm số cell (ví dụ "# 4.2") để xem nguyên cell.
- **Bộ đánh giá** (thẻ ở notebook cell 10.5, `figures/suite_*.png`): E3.

---

> **Cô yêu cầu → notebook đáp ứng**
> - Nội dung cần trình bày: *Next-token prediction, dữ liệu huấn luyện và cross-entropy loss*.
> - Yêu cầu minh họa: *Tạo các cặp ngữ cảnh → token mục tiêu từ một câu*.
> - Đáp ứng: Cell 4.1: 7 cặp + ma trận cặp; 4.2 loss từng token + biểu đồ; 4.3 khớp loss thư viện; 4.4 perplexity; 4.5 câu xáo trộn.
> - Góp ý 11/11 và yêu cầu thêm 3: demo trên model thật, output thật, có phân tích.

## Note lý thuyết để hiểu (dùng cho cả phần 4)

- **Next-token prediction:** cho $x_1..x_i$, đoán $x_{i+1}$. Là **bài toán phân loại** với 151,936 lớp [Slide Ch2: softmax + cross-entropy].
- **Tự giám sát:** nhãn là chính token kế tiếp trong văn bản, không cần người gán → dùng được hàng nghìn tỉ token (GPT-3 ~300 tỉ token [Bổ sung: Brown 2020]; Qwen2.5 18 nghìn tỉ token [Bổ sung: Qwen2.5 report]).
- **Cross-entropy** [Slide Ch2]: nhãn one-hot nên rút gọn thành $-\ln$ xác suất của token đúng:

$$\mathcal{L} = -\frac{1}{n-1}\sum_{i=1}^{n-1} \ln p_\theta(x_{i+1} \mid x_1, \dots, x_i)$$

| $p$ token đúng | 1 | 0.5 | 0.1 | 0.001 |
|---|---|---|---|---|
| $-\ln p$ | 0 | 0.69 | 2.30 | 6.91 |

- Huấn luyện = chỉnh 1.5 tỉ tham số $\theta$ bằng gradient descent để **giảm** $\mathcal{L}$. Theo quy tắc dây chuyền, đó cũng là tăng xác suất của cả văn bản huấn luyện (maximum likelihood). **Notebook không train**, chỉ tính $\mathcal{L}$ của model đã train.
- **Perplexity** [Bổ sung: HF *Perplexity*]: $\text{PPL} = e^{\mathcal{L}}$, đọc là "model phân vân như chọn đều giữa PPL token". Mốc đoán bừa đều: $\mathcal{L} = \ln V$, $\text{PPL} = V$.

## Cell 4.1 — Cặp ngữ cảnh → token mục tiêu (minh họa của đề)

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

**Hình ở cell 4.1** (`figures/context_target_pairs.png`): hàng = cặp, cột = vị trí token; ô xanh = ngữ cảnh đưa vào model, ô cam = token phải đoán. Hình là tam giác giống causal mask ở cell 2.2: cặp thứ $i$ dùng đúng những token mà hàng $i$ của mask cho phép nhìn. Vì vậy **một lần chạy model** có causal mask tính được cả 7 cặp (cell 4.2).

## Cell 4.2 — Loss từng token

> **Lý thuyết ở cell này:** nhờ causal mask (cell 2.2), `logits[i]` chỉ dựa vào token $0..i$ nên chính là dự đoán cho token $i+1$. **Một lần chạy model cho cả 7 dự đoán**: so `logits[:-1]` với `ids[1:]` ("dịch nhãn sang trái 1 vị trí"; gọi là *teacher forcing*: ngữ cảnh luôn là token thật, không phải token model tự sinh). RNN cũng ra 7 dự đoán trong một lượt, nhưng phải tính **lần lượt** từng vị trí; Transformer tính **cả 7 vị trí song song** trên GPU. Đó mới là lý do huấn luyện Transformer nhanh hơn RNN.

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

**Giải thích từng dòng output** (cột "Vì sao" là [Suy luận] dựa trên hiểu biết tiếng Việt; số $p$ là output thật):
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

**Biểu đồ ở cell 4.2** (`figures/per_token_loss.png`): mỗi cột là loss của một cặp, ghi kèm $P$; đường ngang là trung bình 2.985. Hai cột cao nhất ("ôm" 8.19, " trời" 6.34) là chỗ ngữ cảnh còn rất nhiều khả năng, không phải model kém; cột thấp nhất ("ay" 0.004) là chỗ gần như chắc chắn.

## Cell 4.3 — So với loss của `transformers`

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

## Cell 4.4 — Perplexity

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

## Cell 4.5 — Câu xáo trộn, câu Anh

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


## Cell 4.6 — Top-1 / top-5 accuracy của dự đoán token tiếp theo (thêm 24/11)

> **Lý thuyết:** cross-entropy gộp hai thông tin: đoán trúng hay không, và chắc đến đâu. Top-k accuracy tách riêng phần "đoán trúng": tỉ lệ vị trí mà token thật nằm trong k token xác suất cao nhất. Luôn có top-1 ≤ top-5.

**Cách làm:** cùng một lượt forward như 4.2, lấy `topk(logits, 5)` ở mỗi vị trí, so với token thật.

**Kết quả** (`next_token_accuracy.csv`):

| Câu | Số vị trí | Top-1 | Top-5 | Loss |
|---|---|---|---|---|
| "Hôm nay trời rất đẹp." | 7 | 0.43 (3/7) | 0.71 (5/7) | 2.985 |
| Câu xáo trộn | 8 | 0.12 (1/8) | 0.25 (2/8) | 6.701 |
| "The weather is very nice today." | 6 | 0.33 (2/6) | 0.67 (4/6) | 3.129 |

**Nhận xét:** câu đúng được đoán trúng nhiều hơn hẳn câu xáo trộn, cùng chiều với loss. Top-k dễ nói hơn ("3/7 vị trí đoán trúng ngay lần đầu") nhưng bỏ mất thông tin "chắc đến đâu", nên chỉ đi kèm cross-entropy chứ không thay.

## Câu hỏi tự kiểm

| Câu hỏi | Ý trả lời |
|---|---|
| Sao 8 token chỉ có 7 loss? | Token đầu không có ngữ cảnh để đoán |
| Sao chạy một lần mà ra loss mọi vị trí? | Causal mask: vị trí $t$ chỉ nhìn token $1..t$, nên logits ở $t$ là dự đoán cho $t+1$ |
| Perplexity 19.8 nghĩa là gì? | Model phân vân như chọn đều giữa khoảng 20 token |
| So loss câu Việt với câu Anh được không? | Không nên: khác token, khác số token; chỉ so câu đúng với câu xáo trộn cùng từ |

---
