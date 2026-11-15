# Mục 4 — Nguyên lý học: next-token prediction, dữ liệu huấn luyện, cross-entropy loss (nội dung chi tiết, v1)

- Người phụ trách: Phúc | Slide 8–9 | Thời lượng: khoảng 2 phút | Notebook: phần 4 (`src/learning.py`)
- Đề của cô: "Next-token prediction, dữ liệu huấn luyện và cross-entropy loss". Minh hoạ: tạo các cặp ngữ cảnh → token mục tiêu từ một câu.
- Ký hiệu nguồn:
  - **[Slide ChX sN]**: slide môn.
  - **[Bổ sung]**: nguồn ngoài, đã kiểm 2026-11-15 (§9).
  - **[Tự tính]**: đã chạy lại bằng Python / PyTorch.
- **Phạm vi:** nhóm không huấn luyện model (góp ý của cô ngày 11/11). Mục này minh hoạ nguyên lý học bằng cách **tính loss** của model đã huấn luyện sẵn.

---

## 0. Mục này cần làm người nghe hiểu được gì
1. LLM học bằng **một** nhiệm vụ duy nhất: đoán token tiếp theo.
2. Nhãn có sẵn trong chính văn bản, nên dùng được dữ liệu không nhãn khổng lồ (tự giám sát).
3. Một câu cho ra nhiều cặp (ngữ cảnh → token mục tiêu), và nhờ causal mask cả câu được học **song song**.
4. Loss là **cross-entropy**, giống phân loại nhiều lớp ở Ch2; số lớp = số token trong từ điển.
5. Perplexity là cách đọc loss dễ hiểu hơn.

---

## 1. Next-token prediction

### 1.1 Nhiệm vụ
Cho các token đứng trước, model đưa ra **phân phối xác suất** cho token kế tiếp trên toàn bộ từ điển (151,936 token với Qwen), rồi được chấm điểm dựa trên xác suất nó gán cho **token đúng**.

Đây chính là công thức chain rule ở mục 1: P(câu) = ∏ P(token_t | các token trước). Học tốt từng thừa số là học được cả câu.

### 1.2 Vì sao gọi là tự giám sát (self-supervised)
- Không cần người gán nhãn: token tiếp theo trong văn bản **chính là nhãn**.
- Nhờ vậy dùng được văn bản thô với quy mô cực lớn (§3).
- Liên hệ Hugging Face LLM Course: decoder-only được pretrain bằng cách đoán token tiếp theo trên kho văn bản lớn.

### 1.3 Đây là bài toán phân loại  [Slide Ch2 s9, s11–13]
| Phân loại ở Ch2 | Next-token prediction |
|---|---|
| Lớp output Softmax cho phân phối xác suất trên các lớp [Ch2 s9, s11] | Softmax trên 151,936 token |
| Số lớp: vài đến vài chục | Số lớp = kích thước từ điển |
| Nhãn: lớp đúng của mẫu | Nhãn: token đúng đứng tiếp theo |
| Loss: Sparse Categorical Cross-Entropy khi nhãn là số nguyên [Ch2 s13] | Cùng loss đó, mỗi vị trí một lần |

Câu nói khi trình bày: **"Mỗi bước, LLM giải một bài phân loại với khoảng 150 nghìn lớp."**

---

## 2. Tạo cặp ngữ cảnh → token mục tiêu (minh hoạ của đề)

### 2.1 Ví dụ (giả sử mỗi từ là một token để dễ đọc)
Câu: "Hôm nay trời rất đẹp."

| # | Ngữ cảnh (đầu vào) | Token mục tiêu (nhãn) |
|---|---|---|
| 1 | Hôm | nay |
| 2 | Hôm nay | trời |
| 3 | Hôm nay trời | rất |
| 4 | Hôm nay trời rất | đẹp |
| 5 | Hôm nay trời rất đẹp | . |

Câu có n token cho **n − 1 cặp**. Ở notebook, dùng **token thật** của Qwen (mục 3), nên số cặp có thể khác 5.

### 2.2 Cài đặt: dịch nhãn sang trái một vị trí
```
input_ids :  [t1, t2, t3, t4, t5, t6]
đầu vào   :  [t1, t2, t3, t4, t5]        ← bỏ token cuối
nhãn      :      [t2, t3, t4, t5, t6]    ← bỏ token đầu
```
- Nhờ **causal mask** (mục 2), vị trí i chỉ nhìn t1…ti. Một lần chạy model cho **cùng lúc** n − 1 dự đoán; không cần chạy n − 1 lần.
- `transformers`: truyền `labels=input_ids` thì model **tự dịch nhãn** sang trái 1 vị trí và trả về loss trung bình trên các token (Hugging Face, *Perplexity of fixed-length models*).
- Vị trí có nhãn `-100` bị bỏ qua khi tính loss. Dùng để không tính loss trên phần prompt; liên hệ SFT ở mục 5.

### 2.3 Hàm trong plan
```python
def make_context_target_pairs(token_ids: list[int]) -> list[tuple[list[int], int]]:
    """[a,b,c,d] -> [([a],b), ([a,b],c), ([a,b,c],d)]"""
    return [(token_ids[:i], token_ids[i]) for i in range(1, len(token_ids))]
```
Test: `make_context_target_pairs([1, 2, 3, 4])` → 3 cặp `([1],2), ([1,2],3), ([1,2,3],4)`.

---

## 3. Dữ liệu huấn luyện

### 3.1 Ví dụ GPT-3  [Bổ sung: Brown et al. 2020, Bảng 2.2]
| Nguồn | Số token | Tỉ trọng khi huấn luyện |
|---|---|---|
| Common Crawl (đã lọc) | 410 tỉ | 60% |
| WebText2 | 19 tỉ | 22% |
| Books1 | 12 tỉ | 8% |
| Books2 | 55 tỉ | 8% |
| Wikipedia | 3 tỉ | 3% |

- Tổng số token đã huấn luyện: 300 tỉ.
- Điểm đáng chú ý: tỉ trọng **không** tỉ lệ với kích thước. Wikipedia nhỏ nhưng được "xem" khoảng 3.4 lần, còn Common Crawl chỉ khoảng 0.44 lần. Nguồn chất lượng cao được lặp nhiều hơn.

### 3.2 Qwen2.5 (model nhóm dùng)  [Bổ sung: Qwen2.5 Technical Report]
- Dữ liệu pretraining tăng từ 7 nghìn tỉ token (Qwen2) lên **18 nghìn tỉ token**.
- Cân lại tỉ lệ:
  - giảm mẫu các mảng web bị đại diện quá mức (thương mại điện tử, mạng xã hội, giải trí);
  - tăng mẫu nội dung công nghệ, khoa học, học thuật;
  - thêm dữ liệu toán và code;
  - dữ liệu tổng hợp được lọc bằng reward model.
- Pretraining chia nhiều giai đoạn để đổi tỉ lệ dữ liệu giữa các giai đoạn.

### 3.3 Ý cần nói
- **Chất lượng và tỉ lệ dữ liệu quan trọng ngang số lượng** (cả GPT-3 và Qwen2.5 đều lọc và cân lại).
- Model học **mọi thứ có trong dữ liệu**, kể cả lỗi và định kiến → nối sang mục 9 (thiên lệch, hallucination).
- Liên hệ Ch3–4 về dữ liệu: dữ liệu quyết định model; ở đây là ở quy mô nghìn tỉ token.

---

## 4. Cross-entropy loss

### 4.1 Công thức  [Slide Ch2 s7 + Bổ sung]
Slide Ch2 s7 có công thức cross-entropy dạng −Σ y·log(ŷ), cùng đồ thị loss theo giá trị dự đoán khi nhãn = 1.

**Lưu ý ký hiệu:** slide đặt y là dự đoán và ŷ là nhãn, **ngược** với cách viết thông thường. Khi trình bày, nói rõ đang dùng ký hiệu nào.

Với nhãn one-hot (chỉ token đúng có y = 1), tổng rút gọn thành **một số hạng**:

```
loss tại vị trí t = − log P_model(token đúng tại t | các token trước)
loss của câu      = trung bình các vị trí:  L = −(1/(n−1)) Σ log P(x_t | x_<t)
```

### 4.2 Trực giác bằng số  [Tự tính]
| Xác suất model gán cho token đúng | Loss = −ln p |
|---|---|
| 0.9 | 0.105 |
| 0.5 | 0.693 |
| 0.1 | 2.303 |
| 0.01 | 4.605 |

- Đoán **đúng và chắc** → loss gần 0. Đoán sai hoặc phân vân → loss lớn.
- **Mốc so sánh** [Tự tính]: model đoán bừa đều trên 151,936 token có loss = ln(151,936) ≈ **11.93**. Loss thật của Qwen trên câu ví dụ phải thấp hơn nhiều; đây là bằng chứng model đã học.

### 4.3 Perplexity  [Bổ sung: HF *Perplexity*]
- Perplexity là **e mũ của trung bình negative log-likelihood**, tương đương e mũ của cross-entropy:
  `PPL = exp(L)`.
- Cách đọc: "model phân vân trung bình như đang chọn đều giữa khoảng PPL token". Đoán bừa đều thì PPL = 151,936; PPL = 4 nghĩa là phân vân cỡ 4 lựa chọn.
- Perplexity phụ thuộc tokenizer, nên **không so trực tiếp** perplexity giữa hai model khác tokenizer.

### 4.4 Kiểm chứng cách tính loss  [Tự tính, PyTorch]
Với logits ngẫu nhiên, từ điển 5 token, câu 4 token:
- tự dịch nhãn rồi gọi `F.cross_entropy` → **1.3679**;
- tự tính −log p của từng vị trí: 1.7914, 0.4245, 1.8878, trung bình **1.3679** → khớp.

Ở notebook làm đúng như vậy với Qwen: so loss tự tính với `outputs.loss` của model. Trùng nhau thì vừa chứng minh hiểu đúng công thức, vừa chứng minh `transformers` tự dịch nhãn.

---

## 5. Minh hoạ trong notebook (phần 4)  [đề xuất — Phúc chốt]

| Cell | Làm gì | Output cho slide |
|---|---|---|
| 4.1 | Tokenize câu ví dụ, `make_context_target_pairs` → in bảng (ngữ cảnh dạng chữ → token mục tiêu) | Bảng cặp, slide 8 |
| 4.2 | `per_token_loss(model, tok, câu)`: mỗi vị trí in token mục tiêu, P(token đúng), −log p | Bảng loss từng token, slide 9 |
| 4.3 | `sequence_loss(...)` (= trung bình), so với `model(ids, labels=ids).loss` | Hai số phải trùng |
| 4.4 | Perplexity = exp(loss); so với mốc đoán bừa 11.93 / 151,936 | Một dòng so sánh |
| 4.5 (tuỳ chọn) | Loss của câu tiếng Việt và câu tiếng Anh; loss của câu xáo trộn từ (ví dụ "đẹp rất trời nay Hôm.") | Câu xáo trộn loss cao hơn → model đã học trật tự từ |

Cell "Phân tích" (khung, điền theo số thật):
> "Token … có loss thấp nhất (P = …) vì … dễ đoán; token … có loss cao nhất vì … . Loss trung bình … , perplexity … , thấp hơn rất nhiều so với đoán bừa (11.93) → model đã học được quy luật tiếng Việt từ dữ liệu."

**Chú ý:**
- Chạy loss ở chế độ `model.eval()` và trong `torch.no_grad()`.
- Không train, không gọi `loss.backward()`.

---

## 6. Nội dung slide

### Slide 8 — Next-token prediction và dữ liệu
- **Tiêu đề:** 4. LLM học bằng cách đoán token tiếp theo
- **Trái:** bảng cặp ngữ cảnh → token mục tiêu (token thật từ notebook).
- **Phải (3 dòng):**
  - Nhãn nằm sẵn trong văn bản → tự giám sát, dùng được dữ liệu khổng lồ.
  - Causal mask → một câu n token cho n − 1 dự đoán cùng lúc.
  - Dữ liệu: GPT-3 300 tỉ token (Common Crawl 60%, WebText2 22%, sách 16%, Wikipedia 3%); Qwen2.5 18 nghìn tỉ token.
- **Chân:** Brown et al. 2020; Qwen2.5 report · Notebook phần 4, cell 4.1

### Slide 9 — Cross-entropy loss
- **Tiêu đề:** Cross-entropy: phạt khi gán xác suất thấp cho token đúng
- **Trên:** công thức L = −(1/(n−1)) Σ log P(x_t | x_<t); PPL = exp(L).
- **Giữa:** bảng loss từng token thật của Qwen.
- **Dưới:** "Loss câu ví dụ = … (PPL …) ≪ đoán bừa 11.93" + "Mỗi bước = phân loại với 151,936 lớp (giống softmax + cross-entropy ở Ch2)".
- **Chân:** Slide Ch2 s7, s13; HF Perplexity · Notebook phần 4, cell 4.2–4.4

---

## 7. Lời thoại (khoảng 2 phút)

> **[Slide 8]** "LLM chỉ học một nhiệm vụ: đoán token tiếp theo. Từ một câu, ta tạo được các cặp: ngữ cảnh 'Hôm' thì đích là 'nay', ngữ cảnh 'Hôm nay' thì đích là 'trời', cứ thế đến hết câu. Nhãn nằm sẵn trong văn bản, nên không cần người gán nhãn, và dùng được lượng văn bản khổng lồ: GPT-3 học 300 tỉ token, Qwen2.5 học 18 nghìn tỉ token. Nhờ causal mask ở mục 2, cả câu được học song song trong một lần chạy.
> **[Slide 9]** Mỗi bước thực chất là bài phân loại như ở chương 2, chỉ là với khoảng 150 nghìn lớp, nên loss cũng là cross-entropy: lấy trừ log của xác suất model gán cho token đúng. Gán 0.9 thì loss 0.1; gán 0.01 thì loss 4.6. Đây là loss thật của Qwen trên từng token của câu ví dụ … Loss trung bình … , nếu đoán bừa sẽ là 11.9. Và loss em tự tính khớp với loss model trả về."

---

## 8. Q&A dự kiến

| # | Câu hỏi | Ý trả lời | Nguồn |
|---|---|---|---|
| 1 | Nhãn lấy từ đâu khi pretraining? | Chính token tiếp theo trong văn bản → tự giám sát | §1.2 |
| 2 | Câu n token cho mấy cặp? Có phải chạy n − 1 lần không? | n − 1 cặp; chạy một lần nhờ causal mask, mỗi vị trí cho một dự đoán | §2.2 |
| 3 | Vì sao loss là cross-entropy? | Mỗi bước là phân loại nhiều lớp trên từ điển; cross-entropy phạt khi xác suất cho lớp đúng thấp | Slide Ch2 s7, s13 |
| 4 | `labels=input_ids` có bị lệch không? | Không; model tự dịch nhãn sang trái 1 vị trí. Nhóm kiểm bằng cách tự tính và so | HF Perplexity; §4.4 |
| 5 | Perplexity là gì? | exp(loss); như số lựa chọn đều mà model đang phân vân | HF Perplexity |
| 6 | Loss bao nhiêu là tốt? | So với mốc đoán bừa ln(151,936) ≈ 11.93; không so perplexity giữa hai tokenizer khác nhau | Tự tính; HF Perplexity |
| 7 | Vì sao không tự train để minh hoạ? | Cô góp ý không cần fine-tune; tính loss của model đã train là đủ để minh hoạ đúng công thức dùng lúc huấn luyện | Góp ý 11/11 |
| 8 | `-100` để làm gì? | Đánh dấu vị trí không tính loss, ví dụ phần prompt khi SFT | HF Perplexity; mục 5 |

---

## 9. Tài liệu tham khảo cho mục 4

| Tài liệu | Dùng cho |
|---|---|
| Brown et al., *Language Models are Few-Shot Learners*, 2020. https://arxiv.org/abs/2005.14165 | Bảng dữ liệu GPT-3, 300 tỉ token |
| Qwen Team, *Qwen2.5 Technical Report*, 2024. https://arxiv.org/abs/2412.15115 | 18 nghìn tỉ token; cân lại dữ liệu |
| Hugging Face, *Perplexity of fixed-length models*. https://huggingface.co/docs/transformers/perplexity | Định nghĩa PPL; `labels=input_ids` tự dịch nhãn; `-100` |
| Hugging Face LLM Course, ch. 1 "Transformer Architectures". https://huggingface.co/learn/llm-course/chapter1/6 | Decoder-only pretrain bằng next-token prediction |
| Karpathy, *Let's build GPT: from scratch* (video). https://www.youtube.com/watch?v=kCc8FmEb1nY | Minh hoạ tạo cặp ngữ cảnh–đích và loss |
| Slide môn Ch2 s7 (cross-entropy), s9, s11–13 (softmax, sparse categorical CE) | Liên hệ phân loại |
