# Mục 2 — Nền tảng kiến trúc: Transformer, Decoder-only, Causal attention (nội dung chi tiết, v2)

- Người phụ trách mục (lý thuyết, slide, thuyết trình): Muội | Code: Toàn | Slide 5–6 | Thời lượng: khoảng 2.5 phút | Notebook: phần 2 trong `notebook/llm_presentation.ipynb`
- Cập nhật v2 (2026-11-16T01:40:17+07:00, Toàn, AI hỗ trợ): bỏ `src/` (code chỉ trong notebook, Toàn viết toàn bộ code, quyết định 15/11); sửa dòng Attention mỗi block ở §5.2 (5,507,072); thêm §11 so sánh LLM với mô hình khác. v1 giữ nguyên, không dùng.
- Đề của cô: "Vai trò của Transformer; tập trung vào Decoder-only và causal attention". Minh hoạ: sơ đồ từ token đầu vào đến xác suất token tiếp theo.
- Ký hiệu nguồn:
  - **[Slide ChX sN]**: slide môn (`Research/slides-ocr/`).
  - **[Bổ sung]**: nguồn ngoài, đã kiểm 2026-11-14/15 (§9).
  - **[Tự tính]**: đã chạy lại bằng Python.

---

## 0. Mục này cần làm người nghe hiểu được gì
1. Vì sao có Transformer: hạn chế của RNN encoder-decoder.
2. Self-attention là gì; công thức softmax(QKᵀ/√d_k)V.
3. Ba họ Transformer; LLM hiện nay chủ yếu là **decoder-only**.
4. **Causal attention**: mỗi token chỉ nhìn chính nó và các token trước, và vì sao cần như vậy.
5. Luồng dữ liệu: token → embedding → các decoder block → logits → softmax → xác suất token tiếp theo, kèm thông số thật của Qwen2.5-1.5B.

---

## 1. Từ RNN đến Transformer

### 1.1 Slide môn đã có gì
- **[Slide Ch6 s16]** Encoder-decoder bằng RNN:
  - encoder nén cả chuỗi đầu vào thành **một vector cố định** (context vector);
  - decoder sinh đầu ra **từng phần tử một**, dùng hidden state và phần tử vừa sinh để đoán phần tử tiếp theo, cho tới token kết thúc.
- **[Slide Ch6 s11]** Biến thể "LSTM with Attention": cho model tập trung vào phần liên quan nhất của chuỗi đầu vào khi dự đoán.
- **[Slide Ch1 s12]** Transformer hợp nhất cho NLP (dịch, tóm tắt, hỏi đáp); đặc trưng chính là cơ chế attention; ví dụ BERT, GPT, ViT.
- **[Slide Ch1 s7]** Transformer đã làm thay đổi NLP và nhiều lĩnh vực khác.

### 1.2 Vì sao cần Transformer
| Vấn đề của RNN encoder-decoder | Transformer giải quyết thế nào |
|---|---|
| Nén cả câu vào **một vector cố định** [Slide Ch6 s16] → câu dài dễ mất thông tin | Attention cho mỗi vị trí nhìn trực tiếp mọi vị trí khác |
| Xử lý tuần tự từng bước → khó tính song song | Bỏ phần hồi quy; mọi vị trí tính cùng lúc trên GPU |
| Phụ thuộc xa phải đi qua nhiều bước (vanishing gradient, Ch6) | Hai vị trí bất kỳ cách nhau đúng một bước attention |

**[Bổ sung]** Vaswani et al. (2017), *Attention Is All You Need*, đề xuất Transformer: kiến trúc **không dùng hồi quy hay tích chập**, chỉ dùng attention.

---

## 2. Self-attention

### 2.1 Ý tưởng
Mỗi token "hỏi" các token khác để quyết định nên lấy thông tin từ đâu. Ví dụ, trong "Hôm nay trời rất đẹp", để hiểu "đẹp" thì model cần chú ý tới "trời".

### 2.2 Công thức  [Bổ sung: Vaswani et al. 2017, §3.2.1]
Mỗi token được biến thành 3 vector bằng 3 ma trận học được:
- **Q (query):** token này đang tìm gì;
- **K (key):** token này có gì để được tìm;
- **V (value):** nội dung token này đóng góp.

```
Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k ) · V
```

| Bước | Ý nghĩa |
|---|---|
| Q·Kᵀ | Điểm liên quan giữa mọi cặp token (tích vô hướng) |
| / √d_k | Chia cho căn số chiều của key. Nếu không chia, tích vô hướng lớn dần theo d_k, đẩy softmax vào vùng gradient rất nhỏ (với thành phần có kỳ vọng 0, phương sai 1 thì tích vô hướng có phương sai bằng d_k) |
| softmax | Biến điểm thành trọng số, mỗi hàng cộng lại bằng 1 |
| · V | Output = tổng có trọng số các vector value |

### 2.3 Multi-head attention  [Bổ sung: §3.2.2]
- Chạy nhiều attention song song, mỗi "head" dùng phép chiếu Q, K, V riêng; nối kết quả lại rồi chiếu thêm một lần.
- Lợi ích: mỗi head có thể chú ý tới một kiểu quan hệ khác nhau.
- Bài gốc dùng 8 head với d_model = 512. Qwen2.5-1.5B dùng 12 head cho Q (§5).

### 2.4 Thông tin vị trí  [Bổ sung: §3.5]
- Attention tự nó không biết thứ tự từ, vì không có hồi quy hay tích chập.
- Bài gốc **cộng positional encoding** (hàm sin/cos) vào embedding.
- Qwen2.5 dùng **RoPE** (rotary position embedding) theo model card, là một cách khác để đưa thông tin vị trí vào.

### 2.5 Residual và chuẩn hoá  [Bổ sung: §3.1]
- Mỗi lớp con được bọc bởi residual connection và chuẩn hoá: LayerNorm(x + Sublayer(x)).
- Liên hệ môn học: ý tưởng residual giống ResNet (Ch5A).
- Qwen2.5 dùng **RMSNorm**, một biến thể của LayerNorm (model card).

---

## 3. Ba họ Transformer  [Bổ sung: Hugging Face LLM Course, "Transformer Architectures"]

| Họ | Ví dụ | Attention | Cách pretrain | Tác vụ điển hình |
|---|---|---|---|---|
| Encoder-only | BERT, DistilBERT | **Hai chiều**: mỗi token nhìn cả câu, trái và phải | Làm hỏng đầu vào (che từ) rồi khôi phục | Phân loại câu, nhận dạng thực thể, hỏi đáp trích xuất |
| **Decoder-only** | GPT-2, Llama, Gemma, Qwen | **Một chiều**: mỗi token chỉ nhìn token trước | **Đoán token tiếp theo** trên kho văn bản lớn | Sinh văn bản, tóm tắt, dịch, sinh code, hội thoại |
| Encoder-decoder | BART, T5, Marian | Encoder hai chiều; decoder nhìn token output trước và phần đã mã hoá | Khôi phục đoạn bị che | Dịch, tóm tắt, hỏi đáp sinh |

- Hầu hết LLM hiện nay dùng **decoder-only**: pretrain bằng next-token prediction rồi instruction-tune (HF LLM Course).
- Cách gọi khác: encoder-only ≈ "representation model", decoder-only ≈ "generative model" (Alammar & Grootendorst, *Hands-On Large Language Models*, 2024). Không bắt buộc theo đề; nếu dùng thì chỉ một dòng.

---

## 4. Causal attention (masked self-attention)

### 4.1 Định nghĩa  [Bổ sung: Vaswani et al. 2017, §3.2.3]
- Trong decoder, self-attention bị giới hạn: **mỗi vị trí chỉ được chú ý tới chính nó và các vị trí trước**.
- Bài gốc mô tả mục đích là ngăn thông tin "chảy ngược" từ tương lai.
- **Cách làm:** trước softmax, gán điểm của các vị trí tương lai bằng **−∞**. Vì e^(−∞) = 0, trọng số tại các vị trí đó bằng 0.

### 4.2 Vì sao cần
1. **Lúc huấn luyện:** cả câu được đưa vào cùng lúc để tính song song, và mỗi vị trí phải đoán token kế tiếp. Không có mask thì vị trí "trời" nhìn thấy luôn "rất", tức là chép đáp án; loss rất thấp nhưng model không học được cách đoán.
2. **Lúc sinh văn bản:** token tương lai chưa tồn tại. Mask làm cách model chạy lúc huấn luyện **giống** lúc dùng.
3. **Hệ quả kỹ thuật:** token cũ không bao giờ phụ thuộc token mới, nên khi sinh có thể lưu lại K, V đã tính (KV cache) thay vì tính lại từ đầu.

### 4.3 Vì sao gán −∞ mà không gán 0
- Gán **điểm** bằng 0 thì sau softmax e⁰ = 1, vị trí đó vẫn nhận trọng số dương, tức là vẫn "nhìn" được tương lai.
- Chỉ −∞ mới cho trọng số đúng bằng 0.

### 4.4 Minh hoạ bằng code  [Tự tính: PyTorch 2.15 CPU, điểm attention là số ngẫu nhiên, KHÔNG phải của Qwen]

```python
import torch

torch.manual_seed(0)
n = 4                                                              # "Hôm", "nay", "trời", "đẹp"
scores = torch.randn(n, n)                                         # điểm Q·Kᵀ/√d_k giả lập
mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)  # True = vị trí tương lai
weights = torch.softmax(scores.masked_fill(mask, float("-inf")), dim=-1)
```

```
Mask (1 = bị chặn)      Trọng số sau softmax
[[0, 1, 1, 1],          [[1.00, 0.00, 0.00, 0.00],   ← "Hôm" chỉ nhìn chính nó
 [0, 0, 1, 1],           [0.54, 0.46, 0.00, 0.00],
 [0, 0, 0, 1],           [0.45, 0.09, 0.46, 0.00],
 [0, 0, 0, 0]]           [0.13, 0.41, 0.36, 0.09]]   ← "đẹp" nhìn cả câu
```

Đọc kết quả:
- nửa trên đường chéo đều bằng 0;
- mỗi hàng cộng lại bằng 1;
- token càng về sau càng nhìn được nhiều token.

Ở notebook, Muội lấy trọng số **thật** của Qwen để vẽ. Muốn vậy phải nạp model với `attn_implementation="eager"`, vì kiểu attention mặc định (SDPA) không trả trọng số (`research_presentation_llm_v1.md` §1 ý 3).

---

## 5. Luồng từ token đến xác suất token tiếp theo (sơ đồ slide 5)

```
"Hôm nay trời rất"
      │ tokenizer (mục 3)
      ▼
token IDs: [id₁, id₂, …, id_n]
      │ bảng embedding 151,936 × 1,536
      ▼
vector 1,536 chiều cho mỗi token  (+ thông tin vị trí bằng RoPE)
      │
      ▼
┌───────────── Decoder block × 28 ─────────────┐
│ RMSNorm → Masked (causal) self-attention     │
│           12 head Q, 2 head KV (GQA) + residual │
│ RMSNorm → FFN SwiGLU (1,536 → 8,960 → 1,536)  │
│           + residual                          │
└───────────────────────────────────────────────┘
      │ RMSNorm cuối
      ▼
vector của token CUỐI (1,536 chiều)
      │ lớp tuyến tính (LM head) 1,536 → 151,936
      ▼
logits: 151,936 điểm, mỗi token trong từ điển một điểm
      │ softmax
      ▼
P(token tiếp theo | "Hôm nay trời rất")  → top-5: "đẹp" …, … (lấy từ notebook)
```

- Bài gốc (§3.4) mô tả đúng bước cuối: output của decoder đi qua một phép biến đổi tuyến tính rồi softmax để ra xác suất token tiếp theo; ma trận này dùng chung trọng số với lớp embedding.
- Qwen2.5-1.5B cũng **dùng chung** (`tie_word_embeddings`).

### 5.1 Thông số thật của Qwen2.5-1.5B  [Bổ sung: model card + `config.json`]
| Thông số | Giá trị |
|---|---|
| Số tham số | 1.54 tỉ (1.31 tỉ không tính embedding) |
| Số decoder block | 28 |
| `hidden_size` | 1,536 |
| Head | 12 head cho Q, 2 head cho KV (grouped-query attention); mỗi head 128 chiều (1,536 / 12) |
| FFN | `intermediate_size` 8,960, kích hoạt SwiGLU |
| Từ điển | 151,936 |
| Vị trí | RoPE, `rope_theta` = 1,000,000 |
| Chuẩn hoá | RMSNorm |
| Ngữ cảnh tối đa | 32,768 token |

### 5.2 Kiểm tra lại số tham số  [Tự tính từ `config.json`]
| Thành phần | Cách tính | Số tham số |
|---|---|---|
| Embedding (dùng chung với LM head) | 151,936 × 1,536 | 233,373,696 |
| Attention mỗi block | Q: 1,536×1,536 · K, V: 1,536×256 mỗi cái · O: 1,536×1,536 · bias Q, K, V | 5,507,072 |
| FFN mỗi block | 3 × 1,536 × 8,960 (gate, up, down) | 41,287,680 |
| RMSNorm mỗi block | 2 × 1,536 | 3,072 |
| Một block | | 46,797,824 |
| 28 block + norm cuối + embedding | | **1,543,714,304 ≈ 1.54 tỉ** |
| Không tính embedding | | **1,310,340,608 ≈ 1.31 tỉ** |

Kết quả khớp model card → hiểu đúng kiến trúc. Bảng này dùng được khi cô hỏi "1.5 tỉ tham số nằm ở đâu". Nhận xét: **FFN chiếm phần lớn** mỗi block (khoảng 88%); embedding chiếm khoảng 15% toàn model.

**GQA là gì (ngắn):** 12 head query nhưng chỉ 2 head key/value, nên mỗi 6 head Q dùng chung một cặp K, V. Cách này giảm bộ nhớ KV cache khi sinh. Ở mức trình bày chỉ cần nói ý này.

---

## 6. Minh hoạ trong notebook (phần 2)  [đề xuất — Muội chốt]

| Cell | Làm gì | Output đưa lên slide |
|---|---|---|
| 2.1 | `model_config_summary(model)`: in số layer, hidden, head, vocab, số tham số đếm thật | Bảng §5.1 (số thật từ code) |
| 2.2 | `causal_mask(n)` cho câu ví dụ, vẽ heatmap | Ma trận mask, slide 6 |
| 2.3 | Lấy trọng số attention thật (`output_attentions=True`, eager) của 1 layer, 1 head trên "Hôm nay trời rất đẹp." | Heatmap tam giác dưới |
| 2.4 | `top_k_next_tokens(model, tok, "Hôm nay trời rất", k=5)` | Bảng top-5 token + xác suất, slide 6 |

Cell "Phân tích" sau 2.3 và 2.4 (điền theo output thật):
- heatmap đúng hình tam giác → causal mask hoạt động;
- các xác suất top-5 cộng lại chưa bằng 1, vì phân phối trải trên 151,936 token.

---

## 7. Nội dung slide

### Slide 5 — Transformer và Decoder-only
- **Tiêu đề:** 2. Nền tảng kiến trúc: Transformer Decoder-only
- **Trái (3 dòng):**
  - Transformer (Vaswani et al. 2017): chỉ dùng attention, không hồi quy → song song, nhìn xa trực tiếp.
  - Ba họ: encoder-only (BERT), **decoder-only (GPT, Qwen)**, encoder-decoder (T5).
  - LLM hiện nay chủ yếu là decoder-only: học bằng cách đoán token tiếp theo.
- **Phải:** sơ đồ §5, kèm số thật của Qwen.
- **Chân:** Nguồn: Vaswani et al. 2017; HF LLM Course; model card Qwen2.5 · Notebook phần 2, cell 2.1

### Slide 6 — Causal attention
- **Tiêu đề:** Causal attention: chỉ nhìn quá khứ
- **Trái:** công thức `softmax(QKᵀ/√d_k + M)·V`, với M = 0 ở vị trí được nhìn và −∞ ở vị trí tương lai, kèm 2 lý do (huấn luyện không chép đáp án; giống lúc sinh).
- **Giữa:** heatmap attention thật của Qwen trên câu ví dụ.
- **Phải:** bảng top-5 token tiếp theo sau "Hôm nay trời rất" (số thật).
- **Chân:** Notebook phần 2, cell 2.2–2.4

---

## 8. Lời thoại (khoảng 2.5 phút)

> **[Slide 5]** "Ở môn học, ta đã gặp encoder-decoder bằng RNN: encoder nén cả câu vào một vector cố định, decoder sinh từng phần tử một. Câu dài thì vector đó không chứa nổi, và RNN phải chạy tuần tự. Transformer bỏ hẳn phần hồi quy, chỉ dùng attention: mỗi token nhìn trực tiếp mọi token khác, và tính song song được.
> Transformer có ba họ. Encoder-only như BERT nhìn cả hai chiều, hợp để hiểu văn bản. Decoder-only như GPT và Qwen chỉ nhìn phía trước, học bằng cách đoán token tiếp theo, nên sinh được văn bản; hầu hết LLM hiện nay là loại này.
> Đây là luồng dữ liệu của Qwen 1.5B nhóm dùng: câu được tách thành token, mỗi token thành vector 1,536 chiều, đi qua 28 decoder block. Vector của token cuối đi qua một lớp tuyến tính ra 151,936 điểm, mỗi token trong từ điển một điểm, rồi softmax thành xác suất token tiếp theo.
> **[Slide 6]** Trong mỗi block, attention có mask: token chỉ được nhìn chính nó và các token trước. Trước softmax, điểm ở vị trí tương lai được gán âm vô cùng, nên trọng số bằng 0. Nếu không có mask, lúc huấn luyện model sẽ nhìn thấy đáp án. Đây là trọng số attention thật của Qwen trên câu ví dụ, đúng hình tam giác. Và đây là 5 token có xác suất cao nhất sau 'Hôm nay trời rất' … [đọc số thật]."

---

## 9. Q&A dự kiến

| # | Câu hỏi | Ý trả lời | Nguồn |
|---|---|---|---|
| 1 | Vì sao chia cho √d_k? | Tích vô hướng lớn dần theo số chiều, làm softmax bão hoà và gradient rất nhỏ | Vaswani §3.2.1 |
| 2 | Vì sao gán −∞ mà không gán 0? | e⁰ = 1 vẫn cho trọng số dương; chỉ −∞ cho trọng số 0 | §4.3 |
| 3 | Decoder-only khác encoder-only thế nào? | Causal (một chiều) và đoán token tiếp theo, so với hai chiều và che từ rồi khôi phục; một bên sinh văn bản, một bên hiểu và phân loại | HF LLM Course |
| 4 | Transformer biết thứ tự từ bằng cách nào? | Thêm thông tin vị trí: bài gốc dùng sin/cos cộng vào embedding; Qwen dùng RoPE | Vaswani §3.5; model card |
| 5 | Multi-head để làm gì? | Nhiều phép chiếu song song, mỗi head học một kiểu quan hệ; nối lại rồi chiếu | Vaswani §3.2.2 |
| 6 | 1.5 tỉ tham số nằm ở đâu? | Embedding khoảng 233 triệu; mỗi block khoảng 46.8 triệu, trong đó FFN khoảng 41.3 triệu; 28 block | Tự tính §5.2 |
| 7 | GQA là gì? | 12 head Q dùng chung 2 head KV → giảm bộ nhớ KV cache khi sinh | Model card |
| 8 | Liên hệ gì với RNN ở Ch6? | Cả hai đều sinh từng phần tử một; RNN nén vào vector cố định và chạy tuần tự, Transformer nhìn trực tiếp mọi vị trí và song song | Slide Ch6 s16 |

---

## 11. So sánh LLM với các loại mô hình khác (điểm cộng, có demo)  (Toàn duyệt 16/11; Muội trình bày)

Mục đích: cho thấy LLM decoder-only nằm ở đâu so với các mô hình đã học (RNN/LSTM ở Ch6) và các họ mô hình gần nó.

### 11.1 Bảng so sánh
| | RNN / LSTM | Encoder-only (BERT) | Encoder-decoder (T5) | Decoder-only LLM (Qwen2.5) | VLM (Qwen2.5-VL) |
|---|---|---|---|---|---|
| Đầu vào | Chuỗi, xử lý từng bước | Văn bản | Văn bản | Văn bản (token) | Ảnh / video + văn bản |
| Cơ chế chính | Hidden state hồi quy; LSTM thêm cell state và 3 cổng [Slide Ch6 s11–14] | Self-attention **hai chiều** | Encoder hai chiều + decoder sinh từng token | Self-attention có **causal mask** (§4) | Bộ mã hoá ảnh (ViT) + LLM Qwen2.5 [Bổ sung: Qwen2.5-VL Technical Report] |
| Cách pretrain | Thường train cho một tác vụ cụ thể | Che từ rồi đoán lại (masked LM) | Khôi phục đoạn bị che | Đoán token tiếp theo | LLM khởi tạo từ trọng số Qwen2.5, huấn luyện thêm trên dữ liệu ảnh–chữ |
| Sinh văn bản | Được, nhưng ngữ cảnh ngắn | Không (dùng để hiểu, phân loại) | Được | Được (tự hồi quy) | Được (sinh chữ, không sinh ảnh) |
| Tính song song khi train | Không, phải chạy tuần tự theo thời gian | Có | Có | Có | Có |
| Phụ thuộc xa | Yếu do vanishing gradient; LSTM đỡ hơn [Slide Ch6] | Nhìn trực tiếp mọi vị trí trong cửa sổ | Như BERT | Nhìn mọi token trước, ngữ cảnh 32,768 token (Qwen2.5-1.5B) | Như LLM, thêm token ảnh |
| Ví dụ | char-rnn, LSTM phân loại chuỗi (Ch6) | BERT, DistilBERT | T5, BART | GPT-2, Llama, Qwen2.5 | Qwen2.5-VL, LLaVA |

- Ý chính để nói: LLM hiện nay là Transformer **decoder-only**; khác RNN ở chỗ bỏ hồi quy để tính song song và nhìn xa trực tiếp; khác BERT ở chỗ chỉ nhìn quá khứ nên sinh được văn bản; VLM là LLM được **ghép thêm** bộ mã hoá ảnh, phần sinh chữ vẫn là LLM.
- Theo báo cáo Qwen2.5-VL: mô hình gồm LLM làm thành phần nền, khởi tạo từ trọng số Qwen2.5; bộ mã hoá ảnh ViT được huấn luyện từ đầu. Tên tầng nối giữa ViT và LLM: **[cần mở báo cáo §2.1 để ghi đúng]**.

### 11.2 Gợi ý so sánh thêm (tuỳ chọn, chỉ làm nếu đủ thời gian)
| So sánh | Ý chính | Có demo được trên Kaggle T4 không |
|---|---|---|
| Model nhỏ so với lớn cùng họ (Qwen2.5 có nhiều cỡ từ 0.5B đến 72B) | Cùng kiến trúc, khác số tham số → khác chất lượng và tốc độ | Dễ nhất: Qwen2.5-0.5B-Instruct so với 1.5B trên cùng prompt **[cần chạy thử]** |
| Dense so với Mixture-of-Experts (MoE) | MoE chỉ kích hoạt một phần tham số cho mỗi token | Không (model MoE quá lớn); chỉ nêu lý thuyết, cần thêm nguồn |
| Model suy luận nhiều bước (reasoning model) | Sinh chuỗi suy luận trước khi trả lời | Chưa có nguồn đã mở; bỏ nếu không kịp |

### 11.3 Demo (bắt buộc, Toàn chốt 16/11 "phải có demo")  (thiết kế demo: Toàn duyệt 16/11)
Thêm **1 slide (6b)** sau slide 6, khoảng 1 phút; tổng bài thành 22 slide. Code do Toàn viết, đặt cuối phần 2 notebook (cell 2.5–2.8); log theo `templates/RUN_LOG_TEMPLATE_v2.md`.

**Demo A — Encoder-only (BERT) so với decoder-only (Qwen): nhìn hai chiều hay chỉ nhìn quá khứ**
- Model: `google-bert/bert-base-multilingual-cased` (khoảng 0.2B tham số, Apache-2.0; danh sách ngôn ngữ của bản multilingual có tiếng Việt) và `Qwen/Qwen2.5-1.5B-Instruct` (đã nạp ở phần 0).
- Cách làm: cùng một câu có từ cần đoán ở giữa, ví dụ `Thủ đô của [MASK] là Hà Nội.`
  - BERT: `pipeline("fill-mask")` đoán `[MASK]`, nhìn được cả "là Hà Nội" ở bên phải.
  - Qwen: chỉ đưa `Thủ đô của`, lấy top-5 token tiếp theo; Qwen không thấy phần bên phải.
- Output: `outputs/results/compare_bert_qwen.csv` (câu, model, top-5 token, xác suất).
- Ý cần thấy: BERT dùng được ngữ cảnh bên phải nên hợp để hiểu / điền chỗ trống; Qwen chỉ nhìn quá khứ nên sinh được văn bản tiếp. Kết quả nào cũng ghi đúng như thật.
- Lưu ý: tokenizer của mBERT có thể tách "Việt Nam" thành nhiều token, mà `[MASK]` chỉ đoán một token → chọn câu mẫu sao cho đáp án là một token, kiểm bằng tokenizer trước **[cần kiểm khi code]**.

**Demo B — LLM chỉ đọc chữ so với VLM đọc ảnh + chữ**
- Model VLM: `Qwen/Qwen2-VL-2B-Instruct` (2B tham số, Apache-2.0; model card ghi hiểu được chữ tiếng Việt trong ảnh). Không dùng `Qwen2.5-VL-3B-Instruct` vì trang model ghi khoảng 4B tham số, vượt luật "không dùng model lớn hơn 3B".
- Cách làm: một ảnh nhóm tự tạo (`data/vlm_image.png`, ví dụ ảnh chụp bảng có chữ tiếng Việt, không bản quyền) + câu hỏi về nội dung ảnh (`data/compare_prompts.json`, có đáp án ghi sẵn).
  - Qwen2.5-1.5B-Instruct: chỉ nhận câu hỏi (không thấy ảnh).
  - Qwen2-VL-2B-Instruct: nhận ảnh + câu hỏi.
  - Cả hai `do_sample=False`, cùng `max_new_tokens`.
- Output: `outputs/results/compare_vlm.csv` (câu hỏi, model, output, đúng/sai theo đáp án ghi sẵn).
- Phải kiểm ở smoke run: phiên bản `transformers` trên Kaggle có `Qwen2VLForConditionalGeneration` không (bản cũ báo `KeyError: 'qwen2_vl'`); VRAM T4 đủ không (giải phóng model eager và bản base trước khi nạp).

**RNN/LSTM:** không demo, vì muốn LSTM sinh được chữ phải train, trái với "không fine-tune / không train". Chỉ trình bày bảng §11.1 và nối với slide Ch6.

**Đánh giá demo:** có output thật cho cả 2 demo; demo A ghi top-5 kèm xác suất của cả hai model; demo B chấm đúng/sai theo đáp án ghi trước khi chạy.

### 11.4 Q&A thêm
| Câu hỏi | Ý trả lời | Nguồn |
|---|---|---|
| Vì sao LLM không dùng LSTM? | LSTM phải chạy tuần tự nên khó train song song trên dữ liệu rất lớn; phụ thuộc xa phải đi qua nhiều bước | Slide Ch6; Vaswani et al. 2017 |
| BERT có sinh văn bản được không? | Không theo cách tự hồi quy: BERT nhìn hai chiều, học bằng che từ, dùng để hiểu và phân loại | Devlin et al. 2019 |
| VLM khác LLM ở đâu? | VLM thêm bộ mã hoá ảnh, đưa ảnh thành token cho LLM; phần sinh chữ vẫn là LLM | Qwen2.5-VL Technical Report |

---

## 10. Tài liệu tham khảo cho mục 2

| Tài liệu | Dùng cho |
|---|---|
| Vaswani et al., *Attention Is All You Need*, NeurIPS 2017. https://arxiv.org/abs/1706.03762 | Công thức attention, √d_k, multi-head, mask, positional encoding, residual, linear + softmax |
| Alammar, *The Illustrated GPT-2*. https://jalammar.github.io/illustrated-gpt2/ | Hình minh hoạ masked self-attention trong decoder-only |
| Alammar, *The Illustrated Transformer*. https://jalammar.github.io/illustrated-transformer/ | Hình minh hoạ Q, K, V |
| Hugging Face LLM Course, ch. 1 "Transformer Architectures". https://huggingface.co/learn/llm-course/chapter1/6 | Ba họ Transformer |
| Qwen Team, *Qwen2.5 Technical Report*, 2024. https://arxiv.org/abs/2412.15115 | Kiến trúc Qwen2.5 |
| Model card và `config.json` của Qwen2.5-1.5B(-Instruct). https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct | Thông số thật |
| Slide môn Ch1 s7, s12; Ch6 s11, s16 | Transformer trong NLP; RNN encoder-decoder; LSTM + attention |
| Slide môn Ch6 s11–s16 | RNN, LSTM, GRU, encoder-decoder (§11) |
| Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, NAACL 2019. https://arxiv.org/abs/1810.04805 (chưa mở kiểm) | Encoder-only (§11) |
| Raffel et al., *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* (T5), JMLR 2020. https://arxiv.org/abs/1910.10683 (chưa mở kiểm) | Encoder-decoder (§11) |
| Qwen Team, *Qwen2.5-VL Technical Report*, 2025. https://arxiv.org/abs/2502.13923 (mở 16/11) | VLM (§11) |
