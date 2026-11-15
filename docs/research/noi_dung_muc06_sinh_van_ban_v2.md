# Mục 6 — Quá trình sinh văn bản: autoregressive, greedy, temperature, top-p (nội dung chi tiết, v2)

- Người phụ trách mục (lý thuyết, slide, thuyết trình): Phúc | Code: Toàn | Slide 12–13 | Thời lượng: khoảng 1.5 phút | Notebook: phần 6 trong `notebook/llm_presentation.ipynb`
- Cập nhật v2 (2026-11-16T01:40:17+07:00, Toàn, AI hỗ trợ): bỏ `src/` (code chỉ trong notebook, Toàn viết toàn bộ code, quyết định 15/11). v1 giữ nguyên, không dùng.
- Đề của cô: "Autoregressive generation; greedy decoding, temperature và top-p sampling". Minh hoạ: cùng một prompt với các cấu hình sinh khác nhau.
- Ký hiệu nguồn:
  - **[Slide ChX sN]**: slide môn.
  - **[Bổ sung]**: nguồn ngoài, đã kiểm 2026-11-15 (§9).
  - **[Tự tính]**: đã chạy lại bằng Python.

---

## 0. Mục này cần làm người nghe hiểu được gì
1. LLM sinh văn bản **từng token một**: đoán → chọn → nối vào → đoán tiếp (autoregressive).
2. Model chỉ cho **phân phối xác suất**; cách **chọn** token từ phân phối đó (decoding) là một bước riêng, chọn khác thì văn bản khác.
3. Greedy: ổn định nhưng dễ lặp. Sampling: đa dạng nhưng dễ lạc. Temperature và top-p là hai "núm vặn" ở giữa.
4. Thấy được điều đó trên output thật của cùng một prompt.

---

## 1. Autoregressive generation

### 1.1 Slide môn  [Slide Ch6 s16]
Ở encoder-decoder RNN, decoder sinh đầu ra **từng phần tử một**: mỗi bước dùng hidden state hiện tại và **phần tử vừa sinh** để đoán phần tử tiếp theo, dừng khi sinh ra token kết thúc. LLM decoder-only làm đúng như vậy, chỉ thay RNN bằng Transformer.

### 1.2 Vòng lặp sinh
```
ngữ cảnh = prompt
lặp:
    phân phối = softmax(model(ngữ cảnh) tại vị trí cuối)   ← mục 2
    token = CHỌN(phân phối)                                 ← decoding: greedy / sampling
    ngữ cảnh = ngữ cảnh + token
    dừng nếu token = <kết thúc> hoặc đủ max_new_tokens
```
- Mỗi token mới cần một lần chạy model, nên văn bản càng dài càng chậm. Tokenizer tách câu thành nhiều token cũng làm sinh chậm hơn (mục 3).
- Nhờ causal mask, K và V của các token cũ không đổi, nên được lưu lại (KV cache) thay vì tính lại (mục 2).
- **Lỗi dồn tích:** token đã chọn thì không sửa lại được. Một token sai kéo theo phần sau; liên hệ hallucination ở mục 9.

---

## 2. Các cách chọn token (decoding)

### 2.1 Ví dụ đồ chơi dùng xuyên suốt  [Tự tính]
Giả sử sau "Hôm nay trời rất", model cho logits của 5 token: đẹp 3.0, nóng 2.0, lạnh 1.5, mưa 1.0, xanh 0.0. Đây là số tự đặt để minh hoạ, **không** phải của Qwen.

Softmax với temperature T = 1:

| Token | đẹp | nóng | lạnh | mưa | xanh |
|---|---|---|---|---|---|
| P | 0.56 | 0.21 | 0.13 | 0.08 | 0.03 |

### 2.2 Greedy decoding
- **Cách làm:** luôn chọn token có xác suất cao nhất → "đẹp".
- **Ưu:** tất định: cùng prompt luôn ra cùng output. Hợp cho tác vụ cần đáp án cố định, ví dụ phân loại ở mục 7/10.
- **Nhược:**
  - Hugging Face ghi greedy hợp output ngắn, nhưng **với chuỗi dài dễ lặp lại**.
  - Holtzman et al. (2020): các cách giải mã tối đa hoá xác suất (greedy, beam search) cho văn bản **nhạt và hay rơi vào vòng lặp**. Lý do: văn bản người viết **không phải** văn bản có xác suất cao nhất.
- **Trong `transformers`:** `do_sample=False`.

### 2.3 Pure sampling (lấy mẫu thuần)
- **Cách làm:** bốc ngẫu nhiên theo đúng phân phối. Với ví dụ trên, "đẹp" 56% số lần, "xanh" 3%.
- **Ưu:** đa dạng, ít lặp.
- **Nhược:** Holtzman et al. chỉ ra "**phần đuôi không đáng tin**": rất nhiều token xác suất thấp, cộng lại thành một khối xác suất đáng kể, nên thỉnh thoảng bốc phải token vô lý và văn bản mất mạch lạc.
- **Trong `transformers`:** `do_sample=True`.

### 2.4 Temperature
- **Cách làm:** chia logits cho T **trước** softmax: `P = softmax(logits / T)` (Holtzman §3.3).
- **Ví dụ** [Tự tính]:

| T | đẹp | nóng | lạnh | mưa | xanh | Tác dụng |
|---|---|---|---|---|---|---|
| 0.5 | **0.83** | 0.11 | 0.04 | 0.02 | 0.00 | Nhọn hơn → gần greedy, an toàn |
| 1.0 | 0.56 | 0.21 | 0.13 | 0.08 | 0.03 | Phân phối gốc |
| 1.5 | 0.44 | 0.23 | 0.16 | 0.12 | **0.06** | Phẳng hơn → đa dạng, dễ lạc |

- T < 1: dễ đoán hơn. T > 1: ngẫu nhiên hơn.
- Holtzman et al. lưu ý T thấp làm **giảm đa dạng và tăng lặp**.
- T → 0 thì gần như luôn chọn token cao nhất, giống greedy.

### 2.5 Top-p (nucleus sampling)
- **Cách làm** (Holtzman §3.1): mỗi bước lấy **tập nhỏ nhất** các token xác suất cao nhất có **tổng xác suất ≥ p**, chuẩn hoá lại trong tập đó, rồi bốc. Kích thước tập **thay đổi theo hình dạng phân phối**.
- **Ví dụ** [Tự tính, T = 1]:

| p | Tập giữ lại | Tổng | Xác suất sau chuẩn hoá |
|---|---|---|---|
| 0.9 | đẹp, nóng, lạnh, mưa | 0.97 | 0.58 / 0.21 / 0.13 / 0.08 — loại "xanh" (đuôi) |
| 0.5 | chỉ "đẹp" | 0.56 | 1.00 — **trong ví dụ này thành greedy** |

- **Vì sao tốt hơn top-k** (Holtzman §3.2): top-k giữ **cố định** k token.
  - Phân phối phẳng mà k nhỏ → quá hạn chế, văn bản chung chung.
  - Phân phối nhọn mà k lớn → giữ cả token kém, rồi chuẩn hoá lại còn làm chúng mạnh lên.
  - Top-p tự co giãn theo phân phối.

### 2.6 Bảng tổng (cho slide 13)
| Cách | Tham số `transformers` | Tất định? | Ưu | Nhược |
|---|---|---|---|---|
| Greedy | `do_sample=False` | Có | Ổn định, lặp lại được | Nhạt, dễ lặp khi dài |
| Sampling + temperature | `do_sample=True, temperature=T` | Không | T điều chỉnh độ "liều" | T cao dễ vô nghĩa; T thấp dễ lặp |
| Top-p | `do_sample=True, top_p=p` | Không | Cắt đuôi vô lý, tập co giãn | p nhỏ gần như greedy |
| Top-k | `do_sample=True, top_k=k` | Không | Đơn giản | k cố định không hợp mọi ngữ cảnh |

---

## 3. Minh hoạ: cùng prompt, 5 cấu hình (slide 13, notebook phần 6)

### 3.1 Cấu hình (theo kế hoạch nhóm)  [đề xuất — Phúc chốt]
| Tên | Tham số truyền tường minh |
|---|---|
| greedy | `do_sample=False` |
| temp0.7_top_p0.9 | `do_sample=True, temperature=0.7, top_p=0.9` |
| temp1.0 | `do_sample=True, temperature=1.0, top_p=1.0` |
| temp1.5 | `do_sample=True, temperature=1.5, top_p=1.0` |
| top_p0.5 | `do_sample=True, temperature=1.0, top_p=0.5` |

- Mỗi cấu hình sinh **5 mẫu**, cùng `max_new_tokens`, seed cố định cho từng mẫu (`seed + i`) để chạy lại ra đúng kết quả.

### 3.2 Chú ý kỹ thuật (quan trọng)
- `generation_config.json` của Qwen2.5-1.5B-Instruct đặt sẵn `do_sample: true`, `temperature: 0.7`, `top_p: 0.8`, `top_k: 20`, `repetition_penalty: 1.1`. Tham số truyền vào `generate()` **ghi đè** mặc định, nhưng tham số **không truyền** thì vẫn bị mặc định chen vào.
- → Mỗi cấu hình phải đặt rõ cả `top_k` và `repetition_penalty` (ví dụ `repetition_penalty=1.0`, và tắt top-k). Nếu không, mọi cấu hình đều ngầm có top-k = 20 và phạt lặp 1.1, và so sánh không còn đúng là "chỉ đổi temperature/top-p".
- Cách tắt top-k đúng cho phiên bản `transformers` đang dùng (ví dụ `top_k=0` hay `None`) **phải kiểm lúc chạy**: in `model.generation_config` và kiểm output greedy giống nhau giữa hai lần chạy.
- Tương tự mục 5: in cấu hình thật vào notebook làm bằng chứng.

### 3.3 Prompt  [đề xuất]
Một prompt tiếng Việt mở, để thấy rõ khác biệt, ví dụ: `Viết tiếp câu chuyện: Hôm nay trời rất đẹp, em quyết định` (định dạng bằng chat template, hoặc dùng bản base nếu muốn "viết tiếp" thuần).

### 3.4 Cách phân tích (khung cell "Phân tích")
| Quan sát | Đo thế nào | Kỳ vọng theo lý thuyết (kiểm bằng output thật) |
|---|---|---|
| Độ đa dạng | Số mẫu khác nhau trong 5 mẫu | greedy = 1 mẫu duy nhất; temp1.5 khác nhau nhiều nhất |
| Lặp lại | Có cụm lặp không | greedy dễ lặp khi sinh dài |
| Mạch lạc | Đọc bằng mắt | temp1.5 dễ có từ vô nghĩa hoặc lạc đề |
| top_p0.5 | So với greedy | Có thể gần như greedy khi phân phối nhọn (§2.5) |

> Khung viết: "Greedy cho 5 mẫu giống hệt nhau … . Temperature 1.5 cho … mẫu khác nhau nhưng có … (từ lạ, lạc đề). Top-p 0.9 với T = 0.7 cân bằng hơn … . Điều này khớp với Holtzman et al.: tối đa hoá xác suất thì nhạt, lấy mẫu thuần thì lạc, cắt đuôi thì cân bằng."

---

## 4. Nội dung slide

### Slide 12 — Sinh từng token
- **Tiêu đề:** 6. Sinh văn bản: từng token một (autoregressive)
- **Trái:** vòng lặp 4 bước (đoán phân phối → chọn → nối vào → lặp), kèm liên hệ "decoder ở Ch6 cũng sinh từng phần tử".
- **Phải:** bảng sinh từng bước thật từ notebook: bước | token chọn | xác suất.
- **Chân:** Notebook phần 6, cell 6.1 · Theo kế hoạch nhóm, nếu quá giờ thì rút slide này trước.

### Slide 13 — Greedy, temperature, top-p
- **Tiêu đề:** Cách chọn token quyết định văn bản
- **Trên:** ví dụ đồ chơi: phân phối 5 token ở T = 0.5 / 1 / 1.5 (hình cột nhỏ) và tập giữ lại với top-p 0.9.
- **Giữa:** bảng 5 cấu hình × 1–2 mẫu output thật (rút gọn).
- **Dưới (2 dòng):**
  - Greedy: ổn định nhưng nhạt, dễ lặp · Sampling: đa dạng nhưng dễ lạc · Top-p: cắt phần đuôi vô lý.
  - Nhóm dùng greedy cho thí nghiệm mục 10 để kết quả lặp lại được.
- **Chân:** Holtzman et al. 2020; HF generation strategies · Notebook phần 6, cell 6.2

---

## 5. Lời thoại (khoảng 1.5 phút)

> **[Slide 12]** "LLM sinh văn bản từng token một: model cho phân phối xác suất của token tiếp theo, ta chọn một token, nối vào ngữ cảnh rồi lặp lại. Decoder RNN ở chương 6 cũng sinh từng phần tử như vậy. Đây là các bước thật của Qwen …
> **[Slide 13]** Model chỉ cho xác suất; chọn thế nào là việc của ta. Greedy luôn lấy token cao nhất: ổn định, nhưng văn bản dài dễ nhạt và lặp. Lấy mẫu ngẫu nhiên thì đa dạng, nhưng đôi khi bốc trúng token vô lý ở phần đuôi. Temperature chia logits trước softmax: nhỏ hơn 1 thì phân phối nhọn, lớn hơn 1 thì phẳng. Top-p chỉ giữ nhóm token nhỏ nhất có tổng xác suất đạt p, tức là cắt bỏ phần đuôi. Đây là cùng một prompt với 5 cấu hình … [đọc output thật]. Vì cần kết quả lặp lại được, thí nghiệm chính của nhóm dùng greedy."

---

## 6. Q&A dự kiến

| # | Câu hỏi | Ý trả lời | Nguồn |
|---|---|---|---|
| 1 | Autoregressive là gì? | Sinh từng token, mỗi token mới dựa trên mọi token trước, kể cả token vừa sinh | Slide Ch6 s16 |
| 2 | Vì sao greedy dễ lặp? | Tối đa hoá xác suất từng bước cho văn bản nhạt, dễ vòng lặp; văn bản người viết không phải chuỗi xác suất cao nhất | Holtzman 2020; HF docs |
| 3 | Temperature tác động ở đâu? | Chia logits trước softmax; T < 1 nhọn, T > 1 phẳng | Holtzman §3.3 |
| 4 | Top-p khác top-k thế nào? | Top-k giữ số token cố định; top-p giữ tập nhỏ nhất đạt tổng xác suất p, co giãn theo phân phối | Holtzman §3.1–3.2 |
| 5 | Top-p = 0.5 có khác greedy không? | Khi token đầu đã có xác suất ≥ 0.5 thì tập chỉ còn 1 token, giống greedy (ví dụ §2.5) | Tự tính |
| 6 | Vì sao thí nghiệm mục 10 dùng greedy? | Tất định: cùng input ra cùng output, nên so sánh zero/few-shot không bị nhiễu do ngẫu nhiên | §2.2 |
| 7 | Gọi `generate()` không tham số là greedy? | Mặc định chung là greedy, nhưng `generation_config` của model ghi đè được; bản Instruct của Qwen mặc định lấy mẫu | HF docs; `generation_config.json` |
| 8 | Sinh càng dài càng chậm vì sao? | Mỗi token một lần chạy model; KV cache đỡ việc tính lại phần cũ | §1.2 |

---

## 7. Tài liệu tham khảo cho mục 6

| Tài liệu | Dùng cho |
|---|---|
| Holtzman, Buys, Du, Forbes, Choi, *The Curious Case of Neural Text Degeneration*, ICLR 2020. https://arxiv.org/abs/1904.09751 | Greedy/beam nhạt, lặp; đuôi không đáng tin; top-p; temperature; nhược điểm top-k |
| Hugging Face, *Generation strategies*. https://huggingface.co/docs/transformers/generation_strategies | Greedy mặc định, dễ lặp khi dài; `do_sample`; `generation_config` |
| `generation_config.json` của Qwen2.5-1.5B-Instruct. https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/generation_config.json | Mặc định sampling, top_k 20, repetition_penalty 1.1 |
| Slide môn Ch6 s16 | Decoder sinh từng phần tử |
