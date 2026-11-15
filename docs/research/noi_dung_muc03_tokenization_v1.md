# Mục 3 — Tokenization: token, token ID và embedding (nội dung chi tiết, v1)

- Người phụ trách: Muội | Slide 7 | Thời lượng: khoảng 1.5 phút | Notebook: phần 3 (`src/tokenization.py`)
- Đề của cô: "Cách văn bản được chia thành token; token ID và embedding". Minh hoạ: tokenize một câu tiếng Việt và một câu tiếng Anh.
- Ký hiệu nguồn:
  - **[Slide ChX sN]**: slide môn.
  - **[Bổ sung]**: nguồn ngoài, đã kiểm 2026-11-15 (§9).
  - **[Tự tính]**: đã chạy lại bằng Python.
- **Chưa chạy tokenizer thật:** máy làm research không tải được tokenizer từ Hugging Face. Số token của GPT-2 và Qwen chỉ điền từ notebook.

---

## 0. Mục này cần làm người nghe hiểu được gì
1. Model không đọc chữ mà đọc **số**: văn bản → token → token ID → vector embedding.
2. LLM chia văn bản thành **subword**, không theo từ hay ký tự, và lý do của lựa chọn đó.
3. BPE hoạt động thế nào, bản byte-level dùng ở GPT-2 và Qwen.
4. Vì sao cùng một ý mà câu tiếng Việt có thể tốn nhiều token hơn, và hệ quả của điều đó.

---

## 1. Slide môn đã có gì  [Slide Ch6 s17]
Phần "biểu diễn chuỗi" của Ch6 đã giới thiệu các bước:
- **Integer encoding:** mỗi phần tử (từ, nhãn) thành một số nguyên. Gọn, nhưng tạo ra thứ tự giả.
- **One-hot:** vector nhị phân; rất thưa khi từ điển lớn.
- **Embedding:** vector dày, ít chiều, **học được**, mang quan hệ ngữ nghĩa; tốt nhất khi từ điển lớn.
- **Padding / truncation** để các chuỗi dài bằng nhau.

Mục 3 nối tiếp: LLM làm đúng chuỗi "integer encoding → embedding" này. Chỗ khác là đơn vị là **subword** thay cho từ, và từ điển rất lớn (Qwen: 151,936).

---

## 2. Ba cách chia văn bản  [Bổ sung: Hugging Face, *Summary of the tokenizers*]

| Cách chia | Ưu | Nhược |
|---|---|---|
| Theo **từ** | Mỗi đơn vị có nghĩa; chuỗi ngắn | Từ điển phình rất to ("love", "loving", "loved", "lovingly" mỗi từ một token) nên ma trận embedding khổng lồ; từ chưa gặp thành `<unk>` |
| Theo **ký tự** | Từ điển nhỏ, không bao giờ có `<unk>` | Chuỗi rất dài; một ký tự mang ít nghĩa nên model kém hơn |
| Theo **subword** (LLM dùng) | Từ phổ biến giữ nguyên; từ hiếm tách thành mảnh đã biết, ví dụ "annoyingly" → "annoying" + "ly" | Một từ có thể thành nhiều token; cách tách phụ thuộc dữ liệu huấn luyện tokenizer |

Subword cũng giải quyết vấn đề từ hiếm. Sennrich et al. (2016) tách từ hiếm thành đơn vị subword (dùng BPE) để model dịch được với từ điển mở, và cho kết quả tốt hơn cách tra từ điển dự phòng trên WMT 15.

---

## 3. BPE (Byte Pair Encoding)

### 3.1 Cách huấn luyện tokenizer BPE  [Bổ sung: HF tokenizer summary]
Ví dụ gốc của tài liệu Hugging Face:
1. Đếm từ trong dữ liệu: ("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5).
2. Từ điển ban đầu gồm **mọi ký tự**: b, g, h, n, p, s, u.
3. Tìm **cặp liền kề xuất hiện nhiều nhất** và gộp lại: "u" + "g" → "ug".
4. Lặp lại: "u" + "n" → "un", …
5. Dừng khi từ điển đạt kích thước mong muốn. Kích thước từ điển = số ký tự gốc + số lần gộp.

Tokenizer BPE **được huấn luyện một lần** trên dữ liệu, rồi giữ cố định. Model học với đúng tokenizer đó, nên **không được dùng tokenizer của model khác** (Lab 3 / Practice 3 cũng nhắc tokenizer phải khớp model).

### 3.2 Byte-level BPE  [Bổ sung]
- Nếu từ điển gốc gồm mọi ký tự Unicode thì rất lớn. **Byte-level BPE** lấy **256 giá trị byte** làm từ điển gốc, nên chữ nào cũng tách được, không bao giờ cần `<unk>`.
- **GPT-2:** từ điển 50,257 = 256 byte + 50,000 lần gộp + 1 token đặc biệt kết thúc văn bản.
- **Qwen2.5:** byte-level BPE (BBPE) với 151,643 token thường, cộng 22 token điều khiển (Qwen2.5 Technical Report). Trong `config.json`, `vocab_size` là 151,936; con số này lớn hơn tổng trên, nhiều khả năng do làm tròn kích thước ma trận (chưa thấy tài liệu ghi lý do, nên nói "lớn hơn" chứ không giải thích chắc).

### 3.3 Vì sao tiếng Việt có thể tốn nhiều token hơn  [Tự tính + suy luận]
Chữ có dấu tốn **nhiều byte** trong UTF-8:

| Ký tự | Byte UTF-8 |
|---|---|
| ô | 2 byte |
| ờ, ấ, ẹ | 3 byte mỗi chữ |

| Câu | Số ký tự | Số byte UTF-8 |
|---|---|---|
| "Hôm nay trời rất đẹp." | 21 | **29** |
| "The weather is very nice today." | 31 | **31** |

Suy luận (**chưa kiểm bằng tokenizer thật**):
- Tokenizer GPT-2 được huấn luyện chủ yếu trên văn bản tiếng Anh, nên ít cặp byte tiếng Việt được gộp. Câu tiếng Việt khi đó có thể bị tách vụn thành nhiều token, thậm chí từng byte.
- Qwen2.5 được huấn luyện đa ngôn ngữ (model card ghi hỗ trợ hơn 29 ngôn ngữ, có tiếng Việt), nên kỳ vọng tách tiếng Việt gọn hơn.
- **Notebook phần 3 phải đo bằng số thật** rồi mới kết luận.

**Hệ quả nếu đúng là tốn nhiều token hơn:**
- Tốn nhiều chỗ trong ngữ cảnh (giới hạn 32,768 token với Qwen; xem mục 9).
- Sinh chậm hơn, vì mỗi token là một bước sinh (mục 6).
- Model phải "ghép" nghĩa từ nhiều mảnh nhỏ.

---

## 4. Token ID và embedding

### 4.1 Token ID
- Mỗi token trong từ điển có một **số nguyên** riêng (token ID), giống integer encoding ở Ch6.
- Ví dụ minh hoạ dạng bảng (số thật điền từ notebook):

| Token | Token ID |
|---|---|
| "H" / "Hôm" / … | … |

- Token thường có dấu cách dính ở đầu. Tokenizer kiểu GPT-2 hiển thị dấu cách bằng ký tự đặc biệt "Ġ". Đây là điều sẽ thấy khi in token ra; ghi chú điều này trong cell phân tích nếu output có.

### 4.2 Embedding  [Slide Ch6 s17 + Bổ sung]
- **Bảng embedding** là một ma trận kích thước (số token trong từ điển) × (số chiều).
- Với Qwen2.5-1.5B: **151,936 × 1,536 = 233,373,696 tham số** [Tự tính], khoảng 15% toàn model.
- Lấy embedding = **tra hàng** theo token ID: token ID k → hàng thứ k, một vector 1,536 chiều. Đây đúng là `nn.Embedding` ở Ch6.
- Embedding **được học** cùng toàn model khi pretraining; token gần nghĩa có xu hướng có vector gần nhau [Slide Ch6 s17].
- Qwen2.5-1.5B **dùng chung** ma trận embedding với lớp đầu ra (LM head) (`tie_word_embeddings`): cùng một ma trận vừa biến ID thành vector ở đầu vào, vừa biến vector thành điểm cho từng token ở đầu ra (mục 2 §5).

### 4.3 Luồng đầy đủ
```
"Hôm nay trời rất đẹp."
   │ tokenizer.tokenize   → ["…", "…", …]                 (token, dạng chữ)
   │ tokenizer.encode     → [id₁, id₂, …]                 (token ID)
   │ model.get_input_embeddings()(ids)
   ▼
tensor kích thước (1, số_token, 1536)                     (embedding)
```

---

## 5. Minh hoạ trong notebook (phần 3)  [đề xuất — Muội chốt]

| Cell | Làm gì | Output cho slide |
|---|---|---|
| 3.1 | `tokenize_text(qwen_tok, câu_vi)` và `(qwen_tok, câu_en)`: bảng token và token ID | Hai bảng token |
| 3.2 | `compare_token_counts({"gpt2": gpt2_tok, "qwen": qwen_tok}, [câu_vi, câu_en])` | Bảng 2×2 số token |
| 3.3 | In `tokenizer.vocab_size` của hai tokenizer (GPT-2 kỳ vọng 50,257) | Kích thước từ điển |
| 3.4 | `model.get_input_embeddings().weight.shape` và shape embedding của câu ví dụ | (151936, 1536), (1, n, 1536) |
| 3.5 | `tokenizer.decode(ids)` trả lại đúng câu gốc | Kiểm tra hai chiều |

Bảng mong đợi (điền số thật):

| Câu | GPT-2 | Qwen2.5 |
|---|---|---|
| "Hôm nay trời rất đẹp." | … token | … token |
| "The weather is very nice today." | … token | … token |

Cell "Phân tích" (khung, điền theo số thật):
> "Câu tiếng Việt có … token với GPT-2 so với … token với Qwen; câu tiếng Anh … và … . Tokenizer GPT-2 chủ yếu học từ tiếng Anh, nên … . Ít token hơn nghĩa là tốn ít ngữ cảnh hơn và sinh nhanh hơn."

**Chú ý:** `len(tokenizer(câu)["input_ids"])` có thể đã gồm token đặc biệt. Đếm bằng `tokenizer(câu, add_special_tokens=False)` để so công bằng giữa hai tokenizer.

---

## 6. Nội dung slide 7

- **Tiêu đề:** 3. Tokenization: từ chữ đến số
- **Trên:** sơ đồ một dòng: Văn bản → token → token ID → embedding (vector 1,536 chiều).
- **Giữa trái:** hai câu sau khi tokenize bằng Qwen (token + ID), lấy từ notebook.
- **Giữa phải:** bảng số token GPT-2 và Qwen (số thật).
- **Dưới (2 dòng):**
  - Subword (byte-level BPE): từ phổ biến giữ nguyên, từ hiếm tách mảnh, không có `<unk>`.
  - Bảng embedding Qwen: 151,936 × 1,536 ≈ 233 triệu tham số; tra hàng theo token ID.
- **Chân:** Nguồn: Sennrich et al. 2016; HF tokenizer summary; Qwen2.5 report · Notebook phần 3

---

## 7. Lời thoại (khoảng 1.5 phút)

> "Model không đọc được chữ, chỉ tính trên số. Vì vậy văn bản phải đi qua tokenizer: tách thành token, mỗi token có một số ID, rồi ID được tra trong bảng embedding để lấy một vector. Ở chương 6 ta đã gặp integer encoding và embedding; LLM làm đúng như vậy, chỉ khác đơn vị là subword.
> Vì sao subword? Chia theo từ thì từ điển quá lớn và gặp từ lạ là bó tay; chia theo ký tự thì chuỗi quá dài. Subword giữ nguyên từ phổ biến và tách từ hiếm thành mảnh đã biết. Cách tách được học bằng BPE: bắt đầu từ ký tự, liên tục gộp cặp xuất hiện nhiều nhất. GPT-2 và Qwen dùng bản byte-level, nên chữ nào cũng tách được.
> Đây là hai câu ví dụ … [đọc số thật]. Với GPT-2, câu tiếng Việt tốn … token, còn Qwen chỉ … . Tốn nhiều token nghĩa là tốn nhiều ngữ cảnh và sinh chậm hơn. Cuối cùng, bảng embedding của Qwen có 151,936 hàng, mỗi hàng 1,536 số."

---

## 8. Q&A dự kiến

| # | Câu hỏi | Ý trả lời | Nguồn |
|---|---|---|---|
| 1 | Vì sao không chia theo từ? | Từ điển phình to, ma trận embedding khổng lồ; từ chưa gặp thành `<unk>` | HF tokenizer summary |
| 2 | BPE học thế nào? | Bắt đầu từ ký tự/byte, lặp lại việc gộp cặp liền kề phổ biến nhất cho tới đủ kích thước từ điển | HF tokenizer summary |
| 3 | Byte-level BPE có lợi gì? | Từ điển gốc chỉ 256 byte; mọi chuỗi đều mã hoá được, không cần `<unk>` | HF tokenizer summary |
| 4 | Vì sao tiếng Việt có thể tốn token hơn? | Chữ có dấu chiếm 2–3 byte UTF-8; tokenizer học chủ yếu từ tiếng Anh ít gộp cặp byte tiếng Việt. Nói theo số đo được trong notebook | Tự tính §3.3 |
| 5 | Dùng tokenizer GPT-2 cho model Qwen được không? | Không. Token ID của hai bên khác nhau; model học với tokenizer nào thì phải dùng đúng tokenizer đó | Practice 3; §3.1 |
| 6 | Embedding khác one-hot thế nào? | One-hot thưa, không mang nghĩa; embedding dày, ít chiều, học được, token gần nghĩa thì gần nhau | Slide Ch6 s17 |
| 7 | Bảng embedding của Qwen bao nhiêu tham số? | 151,936 × 1,536 ≈ 233 triệu; dùng chung với LM head | Tự tính; `config.json` |

---

## 9. Tài liệu tham khảo cho mục 3

| Tài liệu | Dùng cho |
|---|---|
| Sennrich, Haddow, Birch, *Neural Machine Translation of Rare Words with Subword Units*, ACL 2016. https://arxiv.org/abs/1508.07909 | BPE, subword cho từ hiếm |
| Hugging Face, *Summary of the tokenizers*. https://huggingface.co/docs/transformers/tokenizer_summary | Từ / ký tự / subword; ví dụ BPE; byte-level BPE; GPT-2 50,257 |
| Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2), 2019 | Byte-level BPE của GPT-2 (bài gốc; trích nếu cần) |
| Qwen Team, *Qwen2.5 Technical Report*, 2024. https://arxiv.org/abs/2412.15115 | BBPE 151,643 token thường, 22 token điều khiển |
| Model card + `config.json` Qwen2.5-1.5B-Instruct. https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct | `vocab_size` 151,936; hơn 29 ngôn ngữ, có tiếng Việt; embedding dùng chung |
| Hugging Face Tokenizers docs. https://huggingface.co/docs/tokenizers | API tokenizer |
| Slide môn Ch6 s17 | Integer encoding, one-hot, embedding, padding |
