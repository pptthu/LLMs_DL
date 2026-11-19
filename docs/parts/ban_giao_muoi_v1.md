# Bàn giao code cho Muội — mục 1, 2, 3, 8

- **Created**: 2026-11-19 (Toàn, AI hỗ trợ)
- Dành cho: Muội, người trình bày mục 1 (Giới thiệu), 2 (Kiến trúc), 3 (Tokenization), 8 (RAG).
- Đọc xong file này phải giải thích được **từng cell** của phần 1, 2, 3, 8 trong `notebook/llm_presentation.ipynb` (yêu cầu thêm 5 của cô).
- Mọi số trong file lấy từ lần chạy chính thức **`20261119_150356_normal`** (Kaggle, GPU T4). File kết quả ở `outputs/results/`, hình ở `outputs/figures/`, log ở `outputs/logs/20261119_150356_normal/`.

---

## 0. Đọc trước: notebook này hoạt động thế nào

### 0.1 Notebook là gì, chạy ở đâu
- Notebook là file gồm nhiều **ô (cell)**. Ô **markdown** là chữ giải thích; ô **code** là code Python, chạy xong thì kết quả (output) hiện ngay bên dưới.
- Nhóm chạy trên **Kaggle** (máy chủ miễn phí có GPU Tesla T4). GPU giúp model chạy nhanh hơn CPU hàng chục lần.
- Thứ tự chạy: từ trên xuống. Phần sau dùng biến của phần trước (ví dụ `model`, `tok` tạo ở phần 0), nên **không chạy riêng phần 8 được** nếu chưa chạy phần 0.

### 0.2 Mỗi cell có cấu trúc giống nhau
1. Ô markdown trước: **Mục đích**, **Đầu vào**, **Đầu ra**.
2. Ô code: dòng đầu là `# X.Y tên cell`. Có comment tiếng Việt cho dòng khó.
3. Cuối cell thường có `assert ...`: câu lệnh **tự kiểm**. Nếu điều kiện sai, notebook dừng và báo lỗi. Ví dụ `assert len(df) == 4` nghĩa là "bảng phải đúng 4 dòng, không thì dừng".
4. Hết mỗi phần có ô **Phân tích**: nhận xét về output, đây là ý để nói trên slide.

### 0.3 Các biến dùng chung (tạo ở phần 0, Toàn trình bày)
| Biến | Là gì |
|---|---|
| `model` | Model chính `Qwen/Qwen2.5-1.5B-Instruct`: 1,543,714,304 tham số, nằm trên GPU |
| `tok` | Tokenizer của Qwen: biến chữ thành token ID và ngược lại |
| `gpt2_tok` | Tokenizer của GPT-2 (chỉ tokenizer, không có model), để so ở phần 3 |
| `RESULTS_DIR`, `FIGURES_DIR`, `LOG_DIR` | Thư mục ghi kết quả, hình, log |
| `log(text, part=...)` | Hàm ghi một dòng vào file log (và in ra màn hình) |
| `SEED = 42` | Hạt giống ngẫu nhiên: cùng seed thì chạy lại ra đúng kết quả cũ |

### 0.4 Từ khoá cần nhớ
| Từ | Nghĩa đơn giản |
|---|---|
| **Token** | Mảnh chữ mà model đọc. Có thể là một từ, một phần từ, hoặc một byte |
| **Token ID** | Số thứ tự của token trong từ điển (Qwen có 151,643 token thường) |
| **Logits** | Điểm thô model cho **mỗi** token trong từ điển ở bước tiếp theo |
| **Softmax** | Phép biến điểm thô thành xác suất (các số từ 0 đến 1, cộng lại = 1) |
| **Greedy** | Mỗi bước chọn token có xác suất cao nhất. Chạy lại luôn ra đúng một kết quả |
| **Chat template** | Cách Qwen bọc câu hỏi thành dạng hội thoại: `<|im_start|>user ... <|im_end|>` |
| **`torch.no_grad()`** | Chỉ suy luận, không tính đạo hàm, không train |

---

## 1. Phần 1 — Giới thiệu (slide 3)

**Đề:** LLM là gì? Phân biệt language model, LLM và chatbot. Minh họa: ví dụ các tác vụ hỏi đáp, tóm tắt, dịch, sinh mã.

**Ý chính cần nói:** **một** model, **một** cách sinh (greedy), **chỉ đổi câu lệnh (prompt)** là model làm được 4 việc khác nhau.

### Cell 1.1 — Hàm sinh `chat_generate()`
```python
messages = [{"role": "user", "content": prompt}]
enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None, top_k=None)
new_ids = out[0, n_in:]
text = tok.decode(new_ids, skip_special_tokens=True).strip()
```
- Dòng 1: gói câu hỏi thành một tin nhắn của người dùng.
- Dòng 2: `apply_chat_template` bọc tin nhắn theo đúng định dạng hội thoại của Qwen và biến thành token ID. `add_generation_prompt=True` thêm phần mở đầu lượt "assistant" để model biết đến lượt nó trả lời. `.to(model.device)` đưa dữ liệu lên GPU.
- Dòng 3: `generate` sinh câu trả lời. `do_sample=False` là **greedy**. Ba tham số `None` tắt cấu hình lấy mẫu mặc định của bản Instruct (mặc định của nó là lấy mẫu ngẫu nhiên, nên mỗi lần chạy sẽ khác).
- Dòng 4: output gồm cả prompt + phần sinh thêm; cắt bỏ `n_in` token đầu (prompt), chỉ giữ phần model viết.
- Dòng 5: biến token ID ngược thành chữ.
- Cell thử với "Xin chào": model **trả lời bằng tiếng Trung** ("你好！有什么我可以帮助你的吗？"). Đây là ví dụ hạn chế dùng ở mục 9.

### Cell 1.2 — Chạy 4 tác vụ
- Đọc 4 prompt từ `data/tasks_prompts.json` (bản v2). Mỗi tác vụ có trần số token sinh (`max_new_tokens`): hỏi đáp 32, tóm tắt 256, dịch 48, sinh mã 512.
- Output thật:

| Tác vụ | Prompt | Output của model |
|---|---|---|
| Hỏi đáp | Thủ đô của Nhật Bản là gì? Trả lời một câu. | Thủ đô của Nhật Bản là Tokyo. |
| Tóm tắt | (đoạn 5 câu về môn Học sâu) Tóm tắt đoạn trên trong 1 câu. | Một câu dài 104 token, giữ đủ 5 ý nhưng gần như ghép lại các câu gốc |
| Dịch | Dịch sang tiếng Anh: Hôm nay trời rất đẹp. | Today the weather is very beautiful. |
| Sinh mã | Viết hàm Python is_even(n) trả về True nếu n là số chẵn. | `def is_even(n): return n % 2 == 0`, kèm ví dụ và giải thích |

### Cell 1.3 — Chạy thử code do model viết
- `extract_code()` tách khối ```` ```python ... ``` ```` khỏi câu trả lời, rồi `exec` chạy code đó và gọi `is_even(0), is_even(1), is_even(2), is_even(-3)`.
- Kết quả `[True, False, True, False]`, đúng đáp án.
- Lưu ý: `exec` trong "namespace riêng" chỉ tránh đụng biến notebook, **không phải môi trường an toàn**; phải đọc code model in ra trước khi chạy.

### Cell 1.4 — Chấm
- Luật chấm **ghi trước khi chạy**: hỏi đáp có chữ "Tokyo"; dịch có "today" và một trong "weather/nice/beautiful/lovely"; sinh mã chạy đúng 4 giá trị; tóm tắt chấm tay (đúng 1 câu, giữ ý chính, không thêm ý).
- Kết quả: **đạt 4/4**, không output nào bị cắt. File: `outputs/results/tasks_outputs.csv`.

### Cell 1.5 — Thử prompt khác (demo trực tiếp)
- Sửa `MY_PROMPT` rồi chạy lại riêng cell này để demo khi cô hỏi.
- Output thật với "Giải thích ngắn gọn mạng nơ-ron nhân tạo là gì, trong 2 câu.": model bắt đầu bằng tiếng Việt rồi **chuyển sang tiếng Trung giữa câu**.

### Câu nói trên slide 3
- LM (language model) = model đoán token tiếp theo. LLM = LM rất lớn (hàng tỉ tham số), học trên lượng văn bản khổng lồ. Chatbot = LLM đã qua post-training (bản Instruct) + giao diện hội thoại + lớp an toàn.
- Bản base chỉ viết tiếp văn bản (xem mục 5 của Phúc); bản Instruct dùng ở đây làm theo chỉ dẫn.

---

## 2. Phần 2 — Kiến trúc (slide 4)

**Đề:** vai trò của Transformer; tập trung Decoder-only và causal attention. Minh họa: sơ đồ từ token đầu vào đến xác suất token tiếp theo.

### Cell 2.1 — Thông số và 1.5 tỉ tham số nằm ở đâu
- Đọc thông số từ `model.config` (không chép tay) và đếm tham số theo nhóm bằng tên tham số: `embed_tokens` → embedding, `self_attn` → attention, `mlp` → FFN, `norm` → chuẩn hoá.
- Output thật:

| Nhóm | Số tham số |
|---|---|
| embedding | 233,373,696 (= 151,936 × 1,536) |
| attention | 154,198,016 |
| FFN (mạng truyền thẳng) | 1,156,055,040 (khoảng 75%) |
| norm | 87,552 |
| **Tổng** | **1,543,714,304** |

- 28 block decoder, hidden 1,536, 12 head Q nhưng chỉ 2 head K/V: **GQA** (6 head Q dùng chung 1 cặp K, V để tiết kiệm bộ nhớ). `head_dim` = 1536 / 12 = 128.
- Kết quả ghi `outputs/results/model_config.json`.

### Cell 2.2 — Causal mask
```python
def causal_mask(n):
    return torch.tril(torch.ones(n, n, dtype=torch.bool))
```
- `torch.ones(n, n)`: ma trận toàn số 1. `torch.tril`: giữ **tam giác dưới** (kể cả đường chéo), xoá phần trên thành 0.
- Hàng = token đang xét, cột = token được nhìn. 1 = được nhìn (chính nó và token trước), 0 = token tương lai bị chặn.
- Câu "Hôm nay trời rất đẹp." được Qwen chia thành **8 token**: `'H', 'ôm', ' n', 'ay', ' trời', ' rất', ' đẹp', '.'`. Hình: `outputs/figures/causal_mask.png`.
- **Vì sao cần:** khi huấn luyện, model học đoán token tiếp theo; nếu nhìn được token sau thì nó "chép đáp án".

### Cell 2.3 — Trọng số attention thật
- Cách tính attention mặc định (SDPA) không trả trọng số, nên nạp thêm một bản với `attn_implementation="eager"` (tính bằng phép nhân ma trận thường), lấy trọng số của layer 0, head 0, rồi xoá bản này để nhường bộ nhớ.
- Output thật: tổng phần trên đường chéo = **0.000000**; mỗi hàng cộng lại = 1.0. Đây là **bằng chứng causal mask có tác dụng trong model thật**. Hình: `outputs/figures/attention_qwen.png`.

### Cell 2.4 — Top-5 token tiếp theo
```python
logits = model(ids).logits[0, -1].float()
probs = torch.softmax(logits, dim=-1)
top = torch.topk(probs, k)
```
- `logits[0, -1]`: lấy điểm ở **vị trí cuối** (sau token cuối của câu) — mỗi token trong từ điển một điểm.
- `softmax`: điểm → xác suất. `topk`: lấy 5 token xác suất cao nhất.
- Output thật sau "Hôm nay trời rất":

| Hạng | Token | Xác suất |
|---|---|---|
| 1 | đẹp | 0.788 |
| 2 | nắng | 0.063 |
| 3 | lạnh | 0.046 |
| 4 | nóng | 0.029 |
| 5 | (dấu cách) | 0.007 |

- 5 token cộng lại 0.933; phần còn lại chia cho 151,931 token khác. File: `top5_next_tokens.csv`.

### Cell 2.5 — Tổng kết
- Ghi log và kiểm đủ 2 file kết quả + 2 hình.

### Sơ đồ cho slide 4 (token → xác suất token tiếp theo)
```
"Hôm nay trời rất" → tokenizer → token ID [6 số] → embedding (6 × 1536)
 → 28 decoder block (causal self-attention + FFN) → vector ở vị trí cuối
 → LM head (1536 → 151,936 điểm) → softmax → P(" đẹp") = 0.788
```
- Phần so sánh với mBERT, VLM, LSTM nằm ở `notebook/llm_extension_part02.ipynb`, **không nộp**, không trình bày.

---

## 3. Phần 3 — Tokenization (slide 5)

**Đề:** cách văn bản được chia thành token; token ID và embedding. Minh họa: tokenize một câu tiếng Việt và một câu tiếng Anh.

### Cell 3.1 — Bảng token
```python
ids = tokenizer(text, add_special_tokens=False)["input_ids"]
piece = tokenizer.convert_ids_to_tokens(ids)
text  = [tokenizer.decode([i]) for i in ids]
```
- `ids`: danh sách token ID. `add_special_tokens=False`: không thêm token đặc biệt, để đếm công bằng.
- `piece`: token ở dạng **thô** — `Ġ` là dấu cách; chữ có dấu hiện thành ký tự lạ (`Ã´m`, `Ġtrá»Ŀi`) vì tokenizer làm việc trên **byte UTF-8**.
- `text`: chữ thật khi decode riêng token đó.
- **GPT-2 hiện `�`:** chữ "ờ" chiếm 3 byte `E1 BB 9D`; GPT-2 không có token cho "ờ" nên tách thành 3 token, mỗi token 1 byte; decode riêng 1 byte không ra chữ → `�`. Ghép đủ 3 byte thì ra đúng "ờ".

### Cell 3.2 — Đếm token 2 câu × 2 tokenizer
| Câu | Số ký tự | Số byte | GPT-2 | Qwen |
|---|---|---|---|---|
| Hôm nay trời rất đẹp. | 21 | 29 | **22** | **8** |
| The weather is very nice today. | 31 | 31 | 7 | 7 |

- Ý chính: câu tiếng Anh như nhau, câu tiếng Việt GPT-2 tốn gần **3 lần** token. GPT-2 học chủ yếu tiếng Anh; Qwen học đa ngôn ngữ nên có token riêng cho " trời", " rất", " đẹp".
- Hệ quả: ít token → tốn ít chỗ trong ngữ cảnh (mục 9) và sinh nhanh hơn (mỗi token là một bước sinh, mục 6). File: `token_counts.csv`.

### Cell 3.3 — Kích thước từ điển (3 con số của Qwen)
| Con số | Giá trị | Nghĩa |
|---|---|---|
| GPT-2 `vocab_size` | 50,257 | |
| Qwen `tok.vocab_size` | 151,643 | token thường |
| Qwen `len(tok)` | 151,665 | + token đặc biệt (`<|im_start|>`, `<|im_end|>`…) |
| Qwen `model.config.vocab_size` | 151,936 | số hàng bảng embedding; hàng dư không ứng với token nào |

### Cell 3.4 — Embedding
- Bảng embedding là ma trận **151,936 × 1,536**. Lấy embedding của một token = **lấy hàng** có số thứ tự bằng token ID.
- Câu 8 token: token ID shape `(1, 8)` → embedding shape `(1, 8, 1536)` (1 câu, 8 token, mỗi token 1,536 số).
- `assert torch.equal(vi_emb[0], emb_layer.weight[vi_ids[0]])`: kiểm embedding đúng bằng các hàng của bảng.
- Qwen **dùng chung** bảng này cho lớp đầu ra (LM head): `tie_word_embeddings = True`, kiểm bằng địa chỉ bộ nhớ (`data_ptr()`).

### Cell 3.5 — Decode ngược
- Câu → token ID → câu: ra **đúng câu gốc 4/4** (2 câu × 2 tokenizer). Tokenizer không làm mất thông tin.

---

## 4. Phần 8 — RAG (slide 10)

**Đề:** phân biệt prompting, fine-tuning và RAG. Minh họa: bảng so sánh dữ liệu cần có và trường hợp sử dụng. Góp ý của cô: phải có demo trên model thật.

**RAG (Retrieval-Augmented Generation)** = trước khi hỏi model, **tìm** vài đoạn tài liệu liên quan (Retrieval) rồi **dán vào prompt** (Augmented) để model trả lời dựa trên đó (Generation). Không đổi trọng số model.

### Dữ liệu: UIT-ViQuAD 2.0
- Bộ đọc hiểu tiếng Việt (VLSP 2021), lấy từ Wikipedia tiếng Việt. Split validation: **3,814 câu hỏi, 557 đoạn văn, 19 chủ đề**; 1,161 câu là loại **"đoạn văn không có đáp án"**.
- Mỗi dòng có: `title` (chủ đề), `context` (đoạn văn), `question`, `answers` (đáp án + vị trí trong đoạn), `is_impossible`.
- Notebook tải thẳng từ Hugging Face, không cần upload.

### Cell 8.1 — Chọn câu hỏi và dựng kho tài liệu
- **Kho tài liệu (corpus):** 557 đoạn `context` khác nhau, mỗi đoạn một `doc_id`.
- **Lọc câu tự đủ nghĩa:** bỏ câu có đại từ ("ông ấy", "nó", "đó"…), bỏ câu dưới 7 từ, chỉ lấy câu có đáp án ngắn (≤ 4 từ) để chấm tự động được.
- Lấy ngẫu nhiên (seed 42) **15 câu có đáp án + 3 câu đoạn văn không có đáp án**. Danh sách: `outputs/results/rag_questions_used.csv`.
- Khi hỏi, luôn ghi kèm chủ đề (ví dụ "chủ đề: Paris") để câu không mơ hồ.

### Cell 8.2 — Tìm đoạn liên quan bằng TF-IDF
```python
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
doc_matrix = vectorizer.fit_transform(corpus["title"] + ". " + corpus["context"])
scores = linear_kernel(vectorizer.transform([query]), doc_matrix)[0]
top3 = scores.argsort()[::-1][:3]
```
- **TF-IDF** biến mỗi đoạn thành một vector đếm từ: từ xuất hiện nhiều trong đoạn (TF) nhưng hiếm trong cả kho (IDF) được trọng số cao.
- `ngram_range=(1, 2)`: đếm cả từng âm tiết và **cặp 2 âm tiết liền nhau** (tiếng Việt nhiều từ ghép như "thủ đô", "chính trị").
- `linear_kernel`: tích vô hướng; vì vector TF-IDF đã chuẩn hoá độ dài nên đây chính là **độ giống cosine**.
- `argsort()[::-1][:3]`: xếp điểm giảm dần, lấy 3 đoạn cao nhất.
- **recall@3 = 0.722 (13/18)**: 13 trên 18 câu có đoạn gốc nằm trong top-3.

### Cell 8.3 — Hỏi không RAG và có RAG
- **Hai cách nhận cùng một câu dặn:** "Nếu không biết hoặc tài liệu không có thông tin, trả lời 'Không có thông tin'." Như vậy so khả năng từ chối mới công bằng.
- Không RAG: chỉ có câu hỏi + chủ đề. Có RAG: thêm 3 đoạn tài liệu `[1] … [2] … [3] …`.
- Greedy, `repetition_penalty=1.0`, `max_new_tokens=48`. Hàm `ask()` dùng lại ở phần 9.

### Cell 8.4 — Chấm
- Chuẩn hoá chữ (chữ thường, dấu câu thành khoảng trắng). Câu có đáp án: output chứa đáp án → **Đúng**; chứa "không có thông tin" → **Từ chối**; còn lại → **Sai**. Câu đoạn văn không có đáp án: Từ chối là đúng hành vi; trả lời khác → Sai (bịa).
- Sau đó Toàn đọc lại từng câu, sửa 2 câu chấm tự động sai (`MANUAL_JUDGE8`): "AGM-130" đúng nhưng thiếu chữ "tên lửa"; câu vua Charles — đáp án gốc "1640" là **nhãn sai** của bộ dữ liệu (đúng là 1649).
- Kết quả (`outputs/results/rag_results.csv`):

| | Không RAG | Có RAG |
|---|---|---|
| 15 câu có đáp án: Đúng | **0** | **10** |
| Từ chối | 14 | 2 |
| Sai | 1 | 3 |
| 3 câu đoạn văn không có đáp án: Từ chối (đúng hành vi) | 3 | 1 |
| Trả lời bịa | 0 | 2 |

- Ví dụ cho slide: "Muốn ban hành luật thì hoàng đế phải làm gì?" → không RAG: "phải ký tên vào bản thảo luật" (**Sai**); có RAG: "Hoàng đế phải thông qua Hội đồng để ban hành luật." (**Đúng**).

### Ý nói trên slide 10
1. Không có tài liệu, model từ chối gần hết; có tài liệu, đúng 10/15 → RAG **bổ sung tri thức lúc hỏi**, không cần train lại.
2. RAG chỉ tốt bằng bước tìm kiếm: recall@3 = 0.722; đoạn đúng không được tìm thấy thì RAG cũng không cứu được.
3. Với câu mà tài liệu không có đáp án, có RAG lại **bịa từ tài liệu 2/3** → tài liệu "gần đúng" có thể làm model tự tin trả lời sai.
4. Bảng so sánh: **prompting** chỉ đổi câu hỏi, không cần dữ liệu; **RAG** cần kho tài liệu, không đổi trọng số, hợp với tri thức hay thay đổi; **fine-tuning** đổi trọng số, cần dữ liệu có nhãn và GPU để train, hợp khi muốn đổi hành vi/phong cách lâu dài.

---

## 5. Câu cô có thể hỏi (tự trả lời được trước khi trình bày)
| Câu hỏi | Ý trả lời |
|---|---|
| Sao trả lời bằng greedy? | Greedy luôn ra một kết quả, chạy lại ra y hệt, nên chấm và so sánh được |
| `max_new_tokens` là gì? | Trần số token model được sinh thêm; chạm trần thì câu bị cắt |
| Vì sao câu Việt GPT-2 ra 22 token? | GPT-2 không có token cho chữ có dấu, phải tách xuống từng byte UTF-8 |
| Embedding là gì? | Một hàng 1,536 số trong bảng 151,936 × 1,536, tra theo token ID |
| Causal mask để làm gì? | Chặn token nhìn token phía sau; attention thật phần trên đường chéo = 0 |
| TF-IDF khác embedding? | TF-IDF đếm từ (khớp chữ), không hiểu nghĩa; embedding là vector học được, bắt được nghĩa |
| Sao không dùng slide môn học cho RAG? | Bộ UIT-ViQuAD có sẵn đáp án và đoạn nguồn để kiểm chứng, không phải soạn tay |
