# Nắm kiến thức bài Presentation LLMs theo 5W1H (v1)

- Dùng để: tự học lại toàn bộ 10 mục. Mỗi mục gồm:
  - **Cần nắm**: thông tin phải nhớ, kèm con số;
  - **5W1H**: What, Why, When, Where, Who, How;
  - **Tự kiểm tra**: tự trả lời được thì là đã nắm.
- Chi tiết, nguồn và ví dụ của từng mục nằm trong các file `noi_dung_mucXX_*_v1.md`. Phần của Toàn (mục 7, 10) có thêm `noi_dung_toan_muc07_muc10_v1.md`.
- Ký hiệu: **[Slide]** = có trên slide môn; còn lại là kiến thức bổ sung đã kiểm nguồn (ngày tra 14–15/11/2026).
- **Đọc theo thứ tự 1 → 10**: mỗi mục dựa trên mục trước.

```
1 LLM là gì → 2 Kiến trúc → 3 Token → 4 Cách học → 5 Giai đoạn huấn luyện
→ 6 Cách sinh → 7 Khả năng → 8 Thích nghi → 9 Hạn chế → 10 Thực nghiệm
```

---

## Mục 1 — Giới thiệu: LLM là gì (Muội)

### Cần nắm
- **Language model** = model **dự đoán từ/token tiếp theo**. Xác suất cả câu = tích các xác suất "token tiếp theo" (chain rule).
- **LLM** = language model rất lớn, kiến trúc Transformer, học trên dữ liệu khổng lồ. **Chưa có ngưỡng chính thức** cho chữ "large"; survey Zhao et al. lấy mốc > 10 tỉ tham số.
- **Chatbot** = **ứng dụng** xây trên LLM: LLM + SFT + RLHF + chat template + giao diện + bộ lọc an toàn. ChatGPT fine-tune từ dòng GPT-3.5 bằng RLHF.
- **4 thế hệ LM:** n-gram → neural LM (Bengio 2003) → pre-trained LM (BERT, GPT-2) → LLM (GPT-3 trở đi).
- **Qwen2.5-1.5B** (1.54 tỉ tham số) là **LLM cỡ nhỏ**; bản Instruct vẫn chưa phải chatbot.
- **Điểm yếu của n-gram:** cụm chưa gặp có xác suất 0 (ví dụ tự tính: P(đẹp | nay) = 0).

### 5W1H
| | |
|---|---|
| **What** | Phân biệt LM (đoán token tiếp theo), LLM (LM rất lớn), chatbot (ứng dụng) |
| **Why** | Một model làm được nhiều tác vụ chỉ bằng prompt, không cần model riêng cho từng việc |
| **When** | n-gram (cổ điển) → Bengio 2003 → Transformer 2017 → GPT-3 2020 → ChatGPT 30/11/2022 [Slide Ch1 s10: timeline] |
| **Where** | Dịch, tóm tắt, hỏi đáp, sinh code, chatbot [Slide Ch1 s12, s13, s15] |
| **Who** | OpenAI (GPT, ChatGPT), Google (BERT), Alibaba Qwen Team (Qwen); Bengio (neural LM) |
| **How** | Đoán token tiếp theo lặp đi lặp lại; đổi prompt là đổi tác vụ |

### Tự kiểm tra
1. Vì sao nói "mọi LLM là LM nhưng không phải LM nào cũng là LLM"?
2. ChatGPT có phải là một LLM không?
3. Vì sao n-gram không đủ?

---

## Mục 2 — Kiến trúc: Transformer, Decoder-only, causal attention (Muội)

### Cần nắm
- RNN encoder-decoder nén cả câu vào **một vector cố định** và chạy tuần tự [Slide Ch6 s16] → Transformer **chỉ dùng attention**, song song, nhìn trực tiếp mọi vị trí.
- **Công thức attention:** `softmax(QKᵀ/√d_k)·V`. Chia √d_k để softmax không bão hoà, gradient không quá nhỏ.
- **Ba họ Transformer:**

  | Họ | Ví dụ | Attention | Mạnh ở |
  |---|---|---|---|
  | Encoder-only | BERT | Hai chiều | Hiểu văn bản |
  | **Decoder-only** | GPT, Qwen | **Một chiều** | Sinh văn bản |
  | Encoder-decoder | T5 | Encoder hai chiều, decoder một chiều | Dịch, tóm tắt |

- **Causal attention:** token chỉ nhìn chính nó và token trước; gán **−∞** cho vị trí tương lai trước softmax (gán 0 thì e⁰ = 1, vẫn "nhìn" được).
- **Luồng dữ liệu:** token → embedding → 28 decoder block → token cuối → LM head → 151,936 logits → softmax → P(token tiếp theo).
- **Số thật của Qwen2.5-1.5B:**
  - 28 block, hidden 1,536;
  - 12 head Q, 2 head KV (GQA); FFN 8,960;
  - vocab 151,936; RoPE; RMSNorm; ngữ cảnh 32,768;
  - 1.54 tỉ tham số (tự tính khớp model card); FFN chiếm khoảng 88% mỗi block.

### 5W1H
| | |
|---|---|
| **What** | Transformer decoder-only với masked self-attention |
| **Why** | RNN nén vào vector cố định, chạy tuần tự, khó nhớ xa; mask để lúc huấn luyện không "chép đáp án" |
| **When** | 2017 (*Attention Is All You Need*); decoder-only phổ biến từ GPT |
| **Where** | Mọi decoder block của GPT, Llama, Qwen |
| **Who** | Vaswani et al. (Google), 2017 |
| **How** | Q·Kᵀ/√d_k + mask −∞ → softmax → ·V; nhiều head; residual + chuẩn hoá; thông tin vị trí bằng RoPE |

### Tự kiểm tra
1. Vì sao chia cho √d_k?
2. Vì sao gán −∞ mà không gán 0?
3. Decoder-only khác encoder-only ở đâu?
4. 1.5 tỉ tham số của Qwen nằm chủ yếu ở đâu?

---

## Mục 3 — Tokenization (Muội)

### Cần nắm
- **Luồng:** văn bản → token → token ID → vector embedding. Nối tiếp integer encoding và embedding ở [Slide Ch6 s17].
- **3 cách chia:**
  - theo từ: từ điển quá lớn, có `<unk>`;
  - theo ký tự: chuỗi quá dài;
  - **subword**: LLM dùng.
- **BPE:** bắt đầu từ ký tự (hoặc byte), lặp lại việc gộp **cặp liền kề phổ biến nhất** cho tới đủ kích thước từ điển (Sennrich 2016).
- **Byte-level BPE:** từ điển gốc 256 byte → không bao giờ có `<unk>`.
  - GPT-2: 50,257 token.
  - Qwen2.5: 151,643 token thường + 22 token điều khiển; `vocab_size` 151,936.
- **Tiếng Việt:** chữ có dấu chiếm 2–3 byte UTF-8 ("Hôm nay trời rất đẹp." = 21 ký tự / 29 byte). Tokenizer chủ yếu học tiếng Anh **có thể** tách tiếng Việt vụn hơn. **Phải đo bằng notebook.**
- **Bảng embedding Qwen:** 151,936 × 1,536 ≈ 233 triệu tham số; dùng chung với LM head.
- Tokenizer phải **khớp** model.

### 5W1H
| | |
|---|---|
| **What** | Chia văn bản thành token (subword), gán ID, tra embedding |
| **Why** | Model chỉ tính trên số; subword cân bằng giữa kích thước từ điển và độ dài chuỗi, xử lý được từ hiếm |
| **When** | BPE cho NMT: 2016; byte-level BPE: GPT-2 2019; Qwen2.5: 2024 |
| **Where** | Bước đầu tiên trước khi vào model; ảnh hưởng số token, tốc độ, giới hạn ngữ cảnh |
| **Who** | Sennrich et al. (BPE cho NMT); OpenAI (GPT-2); Qwen Team |
| **How** | `tokenizer.tokenize`, `tokenizer.encode`, `model.get_input_embeddings()`; đếm với `add_special_tokens=False` |

### Tự kiểm tra
1. Vì sao không chia theo từ?
2. BPE học từ điển bằng cách nào?
3. Vì sao tiếng Việt có thể tốn token hơn, và hệ quả là gì?
4. Dùng tokenizer GPT-2 cho Qwen được không?

---

## Mục 4 — Nguyên lý học (Phúc)

### Cần nắm
- **Một nhiệm vụ duy nhất:** đoán token tiếp theo. Nhãn có sẵn trong văn bản → **tự giám sát**.
- **Một câu n token cho n − 1 cặp** (ngữ cảnh → token mục tiêu); nhờ causal mask, chạy **một lần** cho cả n − 1 dự đoán.
- **Mỗi bước là phân loại** với khoảng 150 nghìn lớp → softmax + cross-entropy như [Slide Ch2 s7, s9, s13].
- **Công thức:** `L = −(1/(n−1)) Σ log P(x_t | x_<t)`; **perplexity** = exp(L).
- **Trực giác** (tự tính):
  - p = 0.9 → loss 0.105;
  - p = 0.01 → loss 4.6;
  - đoán bừa đều trên 151,936 token → loss 11.93.
- **Dữ liệu:**
  - GPT-3: 300 tỉ token (Common Crawl 60%, WebText2 22%, sách 16%, Wikipedia 3%);
  - Qwen2.5: **18 nghìn tỉ token**, có lọc và cân lại tỉ lệ.
- `labels=input_ids` → `transformers` **tự dịch nhãn** sang trái 1 vị trí; `-100` = bỏ qua vị trí đó.
- Slide Ch2 s7 đặt ký hiệu y và ŷ **ngược** cách viết thông thường.

### 5W1H
| | |
|---|---|
| **What** | Next-token prediction + cross-entropy trên văn bản khổng lồ |
| **Why** | Không cần gán nhãn → dùng được dữ liệu cực lớn; học tốt từng bước là học được cả câu (chain rule) |
| **When** | Giai đoạn pretraining (và cả SFT, cùng loại loss) |
| **Where** | Web, sách, code, Wikipedia; Qwen2.5 18T token |
| **Who** | Mọi LLM decoder-only: GPT-3 (OpenAI), Qwen2.5 (Qwen Team) |
| **How** | Dịch nhãn 1 vị trí → softmax → −log p(token đúng) → trung bình → backprop. Nhóm chỉ **tính loss**, không train |

### Tự kiểm tra
1. Nhãn lấy từ đâu?
2. Câu 6 token cho mấy cặp, chạy model mấy lần?
3. Loss 11.93 có ý nghĩa gì?
4. Perplexity là gì?

---

## Mục 5 — Các giai đoạn huấn luyện (Phúc)

### Cần nắm
| | Pretraining | SFT | Preference alignment |
|---|---|---|---|
| Mục tiêu | Đoán token tiếp theo | Làm theo chỉ dẫn | Trả lời được người ưa hơn |
| Dữ liệu | Văn bản thô khổng lồ | Cặp (chỉ dẫn → trả lời mẫu) | Cặp (tốt hơn, kém hơn) |
| Kết quả | Base model (viết tiếp) | Biết trả lời | Hữu ích, an toàn hơn |

- **RLHF** (InstructGPT):
  - các bước: SFT (khoảng 13 nghìn prompt) → **reward model** từ xếp hạng của người (khoảng 33 nghìn) → **PPO** (khoảng 31 nghìn) + phạt KL;
  - kết quả: InstructGPT 1.3B được thích hơn GPT-3 175B;
  - cái giá: "alignment tax".
- **DPO:** học thẳng từ cặp (chọn, loại); **không** reward model riêng, **không** học tăng cường; có model tham chiếu + β (mặc định 0.1).
- **Qwen2.5-Instruct:** 18T token → SFT hơn 1 triệu mẫu → DPO khoảng 150 nghìn cặp → RL trực tuyến (GRPO).
- Base model **không khuyến nghị** để hội thoại (model card).
- **Demo:** cùng prompt trên base và instruct, **cùng `do_sample=False`**. Bản Instruct mặc định lấy mẫu.

### 5W1H
| | |
|---|---|
| **What** | 3 giai đoạn: Pretraining → SFT → Alignment (RLHF / DPO) |
| **Why** | Pretraining cho kiến thức; SFT dạy làm theo chỉ dẫn; alignment vì so sánh hai câu dễ hơn viết câu mẫu hoàn hảo |
| **When** | InstructGPT / RLHF 2022; DPO 2023; Qwen2.5 2024 |
| **Where** | Mọi model "Instruct" hoặc "Chat" |
| **Who** | Ouyang et al. (OpenAI, RLHF); Rafailov et al. (Stanford, DPO); Qwen Team |
| **How** | SFT = cross-entropy trên câu trả lời mẫu; RLHF = reward model + PPO; DPO = loss kiểu phân loại trên cặp. Đây là fine-tuning như [Slide Ch4] |

### Tự kiểm tra
1. SFT khác pretraining ở đâu?
2. Vì sao cần alignment sau SFT?
3. DPO bỏ được gì so với RLHF?
4. Vì sao phạt KL?

---

## Mục 6 — Quá trình sinh văn bản (Phúc)

### Cần nắm
- **Autoregressive:** đoán phân phối → chọn token → nối vào → lặp, giống decoder ở [Slide Ch6 s16]. Mỗi token một lần chạy; có KV cache; lỗi dồn tích.
- Ví dụ tự tính, logits [3, 2, 1.5, 1, 0] → P = 0.56 / 0.21 / 0.13 / 0.08 / 0.03.
- **Greedy** (`do_sample=False`): luôn lấy max. Tất định, nhưng văn bản dài dễ nhạt và lặp (Holtzman 2020; HF docs).
- **Sampling** (`do_sample=True`): đa dạng, nhưng có **"đuôi không đáng tin"** → đôi khi vô nghĩa.
- **Temperature:** chia logits cho T trước softmax.
  - T = 0.5 → P(đẹp) = 0.83 (nhọn);
  - T = 1.5 → 0.44 (phẳng);
  - T thấp giảm đa dạng, tăng lặp.
- **Top-p:** tập nhỏ nhất có tổng xác suất ≥ p.
  - p = 0.9 → giữ 4 token, loại phần đuôi;
  - p = 0.5 → chỉ còn 1 token, **thành greedy**.
- **Top-k:** k cố định, không hợp mọi phân phối.
- **Chú ý:** Qwen-Instruct mặc định `do_sample: true`, `temperature 0.7`, `top_p 0.8`, `top_k 20`, `repetition_penalty 1.1` → mỗi cấu hình phải đặt **đủ** tham số.

### 5W1H
| | |
|---|---|
| **What** | Sinh từng token; chọn token bằng greedy / sampling / temperature / top-p |
| **Why** | Model chỉ cho xác suất; cách chọn quyết định văn bản ổn định hay đa dạng |
| **When** | Lúc dùng model (inference); thí nghiệm mục 10 dùng greedy để kết quả lặp lại được |
| **Where** | `model.generate()` trong `transformers`; `generation_config.json` của model |
| **Who** | Holtzman et al. 2020 (nucleus / top-p sampling) |
| **How** | `do_sample`, `temperature`, `top_p`, `top_k`, `max_new_tokens`; truyền tường minh để ghi đè mặc định |

### Tự kiểm tra
1. Vì sao greedy dễ lặp?
2. Temperature tác động ở bước nào?
3. Top-p khác top-k thế nào?
4. Vì sao top-p = 0.5 có thể giống greedy?

---

## Mục 7 — Khả năng của LLM (TOÀN)

### Cần nắm
- **In-context learning (ICL):** làm tác vụ mới nhờ chỉ dẫn / ví dụ trong prompt, **không cập nhật gradient** (GPT-3, Brown 2020). Là khả năng "nổi lên" của LLM (Zhao survey).
- **Zero-shot** = chỉ chỉ dẫn; **one-shot** = 1 ví dụ; **few-shot** = k ví dụ. Nhóm dùng **3-shot**.
- **Ví dụ giúp nhờ cho thấy** tập nhãn, kiểu câu, định dạng. Đổi nhãn ví dụ thành ngẫu nhiên chỉ giảm nhẹ (Min 2022).
- **Few-shot rất nhạy** với định dạng, việc chọn và thứ tự ví dụ. Các thiên lệch: **majority label bias**, **recency bias**, **common token bias** (Zhao 2021) → nhóm chạy **3 bộ ví dụ**.
- **Làm theo chỉ dẫn** đến từ instruction tuning (FLAN, Wei 2021) và SFT + RLHF (InstructGPT) → dùng bản Instruct, cần **chat template**.
- **Một model, nhiều tác vụ:** phân loại = sinh ra chữ "positive" hoặc "negative", rồi parse.
- **Khác Lab 3 / Practice 3:** ở đó fine-tune model phân loại với nhiều nhãn; ở đây không train, 0 hoặc 3 ví dụ.

### 5W1H
| | |
|---|---|
| **What** | ICL (zero-shot / few-shot), làm theo chỉ dẫn, đa tác vụ |
| **Why** | Không cần dữ liệu nhãn lớn, không cần train; đổi prompt là đổi tác vụ |
| **When** | Nổi bật từ GPT-3 (2020); instruction tuning từ FLAN (2021) |
| **Where** | Phân loại cảm xúc UIT-VSFC (thí nghiệm nhóm); mọi tác vụ dạng văn bản |
| **Who** | Brown et al. (OpenAI); Wei et al. (Google, FLAN); Zhao et al. 2021; Min et al. 2022 |
| **How** | Cùng chỉ dẫn + cùng câu cuối; few-shot thêm 3 khối "Câu: … / Cảm xúc: …"; greedy; parse nhãn |

### Tự kiểm tra
1. Few-shot có đổi trọng số không? Vậy vì sao gọi là "learning"?
2. Vì sao chạy 3 bộ ví dụ?
3. Khả năng làm theo chỉ dẫn đến từ đâu?
4. Khác gì so với fine-tune BERT ở Lab 3?

---

## Mục 8 — Prompting, Fine-tuning, RAG (Muội)

### Cần nắm
| | Prompting | Fine-tuning | RAG |
|---|---|---|---|
| Đổi trọng số | Không | **Có** | Không |
| Dữ liệu | Chỉ dẫn + vài ví dụ | Nhiều mẫu có nhãn | Kho tài liệu (không nhãn) |
| Cập nhật tri thức | Hạn chế | Train lại | Cập nhật kho |
| Truy được nguồn | Không | Không | **Có** |
| Dùng khi | Thử nhanh | Đổi hành vi, kỹ năng | Hỏi trên tài liệu riêng hoặc mới |

- **LoRA:** giữ nguyên trọng số gốc, chỉ thêm ma trận hạng thấp. So với fine-tune GPT-3 175B: số tham số cần học giảm khoảng 10,000 lần, bộ nhớ GPU giảm khoảng 3 lần, không chậm lúc chạy (Hu 2021).
- **RAG** (Lewis 2020):
  - bộ nhớ tham số (trọng số) + bộ nhớ ngoài (kho văn bản);
  - retriever + chỉ mục (Wikipedia, đoạn khoảng 100 từ) + generator;
  - thay chỉ mục là đổi tri thức, không train lại.
- **Câu chốt:** đổi *cách làm* → fine-tune; đổi *thứ model biết* → RAG; thử nhanh → prompting.
- **TF-IDF tiếng Việt** (tự kiểm): mặc định bỏ từ 1 ký tự ("ở"); tách theo âm tiết → dùng `ngram_range=(1, 2)`.

### 5W1H
| | |
|---|---|
| **What** | Ba cách thích nghi LLM với việc của mình |
| **Why** | Model không biết tài liệu riêng hoặc tin mới; fine-tune tốn kém và phải train lại khi cập nhật |
| **When** | Prompting: luôn thử trước; fine-tune: cần đổi hành vi và có dữ liệu; RAG: hỏi đáp trên tài liệu |
| **Where** | Demo: hỏi về slide môn học (15–20 câu), không RAG vs có RAG |
| **Who** | Lewis et al. (Meta AI, RAG, 2020); Hu et al. (Microsoft, LoRA, 2021) |
| **How** | Chia đoạn → TF-IDF → tìm k đoạn → ghép vào prompt → LLM trả lời; đo số câu đúng và tỉ lệ tìm đúng đoạn |

### Tự kiểm tra
1. Vì sao không fine-tune để model biết slide môn học?
2. LoRA tiết kiệm nhờ đâu?
3. RAG có hết hallucination không?

---

## Mục 9 — Hạn chế và đánh giá (Phúc)

### Cần nắm
- **Hallucination** (Huang 2023):
  - **Factuality:** sai sự thật (contradiction), bịa (fabrication).
  - **Faithfulness:** lệch chỉ dẫn, lệch ngữ cảnh, mâu thuẫn logic.
  - **Nguyên nhân:** dữ liệu (ngộ nhận), huấn luyện (ép trả lời), lúc sinh (ngẫu nhiên cao).
  - **Gốc rễ:** model tối ưu "xác suất cao", không tối ưu "đúng".
- **TruthfulQA:**
  - 817 câu, 38 chủ đề;
  - model tốt nhất trung thực khoảng 58%, người khoảng 94%;
  - **model lớn nhất thường kém trung thực nhất**.
- **Thiên lệch:**
  - embedding học từ Google News đã mang định kiến giới về nghề nghiệp (Bolukbasi 2016);
  - học máy có thể **khuếch đại** định kiến;
  - alignment giảm được, không xoá hẳn.
- **Giới hạn ngữ cảnh:**
  - cứng: 32,768 token (Qwen);
  - mềm: **lost in the middle**, hình chữ U, thông tin ở giữa dễ bị bỏ sót, kể cả với model ngữ cảnh dài (Liu 2023).
- **Phụ thuộc tác vụ và dữ liệu:**
  - nhãn UIT-VSFC đồng thuận trên 91% → có nhãn không chắc;
  - model 1.5B → không suy rộng.
- **Kiểm chứng:**
  - tách khẳng định → đối chiếu nguồn gốc → kiểm trích dẫn;
  - hỏi lại kiểu khác → dùng RAG → tính toán bằng công cụ.

### 5W1H
| | |
|---|---|
| **What** | Hallucination, thiên lệch, giới hạn ngữ cảnh, phụ thuộc tác vụ / dữ liệu |
| **Why** | Model học xác suất từ dữ liệu, gồm cả lỗi và định kiến; ngữ cảnh có giới hạn |
| **When** | Bất cứ khi nào dùng output như sự thật, nhất là câu hỏi hiếm, có tiền đề sai, xin trích dẫn, tính toán |
| **Where** | Demo: ~20 câu có đáp án; prompt nghề nghiệp; sự thật đặt ở đầu / giữa / cuối văn bản dài |
| **Who** | Huang et al. 2023; Lin et al. 2022 (TruthfulQA); Bolukbasi et al. 2016; Liu et al. 2023 |
| **How** | Chạy greedy, chấm đúng / sai / từ chối; đếm; chọn 1 câu sai thật + nguồn đối chiếu lên slide |

### Tự kiểm tra
1. Factuality khác faithfulness ở đâu?
2. Vì sao model to hơn chưa chắc trung thực hơn?
3. "Lost in the middle" là gì?
4. Kiểm chứng một câu trả lời thế nào?

---

## Mục 10 — Thực nghiệm có kiểm soát và kết luận (TOÀN)

### Cần nắm
- **Câu hỏi:** 3-shot có làm accuracy khác zero-shot **có ý nghĩa thống kê** không? H0: tỉ lệ sai như nhau.
- **Dữ liệu UIT-VSFC** (Nguyen et al., KSE 2018):
  - 3 nhãn → **bỏ neutral**;
  - split test: neg 1,409 / neu 167 / pos 1,590; split train: neg 5,325 / neu 458 / pos 5,643.
- **Tập đánh giá:** 500 câu **từ split test**, 250/250, seed 42. **Ví dụ few-shot từ split train**, kiểm trùng câu.
- **Tải dữ liệu:** `pandas.read_parquet` từ nhánh Parquet; script loader lỗi với `datasets` ≥ 4.0.
- **Chỉ đổi 1 biến** (có ví dụ hay không). Cố định: model, prompt, greedy (`do_sample=False`), dữ liệu, seed.
- **4 cấu hình:** zero-shot + 3-shot × 3 bộ. Bộ 1 (P, N, P) và bộ 2 (N, P, N) đối xứng; bộ 3 ngẫu nhiên. Tổng 2,000 lần sinh.
- **Đánh giá:**
  - accuracy (câu không parse được **tính là sai** + báo riêng tỉ lệ);
  - confusion matrix;
  - **McNemar** (b = chỉ A đúng, c = chỉ B đúng; `exact=True`).
- **Sai số 500 câu:** khoảng ±4.4 điểm ở p = 50%, ±3.5 ở 80% → chênh vài điểm chưa đủ kết luận.
- **Vì sao McNemar:** với thuật toán chạy được một lần, đây là kiểm định duy nhất có sai lầm loại I chấp nhận được (Dietterich 1998). 3 lần kiểm định → cân nhắc Bonferroni 0.05/3.
- **Phân tích lỗi:** đọc câu sai, xếp nhóm (vừa khen vừa chê, mỉa mai, quá ngắn, viết tắt, nhãn có thể sai, lỗi định dạng); ví dụ phải là **câu thật**.
- **Tổng hợp:** bảng kết quả của mọi mục, lấy từ CSV.
- **Giới hạn:** 1 model nhỏ, 1 bộ dữ liệu, 1 tác vụ, mỗi cấu hình 1 lần, k = 3.

### 5W1H
| | |
|---|---|
| **What** | Thí nghiệm zero-shot vs 3-shot trên 500 câu, có kiểm định thống kê; tổng hợp kết quả toàn bài |
| **Why** | Vài ví dụ minh hoạ không đủ kết luận; số trên slide phải có căn cứ, truy được về một lần chạy (bài học Lab 1) |
| **When** | Chạy chính thức một lần Save & Run All, sau khi chốt prompt trên split validation |
| **Where** | Kaggle GPU T4; `outputs/results/icl_*.csv`, `icl_metrics.json`; `docs/experiments/error_analysis_v1.md` |
| **Who** | Toàn chạy; dữ liệu của Nguyen et al. (UIT); phương pháp kiểm định của Dietterich |
| **How** | `make_test_split` → `make_fewshot_pools` → `build_prompt` → `run_icl` (greedy) → `parse_label` → `compute_metrics` → `mcnemar_counts` / `mcnemar_pvalue` |

### Tự kiểm tra
1. Vì sao cân bằng 250/250?
2. Vì sao lấy ví dụ few-shot từ train?
3. b, c trong McNemar là gì? Vì sao không dùng t-test?
4. Câu không parse được xử lý thế nào, vì sao?
5. Chênh 2 điểm accuracy có kết luận được không?

---

## Sợi chỉ nối 10 mục (để trả lời câu hỏi chéo)

| Nếu cô hỏi | Nối các mục |
|---|---|
| Vì sao LLM bịa? | 4 (học xác suất, không học đúng sai) → 6 (sampling dễ bốc token lạ) → 9 (hallucination) → 8 (RAG giảm bịa) |
| Causal mask dùng ở đâu? | 2 (định nghĩa) → 4 (học song song n − 1 cặp) → 6 (sinh từng token, KV cache) |
| Bản Instruct khác base ở đâu? | 5 (SFT + alignment) → 7 (làm theo chỉ dẫn) → 6 (`generation_config` mặc định khác) |
| Vì sao dùng greedy cho thí nghiệm? | 6 (greedy tất định) → 10 (kết quả lặp lại được, so sánh công bằng) |
| Tiếng Việt có bất lợi gì? | 3 (nhiều byte, có thể nhiều token) → 9 (giới hạn ngữ cảnh) → 10 (kết quả không suy rộng) |
| Few-shot khác fine-tune? | 7 (không đổi trọng số) → 8 (fine-tune đổi trọng số) → 5 (SFT là fine-tune) |
