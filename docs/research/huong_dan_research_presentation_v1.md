# Hướng dẫn tự research toàn bộ bài Presentation LLMs (v1)

- Dành cho: Toàn (phụ trách mục 7, 10; ghép notebook) | Tạo: 2026-11-15
- Mục đích: tự research từ đầu tới cuối theo các bước, biết **phải nắm gì**, **hiểu tới mức nào**, và **khi nào là đủ**.
- Tài liệu đã có, dùng làm đáp án để đối chiếu **sau khi** tự làm:
  - `nam_kien_thuc_5w1h_v1.md`: tóm tắt 10 mục theo 5W1H;
  - `noi_dung_mucXX_*_v1.md`: chi tiết từng mục;
  - `noi_dung_toan_muc07_muc10_v1.md`: phần của Toàn.

---

## 0. Ba mức "nắm" và baseline

### 0.1 Ba mức hiểu
| Mức | Tên | Làm được gì | Áp dụng cho |
|---|---|---|---|
| **M1** | Biết | Nói được định nghĩa và 1 ví dụ cho mỗi khái niệm | Mục của người khác (1–6, 8, 9) |
| **M2** | Hiểu | Giải thích **vì sao**, nối được với mục khác, trả lời câu hỏi chéo | Toàn bộ 10 mục |
| **M3** | Bảo vệ được | Trả lời 5W1H; giải thích từng dòng code; nói rõ giới hạn; dẫn nguồn | **Mục 7, 10 và code của mình** |

### 0.2 Baseline (chuẩn tối thiểu phải đạt trước buổi thuyết trình)
- [ ] Mục 1–6, 8, 9: đạt **M1**. Trả lời được câu "Tự kiểm tra" trong `nam_kien_thuc_5w1h_v1.md` mà không nhìn tài liệu.
- [ ] Cả 10 mục: đạt **M2** ở các câu trong bảng "Sợi chỉ nối 10 mục" (cuối `nam_kien_thuc_5w1h_v1.md`).
- [ ] Mục 7, 10: đạt **M3**:
  - nói 1 phút không cần slide cho mỗi mục;
  - trả lời được 12 câu Q&A trong `noi_dung_toan_muc07_muc10_v1.md` §D;
  - giải thích từng hàm trong bảng §E.
- [ ] Mỗi con số mình nói trên slide đều chỉ ra được **nguồn**: bài báo, model card, hoặc file kết quả.

### 0.3 Kiến thức nền cần có trước (từ môn học)
Thiếu ý nào thì ôn lại trước khi vào Bước 1. Ghi chú môn học nằm ở `Research/ch0X-*/`; OCR slide ở `Research/slides-ocr/`.

| Cần biết | Ở đâu trong môn | Kiểm nhanh |
|---|---|---|
| Softmax, cross-entropy, phân loại nhiều lớp | Ch2 (slide 7, 9, 11–13) | Tính được −log(0.5) |
| Train / validation / test, rò dữ liệu | Ch3 | Nói được vì sao không chọn cấu hình trên tập test |
| Fine-tuning, transfer learning | Ch4 | Phân biệt freeze và fine-tune |
| RNN, encoder-decoder, embedding, padding | Ch6 (slide 16, 17) | Nói được context vector cố định là gì |
| Hugging Face `pipeline`, `Trainer`, tokenizer | Practice 3, Lab 3 | Biết tokenizer phải khớp model |

---

## 1. Cách research một chủ đề (dùng lại ở mọi bước)

### 1.1 Vòng lặp 5 bước
1. **Hỏi trước:** viết ra 3–5 câu hỏi mình cần trả lời (lấy từ đề của cô và từ 5W1H).
2. **Đọc nguồn theo thứ tự tin cậy:**
   - slide môn;
   - tài liệu chính thức (model card, docs của thư viện);
   - bài báo gốc;
   - bài giải thích (Illustrated GPT-2, HF course);
   - blog hoặc forum: chỉ để tìm hướng, không trích.
3. **Ghi chú bằng lời của mình**, theo khuôn §1.3; mỗi ý kèm nguồn.
4. **Kiểm chứng:** con số thì mở nguồn gốc đối chiếu; hành vi của code thì chạy thử.
5. **Tự kiểm:** gập tài liệu lại, trả lời các câu hỏi ở bước 1. Chưa trả lời được thì quay lại bước 2.

### 1.2 Cách đọc một bài báo trong 20–30 phút
1. **Abstract:** bài giải quyết vấn đề gì, kết quả chính là gì.
2. **Hình 1 và bảng kết quả chính:** chỉ ra được con số quan trọng.
3. **Phần phương pháp:** chỉ đọc đoạn liên quan tới câu hỏi của mình.
4. **Phần kết luận hoặc giới hạn.**
5. Ghi **1 câu tóm tắt + 1 con số + 1 giới hạn**.

Không cần đọc hết bài.

### 1.3 Khuôn ghi chú (mỗi khái niệm một khối)
```
## <Khái niệm>
- What (định nghĩa 1 câu, lời của mình):
- Why (vì sao cần / giải quyết vấn đề gì):
- When (khi nào dùng / ra đời năm nào):
- Where (dùng ở đâu trong bài nhóm: mục, slide, notebook):
- Who (ai đề xuất / ai dùng):
- How (làm thế nào: công thức, các bước, hàm code):
- Con số đáng nhớ + nguồn:
- Câu cô có thể hỏi + ý trả lời:
- Nguồn: [slide ChX sN] / [bài báo, năm] / [docs]
```

### 1.4 Quy tắc không được phá
- Không ghi số nếu chưa mở nguồn gốc. Số nghe từ AI (kể cả từ mình) là **chưa kiểm**.
- Tách **"từ slide"** và **"bổ sung"** trong ghi chú.
- Kết quả của nhóm chỉ lấy từ `outputs/results/` của lần chạy chính thức.

---

## 2. Lộ trình từng bước

Thời gian là ước lượng cho một người đọc lần đầu.

### Bước 1 — Bức tranh tổng (30 phút)
- **Mục tiêu:** biết 10 mục nối với nhau thế nào trước khi đi sâu.
- **Đọc:**
  - đề của cô;
  - `nam_kien_thuc_5w1h_v1.md`: chỉ đọc phần "Cần nắm" của 10 mục và bảng "Sợi chỉ";
  - HF LLM Course chương 1 (https://huggingface.co/learn/llm-course).
- **Phải nắm:** chuỗi "LLM là gì → kiến trúc → token → cách học → giai đoạn huấn luyện → cách sinh → khả năng → thích nghi → hạn chế → thực nghiệm".
- **Kiểm:** vẽ lại chuỗi 10 mục bằng tay, mỗi mục 1 dòng.

### Bước 2 — Nền tảng: mục 1–4 (2–3 giờ), mức M1–M2
| Mục | Câu hỏi phải trả lời | Đọc gì (theo thứ tự) |
|---|---|---|
| 1 | LM, LLM, chatbot khác nhau ở đâu? "Large" là bao nhiêu? | Slide Ch1 s10, s12 → Jurafsky & Martin ch. 3 (trang đầu) → abstract và mục 2.1 survey Zhao et al. 2023 → bài "Introducing ChatGPT" của OpenAI |
| 2 | Vì sao Transformer thay RNN? Causal mask là gì, vì sao gán −∞? | Slide Ch6 s16 → *The Illustrated GPT-2* (phần masked self-attention) → Vaswani 2017 §3.2 (công thức attention) → model card Qwen2.5-1.5B |
| 3 | Token, token ID, embedding; BPE; vì sao tiếng Việt có thể tốn token hơn? | Slide Ch6 s17 → HF *Summary of the tokenizers* (phần BPE, byte-level BPE) |
| 4 | Next-token prediction; vì sao là cross-entropy; perplexity | Slide Ch2 s7, s13 → HF *Perplexity of fixed-length models* → video Karpathy *Let's build GPT* (khoảng 30 phút đầu, phần tạo cặp ngữ cảnh và đích) |

- **Làm tay:**
  - tính softmax và −log p cho 3–4 số;
  - viết các cặp ngữ cảnh → đích cho câu "Hôm nay trời rất đẹp.".
- **Kiểm:** trả lời câu "Tự kiểm tra" của mục 1–4; đối chiếu với `noi_dung_muc01..04`.

### Bước 3 — Huấn luyện và sinh: mục 5–6 (1.5–2 giờ), mức M1–M2
| Mục | Câu hỏi phải trả lời | Đọc gì |
|---|---|---|
| 5 | Pretraining, SFT, alignment khác nhau về mục tiêu và dữ liệu thế nào? RLHF và DPO khác gì? | Abstract + Hình 2 của InstructGPT (Ouyang 2022) → abstract DPO (Rafailov 2023) → phần post-training của Qwen2.5 Technical Report |
| 6 | Greedy, temperature, top-p hoạt động thế nào? Vì sao greedy dễ lặp? | HF *Generation strategies* → abstract + §3 của Holtzman 2020 → `generation_config.json` của Qwen2.5-1.5B-Instruct |

- **Làm tay:** với logits [3, 2, 1.5, 1, 0], tính softmax ở T = 0.5, 1, 1.5, và tập top-p với p = 0.9.
- **Kiểm:** giải thích vì sao nhóm phải truyền `do_sample=False`.

### Bước 4 — Phần của mình: mục 7 (2–3 giờ), mức M3
- **Câu hỏi:**
  - ICL là gì, vì sao không đổi trọng số mà vẫn "học"?
  - Zero-shot khác few-shot thế nào?
  - Vì sao few-shot nhạy với ví dụ?
  - Làm theo chỉ dẫn đến từ đâu?
- **Đọc theo thứ tự:**
  1. GPT-3 (Brown 2020): abstract + phần định nghĩa zero/one/few-shot ở mục 2 + Hình 2.1.
  2. Zhao et al. 2021 *Calibrate Before Use*: abstract + Hình 1 (độ dao động theo ví dụ, thứ tự) + đoạn mô tả majority label bias, recency bias.
  3. Min et al. 2022 *Rethinking the Role of Demonstrations*: abstract + hình chính (nhãn ngẫu nhiên vs nhãn đúng).
  4. FLAN (Wei 2021): abstract.
  5. HF *Chat templates*: phần `add_generation_prompt`.
- **Làm tay:** tự viết prompt zero-shot và 3-shot cho 1 câu phản hồi sinh viên; chỉ ra chỗ khác nhau duy nhất.
- **Kiểm (M3):**
  - nói 1 phút về mục 7 không nhìn tài liệu;
  - trả lời Q&A số 1–4 và 11 trong `noi_dung_toan_muc07_muc10_v1.md` §D.

### Bước 5 — Mục 8–9 (1.5 giờ), mức M1–M2
| Mục | Câu hỏi | Đọc gì |
|---|---|---|
| 8 | Prompting, fine-tuning, RAG cần dữ liệu gì, dùng khi nào? LoRA là gì? | Abstract RAG (Lewis 2020) → abstract LoRA (Hu 2021) → docs `TfidfVectorizer` |
| 9 | Hallucination có những loại nào, vì sao xảy ra? Thiên lệch? Lost in the middle? | Phần phân loại trong survey Huang 2023 → abstract TruthfulQA (Lin 2022) → abstract Liu 2023 → abstract Bolukbasi 2016 |

- **Kiểm:** trả lời các câu trong bảng "Sợi chỉ" có dính mục 8–9.

### Bước 6 — Phần của mình: mục 10 (3–4 giờ), mức M3
- **Câu hỏi:**
  - Thí nghiệm có kiểm soát là gì?
  - Vì sao 250/250?
  - Vì sao ví dụ few-shot lấy từ train?
  - Accuracy có sai số bao nhiêu với 500 câu?
  - McNemar là gì, vì sao dùng?
  - Phân tích lỗi thế nào?
- **Đọc theo thứ tự:**
  1. Trang dataset UIT-VSFC trên Hugging Face: số câu, 3 nhãn, cột dữ liệu.
  2. Bài báo UIT-VSFC (KSE 2018): abstract + phần độ đồng thuận nhãn.
  3. Dietterich 1998: abstract + phần mô tả McNemar (bảng 2×2; vì sao hợp khi chỉ chạy một lần).
  4. Docs `statsmodels` `mcnemar`: tham số `exact`.
  5. Ôn Ch3: tách tập, không chọn trên tập test.
- **Làm tay** (bắt buộc ở mức M3):
  - Tự tính sai số 95% với n = 500, p = 0.5 → khoảng ±4.4 điểm.
  - Tự lập bảng McNemar với dữ liệu giả 10 câu, đếm b, c, rồi chạy `mcnemar` kiểm lại.
  - Tự đọc khoảng 50 câu UIT-VSFC để biết các kiểu câu khó (vừa khen vừa chê, viết tắt…).
- **Kiểm (M3):**
  - trả lời Q&A số 5–10 và 12;
  - giải thích được bảng 10.2 (cố định) và 10.3 (3 bộ ví dụ) trong `bao_cao_toan_muc07_muc10_v1.md`.

### Bước 7 — Research kỹ thuật cho code (2 giờ, làm song song lúc code)
| Câu hỏi | Kiểm bằng |
|---|---|
| Tải Qwen và tokenizer bằng `AutoTokenizer` / `AutoModelForCausalLM` | Docs Transformers + model card; chạy thử |
| `generation_config` mặc định của Instruct là gì | In `model.generation_config` |
| `apply_chat_template` dùng đúng thế nào, có lặp token đặc biệt không | Docs *Chat templates*; in ra token ID của prompt |
| Cắt phần model **mới sinh** ra khỏi output trước khi parse | Docs `generate()`; thử với 1 câu |
| Chạy từng câu hay theo batch; batch thì padding bên nào | Docs Transformers về padding cho model decoder-only; so output batch với từng câu |
| Đọc UIT-VSFC bằng Parquet | Chạy `pd.read_parquet` |
| fp16 hay fp32 trên T4; thời gian 2,000 lần sinh | Chạy thử 10 câu, đo thời gian, so output |

Mỗi câu trả lời ghi vào `progress_status.md` kèm lệnh và output thật.

### Bước 8 — Tổng duyệt (1–2 giờ)
- [ ] Đi lại baseline §0.2 từng ô.
- [ ] Nhờ Thư hỏi ngẫu nhiên 10 câu ở cả 10 mục.
- [ ] Nói thử mục 7 (1 phút) và mục 10 (2.5 phút), bấm giờ.
- [ ] Mở notebook, giải thích ngẫu nhiên 5 dòng code trong `icl_experiment.py`.

---

## 3. Bảng theo dõi tiến độ research

| Bước | Nội dung | Mức cần đạt | Ước lượng | Xong |
|---|---|---|---|---|
| 1 | Bức tranh tổng | — | 30' | [ ] |
| 2 | Mục 1–4 | M1–M2 | 2–3 h | [ ] |
| 3 | Mục 5–6 | M1–M2 | 1.5–2 h | [ ] |
| 4 | **Mục 7** | **M3** | 2–3 h | [ ] |
| 5 | Mục 8–9 | M1–M2 | 1.5 h | [ ] |
| 6 | **Mục 10** | **M3** | 3–4 h | [ ] |
| 7 | Kỹ thuật code | Chạy được | 2 h | [ ] |
| 8 | Tổng duyệt | Baseline §0.2 | 1–2 h | [ ] |

Tổng khoảng 14–18 giờ. Nếu thiếu thời gian: giữ Bước 4, 6, 7 đầy đủ, rút Bước 2, 3, 5 xuống chỉ đọc "Cần nắm" trong `nam_kien_thuc_5w1h_v1.md`.

---

## 4. Danh sách nguồn theo thứ tự đọc

| Bước | Nguồn | Link |
|---|---|---|
| 1 | HF LLM Course, chương 1 | https://huggingface.co/learn/llm-course |
| 2 | Jurafsky & Martin, SLP3 ch. 3 | https://web.stanford.edu/~jurafsky/slp3/3.pdf |
| 2 | Zhao et al., *A Survey of LLMs* | https://arxiv.org/abs/2303.18223 |
| 2 | OpenAI, *Introducing ChatGPT* | https://openai.com/blog/chatgpt/ |
| 2 | Alammar, *The Illustrated GPT-2* | https://jalammar.github.io/illustrated-gpt2/ |
| 2 | Vaswani et al. 2017 | https://arxiv.org/abs/1706.03762 |
| 2 | HF *Summary of the tokenizers* | https://huggingface.co/docs/transformers/tokenizer_summary |
| 2 | HF *Perplexity* | https://huggingface.co/docs/transformers/perplexity |
| 2 | Karpathy, *Let's build GPT* | https://www.youtube.com/watch?v=kCc8FmEb1nY |
| 3 | Ouyang et al. 2022 (InstructGPT) | https://arxiv.org/abs/2203.02155 |
| 3 | Rafailov et al. 2023 (DPO) | https://arxiv.org/abs/2305.18290 |
| 3 | Qwen2.5 Technical Report | https://arxiv.org/abs/2412.15115 |
| 3 | HF *Generation strategies* | https://huggingface.co/docs/transformers/generation_strategies |
| 3 | Holtzman et al. 2020 | https://arxiv.org/abs/1904.09751 |
| 4 | Brown et al. 2020 (GPT-3) | https://arxiv.org/abs/2005.14165 |
| 4 | Zhao et al. 2021 (*Calibrate Before Use*) | https://arxiv.org/abs/2102.09690 |
| 4 | Min et al. 2022 | https://arxiv.org/abs/2202.12837 |
| 4 | Wei et al. 2021 (FLAN) | https://arxiv.org/abs/2109.01652 |
| 4 | HF *Chat templates* | https://huggingface.co/docs/transformers/chat_templating |
| 5 | Lewis et al. 2020 (RAG) | https://arxiv.org/abs/2005.11401 |
| 5 | Hu et al. 2021 (LoRA) | https://arxiv.org/abs/2106.09685 |
| 5 | Huang et al. 2023 (hallucination) | https://arxiv.org/abs/2311.05232 |
| 5 | Lin et al. 2022 (TruthfulQA) | https://arxiv.org/abs/2109.07958 |
| 5 | Liu et al. 2023 (*Lost in the Middle*) | https://arxiv.org/abs/2307.03172 |
| 5 | Bolukbasi et al. 2016 | https://arxiv.org/abs/1607.06520 |
| 6 | UIT-VSFC trên Hugging Face | https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback |
| 6 | Nguyen et al. 2018 (UIT-VSFC) | doi:10.1109/KSE.2018.8573337 |
| 6 | Dietterich 1998 | doi:10.1162/089976698300017197 |
| 6 | statsmodels `mcnemar` | https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html |
| 7 | Model card Qwen2.5-1.5B-Instruct | https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct |
