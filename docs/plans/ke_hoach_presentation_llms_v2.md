# Kế hoạch Presentation LLMs – Nhóm DL (v2)

Môn: Deep Learning · Chủ đề: Tìm hiểu mô hình ngôn ngữ lớn (LLMs)
Cập nhật: 2026-11-14 (thêm yêu cầu của cô về thư viện, môi trường, demo và thực nghiệm trên slide, giải thích code)

> **v2 (2026-11-16T01:40:17+07:00, Toàn, AI hỗ trợ):** sửa theo quyết định 15/11 **Toàn viết toàn bộ code notebook (phần 0–10)**; Muội, Phúc viết dàn ý desc, lý thuyết, slide, thuyết trình mục mình và nhận bàn giao code. Thêm so sánh LLM với RNN/LSTM, BERT, VLM vào mục 2 (16/11). v1 giữ nguyên, không dùng. Lịch giữ nguyên (Toàn chốt 16/11). **Duyệt: Toàn, 2026-11-16**.

**Ngày 1** = ngày bắt đầu làm: **15/11/2026** (Toàn chốt 16/11). Mỗi người tự làm đúng hạn của mình.

---

## 1. Thông tin chung

| Hạng mục | Nội dung |
|---|---|
| Mục tiêu của cô | Giải thích cách LLM học và sinh văn bản, phân biệt các giai đoạn huấn luyện, nhận biết khả năng và hạn chế |
| Thời lượng trình bày | 15 đến 20 phút (khoảng 1 đến 2 slide mỗi mục) |
| Thời gian làm | 1 tuần |
| Vấn đáp | Mỗi người chuẩn bị khoảng 5 câu Q&A cho phần mình |
| Demo | Notebook đã chạy sẵn có output |
| Góp ý của cô (11/11) | Không cần fine-tune, nhưng phải có demo và phân tích model thực tế: mọi mục đều có demo trên model thật kèm phân tích output, kể cả mục 5 và 8 |
| Yêu cầu thêm của cô | (1) Slide nêu rõ **thư viện sử dụng** và (2) **môi trường chạy**; (3) mỗi mục có **demo trên slide** (output thật); (4) slide phải có **thực nghiệm** (thiết lập, kết quả, phân tích); (5) **giải thích được từng dòng** trong notebook; (6) phần nào slide không có code thì **mở notebook trình bày code** |
| Đang chờ cô xác nhận | Cách demo mục 5 (base so với instruct) và mục 8 (không RAG so với có RAG); nghĩa của "phân tích model thực tế"; dùng bộ dữ liệu UIT-VSFC |

**Sản phẩm nộp:** (1) Slide PDF hoặc PPTX · (2) Notebook kèm hướng dẫn chạy · (3) Bộ câu thử nghiệm, kết quả và phân tích lỗi · (4) Danh sách tài liệu tham khảo.

---

## 2. Dàn ý 10 mục của cô

| # | Mục | Nội dung cần trình bày | Yêu cầu minh họa |
|---|---|---|---|
| 1 | Giới thiệu | LLM là gì? Phân biệt language model, LLM và ứng dụng chatbot | Ví dụ các tác vụ: hỏi đáp, tóm tắt, dịch, sinh mã |
| 2 | Nền tảng kiến trúc | Vai trò của Transformer; tập trung vào Decoder-only và causal attention | Sơ đồ từ token đầu vào đến xác suất token tiếp theo |
| 3 | Tokenization | Cách văn bản được chia thành token; token ID và embedding | Tokenize một câu tiếng Việt và một câu tiếng Anh |
| 4 | Nguyên lý học | Next-token prediction, dữ liệu huấn luyện và cross-entropy loss | Tạo các cặp ngữ cảnh và token mục tiêu từ một câu |
| 5 | Các giai đoạn huấn luyện | Pretraining, SFT và preference alignment; giới thiệu RLHF/DPO ở mức ý tưởng | Sơ đồ mục tiêu và dữ liệu của từng giai đoạn |
| 6 | Quá trình sinh văn bản | Autoregressive generation; greedy decoding, temperature và top-p sampling | Cùng một prompt với các cấu hình sinh khác nhau |
| 7 | Khả năng của LLM | In-context learning, làm theo chỉ dẫn, xử lý các tác vụ ngôn ngữ | So sánh zero-shot và few-shot trên cùng tác vụ |
| 8 | Cách thích nghi và bổ sung tri thức | Phân biệt prompting, fine-tuning và RAG | Bảng so sánh dữ liệu cần có và trường hợp sử dụng |
| 9 | Hạn chế và đánh giá | Hallucination, thiên lệch, giới hạn ngữ cảnh; chất lượng phụ thuộc tác vụ và dữ liệu | Một câu trả lời sai và cách kiểm chứng |
| 10 | Demo và kết luận | Thử nghiệm có kiểm soát, tổng hợp kết quả và nhận xét | Bảng kết quả, ví dụ đầu ra và phân tích lỗi |

---

## 3. Phân công

| Người | Mục lý thuyết | Phần notebook | Ghi chú |
|---|---|---|---|
| **Muội** | 1, 2 (gồm so sánh LLM với mô hình khác), 3, 8; ghép slide | — (nhận bàn giao code phần 1, 2, 3, 8); README | Demo RAG là phần nặng nhất |
| **Phúc** (Member 2) | 4, 5, 6, 9 | — (nhận bàn giao code phần 4, 5, 6, 9) | Soạn bộ câu mục 9 |
| **Toàn** (Member 3) | 7, 10 | **Phần 0–10** (viết toàn bộ code) | Thí nghiệm chính; smoke run, normal run, log; bàn giao code |
| **Thư** (hỗ trợ) | Review, hỏi Q&A thử, bấm giờ | Không nhận phần notebook nào | Không nhận mục nào |

Người làm chính và thuyết trình: **Muội, Phúc, Toàn**; mỗi người trình bày mục của mình, kể cả mở notebook giải thích code phần mình trình bày (Toàn viết code và bàn giao trước). Thư chỉ hỗ trợ (review, hỏi Q&A thử, bấm giờ), không nhận mục nào và không thuyết trình chính. Cách chia vai (ai làm mục nào) sẽ do một bạn trong nhóm chốt; bảng trên là đề xuất ban đầu, có thể đổi.

---

## 4. Phương án demo (dự kiến, chạy thử trước khi chốt)

Cả nhóm dùng chung 1 model, 1 môi trường, 1 câu ví dụ để kết quả các mục khớp nhau. Mỗi output demo kèm 2 đến 3 câu phân tích: quan sát thấy gì, vì sao, liên hệ lý thuyết nào.

| Hạng mục | Phương án |
|---|---|
| Model chính | `Qwen/Qwen2.5-1.5B-Instruct` |
| Model so sánh (mục 5) | `Qwen/Qwen2.5-1.5B` (bản base) |
| Dự phòng | `Qwen2.5-0.5B-Instruct` nếu chạy chậm; `Qwen2.5-3B-Instruct` nếu kết quả quá kém |
| Tokenizer phụ (mục 3) | `gpt2` |
| Môi trường | Kaggle Notebook, GPU T4. Phiên bản Python, CUDA, tên GPU, RAM: in ra ở phần 0 trong lần chạy thật rồi chép lên slide 2 (không ghi theo trí nhớ) |
| Thư viện (dự kiến) | `torch`, `transformers`, `accelerate` (load model lên GPU); `datasets` hoặc `pandas` (đọc UIT-VSFC); `scikit-learn` (accuracy, confusion matrix, TF-IDF cho RAG); `statsmodels` (McNemar); `matplotlib` (vẽ confusion matrix, causal mask). Tùy chọn: `sentence-transformers` nếu RAG dùng embedding. Phiên bản: ghim trong phần 0, lấy từ `pip freeze` lần chạy thật |
| Câu ví dụ (mục 2, 3, 4, 6) | "Hôm nay trời rất đẹp." / "The weather is very nice today." |

**Thí nghiệm có kiểm soát (mục 7, 10):** phân loại cảm xúc tiếng Việt bằng UIT-VSFC (2 nhãn positive/negative; dự phòng Rotten Tomatoes). Test set 500 câu (250/250), seed cố định. So zero-shot với 3-shot trên cùng 500 câu, chạy 3 bộ ví dụ few-shot khác nhau. Greedy decoding, cùng prompt template. Metric: accuracy, confusion matrix, tỉ lệ output không parse được, McNemar test. Lưu CSV gồm câu, nhãn thật, prompt, output, dự đoán.

**Demo cấu hình sinh (mục 6):** 1 prompt tiếng Việt, 5 cấu hình (greedy; temp 0.7 + top-p 0.9; temp 1.0; temp 1.5; top-p 0.5), mỗi cấu hình 5 mẫu.

**Demo RAG (mục 8):** 15 đến 20 câu hỏi về nội dung slide môn học; so hỏi thẳng với hỏi kèm đoạn slide liên quan (TF-IDF hoặc embedding); đếm số câu đúng mỗi cách.

**Mục 9:** hallucination (khoảng 20 câu có đáp án kiểm chứng được), thiên lệch (prompt về nghề nghiệp), giới hạn ngữ cảnh (thông tin ở đầu văn bản dài), nhãn nhiễu trong UIT-VSFC.

**Phạm vi:** không fine-tune; phần huấn luyện (mục 4, 5) minh họa bằng inference.

---

## 5. Khung notebook

Một notebook chung, **Toàn viết toàn bộ code** (quyết định 15/11). Cột dưới là người phụ trách lý thuyết + thuyết trình.

| Phần | Nội dung | Lý thuyết + thuyết trình (code: Toàn) |
|---|---|---|
| 0. Setup | Cài thư viện (ghim phiên bản), seed, load tokenizer và 2 model một lần | Toàn (lý thuyết + code, chốt 16/11) |
| 1. Tác vụ mẫu (mục 1) | 4 tác vụ trên bản instruct | Muội |
| 2. Kiến trúc (mục 2) | In config, causal mask, top-5 token tiếp theo; so sánh LLM với RNN/LSTM, BERT, VLM (có demo) | Muội |
| 3. Tokenization (mục 3) | Tokenize Việt và Anh, so GPT-2 và Qwen, embedding | Muội |
| 4. Nguyên lý học (mục 4) | Cặp ngữ cảnh và token mục tiêu, tính loss | Phúc |
| 5. Base so với instruct (mục 5) (chờ cô) | Cùng bộ prompt trên 2 bản | Phúc |
| 6. Sinh văn bản (mục 6) | Sinh từng bước, 5 cấu hình decoding | Phúc |
| 7. Zero-shot và few-shot (mục 7, 10) (chờ cô) | Pipeline tham số hóa, chạy 500 câu, metric, McNemar, lưu CSV | Toàn |
| 8. RAG (mục 8) (chờ cô) | Chia slide thành đoạn, tìm đoạn liên quan, so không RAG và có RAG | Muội |
| 9. Hạn chế (mục 9) | Hallucination, thiên lệch, giới hạn ngữ cảnh | Phúc |
| 10. Tổng hợp (mục 10) | Bảng tổng hợp kết quả các phần | Toàn |
| Hướng dẫn chạy | README: chạy trên Kaggle, thời gian chạy dự kiến | Muội |

**Quy ước viết notebook (để giải thích được từng dòng):**
- Mỗi cell code có 1 cell markdown đứng trước: mục đích, đầu vào, đầu ra, liên hệ mục lý thuyết nào.
- Mỗi dòng code không hiển nhiên có comment tiếng Việt ngắn (vd `# lấy logits của token cuối để dự đoán token tiếp theo`).
- Mỗi output có 1 cell markdown "Phân tích" ngay sau: 2 đến 3 câu (quan sát gì, vì sao, liên hệ lý thuyết nào).
- Không dùng hàm hoặc tham số mà người viết không tự giải thích được; code chép từ nguồn ngoài phải ghi nguồn.
- Người thuyết trình phần nào phải giải thích được từng dòng phần đó khi cô hỏi (sau khi Toàn bàn giao); Thư review bằng cách hỏi ngẫu nhiên dòng bất kỳ.

**Trình bày code:** slide chỉ đặt output (và đoạn code ngắn nếu cần). Phần nào slide không có code thì mở notebook đúng cell đó để trình bày (cột "Mở notebook" ở mục 6). Notebook mở sẵn bản đã Save & Run All trên Kaggle; dự phòng bản HTML xuất sẵn trên máy nếu mất mạng.

---

## 6. Khung slide (khoảng 21 slide, khoảng 20 phút)

| Mục | Slide | Nội dung chính | Demo / thực nghiệm trên slide | Mở notebook | Thời gian |
|---|---|---|---|---|---|
| Mở đầu | 1–2 | Tiêu đề, thành viên; mục tiêu, lộ trình; **thiết lập: model, thư viện, môi trường** | Bảng model, thư viện + phiên bản, Kaggle GPU T4 (số liệu lấy từ phần 0) | Phần 0 (cell in môi trường) | 1.5' |
| 1 | 3–4 | Bảng phân biệt language model, LLM, chatbot; 4 tác vụ | Output thật 4 tác vụ | Phần 1 | 2' |
| 2 | 5–6, 6b | Sơ đồ token đến softmax, thông số thật của Qwen; causal mask; **6b: so sánh LLM với RNN/LSTM, BERT, VLM** | Ma trận causal mask, bảng top-5 token; 6b: output demo BERT vs Qwen, Qwen2.5 vs Qwen2-VL | Phần 2 | 3.5' |
| 3 | 7 | Tokenization, token ID, embedding | Câu Việt và Anh sau tokenize, số token GPT-2 và Qwen | Phần 3 | 1.5' |
| 4 | 8–9 | Next-token prediction; cross-entropy | Các cặp ngữ cảnh → token mục tiêu; loss thật | Phần 4 | 2' |
| 5 | 10–11 | Sơ đồ pretraining, SFT, alignment | Output base và instruct đặt cạnh nhau (chờ cô) | Phần 5 | 2.5' |
| 6 | 12–13 | Sinh từng token; greedy, temperature, top-p | Output từng bước; output 5 cấu hình | Phần 6 | 1.5' |
| 7 | 14 | In-context learning, làm theo chỉ dẫn | Prompt zero-shot và 3-shot + **kết quả thật** trên vài câu (accuracy đầy đủ ở slide 19) | Phần 7 | 1' |
| 8 | 15–16 | Bảng so sánh prompting, fine-tuning, RAG | Bảng số câu đúng không RAG / có RAG + 1 ví dụ (chờ cô) | Phần 8 | 2' |
| 9 | 17–18 | Hallucination; thiên lệch, giới hạn ngữ cảnh | Câu trả lời sai + nguồn đối chiếu; output thật từng ý | Phần 9 | 2' |
| 10 | 19–20 | **Thực nghiệm**: thiết lập (dữ liệu, số câu, seed, prompt, metric), kết quả, phân tích lỗi; kết luận | Bảng accuracy, confusion matrix, McNemar, ví dụ lỗi (chờ cô) | Phần 7, 10 | 2.5' |
| Kết | 21 | Tài liệu tham khảo | — | — | |

Mỗi slide demo ghi nhỏ ở góc: "Notebook phần X, cell Y" để mở đúng chỗ. Tổng khoảng 21', chưa tính thời gian chuyển sang notebook, nên phải tập bấm giờ. Nếu quá giờ, rút slide 12 trước; slide 2 **không rút** (chứa thư viện và môi trường cô yêu cầu), chỉ bỏ phần lộ trình.

---

## 7. Hạn từng người

Ngày 1 là ngày bắt đầu làm (điền ở đầu file). Việc nào ghi "(chờ cô)" thì phải đợi cô xác nhận mới chạy chính thức, nhưng code vẫn viết trước được. Nếu đến Ngày 4 cô vẫn chưa trả lời thì chạy theo phương án hiện tại (UIT-VSFC, base so với instruct, RAG) và ghi rõ trong slide đó là phương án nhóm chọn.

### Muội

| Việc | Hạn |
|---|---|
| Chốt cách chia vai | Ngày 1 |
| Lý thuyết và slide mục 1, 2, 3 (slide 3 đến 7) | Ngày 2 |
| Lý thuyết và bảng so sánh mục 8 (slide 15) | Ngày 2 |
| Dàn ý desc mục 1, 2, 3, 8 trong `docs/parts/` (khuôn `PART_TEMPLATE_v2.md`) | Ngày 2 |
| Nhận bàn giao code phần 1, 2, 3 từ Toàn, tự giải thích lại được | Ngày 3 |
| Soạn 15 đến 20 câu hỏi và đáp án từ slide môn học; nhận bàn giao code phần 8 (RAG) (chờ cô) | Ngày 4 |
| Khoảng 5 câu Q&A kèm đáp án cho mục của mình | Ngày 5 |
| Ghép slide, thống nhất định dạng; làm slide 1, 2 (gồm thư viện, môi trường), 21; kiểm mỗi slide demo có output thật và ghi "Notebook phần X, cell Y" | Ngày 5 |
| README hướng dẫn chạy trên Kaggle | Ngày 6 |
| Sửa slide theo review và buổi tập | Ngày 6 đến 7 |

### Phúc

| Việc | Hạn |
|---|---|
| Chốt cách chia vai | Ngày 1 |
| Dàn ý desc mục 4, 5, 6, 9 trong `docs/parts/` (khuôn `PART_TEMPLATE_v2.md`) | Ngày 2 |
| Lý thuyết và slide mục 4, 5, 6 (slide 8 đến 13) | Ngày 2 |
| Soạn khoảng 20 câu hỏi hallucination, prompt thiên lệch, giới hạn ngữ cảnh | Ngày 2 |
| Nhận bàn giao code phần 4, 6 từ Toàn, tự giải thích lại được | Ngày 3 |
| Slide 17 và 18; nhận bàn giao code phần 9 | Ngày 4 |
| Nhận bàn giao code phần 5 (base so với instruct) (chờ cô) | Ngày 4 |
| Khoảng 5 câu Q&A kèm đáp án cho mục của mình | Ngày 5 |
| Góp ý sửa notebook phần mình (Toàn sửa) | Ngày 6 |

### Toàn

| Việc | Hạn |
|---|---|
| Chốt cách chia vai | Ngày 1 |
| Lý thuyết và slide mục 7, 10 (slide 14, 19, 20) | Ngày 2 |
| Chạy thử Qwen 1.5B (base và instruct) trên Kaggle; notebook phần 0 (setup + log), có cell in môi trường và phiên bản thư viện; gửi Muội để làm slide 2 | Ngày 1 |
| Duyệt dàn ý desc các mục | Ngày 2 |
| Code phần 1, 2, 3, 4, 6 (mỗi phần smoke run), bàn giao cho Muội, Phúc | Ngày 3 |
| Code phần 5, 8, 9; viết pipeline thí nghiệm tham số hóa theo bộ data (phần 7); bàn giao | Ngày 4 |
| Chạy 500 câu, phân tích lỗi (chờ cô) | Ngày 4 đến 5 |
| Khoảng 5 câu Q&A kèm đáp án cho mục của mình | Ngày 5 |
| Smoke run toàn notebook rồi Save & Run All trên Kaggle (log trong `outputs/logs/`); audit theo desc | Ngày 6 |
| Xuất notebook đã chạy ra HTML làm bản dự phòng khi trình bày | Ngày 6 |

### Thư (hỗ trợ: review, hỏi Q&A thử)

| Việc | Hạn |
|---|---|
| Chốt cách chia vai; theo dõi trả lời của cô | Ngày 1 |
| Đọc lý thuyết cả 10 mục | Ngày 2 |
| Review các phần đã xong | Ngày 3 đến 5 |
| Review toàn bộ slide và notebook; kiểm notebook theo quy ước (markdown trước cell, comment, phân tích sau output) | Ngày 6 |
| Tập trình bày bấm giờ (gồm cả lúc mở notebook), cả nhóm hỏi chéo Q&A; mỗi người giải thích từng dòng code phần mình | Ngày 6 đến 7 |
| Nộp bài cùng cả nhóm | Ngày 7 |

---

## 8. Tài liệu tham khảo

Các link chưa được mở kiểm tra từng cái; mỗi người bấm thử trước khi dùng và đối chiếu với tài liệu của cô. Muội tổng hợp danh sách cuối cùng.

**Mục 1, 2, 3 (Muội)**
- Attention Is All You Need: https://arxiv.org/abs/1706.03762
- The Illustrated Transformer: https://jalammar.github.io/illustrated-transformer/
- The Illustrated GPT-2 (Decoder-only, causal attention): https://jalammar.github.io/illustrated-gpt2/
- Hugging Face Tokenizers: https://huggingface.co/docs/tokenizers
- Hugging Face LLM Course: https://huggingface.co/learn/llm-course
- Stanford CS224N: https://web.stanford.edu/class/cs224n/

**Mục 4, 5, 6, 9 (Phúc)**
- Language Models are Few-Shot Learners (GPT-3): https://arxiv.org/abs/2005.14165
- Finetuned Language Models Are Zero-Shot Learners (FLAN): https://arxiv.org/abs/2109.01652
- InstructGPT (SFT và RLHF): https://arxiv.org/abs/2203.02155
- Direct Preference Optimization (DPO): https://arxiv.org/abs/2305.18290
- The Curious Case of Neural Text Degeneration (top-p): https://arxiv.org/abs/1904.09751
- A Survey on Hallucination in Large Language Models: https://arxiv.org/abs/2311.05232
- Karpathy, Let's build GPT from scratch: https://www.youtube.com/watch?v=kCc8FmEb1nY
- nanoGPT: https://github.com/karpathy/nanoGPT

**Mục 7, 10 (Toàn) và mục 8 (Muội)**
- Language Models are Few-Shot Learners (in-context learning): https://arxiv.org/abs/2005.14165
- Retrieval-Augmented Generation (Lewis et al.): https://arxiv.org/abs/2005.11401
- LoRA (một cách fine-tuning hiệu quả): https://arxiv.org/abs/2106.09685
- A Survey of Large Language Models: https://arxiv.org/abs/2303.18223
- Hugging Face Transformers: https://huggingface.co/docs/transformers
- Hugging Face PEFT: https://huggingface.co/docs/peft
- Stanford CS324: https://stanford-cs324.github.io/winter2022/

**Khác**
- Tài liệu giảng viên cung cấp (đối chiếu trước khi chốt danh sách)
- Model card của Qwen2.5 trên Hugging Face
