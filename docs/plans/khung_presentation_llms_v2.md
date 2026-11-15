# Khung toàn bộ bài Presentation LLMs – chia task (v2)

Cập nhật: 2026-11-14 · Đi kèm file kế hoạch `ke_hoach_presentation_llms.md` (nội dung, phương án demo, thư viện, quy ước notebook)

> **v2 (2026-11-16T01:40:17+07:00, Toàn, AI hỗ trợ):** sửa theo quyết định 15/11 **Toàn viết toàn bộ code notebook (phần 0–10)**; Muội, Phúc viết dàn ý desc, lý thuyết, slide, thuyết trình mục mình và nhận bàn giao code. Thêm so sánh LLM với RNN/LSTM, BERT, VLM vào mục 2 (16/11). v1 giữ nguyên, không dùng. Lịch giữ nguyên (Toàn chốt 16/11). **Duyệt: Toàn, 2026-11-16**.

**Yêu cầu thêm của cô:** slide nêu thư viện sử dụng và môi trường chạy; mỗi mục có demo trên slide; slide phải có thực nghiệm; giải thích được từng dòng trong notebook; phần nào slide không có code thì mở notebook trình bày code.

**Ngày 1** = ngày bắt đầu làm: **15/11/2026** (Toàn chốt 16/11). Mỗi người tự làm đúng hạn của mình. Muốn thêm hoặc sửa gì thì sửa trực tiếp trong file này.

**Nhóm:** Muội, Phúc (Member 2), Toàn (Member 3) chia 10 mục. Muội, Phúc, Toàn là người làm chính và tự thuyết trình mục của mình (kể cả mở notebook giải thích code; Toàn viết code và bàn giao trước). Thư chỉ hỗ trợ (review, hỏi Q&A thử, bấm giờ), không nhận mục nào. Phân công dưới đây là đề xuất; bạn chia team chốt lại thì sửa tên trong các bảng.

## 1. Khung slide (15–20 phút, khoảng 20 slide)

| Slide | Mục | Nội dung chính | Minh hoạ trên slide | Mở notebook | Thời gian |
| --- | --- | --- | --- | --- | --- |
| 1 | — | Tiêu đề, thành viên | — | — | 0.5' |
| 2 | — | Mục tiêu, lộ trình 10 mục; **thiết lập: model, thư viện, môi trường** | Bảng model, thư viện + phiên bản, Kaggle GPU T4 (lấy từ phần 0 lần chạy thật) | Phần 0 | 1' |
| 3 | 1 | LLM là gì; language model, LLM, chatbot khác nhau thế nào | Bảng phân biệt 3 khái niệm | — | 1' |
| 4 | 1 | LLM làm được những tác vụ gì | Output thật: hỏi đáp, tóm tắt, dịch, sinh mã | Phần 1 | 1' |
| 5 | 2 | Vai trò của Transformer; Decoder-only | Sơ đồ token → embedding → decoder blocks → logits → softmax; thông số thật của Qwen | Phần 2 | 1.5' |
| 6 | 2 | Causal attention | Ma trận causal mask của câu ví dụ; bảng top-5 token tiếp theo | Phần 2 | 1' |
| 6b | 2 | So sánh LLM với RNN/LSTM, BERT, VLM (điểm cộng) | Bảng so sánh; output demo BERT vs Qwen, Qwen2.5 vs Qwen2-VL | Phần 2 (cell 2.5–2.8) | 1' |
| 7 | 3 | Tokenization, token ID, embedding | Câu Việt và Anh sau khi tokenize; so số token GPT-2 và Qwen; kích thước vector embedding | Phần 3 | 1.5' |
| 8 | 4 | Next-token prediction, dữ liệu huấn luyện | Các cặp ngữ cảnh → token mục tiêu từ câu ví dụ | Phần 4 | 1' |
| 9 | 4 | Cross-entropy loss | Công thức + loss thật của model trên câu ví dụ | Phần 4 | 1' |
| 10 | 5 | Pretraining → SFT → preference alignment; RLHF/DPO ý tưởng | Sơ đồ mục tiêu và dữ liệu từng giai đoạn | — | 1.5' |
| 11 | 5 | Demo base so với instruct (chờ cô) | Output hai bản đặt cạnh nhau + phân tích | Phần 5 | 1' |
| 12 | 6 | Autoregressive generation | Sinh từng token, in từng bước | Phần 6 | 0.5' |
| 13 | 6 | Greedy, temperature, top-p | Output 5 cấu hình trên cùng prompt | Phần 6 | 1' |
| 14 | 7 | In-context learning, làm theo chỉ dẫn | Prompt zero-shot và 3-shot + **kết quả thật** trên vài câu (accuracy đầy đủ ở slide 19) | Phần 7 | 1' |
| 15 | 8 | Prompting, fine-tuning, RAG | Bảng so sánh 3 cách | — | 1' |
| 16 | 8 | Demo không RAG so với có RAG (chờ cô) | Bảng số câu đúng + 1 ví dụ | Phần 8 | 1' |
| 17 | 9 | Hallucination, cách kiểm chứng | 1 câu trả lời sai + nguồn đối chiếu | Phần 9 | 1' |
| 18 | 9 | Thiên lệch, giới hạn ngữ cảnh, phụ thuộc tác vụ và dữ liệu | Output thật cho từng ý | Phần 9 | 1' |
| 19 | 10 | **Thực nghiệm** có kiểm soát: thiết lập (dữ liệu, số câu, seed, prompt, metric), kết quả, phân tích lỗi (chờ cô) | Bảng accuracy, confusion matrix, McNemar, ví dụ lỗi | Phần 7, 10 | 2' |
| 20 | 10 | Kết luận | 3–4 ý rút ra từ các demo | — | 0.5' |
| 21 | — | Tài liệu tham khảo | — | — | — |

Slide 3, 10, 15, 20 là lý thuyết thuần, không cần demo; các slide còn lại có output thật. Mỗi slide demo ghi nhỏ ở góc "Notebook phần X, cell Y" để mở đúng chỗ. Slide không có code thì mở notebook phần đó để trình bày code.

Tổng khoảng 20.5 phút, chưa tính thời gian chuyển sang notebook. Nếu quá giờ, rút slide 12 trước; slide 2 **không rút** (chứa thư viện và môi trường cô yêu cầu), chỉ bỏ phần lộ trình.

## 2. Khung notebook

Một notebook chung, **Toàn viết toàn bộ code** (quyết định 15/11). Cột dưới là người phụ trách lý thuyết + thuyết trình.

| Phần | Nội dung | Lý thuyết + thuyết trình (code: Toàn) |
| --- | --- | --- |
| 0. Setup | Cài thư viện (ghim phiên bản), seed, load tokenizer và 2 model một lần | Toàn (lý thuyết + code, chốt 16/11) |
| 1. Tác vụ mẫu (mục 1) | 4 tác vụ trên bản instruct | Muội |
| 2. Kiến trúc (mục 2) | In config, causal mask, top-5 token tiếp theo; so sánh LLM với RNN/LSTM, BERT, VLM (có demo) | Muội |
| 3. Tokenization (mục 3) | Tokenize Việt và Anh, so GPT-2 và Qwen, embedding | Muội |
| 4. Nguyên lý học (mục 4) | Cặp ngữ cảnh → token mục tiêu, tính loss | Phúc |
| 5. Base so với instruct (mục 5) (chờ cô) | Cùng bộ prompt trên 2 bản | Phúc |
| 6. Sinh văn bản (mục 6) | Sinh từng bước, 5 cấu hình decoding | Phúc |
| 7. Zero-shot / few-shot (mục 7, 10) (chờ cô) | Pipeline tham số hoá theo bộ data, chạy 500 câu, metric, McNemar, lưu CSV | Toàn |
| 8. RAG (mục 8) (chờ cô) | Chia slide thành đoạn, tìm đoạn liên quan, so không RAG và có RAG | Muội |
| 9. Hạn chế (mục 9) | Hallucination, thiên lệch, giới hạn ngữ cảnh | Phúc |
| 10. Tổng hợp (mục 10) | Bảng tổng hợp kết quả các phần | Toàn |
| Hướng dẫn chạy | README: chạy trên Kaggle, thời gian chạy dự kiến | Muội |

**Quy ước để giải thích được từng dòng** (chi tiết ở file kế hoạch, mục 5): markdown trước mỗi cell (mục đích, đầu vào, đầu ra); comment tiếng Việt cho dòng không hiển nhiên; markdown "Phân tích" sau mỗi output. Ai viết phần nào phải tự giải thích được từng dòng phần đó.

## 3. Danh sách việc và hạn

Việc nào ghi "chờ cô" thì phải đợi cô xác nhận mới chạy chính thức, nhưng code vẫn viết trước được.

| Việc | Mục | Người | Hạn | Chờ cô |
| --- | --- | --- | --- | --- |
| Chốt cách chia vai | — | Cả nhóm | Ngày 1 | |
| Theo dõi trả lời của cô | — | Thư (hỗ trợ) | Ngày 1 | |
| Chạy thử Qwen 1.5B base và instruct trên Kaggle | — | Toàn | Ngày 1 | |
| Notebook phần 0 (setup), có cell in môi trường (Python, CUDA, GPU) và phiên bản thư viện; gửi Muội làm slide 2; mở log | — | Toàn | Ngày 1 | |
| Dàn ý desc mục mình trong `docs/parts/` | Tất cả | Muội, Phúc, Toàn | Ngày 2 | |
| Lý thuyết và slide 3–7 | 1, 2, 3 | Muội | Ngày 2 | |
| Lý thuyết và bảng so sánh, slide 15 | 8 | Muội | Ngày 2 | |
| Lý thuyết và slide 8–13 | 4, 5, 6 | Phúc | Ngày 2 | |
| Soạn khoảng 20 câu hỏi hallucination, prompt thiên lệch, giới hạn ngữ cảnh | 9 | Phúc | Ngày 2 | |
| Lý thuyết và slide 14, 19, 20 | 7, 10 | Toàn | Ngày 2 | |
| Đọc lý thuyết cả 10 mục | Tất cả | Thư (hỗ trợ) | Ngày 2 | |
| Code phần 1, 2, 3, 4, 6 (smoke run từng phần) | 1–4, 6 | Toàn | Ngày 3 | |
| Nhận bàn giao code phần 1, 2, 3 / phần 4, 6 | 1–4, 6 | Muội / Phúc | Ngày 3 | |
| Slide 17–18 | 9 | Phúc | Ngày 3 | |
| Code phần 5, 8, 9; viết pipeline thí nghiệm tham số hoá (phần 7) | 5, 7–10 | Toàn | Ngày 4 | Có |
| Soạn 15–20 câu hỏi + đáp án từ slide môn học; nhận bàn giao code phần 8 | 8 | Muội | Ngày 4 | Có |
| Nhận bàn giao code phần 5 (base so với instruct), phần 9 | 5, 9 | Phúc | Ngày 4 | Có |
| Chạy 500 câu, phân tích lỗi | 7, 10 | Toàn | Ngày 5 | Có |
| Khoảng 5 câu Q&A kèm đáp án cho mỗi mục mình phụ trách | Tất cả | Mỗi người | Ngày 5 | |
| Ghép slide, thống nhất định dạng; slide 1, 2 (gồm thư viện, môi trường), 21; kiểm mỗi slide demo có output thật và ghi "Notebook phần X, cell Y" | — | Muội | Ngày 5 | |
| Review các phần đã xong | — | Thư (hỗ trợ) | Ngày 3–5 | |
| Smoke run toàn notebook rồi Save & Run All trên Kaggle (log trong `outputs/logs/`); audit theo desc | — | Toàn | Ngày 6 | |
| Xuất notebook đã chạy ra HTML làm bản dự phòng khi trình bày | — | Toàn | Ngày 6 | |
| README hướng dẫn chạy | — | Muội | Ngày 6 | |
| Review toàn bộ slide và notebook; kiểm notebook theo quy ước giải thích từng dòng | — | Thư (hỗ trợ) | Ngày 6 | |
| Sửa slide theo review (mỗi người); sửa notebook theo review (Toàn) | — | Muội, Phúc, Toàn | Ngày 6–7 | |
| Tập trình bày bấm giờ (gồm lúc mở notebook), cả nhóm hỏi chéo Q&A; mỗi người giải thích từng dòng code phần mình | — | Muội, Phúc, Toàn trình bày; Thư bấm giờ, hỏi thử | Ngày 7 | |
| Nộp bài | — | Cả nhóm | Ngày 7 | |

Nếu đến Ngày 4 cô vẫn chưa trả lời: chạy theo phương án hiện tại (UIT-VSFC, base so với instruct, RAG), ghi rõ trong slide là phương án nhóm chọn.

## 4. Tải công việc

| Người | Mục | Ghi chú |
| --- | --- | --- |
| Muội | 1, 2 (gồm so sánh mô hình), 3, 8 + ghép slide | Không code; nhận bàn giao code phần 1, 2, 3, 8 |
| Phúc | 4, 5, 6, 9 | Không code; nhận bàn giao code phần 4, 5, 6, 9 |
| Toàn | 7, 10 + **toàn bộ code phần 0–10** | Nặng nhất: code mọi phần, thí nghiệm chính, chạy và log |
| Thư (hỗ trợ) | Review, hỏi Q&A thử, bấm giờ | Không nhận mục nào |

Mục 9 chuyển từ Toàn sang Phúc để Toàn tập trung vào thí nghiệm chính.
