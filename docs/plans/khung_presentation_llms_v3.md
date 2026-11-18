# Khung presentation LLMs (v3)

- **Created**: 2026-11-18 (Toàn, AI hỗ trợ) · Trạng thái: **[đề xuất]**, chờ Toàn duyệt
- Chỉ dựng từ [`yeu_cau_cua_co_v1.md`](yeu_cau_cua_co_v1.md): 10 mục của đề, góp ý 11/11, yêu cầu thêm, sản phẩm nộp. Không thêm nội dung ngoài các yêu cầu đó.
- Kế hoạch làm đến thứ 7: [`ke_hoach_18_21_11_v1.md`](ke_hoach_18_21_11_v1.md).
- §2 chép khung slide của Thư để đối chiếu.

## 1. Khung slide (14 slide, ~16.5 phút nói + ~3 phút mở notebook)

| Slide | Mục | Nội dung cần trình bày (đề) | Minh họa (đề) | Demo trên model thật (góp ý 11/11) | Mở notebook | Phút |
|---|---|---|---|---|---|---|
| 1 | — | Tiêu đề, thành viên | — | — | — | 0.5 |
| 2 | — | Thư viện sử dụng, môi trường chạy (yêu cầu thêm 1, 2) | — | Bảng lấy từ `results/env.json` | — | 1 |
| 3 | 1 | LLM là gì? Phân biệt language model, LLM và chatbot | Hỏi đáp, tóm tắt, dịch, sinh mã | Output thật 4 tác vụ | — | 1.5 |
| 4 | 2 | Transformer; Decoder-only và causal attention | Sơ đồ token đầu vào → xác suất token tiếp theo | Causal mask + top-5 token tiếp theo | — | 1.5 |
| 5 | 3 | Token, token ID, embedding | Tokenize 1 câu Việt + 1 câu Anh | Token, ID, kích thước embedding thật | — | 1 |
| 6 | 4 | Next-token prediction, dữ liệu huấn luyện, cross-entropy loss | Cặp ngữ cảnh → token mục tiêu từ 1 câu | Loss thật từng token | Có (≤ 1') | 1.5 |
| 7 | 5 | Pretraining, SFT, preference alignment; RLHF/DPO ý tưởng | Sơ đồ mục tiêu và dữ liệu từng giai đoạn | Base vs Instruct cùng prompt | — | 1.5 |
| 8 | 6 | Autoregressive; greedy, temperature, top-p | Cùng 1 prompt, nhiều cấu hình sinh | Output từng cấu hình | Có (≤ 1') | 1.5 |
| 9 | 7 | In-context learning, làm theo chỉ dẫn, tác vụ ngôn ngữ | Zero-shot vs few-shot cùng tác vụ | Prompt + accuracy thật | — | 1 |
| 10 | 8 | Prompting vs fine-tuning vs RAG | Bảng dữ liệu cần có + trường hợp sử dụng | Không RAG vs có RAG | — | 1.5 |
| 11 | 9 | Hallucination, thiên lệch, giới hạn ngữ cảnh; phụ thuộc tác vụ và dữ liệu | 1 câu trả lời sai + cách kiểm chứng | Output sai thật + nguồn đối chiếu | — | 1.5 |
| 12 | 10 | Thử nghiệm có kiểm soát (yêu cầu thêm 4: thiết lập, kết quả) | Bảng kết quả | Bảng accuracy zero vs few-shot, confusion matrix | Có (≤ 1') | 1.5 |
| 13 | 10 | Tổng hợp kết quả và nhận xét; kết luận | Ví dụ đầu ra + phân tích lỗi | Câu sai thật, phân loại lỗi | — | 1.5 |
| 14 | — | Tài liệu tham khảo | — | — | — | 0 |

- Mỗi slide demo ghi góc "Notebook phần X, cell Y" (yêu cầu thêm 6). Notebook chạy sẵn (Save & Run All), không chạy live.
- Mỗi người giải thích được từng dòng code phần mình (yêu cầu thêm 5).
- Tổng ~19.5 phút, chưa đo. Bấm giờ ở buổi tập; lố thì mở notebook 2 lần thay vì 3.

## 2. Khung của Thư (đối chiếu)

Chép nguyên từ `ke_hoach_presentation_llms_v2.md` §6 (bản của Thư, Toàn sửa v2 ngày 16/11 thêm slide 6b).

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

**Khác nhau chính giữa hai khung:**

| | Khung v3 (§1) | Khung của Thư (§2) |
|---|---|---|
| Số slide | 14 | 21 (gồm 6b) |
| Thời gian | ~19.5' gồm mở notebook | ~22' chưa gồm mở notebook (`HANDOFF_CLAUDE_CODE.md:118`) |
| So sánh RNN/LSTM, BERT, VLM | Không (đề không yêu cầu) | Có (slide 6b) |
| McNemar | Không | Có |
| Mở notebook | 3 lần | Gần như mọi mục |
