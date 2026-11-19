# Kế hoạch 18/11 → 21/11 (trình bày thứ 7)

- **Created**: 2026-11-18 (Toàn, AI hỗ trợ) · Trạng thái: **[đề xuất]**, chờ Toàn duyệt
- Khung đề: [`khung_presentation_llms_v3.md`](khung_presentation_llms_v3.md). Plan này chỉ phục vụ làm đủ khung đó.

## 1. Cắt khỏi phạm vi (đề không yêu cầu)

| Cắt | Mục | Ghi chú |
|---|---|---|
| So sánh LLM với RNN/LSTM, BERT, VLM (cell 2.5–2.9) | 2 | Code đã có thì để trong notebook, không lên slide |
| RAG quy mô lớn (15–20 câu, RAGAS, chấm mù) | 8 | Giữ demo nhỏ: 5 câu, không RAG vs có RAG (góp ý 11/11 bắt buộc demo mục 8) |
| Thiên lệch, ngữ cảnh dài chạy thật; chấm mù, kappa | 9 | Đề chỉ yêu cầu 1 câu sai + cách kiểm chứng |
| 3 bộ few-shot, 500 câu, McNemar, Bonferroni | 7, 10 | Còn zero-shot vs 3-shot, 1 bộ ví dụ, ~100 câu UIT-VSFC |

Không cắt: demo base vs instruct (mục 5) và demo RAG (mục 8), vì góp ý 11/11 của cô yêu cầu mọi mục có demo trên model thật, kể cả mục 5 và 8.

## 2. Notebook tối thiểu

| Phần | Trạng thái | Việc còn lại |
|---|---|---|
| 0, 1 | xong, smoke đạt | — |
| 2–5 | code xong | Smoke run Kaggle; phần 2 chỉ cần cell 2.1–2.4 |
| 6 | chưa | 1 prompt; in sinh từng bước; greedy, T=0.7, T=1.5, top-p=0.9 → `results/decoding_samples.csv` |
| 7 | chưa | ~100 câu test UIT-VSFC, zero-shot vs 3-shot (ví dụ từ train, `assert` không trùng test) → `results/icl_*.csv` |
| 9 | chưa | 3–5 câu hỏi sự thật có đáp án + nguồn (`data/hallucination.csv`) |
| 10 | chưa | Accuracy, macro-F1, confusion matrix, lọc câu sai → `results/summary_table.csv` |
| 8 | chưa | 5 câu hỏi về slide môn học (`data/rag_questions.csv`, Muội soạn); TF-IDF lấy 1 đoạn; so không RAG vs có RAG → `results/rag_results.csv` |

## 3. Lịch

Quy tắc cắt: **12h trưa thứ 6** phần nào chưa smoke đạt thì bỏ demo, giữ slide lý thuyết.

| Ngày | Toàn | Muội (mục 1, 2, 3, 8) | Phúc (mục 4, 5, 6, 9) | Thư |
|---|---|---|---|---|
| **T4 18/11** tối | Duyệt khung + plan; smoke phần 2–5; code phần 6 | Slide mục 1, 8 (bảng); soạn 5 câu RAG + đoạn slide nguồn | Slide mục 5 (sơ đồ); soạn câu hỏi mục 9 | Báo nhóm |
| **T5 19/11** | Code + smoke phần 7, 8, 9, 10; bàn giao phần 1–6 | Slide mục 2, 3; nhận bàn giao 1–3 | Slide mục 4, 6; nhận bàn giao 4–6 | Review slide |
| **T6 20/11** | Sáng: Save & Run All, xuất HTML; chiều: slide 2 (thư viện, môi trường), mục 7, 10; bàn giao phần 8, 9 | Thay số chính thức; slide demo mục 8; README; tài liệu tham khảo; ghép slide (hạn 20h) | Slide mục 9; thay số chính thức | Review notebook + slide |
| **T6 tối** | Tập lần 1, bấm giờ | ← | ← | Bấm giờ, hỏi thử |
| **T7 21/11** sáng | Xuất PDF, nộp 4 sản phẩm | Tập lại phần mình | Tập lại phần mình | Kiểm checklist nộp |

Mọi số trên slide lấy từ `outputs/results/` của lần Save & Run All.

## 4. Chia giờ

Xem cột "Phút" và "Mở notebook" ở [`khung_presentation_llms_v3.md`](khung_presentation_llms_v3.md) §1. Không chạy code live.
