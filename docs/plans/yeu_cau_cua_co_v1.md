# Yêu cầu của cô — Presentation LLMs

- **Created**: 2026-11-18 (Toàn, AI hỗ trợ)
- Chỉ chứa yêu cầu của cô, không thêm nội dung khác. Nguồn: đề cô gửi; góp ý và yêu cầu thêm của cô chép từ kế hoạch của Thư (`present_llms.md`, trong repo là `ke_hoach_presentation_llms_v2.md` §1).

**Topic:** Tìm hiểu mô hình ngôn ngữ lớn — Large Language Models (LLMs)

**Mục tiêu:** Giải thích được cách LLM học và sinh văn bản, phân biệt các giai đoạn huấn luyện, đồng thời nhận biết khả năng và hạn chế của mô hình.

## 1. Mười mục (đề)

| Các mục | Nội dung sinh viên cần trình bày | Yêu cầu minh họa |
|---|---|---|
| 1. Giới thiệu | LLM là gì? Phân biệt language model, LLM và ứng dụng chatbot | Ví dụ các tác vụ: hỏi đáp, tóm tắt, dịch, sinh mã |
| 2. Nền tảng kiến trúc | Vai trò của Transformer; tập trung vào Decoder-only và causal attention | Sơ đồ từ token đầu vào đến xác suất token tiếp theo |
| 3. Tokenization | Cách văn bản được chia thành token; token ID và embedding | Tokenize một câu tiếng Việt và một câu tiếng Anh |
| 4. Nguyên lý học | Next-token prediction, dữ liệu huấn luyện và cross-entropy loss | Tạo các cặp ngữ cảnh → token mục tiêu từ một câu |
| 5. Các giai đoạn huấn luyện | Pretraining, Supervised Fine-Tuning (SFT) và preference alignment; giới thiệu RLHF/DPO ở mức ý tưởng | Sơ đồ mục tiêu và dữ liệu của từng giai đoạn |
| 6. Quá trình sinh văn bản | Autoregressive generation; greedy decoding, temperature và top-p sampling | Cùng một prompt với các cấu hình sinh khác nhau |
| 7. Khả năng của LLM | In-context learning, làm theo chỉ dẫn, xử lý các tác vụ ngôn ngữ | So sánh zero-shot và few-shot trên cùng tác vụ |
| 8. Cách thích nghi và bổ sung tri thức | Phân biệt prompting, fine-tuning và Retrieval-Augmented Generation (RAG) | Bảng so sánh dữ liệu cần có và trường hợp sử dụng |
| 9. Hạn chế và đánh giá | Hallucination, thiên lệch, giới hạn ngữ cảnh; chất lượng phụ thuộc tác vụ và dữ liệu | Một câu trả lời sai và cách kiểm chứng |
| 10. Demo và kết luận | Thử nghiệm có kiểm soát, tổng hợp kết quả và nhận xét | Bảng kết quả, ví dụ đầu ra và phân tích lỗi |

## 2. Sản phẩm nộp (đề)

- [ ] Slide trình bày dạng PDF hoặc PPTX.
- [ ] Notebook hoặc mã nguồn demo kèm hướng dẫn chạy.
- [ ] Bộ câu thử nghiệm, kết quả và phân tích lỗi.
- [ ] Danh sách tài liệu tham khảo.

## 3. Góp ý của cô (11/11)

- Không cần fine-tune.
- Phải có demo và phân tích model thực tế: **mọi mục đều có demo trên model thật kèm phân tích output, kể cả mục 5 và 8**.
- Cô đã trả lời "cứ làm" cho demo mục 5, mục 8 và việc dùng UIT-VSFC (theo góp ý của Thư, `gop_y_thu_cho_toan_v1.md` §1.5).

## 4. Yêu cầu thêm của cô

1. Slide nêu rõ thư viện sử dụng.
2. Slide nêu rõ môi trường chạy.
3. Mỗi mục có demo trên slide (output thật).
4. Slide phải có thực nghiệm (thiết lập, kết quả, phân tích).
5. Giải thích được từng dòng trong notebook.
6. Phần nào slide không có code thì mở notebook trình bày code.

## 5. Thời lượng

- 15–20 phút (ghi trong kế hoạch của Thư).
