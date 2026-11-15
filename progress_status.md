# Tiến độ bài Presentation LLMs

- **Created**: 2026-11-15T16:30:00+07:00 (Toàn)

Cập nhật file này sau MỖI bước. Bảng dưới đây là nguồn thông tin duy nhất về việc ai đang làm gì.

| Phần notebook | Mục | Phụ trách | Desc (`docs/parts/`) | Code (chưa làm / đang làm / xong) | Smoke run (đạt / lỗi / chưa chạy) | Normal run (đạt / lỗi / chưa chạy) | Slide | Ghi chú |
|---|---|---|---|---|---|---|---|---|
| 0. Setup | — | Phúc | — | chưa làm | chưa chạy | chưa chạy | 2 | Ưu tiên số 1: cách lấy model / tokenizer / dữ liệu |
| 1. Tác vụ mẫu | 1 | Muội | chưa có | chưa làm | chưa chạy | chưa chạy | 3–4 | |
| 2. Kiến trúc | 2 | Muội | chưa có | chưa làm | chưa chạy | chưa chạy | 5–6 | Cần `attn_implementation="eager"` |
| 3. Tokenization | 3 | Muội | chưa có | chưa làm | chưa chạy | chưa chạy | 7 | |
| 4. Nguyên lý học | 4 | Phúc | chưa có | chưa làm | chưa chạy | chưa chạy | 8–9 | |
| 5. Base vs instruct | 5 | Phúc | chưa có | chưa làm | chưa chạy | chưa chạy | 10–11 | Chờ cô xác nhận |
| 6. Sinh văn bản | 6 | Phúc | chưa có | chưa làm | chưa chạy | chưa chạy | 12–13 | |
| 7. Zero / few-shot | 7 | Toàn | chưa có | chưa làm | chưa chạy | chưa chạy | 14 | Chờ cô xác nhận UIT-VSFC |
| 8. RAG | 8 | Muội | chưa có | chưa làm | chưa chạy | chưa chạy | 15–16 | Chờ cô xác nhận |
| 9. Hạn chế | 9 | Phúc | chưa có | chưa làm | chưa chạy | chưa chạy | 17–18 | |
| 10. Tổng hợp | 10 | Toàn | chưa có | chưa làm | chưa chạy | chưa chạy | 19–20 | |
| Ghép notebook, Save & Run All | — | Toàn | — | chưa làm | — | chưa chạy | — | |
| README | — | Muội | — | khung (Toàn) | — | — | — | |
| Ghép slide | — | Muội | — | chưa làm | — | — | 1, 2, 21 | |

## Điều kiện trước khi ghép notebook
- [ ] Phần 0 của Phúc chạy được trên Kaggle; mọi người dùng chung cell setup đó
- [ ] Mỗi phần smoke run đạt, ghi đủ file kết quả theo `project_description.md` §2.3
- [ ] Mỗi phần có markdown, comment, cell "Phân tích" theo `docs/rules/RULES_v1.md` §2
- [ ] Mỗi người gửi tin bàn giao: phần đã làm, file `part_NN_*.ipynb`, vấn đề đã biết

## Kiểm chứng
_Chưa có lần chạy nào._ Ghi theo `docs/templates/RUN_LOG_TEMPLATE_v1.md` (chi tiết ở `docs/experiments/experiment_log.md`).

## Vấn đề mở
- Chờ cô xác nhận: cách demo mục 5, mục 8; dùng UIT-VSFC.
- Cần chốt (`docs/plans/plan_presentation_llm_v2.md` §8):
  - viết notebook riêng rồi ghép;
  - cách chấm RAG / hallucination;
  - Bonferroni ở mục 10.

## Nhật ký (ghi ngày, mỗi bước một dòng)
- 2026-11-15 Toàn: clone repo `pptthu/LLMs_DL` (repo trống) về `D:\File\File\UTH\UTH\Học sâu - DL\LLMs_DL`; tạo khung thư mục và tài liệu theo rule Lab 1 đã sửa cho bài presentation (AI hỗ trợ) → chỉ tạo file, chưa commit, chưa chạy code; chờ Thư duyệt
