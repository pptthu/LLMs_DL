# Presentation LLMs — Tìm hiểu mô hình ngôn ngữ lớn  (v1)
- Tác giả: Toàn (AI hỗ trợ) | Tạo: 2026-11-15T16:30:00+07:00 | Trạng thái: **[đề xuất]**, chờ Thư duyệt
- Trạng thái công việc: In Progress

## Background
- Môn Học sâu (UTH). Đề: "Tìm hiểu mô hình ngôn ngữ lớn — Large Language Models (LLMs)".
- Mục tiêu của cô: giải thích được cách LLM học và sinh văn bản, phân biệt các giai đoạn huấn luyện, nhận biết khả năng và hạn chế của mô hình. Gồm 10 mục, mỗi mục có yêu cầu minh hoạ.
- **Góp ý của cô (11/11):** không cần fine-tune, nhưng mọi mục phải có demo trên model thật kèm phân tích output.
- **Yêu cầu thêm:**
  - slide nêu thư viện sử dụng và môi trường chạy;
  - mỗi mục có demo trên slide; slide phải có thực nghiệm;
  - giải thích được từng dòng notebook;
  - phần không có code trên slide thì mở notebook trình bày.

## Purpose
- **Làm:**
  - slide khoảng 21 trang, 15–20 phút;
  - 1 notebook chạy trên Kaggle (phần 0–10), có output;
  - bộ câu thử nghiệm + kết quả + phân tích lỗi;
  - tài liệu tham khảo;
  - mỗi người khoảng 5 câu Q&A.
- **Không làm:**
  - fine-tune hay huấn luyện model;
  - mã nguồn ngoài notebook;
  - model lớn hơn 3B;
  - đánh giá trên nhiều bộ dữ liệu.

## Input / Output
- **Input:**
  - `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-1.5B`, tokenizer `openai-community/gpt2` (Hugging Face Hub);
  - UIT-VSFC (Parquet trên Hugging Face);
  - bộ câu nhóm tự soạn trong `data/`.
- **Output:** `outputs/results/`, `outputs/figures/`, `slides/`, `docs/experiments/`.
- **Code:** `notebook/llm_presentation.ipynb`.

## Success criteria
- 10 mục, mỗi mục có minh hoạ đúng cột "Yêu cầu minh họa" của đề, với output thật từ notebook (bảng đối chiếu ở `../project_description.md` §1).
- Slide 2 có thư viện + phiên bản và môi trường, lấy từ `outputs/results/env.json`.
- Thực nghiệm có kiểm soát (mục 7, 10): 500 câu UIT-VSFC (250/250), zero-shot vs 3-shot × 3 bộ, greedy; có accuracy, confusion matrix, tỉ lệ không parse được, McNemar; bảng kết quả + ví dụ lỗi thật.
- Notebook chạy Save & Run All không lỗi; mỗi cell code có markdown trước, comment, phân tích sau output.
- Mọi số trên slide có đường dẫn file kết quả của lần chạy chính thức.
- Tập thuyết trình trong 20 phút, gồm cả thời gian mở notebook.

## How-to-do
1. Phúc: phần 0 setup, chạy thử tải 2 model + tokenizer + UIT-VSFC trên Kaggle, ghi `env.json`.
2. Mỗi người: desc mục mình trong `parts/` (khuôn `PART_TEMPLATE_v1.md`) → Thư duyệt.
3. Mỗi người: soạn file trong `data/`, viết notebook phần mình trong `notebook/parts/`, smoke run.
4. Toàn: chốt prompt mục 7 trên split validation; normal run 500 câu × 4.
5. Toàn: ghép notebook, Save & Run All trên Kaggle, xuất HTML dự phòng.
6. Mỗi người: điền số thật vào slide mục mình; Muội ghép slide; Thư review.
7. Tập trình bày, hỏi chéo Q&A, nộp.

## Quyết định (người ghi)
- 2026-11-11: không fine-tune; mọi mục có demo trên model thật (góp ý của cô) — Thư ghi.
- 2026-11-15: repo riêng `LLMs_DL`; code hết trong notebook, không có mã nguồn riêng; cấu trúc thư mục gọn; ưu tiên phần setup (lấy model ở đâu, như thế nào); giữ phân công code cho Muội, Phúc, Toàn — góp ý của Thư, Toàn ghi.
