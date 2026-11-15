# Luật làm việc bài Presentation LLMs (v1)

- Tác giả: Toàn (AI hỗ trợ) | Tạo: 2026-11-15T16:30:00+07:00
- Trạng thái: **[đề xuất]**, chờ Thư duyệt
- Dựa trên: `Lab-DL/labs/lab1/docs/rules/RULES_v1.md` (Thư), sửa cho đề bài presentation: không có code nguồn riêng, mọi code nằm trong notebook, không huấn luyện model.
- Áp dụng cho: cả nhóm (Muội, Phúc, Toàn, Thư) và AI agent khi làm bài presentation. Luật này luôn bật, đọc trước mọi việc.

## 1. Quy trình
Mỗi phần (mục 1–10 của đề, ứng với phần 0–10 của notebook) đi theo thứ tự sau, không nhảy cóc:
1. **Desc:** người phụ trách viết dàn ý theo `templates/PART_TEMPLATE_v1.md`, lưu ở `parts/`. Desc cả bài dùng `templates/DESC_TEMPLATE_v1.md`.
2. **Discussion:** AI hỏi lại chỗ chưa rõ, mỗi lượt tối đa 5 câu.
3. **Detail plan:** AI chi tiết hoá từ desc, lưu vào `plans/`.
4. **Duyệt:** Thư đọc plan và nói duyệt hay sửa.
5. **Lưu có version:** tên file `_v0`, `_v1`… Luôn dùng bản số lớn nhất, bản cũ giữ nguyên.
6. **Small prompt:** mỗi prompt chỉ làm một bước (ví dụ: dựng prompt, rồi parse nhãn, rồi tính metric).
7. **Smoke run (chạy thử):** đặt `SMOKE = True` ở đầu notebook, mỗi phần chạy vài câu hoặc vài mẫu, seed 42. Mọi file ghi vào `outputs/_smoke/`, không ghi vào `outputs/results|figures`.
   - Đạt khi: chạy hết không lỗi; mỗi phần ghi đủ file kết quả, đúng cột; `env.json` có đủ phiên bản.
   - Số liệu của smoke run **không** dùng để báo cáo.
8. **Normal run (chạy chính thức):** `SMOKE = False`, **Save & Run All** trên Kaggle (GPU T4). Ghi log theo `templates/RUN_LOG_TEMPLATE_v1.md` vào `experiments/experiment_log.md`.
9. **Audit:** đối chiếu notebook, kết quả và slide với desc. Chỉ đánh dấu Done khi audit xong; lỗi nào không sửa thì ghi lý do vào desc.

## 2. Luật chung
- **Code chỉ nằm trong notebook.** Không có thư mục mã nguồn riêng. Cô hỏi phần nào thì chỉ ngay cell đó. Hàm dùng lại được định nghĩa ngay trong cell trước chỗ dùng, kèm docstring ngắn và 2–3 dòng `assert` để tự kiểm.
- **Notebook vừa giữ output vừa ghi file.** Notebook nộp là bản đã chạy, có output (yêu cầu của cô). Đồng thời mỗi phần ghi kết quả ra `outputs/results/` (CSV/JSON) và `outputs/figures/` (PNG) để làm slide mà không phải chạy lại. Luật này **thay** luật "không để kết quả trong notebook" của Lab 1.
- **Tham số đặt ở một chỗ:** cell cấu hình đầu notebook (tên model, seed, `SMOKE`, số câu, đường dẫn). Các phần khác đọc từ đó, không gõ lại.
- **Tham số sinh văn bản luôn truyền tường minh** khi gọi `generate()`: `do_sample`, `max_new_tokens`, và khi lấy mẫu thì thêm `temperature`, `top_p`, `top_k`, `repetition_penalty`. Bản Qwen2.5-Instruct mặc định lấy mẫu ngẫu nhiên (`generation_config.json`).
- **Số liệu** ghi trên slide hoặc trong tài liệu phải lấy từ file kết quả của **lần chạy chính thức** và ghi đường dẫn file đó. Không chép số từ smoke run, từ trí nhớ, hay từ output của AI.
- **Ghi nguồn:** tách rõ **[Slide ChX sN]** (slide môn) và **[Bổ sung]** (nguồn ngoài, đã mở kiểm tra). Mỗi khẳng định kỹ thuật có nguồn ở `references.md`.
- **Giải thích được từng dòng:** mỗi cell code có markdown đứng trước (mục đích, đầu vào, đầu ra), comment tiếng Việt cho dòng không hiển nhiên, và cell "Phân tích" 2–3 câu sau output. Người viết phần nào phải tự giải thích được từng dòng phần đó.
- **Không commit:** dữ liệu tải về (UIT-VSFC), trọng số model, slide môn học, token, file `.env`.
- **Ngôn ngữ:** tài liệu viết tiếng Việt, giữ thuật ngữ tiếng Anh, không song ngữ (như ngoại lệ của Lab 1). Tên biến, hàm, file viết tiếng Anh.
- **Không fine-tune** (góp ý của cô ngày 11/11). Phần huấn luyện (mục 4, 5) minh hoạ bằng inference: tính loss, so base với instruct.

## 3. Phân vai người và AI
- Người viết dàn ý desc, mục tiêu, success criteria và phần "không làm". AI chỉ chi tiết hoá.
- Người quyết định, ghi kèm ngày và lý do. AI đề xuất và đánh dấu `[đề xuất]` hoặc `[cần chốt]` cho tới khi được duyệt.
- Thư duyệt ở 4 cổng: desc, detail plan, smoke run, audit.
- AI không tự thêm thí nghiệm, không đổi cấu trúc thư mục ngoài plan đã duyệt.
- Chưa chắc thì AI hỏi, không đoán. AI không khẳng định "chạy được" khi chưa có output thật.

## 4. Phân công và làm song song
| Người | Mục lý thuyết | Phần notebook |
|---|---|---|
| Muội | 1, 2, 3, 8; ghép slide | 1, 2, 3, 8; README |
| Phúc | 4, 5, 6, 9 | 0 (setup), 4, 5, 6, 9 |
| Toàn | 7, 10 | 7, 10; ghép notebook |
| Thư (hỗ trợ) | Review, hỏi Q&A thử, bấm giờ | Không nhận phần nào |

- Mỗi người viết trong notebook riêng `notebook/parts/part_NN_<tên>.ipynb` và chỉ sửa file của mình (file `.ipynb` rất khó gộp khi hai người cùng sửa).
- Phần 0 (setup) do Phúc viết trước; các phần khác chép đúng cell setup đó, không tự viết cách tải model khác.
- Toàn ghép các phần thành `notebook/llm_presentation.ipynb` rồi chạy chính thức một lần.
- Mỗi người tự thuyết trình mục của mình, kể cả mở notebook giải thích code mình viết.
