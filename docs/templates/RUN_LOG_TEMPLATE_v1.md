# Khuôn ghi lần chạy (v1)

- Tác giả: Toàn (AI hỗ trợ) | Tạo: 2026-11-15T16:30:00+07:00
- Dựa trên: `Lab-DL/labs/lab1/docs/templates/LOG_TEMPLATE_v1.md` (Thư). Lab 1 ghi log huấn luyện (epoch, early stopping, checkpoint); bài này không train, nên log ghi **môi trường + cấu hình suy luận + file kết quả**.
- Dùng cho: mọi lần chạy notebook (smoke run và normal run).

## 1. File sinh ra mỗi lần chạy

| File | Nội dung | Ai ghi |
|---|---|---|
| `outputs/results/env.json` (smoke: `outputs/_smoke/env.json`) | Python, `torch`, `transformers`, `pandas`, `scikit-learn`, `statsmodels`; GPU, CUDA, RAM; dtype; ngày giờ chạy | Cell cuối phần 0 |
| `outputs/results/<phần>_config.json` | Tham số của phần đó: model, prompt template, tham số sinh, seed, số câu | Cell đầu mỗi phần |
| `outputs/results/<phần>_<tên>.csv` / `.json` | Kết quả của phần | Mỗi phần |
| Dòng trong `docs/experiments/experiment_log.md` | Khuôn §2 | Người chạy |

## 2. Khuôn một mục trong `experiment_log.md`

    ## <YYYY-MM-DD> — <smoke | normal> run — <phần hoặc "toàn notebook">
    - Người chạy: <tên> | Nơi chạy: Kaggle GPU T4 | Notebook: <đường dẫn, commit>
    - Môi trường: xem outputs/results/env.json (Python …, torch …, transformers …, GPU …)
    - Model: Qwen/Qwen2.5-1.5B-Instruct (dtype …) [+ Qwen/Qwen2.5-1.5B, openai-community/gpt2 tokenizer]
    - Tham số chính: SMOKE=…, seed=42, do_sample=…, max_new_tokens=…, <tham số riêng của phần>
    - Thời gian chạy: <giờ bắt đầu> – <giờ kết thúc> (<phút>)
    - File kết quả: <liệt kê đường dẫn>
    - Kết quả chính: <số thật, lấy từ file kết quả>
    - Lỗi / ghi chú: <…>

## 3. Quy tắc
1. Số liệu trên slide chỉ lấy từ **normal run**, kèm đường dẫn file.
2. Sửa code sau normal run thì **chạy lại toàn bộ** notebook rồi mới cập nhật số trên slide.
3. Mỗi phần in `model.generation_config` hoặc tham số sinh đã truyền vào, để chứng minh dùng đúng cấu hình.
4. Mục 7, 10: chốt prompt trên split **validation** của UIT-VSFC; tập test (500 câu) chỉ chạy ở normal run.
