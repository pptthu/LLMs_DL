# Sổ ghi các lần chạy và thí nghiệm

- **Created**: 2026-11-15T16:30:00+07:00 (Toàn)
- Mỗi lần chạy (smoke / normal) ghi một mục tóm tắt theo `docs/templates/RUN_LOG_TEMPLATE_v2.md`; log đầy đủ ở `outputs/logs/<YYYYMMDD_HHMMSS>_<smoke|normal>/` (smoke: `outputs/_smoke/logs/`). Mỗi thí nghiệm có so sánh ghi thêm theo `docs/templates/EXPERIMENT_NOTE_TEMPLATE_v1.md`.
- Số liệu lấy từ `outputs/results/` của chính lần chạy đó. Mục mới ghi ở trên cùng.

---

## 20261116_030403_smoke — phần 0 Setup (chạy lại sau khi sửa cell 0.3, 0.4)

- Người chạy: Toàn | Nơi chạy: Kaggle Notebook, GPU Tesla T4, `gpu_count` = 2, model trên `cuda:0` | Chế độ: smoke
- Môi trường: giống lần 20261116_025328_smoke (Python 3.13.16, torch 2.11.0+cu130, transformers 5.20.2)
- Model: Qwen/Qwen2.5-1.5B-Instruct, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- Kết quả: đạt, cell 0.1–0.8 chạy hết, `[part00_setup] OK 2026-11-16T03:04:10+07:00`. Hai lỗi của lần trước đã hết (có `gpu_count`, revision không còn `None`).
- File: `outputs/_smoke/results/env.json`, `outputs/_smoke/logs/20261116_030403_smoke/{run.log, part00_setup.log, config.json, env.json}`

---

## 20261116_025328_smoke — phần 0 Setup

- Người chạy: Toàn | Nơi chạy: Kaggle Notebook, GPU Tesla T4 x2 (dùng GPU 0, VRAM 14.6 GB), Internet bật | Chế độ: smoke (`SMOKE=True`, `N_TEST=10`)
- Môi trường: Python 3.13.16 | torch 2.11.0+cu130 (CUDA 13.0) | transformers 5.20.2 | accelerate 1.15.0 | pandas 2.3.3 | pyarrow 23.0.1 | scikit-learn 1.6.1 | statsmodels 0.15.0 | matplotlib 3.10.0
- Model: Qwen/Qwen2.5-1.5B-Instruct, fp16; số tham số đếm bằng code 1,543,714,304 (khớp model card); 28 block, hidden 1536, GQA 12/2, FFN 8960, vocab 151,936
- Dữ liệu: UIT-VSFC Parquet, số dòng gốc khớp 11,426 / 1,583 / 3,166; bỏ neutral còn 10,968 / 1,510 / 2,999 (test: 1,590 positive / 1,409 negative)
- Kết quả: đạt. File: `outputs/_smoke/results/env.json`, `outputs/_smoke/logs/20261116_025328_smoke/{run.log, part00_setup.log, config.json, env.json}`
- Sự cố: cell 0.8 báo `NameError: ARCH` vì cell 0.5 chưa chạy trong session hiện tại; chạy lại 0.5 rồi 0.8 thì đạt.
- `generation_config` mặc định: do_sample=True, temperature 0.7, top_p 0.8, top_k 20, repetition_penalty 1.1.
- Sự cố 2: log ghi `revision None` (`model.config._commit_hash` trống trên transformers 5.20.2).
- Thay đổi sau lần chạy: cell 0.4 `device_map="auto"` → `{"": 0}`; cell 0.3 thêm `gpu_count`; cell 0.4 lấy revision từ Hub (`huggingface_hub.model_info(...).sha`) nếu `_commit_hash` trống. Lần chạy sau sẽ có hai thay đổi này.
