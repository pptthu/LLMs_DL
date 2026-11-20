# Khuôn log mỗi lần chạy (v3)

- Tác giả: Toàn (AI hỗ trợ) | Tạo: 2026-11-16T01:40:17+07:00 | Cập nhật: Toàn (AI hỗ trợ), 2026-11-16T01:40:17+07:00
- Trạng thái: **Duyệt: Toàn, 2026-11-16**
- Dựa trên: `RUN_LOG_TEMPLATE_v1.md` và `LOG_TEMPLATE_v1.md` của Lab 1 (log 3 phần: đầu / mỗi bước / cuối).
- Thay đổi so với v1: log tách theo thư mục ngày giờ; ghi rõ log phải chứa gì (tham số, kiến trúc model, mỗi mẫu một dòng, tổng kết); thay "mỗi epoch" bằng "mỗi mẫu" vì không train. v1 giữ nguyên, không dùng.
- Dùng cho: mọi lần chạy notebook, cả smoke run và normal run.

## 1. Nơi lưu và tên file

- Thư mục mỗi lần chạy: `outputs/logs/<YYYYMMDD_HHMMSS>_normal/` (smoke: `outputs/_smoke/logs/<YYYYMMDD_HHMMSS>_smoke/`).
  - `<YYYYMMDD_HHMMSS>` = thời điểm bắt đầu chạy, giờ Việt Nam (UTC+7). Ví dụ `20261118_093000`.
  - Chạy lại thì tạo thư mục mới, **không ghi đè** thư mục cũ.

| File | Ai ghi | Nội dung |
|---|---|---|
| `run.log` | Cell đầu và cell cuối notebook | Khuôn §2 |
| `config.json` | Cell cấu hình (phần 0) | Mọi tham số ở §2.1, dạng JSON |
| `env.json` | Cell cuối phần 0 | Phiên bản và phần cứng ở §2.1 |
| `partNN_<tên>.log` | Cell đầu và cuối mỗi phần | Khuôn §3 |
| `partNN_<tên>_records.csv` | Mỗi phần | Mỗi mẫu một dòng, cùng cột với phần "mỗi mẫu" trong `.log` |

## 2. Khuôn `run.log`

### 2.1 Phần đầu (ghi trước khi chạy phần 1)

    === RUN START 2026-11-18T09:30:00+07:00 ===
    run_id: 20261118_093000_normal | mode: normal (SMOKE=False) | người chạy: Toàn
    nơi chạy: Kaggle Notebook, GPU Tesla T4 (VRAM … GB), CUDA … | notebook: notebook/llm_presentation.ipynb, commit <hash>
    môi trường: Python … | torch … | transformers … | accelerate … | pandas … | scikit-learn … | statsmodels …
    seed: 42 | dtype: float16 | device_map: auto
    model chính: Qwen/Qwen2.5-1.5B-Instruct (revision <commit hash trên Hub>)
      kiến trúc: decoder-only, 28 block, hidden 1536, 12 head Q / 2 head KV, FFN 8960, vocab 151936, context 32768
      số tham số (đếm bằng code): …  | sơ đồ: docs/models/qwen2.5-1.5b-instruct_v2.md
    model phụ: Qwen/Qwen2.5-1.5B (phần 5) | tokenizer phụ: openai-community/gpt2 (phần 3)
    dữ liệu: UIT-VSFC (Parquet, split test, 2,999 câu, id lưu ở results/vsfc_test_ids.csv) | UIT-ViQuAD 2.0 | bộ câu nhóm: data/*.json
    tham số sinh mặc định: do_sample=False, max_new_tokens=…, (lấy mẫu thì ghi thêm temperature, top_p, top_k, repetition_penalty)

- Số liệu kiến trúc **in từ `model.config` và đếm bằng code**, không chép tay.
- Ghi revision (commit hash) của model trên Hub để người khác tải đúng bản.

### 2.2 Phần thân: mỗi phần một dòng tóm tắt

    [part00_setup]   OK  12.3 s  → outputs/results/env.json
    [part01_tasks]   OK  45.0 s  → outputs/results/tasks_outputs.csv (4 dòng)
    [part07_icl]     OK  1820 s  → outputs/results/icl_zero_shot.csv, icl_3shot_set{1,2,3}.csv (500 dòng mỗi file)
    ...

### 2.3 Phần cuối

    === RUN END 2026-11-18T10:15:42+07:00 ===
    tổng thời gian: 45 phút 42 giây | số phần chạy: 11/11 | lỗi: 0 | cảnh báo: <…>
    file kết quả: <liệt kê đường dẫn>
    kết quả chính: <số thật, lấy từ file kết quả, ví dụ accuracy zero-shot / 3-shot>

## 3. Khuôn `partNN_<tên>.log`

    === PART 07 icl START 2026-11-18T09:40:10+07:00 ===
    tham số: model=Qwen/Qwen2.5-1.5B-Instruct | prompt_template=icl_v1 | k_shot=0,3 | fewshot_sets=3 | n_test=500 | do_sample=False | max_new_tokens=5
    dữ liệu vào: data/test_500_ids.json (500 ID, 250 pos / 250 neg), data/fewshot_pools.json
    --- mỗi mẫu ---
    idx,id,setting,n_input_tokens,n_output_tokens,raw_output,pred,label,correct,time_s,acc_cumulative
    0,1234,zero_shot,58,2,"tích cực",pos,pos,1,0.21,1.000
    ...
    === PART 07 icl END 2026-11-18T10:10:30+07:00 ===
    tổng kết: accuracy zero-shot=… | 3-shot (bộ 1/2/3)=… | tỉ lệ không parse được=… | thời gian=… s
    file ghi: outputs/results/icl_zero_shot.csv, ...

**Cột "mỗi mẫu" bắt buộc theo từng phần** (thay cho dòng mỗi epoch của Lab 1):

| Phần | Cột bắt buộc ngoài `idx, id, time_s` |
|---|---|
| 1 Tác vụ mẫu | `task, prompt, n_input_tokens, n_output_tokens, output` |
| 2 Kiến trúc | `text, position, token, top_k_token, top_k_prob` |
| 3 Tokenization | `sentence, lang, tokenizer, n_tokens, tokens` |
| 4 Nguyên lý học | `context, target_token, prob_target, loss_token` (cross-entropy từng token); cuối phần: `loss_mean`, `perplexity`, so với `outputs.loss` |
| 5 Base vs instruct | `prompt_id, model, output, n_output_tokens` |
| 6 Sinh văn bản | `config_name, do_sample, temperature, top_p, top_k, output` |
| 7, 10 Zero/few-shot | `setting, fewshot_set, raw_output, pred, label, correct, acc_cumulative` |
| 8 RAG | `question_id, with_rag, retrieved_chunk_ids, output, judged_correct` |
| 9 Hạn chế | `case_type, prompt, output, judged, note` |

## 4. How to do: từng bước tạo log trong notebook
1. Cell 0.1: tạo `RUN_ID` từ giờ bắt đầu (múi giờ Việt Nam) và chế độ chạy; tạo thư mục `OUT_DIR/logs/RUN_ID/`.
2. Cell 0.2: viết hàm ghi log (ghi thêm dòng vào `run.log` và `partNN_<tên>.log`); ghi phần đầu `run.log` (§2.1).
3. Cuối phần 0: ghi `config.json` (mọi tham số) và `env.json` (phiên bản, GPU).
4. Đầu mỗi phần: ghi dòng `=== PART NN START <thời điểm> ===`, tham số của phần, dữ liệu vào.
5. Trong vòng lặp: mỗi mẫu ghi một dòng vào `.log` và một hàng vào list; cuối phần lưu list thành `partNN_<tên>_records.csv` (cột theo bảng §3).
6. Cuối mỗi phần: ghi dòng tổng kết (metric, thời gian, file đã ghi) và dòng `=== PART NN END ===`; thêm một dòng tóm tắt vào `run.log` (§2.2).
7. Cell cuối notebook: ghi phần cuối `run.log` (§2.3).
8. Chạy xong: mở thư mục log, kiểm đủ file ở §1 và đủ 3 phần đầu / mỗi mẫu / cuối; ghi một mục vào `docs/experiments/experiment_log.md`.

## 5. Quy tắc

1. Số trên slide chỉ lấy từ file kết quả của **normal run**, ghi kèm `run_id`.
2. Sửa code sau normal run thì **chạy lại toàn bộ**, tạo thư mục log mới, rồi mới cập nhật số trên slide.
3. Mỗi phần in và ghi tham số sinh đã truyền vào `generate()` (bản Instruct mặc định lấy mẫu ngẫu nhiên).
4. Mục 7, 10: chốt prompt trên split **validation** của UIT-VSFC; toàn bộ 2,999 câu test chỉ chạy ở normal run (smoke: 20 câu).
5. Smoke run: mỗi phần vài mẫu, seed 42, ghi vào `outputs/_smoke/`. Đạt khi chạy hết không lỗi, đủ file ở §1, log có đủ phần đầu / mỗi mẫu / cuối. Số của smoke run không dùng để báo cáo.
6. Sau mỗi lần chạy, thêm một mục tóm tắt vào `docs/experiments/experiment_log.md`, trỏ tới thư mục log.
7. Không có epoch, training loss, training accuracy: bài không fine-tune (góp ý của cô ngày 11/11). Nếu sau này fine-tune thì thêm khuôn log epoch theo `LOG_TEMPLATE_v1.md` của Lab 1. **[cần chốt nếu xảy ra]**
