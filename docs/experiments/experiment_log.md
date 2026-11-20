# Sổ ghi các lần chạy và thí nghiệm

- **Created**: 2026-11-15T16:30:00+07:00 (Toàn)
- Mỗi lần chạy (smoke / normal) ghi một mục tóm tắt theo `docs/templates/RUN_LOG_TEMPLATE_v2.md`; log đầy đủ ở `outputs/logs/<YYYYMMDD_HHMMSS>_<smoke|normal>/` (smoke: `outputs/_smoke/logs/`). Mỗi thí nghiệm có so sánh ghi thêm theo `docs/templates/EXPERIMENT_NOTE_TEMPLATE_v1.md`.
- Số liệu lấy từ `outputs/results/` của chính lần chạy đó. Mục mới ghi ở trên cùng.

---

## 20261121_010447_normal — phần 0–10 (**lần chạy chính thức cuối**, notebook nộp và số cho slide lấy từ đây)

- Người chạy: Toàn | Kaggle, 2 × T4 (model trên GPU 0), Internet bật | Chế độ: normal | notebook đã chạy và `outputs/` (qua `outputs.zip`) tải về repo 21/11
- Thay đổi so với 20261119_150356_normal (sửa 20–21/11, AI hỗ trợ): bảng in bằng `show_table()` (kẻ vạch); khung ĐẦU VÀO `show_input()` trước mỗi demo (25 khung, có chuỗi sau chat template); cell mới 3.1b (gom token theo từ, byte UTF-8), 5.0 (sơ đồ 3 giai đoạn huấn luyện), 9.4 (thiên lệch: WinoGender + BLS, Nemotron-Personas-Vietnam + 3 khuôn câu tự soạn), 9.5 (kim trong đống rơm, cắt cửa sổ 1,024 token, chi phí theo độ dài), 9.6 (chất lượng theo tác vụ và dữ liệu); 10 hình mới; xoá tên thành viên khỏi notebook (`judge_note` đổi "Toàn đọc lại" → "chấm tay").
- Kết quả: **đạt**, 140 cell chạy hết, không lỗi; 9.2 phút. Phần 1–8 và 9.1–9.3 trùng 20261119_150356_normal (chỉ khác `time_s`): zero-shot 0.8986, few-shot 0.9230, McNemar p = 1.5e-08; RAG 0/15 → 10/15; mục 9 sai 12/15 mỗi cách hỏi; thực thể không tồn tại bịa 5/6.
- Mới, 9.4 thiên lệch (`bias_summary.csv`): tiếng Anh 60 nghề r = 0.69 (p = 8.5e-10), tiếng Việt 122 nghề r = 0.44 (p = 4.0e-07); 7/7 biến thể đạt luật ghi trước (r > 0, p < 0.05). Đối chứng: "person" 0.13, "người đó" 0.01 (model mặc định chọn đại từ nam). Nemotron revision ghi trong `part09_limits.log`.
- Mới, 9.5 (`context_needle.csv`, `context_cost.csv`): câu giấu trong input đúng 6/6 (1,161 và 4,202 token); cắt còn 1,024 token cuối làm mất câu giấu → trả lời "1234"; 8,192 token: 10.7 s, 13.3 GB.
- Mới, 9.6 (`quality_by_task.csv`, `quality_by_data.csv`): câu có từ góp ý accuracy 0.790 (619 câu) so với 0.958 (2,380 câu).
- Sự cố trước khi đạt: `20261121_002559_normal` và `20261121_003544_normal` dừng giữa chừng (lần sau OOM ở cell 9.5, đo 8,192 token khi bộ nhớ GPU bị phân mảnh) → cell 9.5 dọn bộ nhớ trước mỗi độ dài và bắt `OutOfMemoryError` (ghi `OOM` vào bảng thay vì dừng). `20261121_004703_normal` chạy hết, số trùng run này (chưa có khung ĐẦU VÀO). Log 3 run này không chép vào repo.

---

## 20261119_150356_normal — phần 0–10 (lần chạy chính thức trước)

- Người chạy: Toàn | Kaggle T4 | Chế độ: normal | notebook đã chạy và `outputs/` (qua `outputs.zip`) tải về repo 19/11
- Thay đổi so với 20261119_025437_normal: thêm cell 9.3 (3 câu về thực thể không tồn tại × 2 cách hỏi, `MANUAL_JUDGE9F`); cell 10.3 đổi tên nhóm (d) và chuyển row 1695 sang (c); cell 10.4 ghi đủ đúng / sai / từ chối (mục 8) và cả 2 bản (mục 5); markdown chi tiết hơn. Phần sinh của phần 0–8 không đổi.
- Kết quả: **đạt**, chạy hết, không lỗi; 7.4 phút. Mọi số trùng 20261119_025437_normal (chỉ khác `time_s`): zero-shot 0.8986, few-shot 0.9230, McNemar p = 1.5e-08; RAG có đáp án 0 đúng / 1 sai / 14 từ chối vs 10 / 3 / 2; mục 9 sai 12/15 mỗi cách hỏi.
- Mới: thực thể không tồn tại bịa 5/6 lần hỏi, từ chối 1/6 (`fict-3|B`, nhãn tay áp đúng, không có cảnh báo); output trùng lần chạy thử trên CPU.
- Phân tích lỗi: ý ngầm 20, vừa khen vừa chê 4, model sai rõ 4, nhãn gốc có vẻ sai 2.
- Lần chạy `20261119_142011_normal` dừng sau phần 0, không dùng, không chép vào repo.

---

## 20261119_025437_normal — phần 0–10 (lần chạy chính thức trước)

- Người chạy: Toàn | Kaggle T4 | Chế độ: normal | notebook đã chạy và `outputs/` tải về repo 19/11
- Thay đổi so với 20261119_023934_normal: cell Phân tích phần 0–10 đã điền; cell 10.3 thêm `MANUAL_ERROR_GROUPS` (30 câu) và đếm câu góp ý. Phần sinh không đổi.
- Kết quả: **đạt**, chạy hết, không lỗi; 7.1 phút. Mọi số trùng với 20261119_023934_normal (greedy): zero-shot 0.8986, few-shot 0.9230, McNemar p = 1.503e-08; RAG 0/15 vs 10/15; mục 9 sai 12/15 mỗi cách hỏi.
- Phân tích lỗi (`part10_summary.log`): ý ngầm 20, model sai rõ 4, nhãn gốc có vẻ sai 3, vừa khen vừa chê 3; negative bị đoán positive là câu góp ý: 189/248 (zero-shot), 117/145 (few-shot).

---

## 20261119_023934_normal — phần 0–10 (lần chạy chính thức thứ nhất)

- Người chạy: Toàn | Kaggle T4, Internet bật, Dataset `llms-dl-data` | Chế độ: normal | `outputs/` và notebook đã chạy tải về repo 19/11
- Thay đổi so với `20261119_022224_normal`: chỉ phần chấm của cell 8.4 (dấu câu → khoảng trắng, `MANUAL_JUDGE8`) và 9.2 (`MANUAL_JUDGE9`). Phần sinh không đổi.
- Kết quả: **đạt**, cell 0.1–10.4 chạy hết, không lỗi; `RUN END` sau 7.1 phút.
- Mục 7, 10: giống lần trước (greedy): zero-shot 0.8986, few-shot 0.9230 trên 2,999 câu; McNemar b=47, c=120, p=1.5e-08.
- Mục 8: câu có đáp án: không RAG đúng 0/15 (từ chối 14, sai 1), có RAG đúng 10/15 (sai 3, từ chối 2); câu tài liệu không có đáp án: có RAG trả lời bịa 2/3.
- Mục 9: không tài liệu sai 12/15 (cách A), 12/15 (cách B), sai cả 2 lần 11/15; câu vua Charles: đáp án gốc "1640" là nhãn sai của UIT-ViQuAD, model trả lời 1649 (đúng).
- Mục 5: instruct đạt 4/4, base 0/4 (chấm theo `follow_rule`).

---

## 20261119_022224_normal — phần 0–10 (normal run đầu tiên)

- Người chạy: Toàn | Kaggle T4, Internet bật, Dataset `llms-dl-data` | Chế độ: normal (`SMOKE=False`) | `outputs/` tải về repo 19/11
- Kết quả: **đạt**, cell 0.1–10.4 chạy hết, không lỗi; `RUN END` sau 7.2 phút.
- Phần 7, 10 (`icl_metrics.json`): mẫu prompt chọn trên validation: `vi_nhan_vi` (trung bình 0.875, `vi_nhan_en` 0.8725); 2,999 câu test: zero-shot 0.8986 (±0.0108), few-shot 0.9230 (±0.0095), không parse được 0; McNemar exact a=2648, b=47, c=120, d=184, p=1.5e-08.
- Phần 8, 9: **chấm tự động sai vài câu** (xoá dấu câu làm "Charles-de-Gaulle" không khớp "Charles de Gaulle"; "Bảo tàng Louvre", "Georgia" không khớp đáp án; đáp án gốc "1640" của câu vua Charles là nhãn sai, đúng là 1649). Đã sửa cell 8.4 (dấu câu → khoảng trắng) và thêm `MANUAL_JUDGE8`, `MANUAL_JUDGE9` → **phải chạy lại normal run**; số của lần này ở phần 8, 9 không dùng.
  - Tính lại trên CSV của lần này: câu có đáp án, không RAG đúng 0/15 (từ chối 14), có RAG đúng 10/15; hỏi không tài liệu sai 12/15 (A), 12/15 (B), sai cả 2 là 11/15.
- Phân tích lỗi: phần lớn câu few-shot sai là **góp ý** ("nên…", "cần…") có nhãn negative nhưng model đoán positive.

---

## 20261119_021425_smoke — phần 0–10 (notebook có phần 6–10 mới)

- Người chạy: Toàn | Kaggle T4, Internet bật, Dataset `llms-dl-data` | Chế độ: smoke (`N_TEST=20`, `N_VAL_PROMPT=20`, `N_QA=5`, `N_QA_IMPOSSIBLE=1`)
- Kết quả: **đạt**, cell 0.1–10.4 chạy hết, không lỗi; `RUN END` sau 2.3 phút. Notebook đã chạy tải về repo; file `outputs/_smoke/` chưa tải.
- Phần 2: VRAM khi có bản eager 5.76 GB (lần 17/11: 11.51 GB, do còn mBERT).
- Phần 5: base bị cắt 4/4, instruct 0/4; **chưa chấm**.
- Phần 6: sinh từng bước khớp `generate()`; P(token đầu) T=0.5/1.0/1.5 = 0.995/0.847/0.468; greedy 1 output / 3 lần, các cấu hình lấy mẫu 3/3; temp_1.5 ra chuỗi vô nghĩa lẫn nhiều ngôn ngữ.
- Phần 7: chọn mẫu `vi_nhan_en` (validation 20 câu); mẫu `vi_nhan_vi` few-shot không parse được 60%. **Ước tính 2 cách × 2,999 câu: 3.1 phút (0.031 s/câu/cách)** → normal run giữ `N_TEST = None` (toàn bộ test).
- Phần 8: recall@3 = 0.833 (5/6); câu có đáp án: không RAG đúng 0/5 (từ chối 4), có RAG đúng 3/5.
- Phần 9: không tài liệu, sai 5/5 cả 2 cách hỏi; ví dụ: "Muốn ban hành luật thì hoàng đế phải làm gì?" → model "Đưa ra một đạo luật.", đáp án "phải thông qua Hội đồng".
- Số smoke không dùng cho slide.

---

## 20261117_161656_smoke — phần 0–5

- Người chạy: Toàn | Kaggle T4 x2 (dùng GPU 0) | Chế độ: smoke | Môi trường như lần 20261116 (torch 2.11.0+cu130, transformers 5.20.2)
- Kết quả: cell 0.1–5.4 chạy hết, không lỗi (output lưu trong notebook ngày 19/11; file `outputs/_smoke/` chưa tải về máy).
  - Phần 2: attention thật phần trên đường chéo = 0.000000; top-1 sau "Hôm nay trời rất" = " đẹp" (0.788).
  - Phần 3: câu tiếng Việt Qwen 8 token / GPT-2 22 token; tiếng Anh 7 / 7; decode đúng 4/4.
  - Phần 4: loss câu ví dụ 2.985 = `outputs.loss` (chênh 6.3e-08); câu xáo trộn 6.701.
  - Phần 5: base bị cắt ở `max_new_tokens` 4/4, instruct 0/4; **chưa chấm** (`MANUAL_JUDGE5` trống) → phần 5 chưa tính đạt.
- Thời gian sinh: câu ngắn 9–10 token mất 0.38–0.42 s (cell 1.1, 5.3) → căn cứ chọn sinh theo lô ở phần 7.
- VRAM cao nhất 12.18 GB (cell 5.1, khi có cả bản base).
- Ghi nhận cho mục 9: cell 1.1 ("Xin chào") và 1.5 model trả lời lẫn tiếng Trung.

---

## 2026-11-16 smoke run phần 1 Tác vụ mẫu (lần cuối: END 03:54:34)

- Người chạy: Toàn | Kaggle T4, Dataset `llms-dl-data` (`tasks_prompts.json` v2) | Chế độ: smoke | greedy, repetition_penalty 1.1 (mặc định)
- Kết quả (`outputs/_smoke/results/tasks_outputs.csv`): hỏi đáp đạt (10/32 token), dịch đạt (8/48), sinh mã đạt (216/512; `is_even` = [True, False, True, False]), tóm tắt đạt, chấm tay: Toàn (104/256; "đúng 1 câu, giữ đủ 5 ý, không thêm ý; nhưng gần như ghép lại các câu gốc, rút gọn ít"); **đạt 4/4**, bị cắt 0/4; thời gian sinh 13.0 s.
- Quyết định chấm (Toàn, 16/11): sinh mã giữ đạt theo test `is_even`; chữ ngoài khối code chỉ ghi vào `judge_note`, không trừ điểm vì prompt không cấm.
- Sự cố đã xử lý:
  - Lần đầu `FileNotFoundError` `data/tasks_prompts.json`: cell 0.1 chạy trước khi gắn Dataset; chạy lại 0.1 thì đạt.
  - Bản v1 (`max_new_tokens` tóm tắt 96, sinh mã 192): cả hai chạm trần, bị cắt → v2 (256, 512) theo mức benchmark tóm tắt (~62–110 token tiếng Anh) và model card Qwen (512).
  - Kaggle Dataset chưa cập nhật nên 2 lần chạy vẫn đọc v1 → cell 1.2 thêm `assert` version file dữ liệu.
- Quan sát: prompt thử "Xin chào" được trả lời bằng tiếng Trung; sinh mã tự thêm ví dụ `print` và giải thích; tóm tắt gần như ghép lại 5 câu gốc.

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
