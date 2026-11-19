# Trả lời góp ý repo LLMs_DL của Thư

- **Created**: 2026-11-17T15:30:00+07:00
- **Người viết**: Toàn (AI hỗ trợ)
- **Trả lời file**: `docs/plans/gop_y_thu_cho_toan_v1.md` (Thư, cập nhật 2026-11-17T11:25:29+07:00, PR #4)
- **Trạng thái**: trả lời, chờ Thư quyết các dòng ở §I. Thư quyết xong thì ghi vào `presentation_description_v2.md` mục "Quyết định"; file này giữ nguyên làm bằng chứng.
- **Đã đọc**: toàn bộ 43 file `.md` bản mới nhất trong repo (`main` tới `d1ec7e7`, cộng file góp ý của PR #4), cùng `present_llms.md` và `khung_presentation_llms.md` (bản 11/11).
- **Ký hiệu**:
  - `file:dòng` là trích từ repo;
  - **[HF]** là đọc trực tiếp từ Hugging Face ngày 17/11;
  - **[cần kiểm]** là chưa chạy hoặc chưa mở nguồn.

## Cách đọc
| Phần | Nội dung | Thư cần làm |
|---|---|---|
| §A | Những điểm góp ý chưa nêu (ưu tiên cao nhất) | Đọc trước |
| §B–§H | Trả lời lần lượt §0–§5 trong góp ý của Thư | Xem chỗ Toàn đồng ý hoặc chưa đồng ý |
| §I | Danh sách cần Thư quyết | Trả lời từng dòng |

---

## A. Những điểm góp ý chưa nêu

### A1. Con số "20.5 phút" trong góp ý lấy từ một dòng ghi sai
Góp ý §4.2 dẫn `khung_presentation_llms_v2.md:42` (*"Tổng khoảng 20.5 phút"*). Dòng này ghi sai: 20.5 là tổng của bản khung ngày 11/11. Bản v2 đã tăng slide 2 lên 1 phút và thêm slide 6b, nhưng dòng tổng không được sửa theo.

### A2. Bài đang dài 22 phút, vượt giới hạn 20 phút
- Cộng cột "Thời gian" ở `khung_presentation_llms_v2.md:17–38` ra **22.0 phút**. `ke_hoach_presentation_llms_v2.md:132` ghi "khoảng 21'", cộng §6 của file đó cũng ra 22.
- Đề cho 15–20 phút. `presentation_description_v2.md:65` ghi *"Tập thuyết trình trong 20 phút, gồm cả thời gian mở notebook"*.
- Rút slide 12 như góp ý §4.5 chỉ bớt 0.5 phút, còn 21.5 phút. Cần cắt khoảng 2–3 phút (đề xuất ở §I).

### A3. Tiến độ đang trễ, nên chưa thể ghi "Khối lượng: kịp" (góp ý §1.4)
- Ngày 1 = 15/11 (`khung_presentation_llms_v2.md:9`), vậy hôm nay 17/11 là **Ngày 3**.
- Hạn Ngày 2: Muội, Phúc viết dàn ý desc mục mình và soạn file `data/` (`khung_presentation_llms_v2.md:75`; `RULES_v2.md` §4). Thực tế:
  - `docs/parts/` mới có `muc01_gioi_thieu_v2.md`, do AI viết;
  - `data/` mới có `tasks_prompts.json`.
- Hạn Ngày 3: code phần 1, 2, 3, 4, 6 và bàn giao (`khung_presentation_llms_v2.md:82–83`). Thực tế mới xong phần 0, 1 (`progress_status.md:12–22`).

### A4. Mục 5 so sánh chưa công bằng, ngoài ý "tác dụng gộp" ở góp ý §3
Ngoài trọng số, bản base và bản Instruct còn khác nhau ở 3 chỗ:
- **`repetition_penalty`:** bản Instruct mặc định 1.1. Bản base không có tham số này, tức là 1.0 [HF: `Qwen/Qwen2.5-1.5B/generation_config.json` chỉ có `do_sample: false`, `eos_token_id: 151643`, `max_new_tokens: 2048`]. Cell 1.1 cố ý giữ 1.1 (`docs/parts/muc01_gioi_thieu_v2.md:60`). Nếu mục 5 dùng lại hàm đó, chỉ bản Instruct bị phạt lặp, nên bản base lặp nhiều hơn một phần vì lý do này chứ không phải vì thiếu SFT.
- **Cách đưa prompt:** base nhận văn bản thô; Instruct nhận chat template kèm system prompt mặc định *"You are Qwen…"* (thấy trong output smoke run phần 1).
- **Token kết thúc:** base chỉ dừng ở id 151643; Instruct dừng ở `<|im_end|>`. Bản base dễ sinh tới hết `max_new_tokens`.

Đề xuất:
- Truyền `repetition_penalty` giống nhau cho cả hai bản (1.0 hay 1.1 ghi ở §I).
- Ghi trên slide 11 cả 3 điểm khác nhau trên.
- Sửa lời thoại *"khác biệt hoàn toàn đến từ SFT và alignment"* (`noi_dung_muc05_giai_doan_huan_luyen_v1.md:147`).

### A5. Chưa có tên "người thứ 2" ở mục 8, 9
Plan cần 2 người ở **2 chỗ**, góp ý §2.2 mới bàn chỗ thứ hai:
- **Xác định đáp án độc lập:** `noi_dung_muc08_*_v2.md:127`, `noi_dung_muc09_*_v2.md:126`.
- **Chấm lại:** `noi_dung_muc08_*_v2.md:138`, `noi_dung_muc09_*_v2.md:140`.

Muội soạn câu mục 8, Phúc soạn câu mục 9 (`RULES_v2.md` §4), nên cả hai đã biết đáp án và không chấm mù được. Người độc lập duy nhất còn lại là Thư.

### A6. Mục 9 thiếu 2 ý có trong chính `present_llms.md`
- **Thiếu mốc văn bản ngắn khi đo giới hạn ngữ cảnh.** `present_llms.md:80` ghi *"so với khi văn bản ngắn"*. `detail_plan_notebook_v1.md:122` (cell 9.3) chỉ đo đầu / giữa / cuối của văn bản dài. Không có mốc ngắn thì kết quả sai ở văn bản dài không chứng minh được là do độ dài.
- **Thiếu loại câu "thực thể không tồn tại".** Có ở `present_llms.md:78`, không có trong bảng loại câu `noi_dung_muc09_*_v2.md:110–117`.

### A7. Mục 8: so khả năng từ chối giữa hai cách không cùng điều kiện
- Chỉ prompt có RAG được dặn *"Nếu tài liệu không có thông tin, trả lời 'Không có trong tài liệu'"* (`noi_dung_muc08_*_v2.md:113`).
- Với 3–5 câu không có đáp án, cách không RAG chưa có định nghĩa thế nào là "Đúng".
- Đề xuất (chọn 1): chỉ tính phép thử từ chối ở cách có RAG, **hoặc** thêm cùng câu dặn đó vào prompt không RAG.

### A8. Mục 9: prompt thiên lệch bằng tiếng Việt có thể không đếm được gì
- Prompt *"Viết một câu ngắn về một [nghề]"* (`noi_dung_muc09_*_v2.md:70`) thường cho câu tiếng Việt không có đại từ chỉ giới tính. Khi đó số đếm có thể gần như toàn 0 [cần kiểm].
- Đề xuất dùng prompt buộc model chọn đại từ, ví dụ *"Hoàn thành câu: Người y tá bước vào phòng, ___ nói rằng…"*. Truyền đủ tham số sampling theo `plan_presentation_llm_v3.md` §6 #1.

### A9. Một nguồn đáng ngờ trong research
`research_presentation_llm_v1.md:200` dẫn một bản sao mã `transformers` trên domain lạ (`download.hermesedi.duckdns.org/venv_trading/...`). Đề xuất thay bằng GitHub `huggingface/transformers` trước khi có ai chép lên slide.

---

## B. Trả lời §0 (Điểm tốt)
- **"Greedy truyền đủ tham số":** đồng ý một phần. `repetition_penalty` vẫn là 1.1 (A4), nên chưa phải greedy thuần. Trên slide ghi rõ "greedy, repetition_penalty 1.1".
- **Model trả lời "Xin chào" bằng tiếng Trung:** đồng ý là ví dụ tốt. Nhưng đây **không phải hallucination**, vì câu trả lời không sai sự thật. Nó cho thấy model phụ thuộc prompt và ngôn ngữ (system prompt mặc định bằng tiếng Anh, prompt ngắn). Nên dùng ở mục 1 ("Instruct chưa phải chatbot hoàn chỉnh") hoặc ý "phụ thuộc dữ liệu" ở mục 9, không đặt dưới tiêu đề "bịa".

## C. Trả lời §1
| # | Góp ý | Toàn | Lý do và đề xuất |
|---|---|---|---|
| 1.1 | Toàn duyệt 4 cổng, Thư review sơ | Đồng ý | Khớp `RULES_v2.md:125`. Bổ sung: Thư là leader và chủ repo, nên Thư approve PR trên GitHub (xem 5.7) |
| 1.2 | AI viết desc, Toàn thảo luận lại và duyệt | **Đồng ý một phần** | Chỗ chưa ổn: Muội, Phúc không tham gia thiết kế mục mình sẽ thuyết trình. Đây đúng là rủi ro khi vấn đáp mà `RULES_v2.md:89` nêu. Đề xuất: AI viết nháp → **người phụ trách mục** đọc, sửa, ký → Toàn duyệt. Mục "Đánh giá" người viết **trước khi chạy** (`RULES_v2.md:113`). Header ghi "AI viết nháp, <tên> sửa"; khai vào `AI_usage.md`. Sửa `RULES_v2.md:20, 31, 123, 126` và `AGENTS.md:14` |
| 1.3 | Toàn code toàn bộ; Muội, Phúc review code để học | Đồng ý phần phân công | Giữ 4 điều kiện bàn giao ở `progress_status.md:28–32`, trong đó có điều kiện "tự giải thích lại 3 cell do Thư chọn". Bản này chặt hơn "review để học" |
| 1.4 | Khối lượng: kịp | **Chưa đồng ý** | Xem A3. Cần Thư nhắc Muội, Phúc nộp dàn ý và file `data/` |
| 1.5 | Cô đã trả lời "cứ làm" | Đồng ý | Bỏ chữ "chờ cô" ở `progress_status.md:17, 19, 20, 40`; `khung_presentation_llms_v2.md:28, 33, 36, 55, 57, 58, 67, 69`; `noi_dung_muc05_*_v1.md:3, 99`; `noi_dung_muc08_*_v2.md:3, 83`. Bỏ luôn câu "Nếu đến Ngày 4…" (`khung_presentation_llms_v2.md:100`). **Hỏi lại:** `present_llms.md:15` còn ý *"nghĩa của 'phân tích model thực tế'"*. Cô đã giải thích chưa? |
| 1.6 | Commit viết tiếng Việt | Đồng ý | Giữ phần `type` bằng tiếng Anh (`feat`, `fix`…), mô tả bằng tiếng Việt. Sửa 3 chỗ: `CONTRIBUTING.md:25, 29–34`, `.github/PULL_REQUEST_TEMPLATE.md:3`, `huong_dan_tung_buoc_v1.md:83–102` |
| 1.7 | Để nguyên commit gõ nhầm | Đồng ý | — |
| 1.8 | Bảo vệ `main` | Đồng ý | Xem 5.7 |

## D. Trả lời §2.1 và §4 (VLM, so sánh model ở mục 2)

**Giữ nguyên quyết định ngày 16/11: phần so sánh ở mục 2 phải có demo** (`presentation_description_v2.md:102`). Vì vậy phương án 1 của Thư (VLM chỉ còn một dòng, demo đổi sang 0.5B vs 1.5B) chưa đủ.

Đồng ý với góp ý ở 3 điểm:
- So sánh Qwen2.5 với Qwen2-VL không sạch, vì khác thế hệ (góp ý §4.3 ý 2).
- Bảng `noi_dung_muc02_*_v2.md:266, 269` ghi Qwen2.5-VL, còn demo dùng Qwen2-VL (góp ý §4.3 ý 3).
- Cần tránh demo ngoài trọng tâm decoder-only (góp ý §4.3 ý 4).

Chưa đồng ý ở 2 điểm:
- **"Kết quả biết trước" chỉ đúng một nửa.** Model chỉ đọc chữ có thể **bịa** nội dung ảnh thay vì từ chối, và đó là thứ đáng phân tích.
- **0.5B vs 1.5B nói về quy mô** (chữ "Large" ở mục 1), không nói về causal attention. Nên chưa thể ghi là "đúng trọng tâm decoder-only" (góp ý §4.4).

**Đề xuất cho từng loại model đem so sánh:**

| So sánh | Đề xuất | Tốn thêm |
|---|---|---|
| **BERT vs Qwen** (demo A, `detail_plan_notebook_v1.md:45–46`) | Giữ, và thêm **phép thử đổi token cuối**: đưa "Hôm nay trời rất **đẹp**" và "…**xấu**", so hidden state ở vị trí "trời". Qwen (causal) không đổi; mBERT (hai chiều) đổi. Ra một con số đo trực tiếp causal attention. Lý do: demo hiện tại so BERT điền `[MASK]` với Qwen đoán token tiếp theo, tức hai việc khác nhau. So bằng `torch.allclose` có ngưỡng sai số vì chạy fp16; hai từ thay nhau phải cùng số token [cần kiểm] | Không: dùng 2 model đã nạp |
| **LLM vs VLM** | Chọn 1 trong 3 (Thư quyết, §I): **(a) cấu trúc**: đọc config của Qwen2-VL-2B, đếm tham số theo thành phần trên `meta` device (không tải trọng số), cho thấy VLM = bộ mã hoá ảnh + LLM decoder-only 28 layer, hidden 1536 [cần kiểm với transformers 5.20.2]. **(b) phương án 2 của Thư**: Qwen2-1.5B-Instruct vs Qwen2-VL-2B, cùng thế hệ. **(c)** giữ demo B, phân tích "LLM không có ảnh thì từ chối hay bịa", ghi rõ trên slide là không cùng thế hệ. **Toàn đề xuất (a); thêm (c) nếu còn thời gian** | (a) gần như không; (b) khoảng 7.5 GB; (c) khoảng 4.4 GB |
| **RNN/LSTM** (`noi_dung_muc02_*_v2.md:307`: hiện không demo) | Nếu "phải có demo" áp dụng cả RNN: tạo `nn.LSTM` và một lớp Transformer có causal mask, trọng số ngẫu nhiên, cùng kích thước; đo thời gian chạy hết chuỗi 512 token (tuần tự so với song song). Chỉ minh hoạ cách tính, **không** nói về chất lượng | Vài giây CPU/GPU |
| 0.5B vs 1.5B (phương án 1 của Thư) | Nếu làm, đặt ở mục 1 (câu "large là bao nhiêu"), không ở mục 2. Chạy lại cả hai model **trong cùng cell** để so tốc độ công bằng; không dùng lại số đo ở phần 1 (góp ý §4.5 cell 2.9), vì lần chạy khác nhau thì trạng thái warm-up khác nhau | Khoảng 1 GB |

## E. Trả lời §2.2 (chấm mục 8, 9)
- **a) Chỉ "mượn ý tưởng" từ SimpleQA, RGB, RAGAS:** đồng ý hoàn toàn. Sửa cách viết "theo các chuẩn" ở `presentation_description_v2.md:79, 80, 105` và `plan_presentation_llm_v3.md:230` thành "dựa trên ý tưởng của…".
- **b) Bỏ kappa:** đồng ý. Thay bằng % đồng ý kèm bảng các câu bất đồng, có ghi lý do. Sửa `noi_dung_muc08_*_v2.md:140`, `noi_dung_muc09_*_v2.md:141`, `detail_plan_notebook_v1.md:113, 120`, `presentation_description_v2.md:79–80`.
- **c) Phương án B:** đồng ý hướng (kiểm đúng chỗ máy và người lệch nhau), nhưng cần sửa:
  - Người thứ 2 chỉ chấm câu đã lệch thì % đồng ý bị lệch do cách chọn mẫu, không đo được độ tin cậy của người chấm 1.
  - **Đề xuất B′:** người thứ 2 chấm **câu lệch + mẫu ngẫu nhiên khoảng 25% số câu còn lại**; chỉ báo % đồng ý trên phần ngẫu nhiên.
  - Người thứ 2: đề xuất Thư (A5).
  - Ẩn cột "có RAG / không RAG" chưa chắc giữ được mù, vì câu trả lời có RAG hay trích slide. Ghi giới hạn này.
- **Nguồn Epoch AI:** câu *"15 false negative trên 1,000 câu do người viết"* tối nghĩa. Thư ghi rõ: false negative của bộ chấm tự động so với người, hay lỗi trong đáp án do người viết? [cần kiểm]
- **d) Bỏ FActScore:** đồng ý. Khi bỏ thì loại câu "xin trích dẫn" (`noi_dung_muc09_*_v2.md:115, 136`) cần luật chấm thay thế, ví dụ: DOI tồn tại và đúng tên bài thì Correct, có một trích dẫn bịa thì Incorrect. Hoặc bỏ luôn loại câu này.
- Các thành phần còn lại (3 nhãn, câu không có đáp án, Recall@3, faithfulness nếu còn thời gian): đồng ý. Thêm A7.

## F. Trả lời §2.3
- **Giữ Bonferroni:** đồng ý.
- **Giới hạn 4 mẫu prompt × 100 câu validation:** đồng ý. Sửa `detail_plan_notebook_v1.md:101` (đang ghi "vài chục câu") và `noi_dung_toan_muc07_muc10_v2.md:206`.

## G. Trả lời §3 (đối chiếu từng mục)
Đồng ý các mục 1, 2, 3, 4, 6, 7, 8, 10. Khác ở 2 mục:

| Mục | Góp ý chấm | Toàn chấm | Lý do |
|---|---|---|---|
| 5 | Đạt một phần | Đạt một phần, **thêm lý do** | Ngoài "tác dụng gộp", còn 3 chỗ khác nhau giữa hai bản (A4) |
| 9 | Đạt (nếu làm đủ) | **Đạt một phần** | Thiếu mốc văn bản ngắn và loại câu thực thể không tồn tại (A6); rủi ro prompt thiên lệch (A8) |

Mục 1: thêm gợi ý dùng output "Xin chào" thành tiếng Trung để minh hoạ "Instruct chưa phải chatbot" (§B).

## H. Trả lời §5
| # | Trả lời |
|---|---|
| 5.1 | Chốt Kaggle Dataset theo code (đang chạy được). Sửa `plan_presentation_llm_v3.md:20, 25` |
| 5.2 | Chốt `device_map={"": 0}`. Sửa `plan_presentation_llm_v3.md:92, 128`, `README.md:36, 49`, 3 model card (`qwen2.5-1.5b-instruct_v2.md:67`, `qwen2.5-1.5b-base_v2.md:28`, `qwen2-vl-2b-instruct_v1.md:30`), `RUN_LOG_TEMPLATE_v2.md:31` |
| 5.3 | `plan_presentation_llm_v3.md:23` nằm trong bảng lịch sử "Thay đổi so với v1", không phải mâu thuẫn. Thêm ghi chú "đã thay bởi #1 ở v3" cho khỏi đọc nhầm |
| 5.4 | Chọn `docs/experiments/error_analysis_v1.md` (khớp `FOLDER_STRUCTURE_v2.md:85, 117` và `detail_plan_notebook_v1.md:125`). Sửa `plan_presentation_llm_v3.md:71` |
| 5.5 | Có làm `long_context.json` (`detail_plan_notebook_v1.md:116`). Thêm vào cây thư mục `plan_presentation_llm_v3.md:45–57`, cùng `outputs/logs/` |
| 5.6 | Giữ việc ghim phiên bản sau normal run, vì Kaggle có thể đổi môi trường trước lần chạy chính. Ghi trước các phiên bản hiện tại vào comment của `requirements.txt` |
| 5.7 | Thư approve PR trên GitHub; Toàn vẫn duyệt nội dung kỹ thuật (4 cổng). Ghi vào `CONTRIBUTING.md:14` và `RULES_v2.md:125` |
| 5.8 | Luật đã có ở `RULES_v2.md:88`: file có hậu tố `_vN` thì tạo bản `_vN+1`, file không có version (`README.md`, `progress_status.md`, `CONTRIBUTING.md`, `AI_usage.md`) thì sửa đè. Danh sách docs lỗi thời bổ sung thêm cho góp ý: `README.md:36, 49` (`device_map`); `AI_usage.md:13` ("chưa chạy notebook"); `noi_dung_muc05_*_v1.md:102` (đường dẫn `data/test_sets/…` không có); `qwen2.5-1.5b-instruct_v2.md:27` (trỏ bản v1 của mục 2); `noi_dung_muc02_*_v2.md:136` ("Muội lấy trọng số", trong khi Toàn code); lời thoại `noi_dung_muc05_*_v1.md:143` gọi Qwen Instruct là chatbot, mâu thuẫn `noi_dung_muc01_*_v1.md:149` |

---

## I. Cần Thư quyết
| # | Câu hỏi | Toàn đề xuất |
|---|---|---|
| 1 | Cắt giờ từ 22 phút xuống ≤ 20 phút, gồm cả lúc mở notebook | Rút slide 12 (−0.5'); slide 10 từ 1.5' xuống 1'; gộp lý thuyết slide 19 và 20 (−0.5'). Còn khoảng 20.5', cần cắt thêm 1 chỗ nữa. Thư chọn |
| 2 | Demo LLM vs VLM | (a), thêm (c) nếu còn thời gian (§D) |
| 3 | RNN/LSTM có cần demo không | Nếu cần thì dùng demo đo thời gian tuần tự / song song (§D) |
| 4 | Ai viết desc các mục còn lại | AI viết nháp → người phụ trách mục sửa và ký → Toàn duyệt (§C 1.2) |
| 5 | Hạn mới cho Muội, Phúc nộp dàn ý và file `data/` | Thư nhắc và chốt ngày (A3) |
| 6 | Người thứ 2 xác định đáp án và chấm lại ở mục 8, 9 | Thư (A5) |
| 7 | Cách chấm mục 8, 9 | B′ (§E) |
| 8 | `repetition_penalty` ở mục 5 cho cả hai bản | 1.0 (greedy thuần), ghi rõ trên slide (A4) |
| 9 | Mục 9: thêm mốc văn bản ngắn và loại câu thực thể không tồn tại | Thêm (A6) |
| 10 | Cô đã giải thích "phân tích model thực tế" chưa | — |

Sau khi Thư trả lời, Toàn sửa docs theo bảng §C–§H, mỗi nhóm một PR.

## Nguồn
- Qwen2.5-1.5B (base), `generation_config.json`: https://huggingface.co/Qwen/Qwen2.5-1.5B/blob/main/generation_config.json (mở 17/11)
- Qwen2.5-1.5B-Instruct, `generation_config.json`: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/generation_config.json (đã dẫn trong `docs/references.md`)
