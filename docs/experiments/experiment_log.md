# Sổ ghi các lần chạy và thí nghiệm

- **Created**: 2026-11-15T16:30:00+07:00 (Toàn)
- Mỗi lần chạy (smoke / normal) ghi một mục tóm tắt theo `docs/templates/RUN_LOG_TEMPLATE_v2.md`; log đầy đủ ở `outputs/logs/<YYYYMMDD_HHMMSS>_<smoke|normal>/` (smoke: `outputs/_smoke/logs/`). Mỗi thí nghiệm có so sánh ghi thêm theo `docs/templates/EXPERIMENT_NOTE_TEMPLATE_v1.md`.
- Số liệu lấy từ `outputs/results/` của chính lần chạy đó. Mục mới ghi ở trên cùng.

---

## 20261129_021918_local — chạy lại cell 10.3e → 10.6 với nhãn tay v2 (không gọi model, không gọi giám khảo)

- Người chạy: AI (Claude Code) theo yêu cầu của Toàn | máy cá nhân, venv ngoài repo (pandas, matplotlib) | đọc `outputs/results/` của run `20261129_003355_normal`; exec nguyên code cell 10.3e, 10.4, 10.5c, E1–E14, 10.6 | log: `outputs/logs/20261129_021918_local/`
- Lý do: `data/human_labels_v2.json` (Toàn gắn 29/11 qua `data/human_labels_v2_sheet.xlsx`; AI sửa 2 ô trái luật theo yêu cầu) thêm nhãn theo từng metric: Answer Relevancy, Hallucination, Faithfulness, Semantic equivalence, Completeness. 10.3e so mỗi metric với đúng nhãn của nó.
- Kết quả: chỉ `judge_vs_human.csv`, `judge_vs_human_rows.csv` đổi nội dung (các file khác trùng run gốc); chép 2 file này + `figures/suite_E13.png` vào `outputs/`.
  - Mục 8: Answer Relevancy 22/36, Hallucination 31/36, Faithfulness 14/18, G-Eval Correctness 34/36, Semantic equivalence 36/36, Completeness 36/36. 9.1: Hallucination 24/30, Correctness 29/30. 9.3 False premise 6/6.
  - Answer Relevancy: 13/14 câu lệch là câu từ chối (người: liên quan; giám khảo: không).
  - Giám khảo Qwen2.5-7B (điểm từng câu ở commit 79bfe61, run 20261125_082211, cùng trung bình run 20261125_214650; output 36/36 trùng) so cùng nhãn v2, tính bằng script ngoài notebook: 14/36, 12/36, 11/17, 31/36, 27/36, 32/36.
- Notebook trên Kaggle cần `data/human_labels_v2.json` (Kaggle Dataset New Version) trước lần chạy sau.

---

## 20261129_003355_normal — chạy lại sau khi sửa VRAM (**lần chạy chính thức mới**)

- Người chạy: Toàn | Kaggle, GPU T4 x2 | 41.3 phút (run trước 44.1), phần 0–10 và 10.5–10.6 OK, 0 lỗi | log: `outputs/logs/20261129_003355_normal/`; `outputs/` đã chép kết quả run này
- Thay đổi notebook so với run 20261128_232631: 5.5 xoá `inputs` + `assert` weakref (base được giải phóng); 5.1 base nạp lên GPU 1, 5.5 sinh IFEval 2 bản song song; `cuda_sync()` đồng bộ mọi GPU; 10.3e so Hallucination theo cách B (từ chối = không bịa); đổi tên "bộ 18 metric đánh giá LLM".
- **VRAM đúng:** 6.6 trọng số 2.90 GB, đỉnh 2.90–2.91 GB; 9.5 đỉnh 2.96 / 3.07 / 3.46 / 4.91 / 10.42 GB (512 → 8,192 token), khớp run tham chiếu 20261127_234527 (2.95–10.41). Không cần lấy VRAM từ run khác nữa.
- **Thời gian:** phần 5 11.3 → 8.4 phút (IFEval Instruct cuda:0 393.5 s, base cuda:1 434.8 s, chạy cùng lúc); các phần khác như cũ; giám khảo 17.6 phút.
- **Output model giống hệt run trước** ở mọi file (IFEval 200, HumanEval, RAG, 9.1, ICL 2 × 2,999, 5.4, mục 1, 9.4b, mục 6) → số theo metric tự động không đổi: HumanEval 0.561 (92/164); IFEval base 0.250 / 0.331, Instruct 0.400 / 0.552 (prompt / instruction strict); mục 8 EM / F1 RAG 0.200 / 0.395; ICL 0.899 → 0.923. IFEval base có 10/100 output rỗng (model sinh EOS ngay), cả 2 run như nhau.
- **Giám khảo dao động trên cùng output:** DeepEval (chấm 1 lần) 26/130 câu đổi đạt / không giữa 2 run; trung bình nhóm lệch tới 0.111 (Answer Relevancy không RAG 0.500 → 0.389), Bias 0.857 → 0.950. G-Eval (TB 20 lần) 7/201 cặp đổi đạt / không, trung bình nhóm lệch TB 0.017, max 0.115.
- Số giám khảo run này: DeepEval Answer Relevancy 0.389 → 0.875, Hallucination 0.778 → 0.833, Faithfulness 0.889; 9.1 Hallucination 0.533 / 0.600; Bias 0.950 (10/10). Khớp nhãn tay: DeepEval Hallucination 29/36 (mục 8, cách B), 19/30 (9.1); G-Eval Correctness 34/36, 29/30; False premise 6/6.

---

## 20261128_232631_normal — normal run theo bộ đánh giá của cô (HumanEval, IFEval, 9.4b Bias, 10.3e nhãn tay)

- Người chạy: Toàn | Kaggle, GPU T4 x2, Python 3.13.16, torch 2.11.0+cu130, transformers 5.20.2, deepeval 4.3.5 | 44.1 phút, phần 0–10 và 10.5–10.6 OK | log: `outputs/logs/20261128_232631_normal/`; `outputs/` đã chép kết quả run này
- Kết quả theo metric của cô:
  - HumanEval Pass@1 (greedy) **0.561** (92/164), 0 timeout.
  - IFEval 100 prompt, prompt / instruction-level strict: base 0.250 / 0.331, Instruct **0.400 / 0.552** (loose 0.270 / 0.344 và 0.430 / 0.583). Bị cắt ở 1024 token: base 47, Instruct 6.
  - Mục 8 EM / Token F1 (15 câu có đáp án): không RAG 0.000 / 0.007, RAG 0.200 / 0.395. G-Eval Correctness 0.17 → 0.60. DeepEval: Answer Relevancy 0.50 → 0.86, Hallucination 0.72 → 0.83, Faithfulness 0.86.
  - 9.1 Token F1 A / B 0.079 / 0.097; G-Eval Correctness 0.21 / 0.22; DeepEval Hallucination 0.53 / 0.60. 9.3 G-Eval False premise 0.00 / 0.06 (model chấp nhận tiền đề sai cả 3 câu).
  - 9.4 Pearson r: en 0.69, vi 0.44. 9.4b DeepEval Bias TB 0.857 (9/10 nghề đạt; 1 = không thiên lệch).
  - Mục 7 accuracy 0.899 → 0.923, McNemar p = 1.5e-08 (không đổi so với run trước).
  - Giám khảo: DeepEval 210 lần gọi, 1.3 phút; G-Eval N = 20, 4,230 lần gọi cộng dồn, 15.7 phút, σ TB 0.023, lớn nhất 0.276, 1 lỗi JSON (mục 6 T = 1.5 Coherence).
  - 10.3e khớp nhãn tay (72 nhãn, 0 output đổi): G-Eval Correctness 34/36 (mục 8), 28/30 (9.1); False premise 6/6; DeepEval Hallucination 20/36 (mục 8), 19/30 (9.1).
- **Lỗi đo VRAM (do sửa 5.4 ngày 28/11):** 6.6 trọng số 5.76 GB, 9.5 đỉnh 5.82–13.28 GB, cao hơn run 20261127_234527 khoảng 2.87 GiB (= 1 model 1.5B fp16). Nguyên nhân: khi bỏ chấm tay ở 5.4 đã bỏ luôn `del inputs`, biến `inputs` của 5.4 vẫn trỏ tới `model_base` nên 5.5 xoá xong base vẫn ở GPU 0. Các số khác không bị ảnh hưởng (greedy, không phụ thuộc VRAM); thời gian 6.6 chậm hơn ~5%. Sửa: 5.5 xoá thêm `inputs`, `gc.collect()`, `torch.cuda.empty_cache()`, `assert weakref` base đã được giải phóng.
- Phát hiện: DeepEval Hallucination khớp nhãn tay thấp vì 2 thứ đo khác nhau. Nhãn tay ghi "trả lời đúng"; Hallucination chỉ hỏi "có mâu thuẫn tài liệu không". Mục 8 có 11/16 câu lệch là câu **từ chối** ("Không có thông tin"): người ghi chưa đạt, giám khảo ghi không bịa. 9.1 có 11 câu trả lời sai mà giám khảo vẫn cho đạt.
- Quyết định (Toàn, 29/11): **chạy lại normal run** với notebook đã sửa 5.5 (run này không dùng làm chính thức); 10.3e so Hallucination theo cách B: phía người "không bịa" = nhãn Đúng hoặc Từ chối, chỉ Sai là bịa (Correctness, False premise giữ `human_ok`). Thử 10.3e đã sửa trên dữ liệu run này: Hallucination mục 8 khớp 26/36 (cũ 20/36), 9.1 19/30, G-Eval không đổi.

---

## 20261128_221403_smoke — notebook theo bộ đánh giá của cô (bỏ bộ cũ; HumanEval, IFEval, 9.4b Bias, 10.3e nhãn tay)

- Người chạy: Toàn | Kaggle, GPU T4 x2, Internet bật, secret `JUDGE_API_KEY` | 10.3 phút, phần 0–10 và 10.5–10.6 OK, 0 lỗi | log: `outputs/_smoke/logs/20261128_221403_smoke/` (không commit)
- Thay đổi notebook (28/11): 1.4 bỏ luật từ khoá; **1.7 HumanEval**; 4.6 top-1 = Accuracy; 5.4 bỏ chấm tay; **5.5 IFEval**; 8.4 EM / F1 thay chấm 3 mức, bỏ recall@3; 9.1–9.3 bỏ chấm tay; **9.4b DeepEval Bias**; 9.5 Exact Match; 10.3c thêm Hallucination 9.1, Bias; 10.3d thêm tiêu chí False premise, ca 9.1, 9.3; **10.3e** kiểm giám khảo bằng `data/human_labels_v1.json`; 10.4, thẻ E1–E14 theo metric của cô.
- Smoke (số nhỏ, chỉ kiểm chạy được): HumanEval 8/10 (sai HumanEval/1 AssertionError, HumanEval/9 IndexError); IFEval 8 prompt: base strict 0.000 / 0.154, Instruct 0.250 / 0.385 (prompt / chỉ dẫn), bị cắt ở 1024 token: base 3, Instruct 1; 9.4b Bias TB 0.9 (10 nghề); DeepEval 84 lần gọi 0.7 phút, G-Eval N = 2, 302 lần gọi cộng dồn 1.1 phút, 0 lỗi JSON; 10.3e so 50 cặp (28 output có nhãn tay, cả 28 trùng output đã đọc).
- Sự cố: 10.3e in "bỏ 44 câu do output đổi" — sai: 44 nhãn **không có** trong smoke (smoke chỉ chạy 5 + 1 câu mục 8, 5 câu 9.1), không phải output đổi. Sửa: tách `in_run` và `changed`, chỉ cảnh báo khi output thật sự đổi; kiểm lại trên dữ liệu đủ: 72/72 trùng.
- Kết luận: đạt, sẵn sàng normal run (Restart session trước, `SMOKE = False`).

---

## 20261128_041850_normal — G-Eval N = 20, 16 luồng (**lần chạy chính thức mới**; VRAM 6.6 / 9.5 lấy run 20261127_234527)

- Người chạy: Toàn | Kaggle, GPU T4 x2 | 22.0 phút, 188 cell, 0 lỗi | log: `outputs/logs/20261128_041850_normal/`
- Thay đổi notebook: G-Eval chấm N = 20 lần / tiêu chí lấy trung bình; giám khảo gọi song song 16 luồng (10.3c, 10.3d).
- Kết quả giám khảo: G-Eval 3,300 lần chấm, 0 lỗi JSON, 11.1 phút, σ trung bình 0.022, lớn nhất 0.251 (mục 8 RAG 0009-0012-0001 Semantic equivalence); 5/165 cặp σ > 0.1. Mục 5 Instruction following base 0.16 / Instruct 0.99; mục 8 Correctness 0.17 → 0.59. Khớp nhãn tay: G-Eval Correctness 34/36; DeepEval (chấm 1 lần) Hallucination 15/36, Answer Relevancy 20/36, Faithfulness 12/18.
- **Lỗi đo:** VRAM 6.6 / 9.5 lệch đều +2.97 GiB (trọng số 5.86 thay vì 2.89 GiB). Nguyên nhân: run `20261128_041411_normal` dừng giữa chừng (tới phần 7), rồi Run All lại trong cùng phiên không Restart → model của lần trước còn trên GPU 0. Các file khác trùng run 20261127_234527 trừ cột thời gian.
- Sự cố khác: output 10.3c/10.3d in widget lỗi "Error displaying widget: model not found" (thanh tiến trình DeepEval từ nhiều luồng) → sửa `measure(case, _show_indicator=False)`.
- Quyết định (Toàn, 28/11): **không chạy lại**. Dùng run này làm chính thức; số VRAM 6.6, 9.5 (và thời gian 6.6, 9.5 trên slide) lấy run 20261127_234527_normal, lưu bản đối chiếu `outputs/logs/20261127_234527_normal/vram_ref_efficiency.csv`, `vram_ref_context_cost.csv`; notebook thêm ghi chú ở markdown 6.6, 9.5. `outputs/` đã chép kết quả run này (kèm log `20261128_041411_normal` dừng giữa chừng).
- DeepEval (chấm 1 lần): Answer Relevancy 0.556 → 0.861, Hallucination 0.833 → 0.889, Faithfulness 0.889 (16/18 đạt); câu từ chối: Hallucination 0.82, Answer Relevancy 0.53.

---

## 20261128_040436_pilot — chạy thử chọn số lần chấm N cho G-Eval (giám khảo không trả log-prob)

- Người chạy: Toàn (AI hỗ trợ, Claude Code) | máy cá nhân, không GPU, Python 3.13.7, deepeval 4.3.5, openai 3.60.0 | log: `outputs/logs/20261128_040436_pilot/` (`run.log`, `pilot_geval_records.csv`, `pilot_geval_summary.csv`)
- Lý do: gpt-6.2-sol (và mọi model trên ttmapi.site, thử 28/11) không trả log-prob → G-Eval chỉ có điểm thô. Bài báo G-Eval (Liu et al. 2023) khi GPT-4 không có log-prob thì lấy mẫu 20 lần để ước lượng điểm kỳ vọng. Chạy thử để đo độ dao động σ của giám khảo rồi chọn N.
- Thiết kế: 9 ca (mục 1 tóm tắt, dịch; mục 5 base; mục 6 greedy, T = 0.5 × 2; mục 8 RAG một câu điểm giữa, một câu Đúng, một câu Sai) × 20 lần chấm = 180 lần; dùng nguyên `GEVAL_SPECS` cell 10.3d và `ApiJudge` cell 10.3c; output lấy từ run 20261127_234527_normal; 8 luồng song song. Code: script tạm ngoài repo (không thuộc notebook).
- Kết quả: **đạt**, 1.2 phút, 180 lần gọi, 61,680 / 16,627 token, 0 lỗi JSON.
  - Câu rõ ràng (mục 8 Đúng / Sai): σ = 0, 20 lần ra cùng điểm.
  - Câu điểm giữa (mục 6, 0.4–0.5): σ = 0.08–0.105; một lần chấm dao động 0.3–0.6, **cắt qua ngưỡng đạt 0.5**.
  - σ lớn nhất 0.105 → để sai số chuẩn σ/√N ≤ 0.025 cần N ≥ 18.
- [đề xuất] Đề xuất: N = 20 (bằng bài báo G-Eval, ≥ 18), chạy song song nhiều luồng. Chờ Toàn duyệt trước khi sửa cell 10.3d.

---

## 20261127_234527_normal — giám khảo DeepEval / G-Eval đổi sang API `gpt-6.2-sol` (**lần chạy chính thức mới**, chờ Toàn duyệt dùng cho slide)

- Người chạy: Toàn | Kaggle, 2 × T4, Internet bật | Chế độ: normal | notebook đã chạy và `outputs/` (qua `outputs.zip`) chép vào repo 28/11 (results, figures, log run này; không chép `_smoke`, không chép run dở `20261127_234503_normal`)
- Thay đổi so với 20261125_214650_normal (AI hỗ trợ, Claude Code): cell 10.3c thêm `JUDGE_BACKEND` và lớp `ApiJudge` (OpenAI-compatible `chat/completions`); giám khảo `gpt-6.2-sol` qua dịch vụ trung gian `https://ttmapi.site/v1` (không phải API chính thức OpenAI, không kiểm chứng được model thật phía sau), `temperature=0` (dịch vụ có thể bỏ qua), `max_tokens=1024`; dịch vụ không trả log-prob nên G-Eval vẫn điểm thô; 10.3d dùng chung giám khảo; phần 10.5–10.7; 188 cell.
- Kết quả: **đạt**, 188 cell, 0 lỗi, 29.3 phút. DeepEval 162 lần gọi, 11.3 phút; G-Eval cộng dồn 327 lần gọi, 8.5 phút; token cộng dồn 132,986 vào / 29,883 ra; **0 câu JSON lỗi**. 34/43 file kết quả trùng run trước (chỉ khác cột thời gian, bộ nhớ); `icl_metrics.json` chỉ khác `run_id`.
- DeepEval (cũ Qwen2.5-7B → mới): Answer Relevancy không RAG 0.056 → 0.778, có RAG 0.753 → 0.806; Hallucination không RAG 0.000 → 0.889, có RAG 0.556 → 0.778; Faithfulness 0.627 (n = 17) → 0.778 (n = 18).
- G-Eval mục 8 Correctness 0.22 → 0.17 (không RAG), 0.38 → 0.58 (có RAG); mục 5 Instruction following base 0.00 → 0.18, Instruct 0.82 → 1.00.
- Độ khớp nhãn tay: G-Eval Correctness mục 8 **34/36** (Qwen 31/36). Hallucination "≥ 0.5 ↔ nhãn = Đúng" chỉ 14/36 (Qwen 32/36), Answer Relevancy 15/36: giám khảo mới cho câu **từ chối** điểm cao (không mâu thuẫn nguồn, đúng trọng tâm), đúng định nghĩa metric nhưng không còn dùng được như thước "đúng / sai".
- Smoke trước khi chạy: `20261127_231551_smoke` đạt (14.7 phút, 146 lần gọi, 0 lỗi JSON); không chép vào repo.
- Lưu ý bảo mật: notebook đang chứa API key ở cell 10.3c; phải xoá trước khi commit / nộp.

---

## 20261127_221729_local — chỉ chạy cell 10.5c–10.6 (tổng hợp bộ đánh giá), đọc lại kết quả run 20261125_214650_normal

- Người chạy: Toàn (AI hỗ trợ, Claude Code) | máy cá nhân, CPU, Python 3.13.7, pandas 3.1.1, matplotlib 3.11.2 | **không tải model, không chạy cell 0–10.4** | log: `outputs/logs/20261127_221729_local/`
- Thay đổi notebook: 24 cell markdown đánh giá thêm dòng "Bộ đánh giá: Ex · tên bộ"; chèn 34 cell mới 10.5–10.7 trước cell nén `outputs` (thẻ 14 bộ đánh giá E1–E14: sơ đồ, ma trận, hình, bảng thông tin, 5W1H; bảng bộ cũ 21/11 và danh sách 18 metric của cô; bản đồ bộ × mục; khuyến nghị); 154 → 188 cell. Output cell cũ giữ nguyên.
- Kết quả: **đạt**, 16 cell code chạy hết, không lỗi, 16.8 giây. Mọi số trùng `outputs/results/` của run 20261125_214650_normal; độ khớp giám khảo tính lại: DeepEval Hallucination 32/36, G-Eval Correctness mục 8 31/36.
- File mới: `outputs/results/metric_suites.csv`, `outputs/figures/suite_E1.png` … `suite_E14.png`, `metric_suites_overview.png`.
- Lệch luật: `RUN_LOG_TEMPLATE_v3.md` §5.2 yêu cầu chạy lại toàn bộ khi sửa code; lần này chỉ chạy phần mới (không phụ thuộc model). [cần chốt] Toàn quyết định có Save & Run All trên Kaggle hay giữ như vậy.

---

## 20261125_214650_normal — chạy lại sau sửa đo VRAM (**lần chạy chính thức cuối**, notebook nộp và số cho slide lấy từ đây)

- Người chạy: Toàn | Kaggle, 2 × T4 (model chính GPU 0, giám khảo GPU 1), Internet bật | Chế độ: normal | notebook đã chạy và `outputs/` (qua `outputs.zip`) chép vào repo 25/11
- Thay đổi so với 20261125_082211_normal (theo review của Thư, AI hỗ trợ, Claude Code): cell 5.4 `del inputs` trước khi xoá model base (biến `inputs` của 5.3 giữ `model_base` trên GPU 0 tới hết notebook); cell 6.6 đổi `/ 1e9` → `/ 1024**3` cho VRAM đỉnh và RAM (thống nhất GiB với `vram_used_gb()`); 154 cell.
- Kết quả: **đạt**, chạy hết, không lỗi; 41.2 phút (DeepEval 15.2 phút, 159 lần gọi; G-Eval 13.7 phút, 324 lần gọi cộng dồn). "VRAM sau khi xoá base": 2.88 GiB (run trước 5.76). Mọi file kết quả trùng 20261125_082211_normal, trừ cột thời gian và bộ nhớ.
- 6.6 (`efficiency.csv`): TTFT 0.045–0.048 s; latency 2.31 / 3.46 / 8.44 s; ≈ 26 token/s; VRAM đỉnh 2.90 GiB (trọng số 2.89 GiB), RAM 2.35 GiB (run trước 6.2 GB, 2.67 GB, có model base).
- 9.5 (`context_cost.csv`): 512 token 0.12 s, 2.95 GiB; 8,192 token 9.9 s, 10.41 GiB (run trước 13.28 GiB, có model base).
- `20261125_213700_normal` dừng sau phần 8, không dùng, không chép vào repo.

---

## 20261125_082211_normal — phần 0–10 + metric theo danh sách của cô (lần chạy chính thức trước; VRAM 6.6, 9.5 đo sai vì model base chưa giải phóng)

- Người chạy: Toàn | Kaggle, 2 × T4 (model chính GPU 0, giám khảo GPU 1), Internet bật | Chế độ: normal | notebook đã chạy và `outputs/` (qua `outputs.zip`) chép vào repo 25/11
- Thay đổi so với 20261124_180414_normal (AI hỗ trợ, Claude Code): cell 1.6 (unit-test pass rate 7 test, Pass@k với 10 mẫu T = 0.8, top-p 0.95), 4.6 (top-1 / top-5 accuracy token tiếp theo), 6.5 (distinct-1/2, tỉ lệ lặp), 6.6 (TTFT, latency, throughput, VRAM, RAM, GPU-giây; `GPU_USD_PER_HOUR = None`), 10.3d (G-Eval 5 tiêu chí cho mục 1, 5, 6, 8, cùng giám khảo 10.3c); 154 cell.
- Kết quả: **đạt**, chạy hết, không lỗi; 41.2 phút (DeepEval 14.8 phút, 159 lần gọi; G-Eval 13.5 phút, 324 lần gọi cộng dồn). Mọi file kết quả đã có trùng 20261124_180414_normal, chỉ khác cột thời gian (`time_s`, `context_cost.csv`, `prompt_selection.csv`).
- Mới, 1.6 (`code_passk_summary.csv`): greedy 7/7 test; 10/10 mẫu qua cả 7 test → Pass@1 = Pass@5 = 1.0 (bài dễ nên bão hòa).
- Mới, 4.6 (`next_token_accuracy.csv`): top-1 / top-5 câu đúng 0.43 / 0.71, xáo trộn 0.12 / 0.25, tiếng Anh 0.33 / 0.67.
- Mới, 6.5 (`decoding_diversity.csv`): distinct-2 greedy 0.33, T = 0.5 0.65, top-p 0.5 0.76, top-p 0.95 0.84, T = 1.5 0.99; tỉ lệ lặp ≤ 0.06.
- Mới, 6.6 (`efficiency.csv`, trung vị 5 lần): TTFT 0.045–0.046 s; latency 2.30 / 3.46 / 8.37 s (60 / 90 / 219 token ra); ≈ 26 token/s; VRAM đỉnh 6.2 GB; RAM 2.67 GB.
- Mới, 10.3d (`geval_scores.csv`, `geval_summary.csv`): mục 5 Instruction following base 0.00 / Instruct 0.82, Coherence 0.05 / 0.95; mục 8 Correctness 0.22 → 0.38, Semantic equivalence 0.00 → 0.21, Completeness 0.08 → 0.27; mục 6 Coherence greedy 0.70, T = 1.5 0.00, Instruction following 0–0.20; mục 1 Correctness 1.0 / 0.8 / 0.5 (hỏi đáp / tóm tắt / dịch); 1 câu JSON lỗi (top-p 0.95, Coherence).
- `20261125_082125_normal` dừng sau phần 0, không dùng, không chép vào repo.

---

## 20261124_180414_normal — phần 0–10 + bộ metric nhóm chốt (lần chạy chính thức trước)

- Người chạy: Toàn | Kaggle, 2 × T4 (model chính GPU 0, giám khảo DeepEval GPU 1), Internet bật | Chế độ: normal | notebook đã chạy và `outputs/` (qua `outputs.zip`) chép vào repo 24/11
- Thay đổi so với 20261121_010447_normal (AI hỗ trợ, Claude Code): cell 10.3b (Exact Match, token F1 cho hỏi đáp mục 8; BLEU cho dịch; ROUGE-1/2/L cho tóm tắt, tham chiếu `REF_SUMMARY` nhóm tự viết) và 10.3c (DeepEval 4.3.5: Answer Relevancy, Faithfulness, Hallucination; giám khảo `Qwen/Qwen2.5-7B-Instruct` 4-bit NF4, greedy); 10.4 thêm 2 dòng; 144 cell.
- Kết quả: **đạt**, chạy hết, không lỗi; 26.4 phút (DeepEval 14.9 phút, 159 lần gọi giám khảo). Phần 1–10 trùng từng file với 20261121_010447_normal, chỉ khác cột thời gian. Chi phí ngữ cảnh 8,192 token: 8.7 s (cũ 10.7 s), gấp 83 lần 512 token; bộ nhớ vẫn 13.28 GB.
- Mới, 10.3b (`lexical_metrics.csv`, `qa_em_f1.csv`): hỏi đáp 15 câu EM 0.0 → 0.2, token F1 0.007 → 0.395 (không RAG → RAG); dịch BLEU 26.3; tóm tắt ROUGE-1/2/L F1 0.550 / 0.341 / 0.504.
- Mới, 10.3c (`deepeval_scores.csv`, `deepeval_summary.csv`), 18 câu, cả 3 metric cao = tốt: Answer Relevancy 0.056 → 0.753; Hallucination 0.000 → 0.556; Faithfulness (chỉ RAG) 0.627, 1 câu lỗi (`0019-0036-0002`, giám khảo trả JSON thiếu khóa `verdicts`). Giám khảo chấm sai vài câu đúng (ví dụ "20.4%" Hallucination 0, "14,9" Answer Relevancy 0): ghi ở `docs/drafts/tieu_chi_danh_gia_v2.md` §10b.
- Smoke run trước khi đạt (`outputs/_smoke/`, không chép vào repo): `20261124_163828_smoke` lỗi `AttributeError: 'LocalJudge' object has no attribute 'model_name'` (deepeval 4.x không tự đặt `model_name`) → sửa; chạy được nhưng Hallucination ngược chiều mô tả (deepeval 4.x tính tỉ lệ khớp) và Faithfulness lỗi JSON 2/6 (output giám khảo bị cắt ở 512 token) → ghim `deepeval==4.3.5`, `truths_extraction_limit=10`, `JUDGE_MAX_NEW_TOKENS = 1024`; `20261124_165525_smoke` đạt, 0 lỗi. Các normal run dở `20261124_154817_normal`, `20261124_162742_normal` (chạy nhầm với `SMOKE = False`) không chép vào repo.

---

## 20261121_010447_normal — phần 0–10 (lần chạy chính thức trước)

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
