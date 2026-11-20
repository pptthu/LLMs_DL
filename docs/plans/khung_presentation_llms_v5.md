# Khung presentation LLMs (v5)

- **Created**: 2026-11-18 (Toàn, AI hỗ trợ) · **Last Updated**: 2026-11-21 (Toàn, AI hỗ trợ)
- Thay `khung_presentation_llms_v4.md` (giữ nguyên). Thay đổi: Toàn trình bày cả 16 slide (21/11: thêm slide 4 sơ đồ kiến trúc; tách mục 9 thành slide 12, 13; mỗi slide nội dung có khung code ngắn và sơ đồ; bỏ dòng "ĐỀ" trên slide); slide 6, 8, 12–15 thêm hình mới của run `20261121_010447_normal`; slide 12–13 thêm thiên lệch, giới hạn ngữ cảnh, chất lượng theo dữ liệu; thời gian theo kịch bản `scripts/kich_ban_toan_v2.md`.
- Chỉ dựng từ yêu cầu của cô trong [`yeu_cau_cua_co_v1.md`](yeu_cau_cua_co_v1.md).

## 1. Khung slide

Tổng thời gian ≤ 20 phút, **gồm cả lúc mở notebook** (yêu cầu thêm 6). Cột "Phút" là mức tối đa cho slide đó.

| Slide | Mục | Nội dung (đề) | Minh họa (đề) | Demo output thật trên slide (yêu cầu thêm 3) | Notebook | Phút |
|---|---|---|---|---|---|---|
| 1 | — | Tên đề tài, thành viên, người trình bày: Toàn | — | — | — | 0.3 |
| 2 | — | Thư viện (yêu cầu thêm 1); môi trường chạy (yêu cầu thêm 2) | — | Phiên bản thật từ `env.json` | Phần 0 | 1.0 |
| 3 | 1. Giới thiệu | LLM là gì? Phân biệt language model, LLM và ứng dụng chatbot | Ví dụ các tác vụ: hỏi đáp, tóm tắt, dịch, sinh mã | Output thật của 4 tác vụ. Phân tích nối với slide 8: base ≈ LM chỉ viết tiếp; Instruct đã qua post-training; chatbot = Instruct + giao diện + lớp an toàn | Phần 1 | 1.5 |
| 4 | 2. Nền tảng kiến trúc (1/2) | Vai trò của Transformer; Decoder-only và causal attention | Sơ đồ kiến trúc | RNN vs self-attention; 3 họ Transformer kèm mask; khối decoder-only của Qwen2.5-1.5B (28 block, số tham số thật cell 2.1); causal attention Q, K, V + mask 4 × 4 | Phần 2.1, 2.2 (code trên slide) | 1.0 |
| 5 | 2. Nền tảng kiến trúc (2/2) | Vai trò của Transformer; Decoder-only và causal attention | Sơ đồ từ token đầu vào đến xác suất token tiếp theo | **Ma trận causal mask** của câu ví dụ, trọng số attention thật (phần trên đường chéo = 0), top-5 token tiếp theo | Phần 2 | 1.0 |
| 6 | 3. Tokenization | Cách chia token; token ID và embedding | Tokenize một câu tiếng Việt và một câu tiếng Anh | Câu tô màu theo token, bảng gom theo từ (cell 3.1b), số token Qwen vs GPT-2, `embedding_qwen.png` | Phần 3 | 1.2 |
| 7 | 4. Nguyên lý học | Next-token prediction, dữ liệu huấn luyện, cross-entropy loss | Tạo các cặp ngữ cảnh → token mục tiêu từ một câu | `context_target_pairs.png`, `per_token_loss.png`, loss tự tính = `outputs.loss` | Phần 4 | 1.3 |
| 8 | 5. Các giai đoạn huấn luyện | Pretraining, SFT, preference alignment; RLHF/DPO ở mức ý tưởng | Sơ đồ mục tiêu và dữ liệu của từng giai đoạn | Base vs Instruct, cùng prompt, cùng `do_sample=False`, `repetition_penalty=1.0`, cùng `max_new_tokens`. Lời thoại: "tác dụng gộp của SFT + alignment" (Qwen2.5-Instruct qua SFT, DPO, GRPO) | Phần 5 | 1.5 |
| 9 | 6. Quá trình sinh văn bản | Autoregressive generation; greedy, temperature, top-p | Cùng một prompt với các cấu hình sinh khác nhau | Sinh từng bước (vòng lặp tự viết = `generate()`); temperature làm phân phối nhọn/phẳng; 5 cấu hình × 3 lần | Phần 6 | 1.5 |
| 10 | 7. Khả năng của LLM | In-context learning, làm theo chỉ dẫn, xử lý tác vụ ngôn ngữ | So sánh zero-shot và few-shot trên cùng tác vụ | 2 prompt cạnh nhau; output thật của vài câu UIT-VSFC | Phần 7 | 1.5 |
| 11 | 8. Thích nghi và bổ sung tri thức | Phân biệt prompting, fine-tuning, RAG | Bảng so sánh dữ liệu cần có và trường hợp sử dụng | Cùng câu hỏi UIT-ViQuAD 2.0: không RAG vs có RAG (TF-IDF top-3), cùng câu dặn "Không có thông tin"; số câu đúng, recall@3 | Phần 8 | 1.5 |
| 12 | 9. Hạn chế và đánh giá (1/2) | Hallucination; phụ thuộc dữ liệu | Một câu trả lời sai và cách kiểm chứng | Bảng 6 bước kiểm chứng (cell 9.2); sai 12/15 mỗi cách hỏi; thực thể không tồn tại bịa 5/6 (9.3) | Phần 9.1–9.3 | 1.2 |
| 13 | 9. Hạn chế và đánh giá (2/2) | Thiên lệch, giới hạn ngữ cảnh; chất lượng phụ thuộc tác vụ và dữ liệu | (cùng mục 9) | `bias_scatter.png` (r = 0.69 / 0.44); kim trong đống rơm (cắt → bịa "1234"), `context_cost.png`; chất lượng theo tác vụ / dữ liệu (9.6) | Phần 9.4–9.6 | 1.6 |
| 14 | 10. Demo và kết luận | Thử nghiệm có kiểm soát (yêu cầu thêm 4: thiết lập, kết quả, phân tích) | Bảng kết quả | Thiết lập + bảng accuracy, không parse được, McNemar (b, c, p). Giới hạn: "1 bộ ví dụ few-shot; đổi ví dụ hoặc thứ tự có thể cho kết quả khác" | Phần 7, 10; `accuracy_icl.png` | 2.0 |
| 15 | 10. Demo và kết luận | Tổng hợp kết quả và nhận xét | Ví dụ đầu ra và phân tích lỗi | Confusion matrix, nhóm lỗi (ghi sẵn trước khi đọc), bảng tổng hợp mọi phần | Phần 10 | 1.5 |
| 16 | — | Tài liệu tham khảo (sản phẩm nộp) | — | — | — | 0.2 |
| | | | | | **Tổng** | **19.4** (theo kịch bản) |

## 2. Quy định áp dụng cho mọi slide (từ yêu cầu của cô)

- Không fine-tune (góp ý 11/11).
- Mỗi output demo có phân tích kèm theo (góp ý 11/11).
- Mỗi slide nội dung có khung code ngắn (rút gọn từ cell); mở notebook ở slide 5, 8, 12, 13, 14 để chỉ code đầy đủ và output thật (yêu cầu thêm 6); thời gian mở notebook tính trong cột "Phút".
- Người trình bày giải thích được từng dòng trong notebook (yêu cầu thêm 5). Notebook nộp chỉ chứa phần trình bày; phần mở rộng mục 2 nằm ở `notebook/llm_extension_part02.ipynb`, không nộp.
- Mọi số trên slide lấy từ `outputs/results/` của run chính thức `20261121_010447_normal`, ghi kèm `run_id`.

## 3. Sản phẩm nộp → đường dẫn

- [ ] Slide PDF/PPTX: `slides/`
- [ ] Notebook + hướng dẫn chạy: `notebook/llm_presentation.ipynb` + `README.md`
- [ ] Bộ câu thử nghiệm, kết quả, phân tích lỗi: `data/`; `outputs/results/` (`vsfc_test_ids.csv`, `fewshot_examples.json`, `rag_questions_used.csv`, các file kết quả); `docs/experiments/error_analysis_v2.md`
- [ ] Tài liệu tham khảo: `docs/references.md`

## 4. Quyết định ngày 19/11 (Toàn duyệt), thêm vào file chốt của Thư

| Quyết định | Lý do |
|---|---|
| Phần 7: chạy toàn bộ 2,999 câu test bằng sinh theo lô (`BATCH_SIZE = 16`); smoke run ước tính > 20 phút thì đặt `N_TEST = 500` | Log phần 1: 0.38–0.42 s cho 9–10 token → chạy từng câu thì sát ngưỡng 20 phút |
| Few-shot 4 ví dụ (2 positive, 2 negative), xen kẽ | 3 ví dụ lệch nhãn 2:1; Zhao et al. (2021): few-shot thiên về nhãn chiếm đa số trong ví dụ |
| Chọn prompt trên 200 câu validation (100/100), 3 mẫu, theo accuracy trung bình zero + few | File chốt chưa ghi số câu và tiêu chí |
| Lưu id câu test, ví dụ few-shot, câu hỏi RAG vào `outputs/results/` | Sản phẩm nộp "bộ câu thử nghiệm"; chạy lại ra đúng số |
| Phân tích lỗi lấy 30 câu sai của few-shot; mục 6 cố định seed, mỗi cấu hình 3 lần | File chốt chưa ghi lấy từ cách nào |
| Mục 8, 9 dùng **UIT-ViQuAD 2.0** (tiếng Việt, có đáp án và đoạn nguồn, có câu không trả lời được) thay cho 3 file soạn tay | Câu hỏi tiếng Việt gốc, không phải dịch; có nguồn để kiểm chứng; bỏ được việc soạn tay |
| Toàn code, soạn dữ liệu, chấm tay mọi phần; Muội, Phúc không còn việc dữ liệu | Toàn quyết định 19/11 |
| Phần 7: `max_new_tokens` 5 → 10 | Smoke `20261119_021425_smoke`: mẫu nhãn tiếng Việt few-shot không parse được 60%, nghi do 5 token cắt mất "tiêu cực"; 10 token để so 3 mẫu công bằng |
