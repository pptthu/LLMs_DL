# Khung presentation LLMs (v4)

- **Created**: 2026-11-18 (Toàn, AI hỗ trợ) · **Last Updated**: 2026-11-19 (Toàn, AI hỗ trợ)
- Trạng thái: **Duyệt: Toàn, 2026-11-19**, theo [`chot_khung_v4_ke_hoach_cho_Toan.md`](chot_khung_v4_ke_hoach_cho_Toan.md) (Thư) và các quyết định ngày 19/11 ở §4.
- Thay [`khung_presentation_llms_v3.md`](khung_presentation_llms_v3.md). v3 giữ nguyên.
- Chỉ dựng từ yêu cầu của cô trong [`yeu_cau_cua_co_v1.md`](yeu_cau_cua_co_v1.md): đề (10 mục, sản phẩm nộp), góp ý 11/11, yêu cầu thêm.
- Kế hoạch làm đến thứ 7: [`ke_hoach_18_21_11_v2.md`](ke_hoach_18_21_11_v2.md).

## 1. Khung slide

Tổng thời gian ≤ 20 phút, **gồm cả lúc mở notebook** (yêu cầu thêm 6). Cột "Phút" là mức tối đa cho slide đó.

| Slide | Mục | Nội dung (đề) | Minh họa (đề) | Demo output thật trên slide (yêu cầu thêm 3) | Notebook | Phút |
|---|---|---|---|---|---|---|
| 1 | — | Tên đề tài, thành viên | — | — | — | 0.3 |
| 2 | — | Thư viện (yêu cầu thêm 1); môi trường chạy (yêu cầu thêm 2) | — | Phiên bản thật từ `env.json` | Phần 0 | 1.0 |
| 3 | 1. Giới thiệu | LLM là gì? Phân biệt language model, LLM và ứng dụng chatbot | Ví dụ các tác vụ: hỏi đáp, tóm tắt, dịch, sinh mã | Output thật của 4 tác vụ. Phân tích nối với slide 7: base ≈ LM chỉ viết tiếp; Instruct đã qua post-training; chatbot = Instruct + giao diện + lớp an toàn | Phần 1 | 1.5 |
| 4 | 2. Nền tảng kiến trúc | Vai trò của Transformer; Decoder-only và causal attention | Sơ đồ từ token đầu vào đến xác suất token tiếp theo | **Ma trận causal mask** của câu ví dụ, trọng số attention thật (phần trên đường chéo = 0), top-5 token tiếp theo | Phần 2 | 1.5 |
| 5 | 3. Tokenization | Cách chia token; token ID và embedding | Tokenize một câu tiếng Việt và một câu tiếng Anh | Token, token ID, embedding thật của 2 câu; số token Qwen vs GPT-2 | Phần 3 | 1.0 |
| 6 | 4. Nguyên lý học | Next-token prediction, dữ liệu huấn luyện, cross-entropy loss | Tạo các cặp ngữ cảnh → token mục tiêu từ một câu | Các cặp ngữ cảnh → token mục tiêu, loss từng token, loss tự tính = `outputs.loss` | Phần 4 | 1.5 |
| 7 | 5. Các giai đoạn huấn luyện | Pretraining, SFT, preference alignment; RLHF/DPO ở mức ý tưởng | Sơ đồ mục tiêu và dữ liệu của từng giai đoạn | Base vs Instruct, cùng prompt, cùng `do_sample=False`, `repetition_penalty=1.0`, cùng `max_new_tokens`. Lời thoại: "tác dụng gộp của SFT + alignment" (Qwen2.5-Instruct qua SFT, DPO, GRPO) | Phần 5 | 1.5 |
| 8 | 6. Quá trình sinh văn bản | Autoregressive generation; greedy, temperature, top-p | Cùng một prompt với các cấu hình sinh khác nhau | Sinh từng bước (vòng lặp tự viết = `generate()`); temperature làm phân phối nhọn/phẳng; 5 cấu hình × 3 lần | Phần 6 | 1.5 |
| 9 | 7. Khả năng của LLM | In-context learning, làm theo chỉ dẫn, xử lý tác vụ ngôn ngữ | So sánh zero-shot và few-shot trên cùng tác vụ | 2 prompt cạnh nhau; output thật của vài câu UIT-VSFC | Phần 7 | 1.5 |
| 10 | 8. Thích nghi và bổ sung tri thức | Phân biệt prompting, fine-tuning, RAG | Bảng so sánh dữ liệu cần có và trường hợp sử dụng | Cùng câu hỏi UIT-ViQuAD 2.0: không RAG vs có RAG (TF-IDF top-3), cùng câu dặn "Không có thông tin"; số câu đúng, recall@3 | Phần 8 | 1.5 |
| 11 | 9. Hạn chế và đánh giá | Hallucination, thiên lệch, giới hạn ngữ cảnh; phụ thuộc tác vụ và dữ liệu | Một câu trả lời sai và cách kiểm chứng | Câu sai thật (hỏi không tài liệu) + đáp án + đoạn Wikipedia nguồn + hỏi lại cách khác. Phụ thuộc dữ liệu: output lẫn tiếng Trung (cell 1.1, 1.5), nhãn có thể sai (phần 10). Thiên lệch, context 32,768 token: nói bằng lời | Phần 9 | 1.5 |
| 12 | 10. Demo và kết luận | Thử nghiệm có kiểm soát (yêu cầu thêm 4: thiết lập, kết quả, phân tích) | Bảng kết quả | Thiết lập + bảng accuracy, không parse được, McNemar (b, c, p). Giới hạn: "1 bộ ví dụ few-shot; đổi ví dụ hoặc thứ tự có thể cho kết quả khác" | Phần 7, 10 | 2.0 |
| 13 | 10. Demo và kết luận | Tổng hợp kết quả và nhận xét | Ví dụ đầu ra và phân tích lỗi | Confusion matrix, nhóm lỗi (ghi sẵn trước khi đọc), bảng tổng hợp mọi phần | Phần 10 | 1.5 |
| 14 | — | Tài liệu tham khảo (sản phẩm nộp) | — | — | — | 0.3 |
| | | | | | **Tổng** | **19.1** |

## 2. Quy định áp dụng cho mọi slide (từ yêu cầu của cô)

- Không fine-tune (góp ý 11/11).
- Mỗi output demo có phân tích kèm theo (góp ý 11/11).
- Phần nào slide không có code thì mở notebook trình bày code (yêu cầu thêm 6); thời gian mở notebook tính trong cột "Phút".
- Người trình bày giải thích được từng dòng trong notebook (yêu cầu thêm 5). Notebook nộp chỉ chứa phần trình bày; phần mở rộng mục 2 nằm ở `notebook/llm_extension_part02.ipynb`, không nộp.
- Mọi số trên slide lấy từ `outputs/results/` của lần Save & Run All chính thức, ghi kèm `run_id`.

## 3. Sản phẩm nộp → đường dẫn

- [ ] Slide PDF/PPTX: `slides/`
- [ ] Notebook + hướng dẫn chạy: `notebook/llm_presentation.ipynb` + `README.md`
- [ ] Bộ câu thử nghiệm, kết quả, phân tích lỗi: `data/`; `outputs/results/` (`vsfc_test_ids.csv`, `fewshot_examples.json`, `rag_questions_used.csv`, các file kết quả); `docs/experiments/error_analysis_v1.md`
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
