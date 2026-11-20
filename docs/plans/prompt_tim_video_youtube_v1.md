# Prompt tìm và kiểm chứng video YouTube theo từng mục — Presentation LLMs (v1)

- **Created**: 2026-11-20 (Toàn, AI hỗ trợ) · Trạng thái: **[đề xuất]**
- Mục đích: tìm video tiếng Việt + tiếng Anh cho từng mục 1–9 của bài thuyết trình, có kiểm chứng nội dung khớp slide.
- Khung mục: [`khung_presentation_llms_v3.md`](khung_presentation_llms_v3.md).

## 1. Danh sách đã kiểm chứng tồn tại (2026-11-20, qua YouTube oEmbed)

Chỉ xác nhận: link còn sống, đúng tiêu đề, đúng kênh. **Chưa** xác nhận nội dung từng phút — dùng prompt mục 2 để Gemini kiểm.

| Mục | Ngôn ngữ | Video | Kênh |
|---|---|---|---|
| 1, 4, 5, 9 | EN | [[1hr Talk] Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g) | Andrej Karpathy |
| 1–9 (dài) | EN | [Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI) | Andrej Karpathy |
| 2, 3 | EN | [Transformers, the tech behind LLMs — Deep Learning Chapter 5](https://www.youtube.com/watch?v=wjZofJX0v4M) | 3Blue1Brown |
| 2 | EN | [Attention in transformers, step-by-step — Deep Learning Chapter 6](https://www.youtube.com/watch?v=eMlx5fFNoYc) | 3Blue1Brown |
| 2 | VI | [Đọc nghiên cứu mô hình Transformer - Phần 1 - Transformer Encoder](https://www.youtube.com/watch?v=_Zt23FA31co) | ProtonX |
| 2 | VI | [Đọc nghiên cứu mô hình Transformer - Phần 2 - Transformer Decoder](https://www.youtube.com/watch?v=HY-uNjFYQ68) | ProtonX |
| 3 | EN | [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) | Andrej Karpathy |
| 5 | EN | [Alignment: SFT, RLHF and DPO](https://www.youtube.com/watch?v=txRxIJdJurA) | Edukator(dot)AI |
| 5 | EN | [Preference Alignment & RLHF in LLMs Explained](https://www.youtube.com/watch?v=P0lxTrGjJ3Y) | Sunny Savita |
| 6 | EN | [Gentle Introduction to LLM Sampling: Greedy, Top-k, Top-p, and Temperature](https://www.youtube.com/watch?v=M4X07FFRKik) | neuralkian |
| 6 | EN | [Temperature, Top-K & Top-P Explained in 10 Minutes](https://www.youtube.com/watch?v=YMIQrH9BQK0) | bababoss |
| 7 | EN | [In-Context Learning (Few-Shot Prompting), Explained #Shorts](https://www.youtube.com/watch?v=MPWckSNZaJo) | Total Technology Zonne |
| 8 | EN | [RAG vs Fine-Tuning vs Prompt Engineering](https://www.youtube.com/watch?v=zYGDpG-pTho) | IBM Technology |
| 8 | VI | [Chi tiết về RAG, thực hành triển khai "code chay" Python và N8N](https://www.youtube.com/watch?v=kADBUMDCJkM) | Mì AI |
| 8 | VI | [RAG cho Người Mới: Tạo AI Agent Nội Bộ từ A - Z](https://www.youtube.com/watch?v=Sg-jLHRTufQ) | Vincent Do (thiên n8n, ít lý thuyết) |
| 9 | EN | [AI Hallucination Explained: Why AI Confidently Makes Things Up](https://www.youtube.com/watch?v=awLAenCsrlg) | Dr. Daniel Downs |

Loại bỏ: `yMQPQuz5WpA` (bản đăng lại 3Blue1Brown, không phải kênh gốc), `Li2Vuc1nd5g` (404), `UQ0XPdlHNMg` (không xác minh được tiêu đề/kênh).

Còn thiếu video tiếng Việt cho mục 1, 3, 4, 5, 6, 7, 9.

## 1b. File đính kèm theo từng mục

Mỗi lần gửi prompt cho một mục, đính kèm 3 loại file: **research** (nội dung mục), **kiến thức** (phần tương ứng trong `docs/giai_thich_du_an_v3.md`), **output** (kết quả notebook thật, run `20261119_150356_normal`). Đường dẫn tính từ thư mục `LLMs_DL/`.

| Mục | Research | Kiến thức (`docs/giai_thich_du_an_v3.md`) | Output |
|---|---|---|---|
| 1. Giới thiệu | `docs/research/noi_dung_muc01_gioi_thieu_v1.md` | §3 Phần 1 | output notebook Phần 1 (4 tác vụ) — chụp cell |
| 2. Kiến trúc | `docs/research/noi_dung_muc02_kien_truc_v2.md` | §4 Phần 2 | `outputs/figures/causal_mask.png`, `outputs/figures/attention_qwen.png`, `outputs/results/model_config.json` |
| 3. Tokenization | `docs/research/noi_dung_muc03_tokenization_v2.md` | §5 Phần 3 | output notebook Phần 3 (token, ID, embedding) — chụp cell |
| 4. Nguyên lý học | `docs/research/noi_dung_muc04_nguyen_ly_hoc_v2.md` | §6 Phần 4 | `outputs/results/per_token_loss.csv`, `outputs/results/loss_summary.csv` |
| 5. Giai đoạn huấn luyện | `docs/research/noi_dung_muc05_giai_doan_huan_luyen_v1.md` | §7 Phần 5 | `outputs/results/base_vs_instruct.csv` |
| 6. Sinh văn bản | `docs/research/noi_dung_muc06_sinh_van_ban_v2.md` | §8 Phần 6 | `outputs/results/decoding_samples.csv`, `outputs/results/decoding_steps.csv` |
| 7. Khả năng (ICL) | `docs/research/noi_dung_toan_muc07_muc10_v2.md` | §9 Phần 7 | `outputs/results/icl_zero_shot.csv`, `outputs/results/icl_few_shot.csv`, `outputs/results/icl_metrics.json` |
| 8. Prompting/FT/RAG | `docs/research/noi_dung_muc08_prompting_finetuning_rag_v2.md` | §10 Phần 8 | `outputs/results/rag_questions_used.csv` |
| 9. Hạn chế | `docs/research/noi_dung_muc09_han_che_danh_gia_v2.md` | §11 Phần 9 | `outputs/results/hallucination_results.csv`, `outputs/results/hallucination_fictitious.csv` |

- `giai_thich_du_an_v3.md` dài (~138 KB): đính kèm nguyên file một lần ở đầu cuộc trò chuyện, sau đó mỗi mục chỉ ghi "xem §N".
- Mục 1 và 3 không có file output riêng trong `outputs/results/` → chụp ảnh output cell trong `notebook/llm_presentation.ipynb`.
- Mục 10 không cần video (là demo của nhóm).

## 2. Prompt cho Gemini (xem được nội dung video YouTube)

Gemini đọc được video YouTube khi dán link → dùng để **kiểm nội dung**. Gửi lần lượt từng mục, kèm file theo bảng 1b.

```text
Tôi chuẩn bị bài thuyết trình "Tìm hiểu mô hình ngôn ngữ lớn (LLMs)" môn Học sâu. Mục cần video:

Mục <N>: <tên mục>
File đính kèm:
  - RESEARCH: <tên file research của mục> — nội dung slide và lời thoại nhóm sẽ nói.
  - KIẾN THỨC: docs/giai_thich_du_an_v3.md, §<N> — giải thích lý thuyết + code notebook.
  - OUTPUT: <tên file output> — kết quả chạy thật trên Qwen/Qwen2.5-1.5B-Instruct (mục 5 có thêm bản Base), không phải số minh họa.
Đọc hết các file đính kèm trước. Coi file RESEARCH và KIẾN THỨC là chuẩn nội dung của nhóm.

Việc 1 — Kiểm video có sẵn. Với mỗi link dưới đây, hãy XEM video và trả lời:
<dán link từ bảng mục 1>
  a) Kênh, thời lượng, ngôn ngữ nói, có phụ đề Việt/Anh không.
  b) Mốc thời gian (mm:ss–mm:ss) của đoạn khớp từng ý trong file RESEARCH. Ý nào video không nói, ghi "không có".
  c) Chỗ video nói KHÁC hoặc NGƯỢC với file RESEARCH/KIẾN THỨC (mốc thời gian + trích câu trong file). Chỉ ra ai đúng, kèm nguồn (bài báo/tài liệu chính thức).
  d) Chỗ video SAI hoặc lỗi thời so với kiến thức hiện nay (mốc thời gian + lý do).
  e) Video có giúp người xem hiểu file OUTPUT không (ví dụ: hiểu vì sao loss từng token khác nhau, vì sao temperature cao cho output khác)? Nêu đoạn cụ thể.
  f) Kết luận: Dùng / Dùng một đoạn (ghi mốc) / Không dùng.

Việc 2 — Tìm thêm video TIẾNG VIỆT cho mục này (tối đa 3). Điều kiện:
  - Người nói tiếng Việt, kênh giáo dục/kỹ thuật (ví dụ ProtonX, Mì AI, AI VIETNAM, kênh giảng viên đại học), không phải video bán khóa học.
  - Giải thích khái niệm, không chỉ demo công cụ.
  - Với mỗi video: link đầy đủ, tiêu đề chính xác, kênh, ngày đăng, rồi trả lời a–f như Việc 1.

Quy tắc bắt buộc:
  - CHỈ đưa link bạn đã mở và xem được. Không đoán link, không tự tạo ID video.
  - Không tìm được thì ghi "không tìm được", không thay bằng video lạc đề.
  - Không sửa, không làm tròn số trong file OUTPUT khi trích.
  - Trả lời dạng bảng: Link | Kênh | Thời lượng | Mốc khớp RESEARCH | Khác/ngược với file | Lỗi/lỗi thời | Giúp hiểu OUTPUT | Kết luận.
```

## 3. Prompt cho ChatGPT (tìm kiếm web, không xem được video)

ChatGPT chỉ đọc trang/mô tả, không xem video → dùng để **tìm ứng viên**, sau đó đưa link sang Gemini (mục 2) để kiểm. Đính kèm các file **research** ở bảng 1b (không cần output) để ChatGPT lấy đúng thuật ngữ tìm kiếm.

```text
Đính kèm là các file research nội dung từng mục của bài thuyết trình nhóm tôi. Đọc trước để biết mỗi mục cần giải thích ý gì.

Bật tìm kiếm web. Tìm video YouTube TIẾNG VIỆT giải thích các chủ đề sau về mô hình ngôn ngữ lớn (LLM), mỗi chủ đề tối đa 3 video. Ưu tiên video nói đúng các ý trong file research của mục đó:

1. LLM là gì; phân biệt language model, LLM, chatbot
3. Tokenization: token, token ID, embedding (BPE)
4. Next-token prediction và cross-entropy loss
5. Pretraining, SFT, RLHF, DPO
6. Sinh văn bản: greedy, temperature, top-p
7. In-context learning: zero-shot vs few-shot
9. Hallucination (ảo giác), thiên lệch, giới hạn ngữ cảnh

Điều kiện: người nói tiếng Việt; kênh giáo dục/kỹ thuật; giải thích khái niệm, không phải quảng cáo khóa học hay chỉ demo công cụ.

Quy tắc bắt buộc:
- CHỈ đưa link bạn đã thấy trong kết quả tìm kiếm thật, kèm nguồn. Không tự tạo hay đoán ID video.
- Với mỗi video ghi: link đầy đủ, tiêu đề chính xác, tên kênh, ngày đăng nếu có, 1 câu vì sao khớp ý nào trong file research.
- Ghi rõ "chỉ dựa trên tiêu đề/mô tả, chưa xem video".
- Chủ đề nào không tìm được thì ghi "không tìm được".
Trả lời dạng bảng: Chủ đề | Link | Tiêu đề | Kênh | Ngày | Lý do.
```

## 4. Sau khi có kết quả

- Mở từng link 1 lần kiểm tra tay (link sống, đúng kênh) trước khi đưa lên slide/README.
- Ghi video được chọn vào [`../references.md`](../references.md) kèm mốc thời gian.
