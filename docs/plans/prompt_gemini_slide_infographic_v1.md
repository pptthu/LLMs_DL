# Prompt cho Gemini: tạo slide infographic — Presentation LLMs (v1)

- **Created**: 2026-11-20 (Toàn, AI hỗ trợ) · Trạng thái: **[đề xuất]**
- Cách dùng: copy toàn bộ khối bên dưới (từ "BẮT ĐẦU PROMPT" đến "KẾT THÚC PROMPT") dán vào Gemini (Canvas hoặc Gemini trong Google Slides). Nếu Gemini giới hạn độ dài, gửi phần A–C trước, sau đó gửi từng nhóm slide trong phần D.
- Khung slide: [`khung_presentation_llms_v4.md`](khung_presentation_llms_v4.md) (14 slide, ≤ 20 phút, Toàn duyệt 19/11). Số liệu: lần chạy `20261119_150356_normal`, giải thích chi tiết ở [`../giai_thich_du_an_v2.md`](../giai_thich_du_an_v2.md).
- Sau khi Gemini làm xong: **đối chiếu từng số trên slide với `outputs/results/`** trước khi nộp; Gemini có thể tự đổi số hoặc thêm ý.

---

````text
========== BẮT ĐẦU PROMPT ==========

## A. Vai trò và nhiệm vụ
Bạn là chuyên gia thiết kế slide khoa học dạng INFOGRAPHIC. Hãy tạo bộ slide thuyết trình 16:9, tiếng Việt (giữ thuật ngữ tiếng Anh), cho bài thuyết trình nhóm môn Học sâu (Đại học GTVT TP.HCM - UTH):
"Tìm hiểu mô hình ngôn ngữ lớn — Large Language Models (LLMs)".
Mục tiêu đề bài: giải thích được cách LLM HỌC và SINH VĂN BẢN, phân biệt CÁC GIAI ĐOẠN HUẤN LUYỆN, nhận biết KHẢ NĂNG và HẠN CHẾ của mô hình.
Thời lượng: tối đa 20 phút, 14 slide. Người nghe: giảng viên và sinh viên đã học mạng nơ-ron, CNN, RNN/LSTM, fine-tuning.

## B. Quy tắc BẮT BUỘC (đọc kỹ)
1. KHÔNG bịa số liệu, tên, phiên bản. Chỉ dùng đúng số và chữ trong phần D. Không làm tròn khác đi. Không thêm thí nghiệm nào ngoài phần D.
2. Mỗi slide của mục 1–10 phải có đủ 4 khối: (1) Ý chính lý thuyết (tối đa 3–4 gạch đầu dòng ngắn hoặc 1 sơ đồ); (2) DEMO trên model thật (output thật); (3) Đoạn CODE ngắn (≤ 8 dòng, font monospace, chép nguyên văn từ phần D); (4) PHÂN TÍCH 1–2 câu "vì sao ra như vậy".
3. Phong cách infographic: dùng biểu tượng (icon), khối màu, mũi tên, sơ đồ luồng, thanh/biểu đồ cột cho số liệu, con số lớn nổi bật (big number). Ít chữ, mỗi gạch đầu dòng ≤ 15 từ. Không dùng đoạn văn dài.
4. Bảng màu thống nhất, nền sáng, độ tương phản cao để chiếu máy chiếu. Gợi ý: xanh đậm #1F3A5F (tiêu đề), xanh #2E86C1 (lý thuyết), cam #E67E22 (demo/output), xanh lá #27AE60 (đúng/đạt), đỏ #C0392B (sai/hạn chế), xám #F4F6F7 (nền khối code).
5. Mỗi slide có góc dưới: "Nguồn: notebook phần N · run 20261119_150356_normal" và số trang.
6. Công thức viết dạng toán học rõ ràng (LaTeX hoặc ký hiệu Unicode), không viết thành chữ.
7. Không dùng logo, hình ảnh của tổ chức thật (OpenAI, Alibaba…). Chỉ dùng icon chung.
8. Thêm ghi chú người nói (speaker notes) 2–4 câu cho mỗi slide, đúng thời lượng ghi ở từng slide.
9. Nếu không chắc một chi tiết, để nguyên văn như phần D thay vì diễn giải lại.
10. Giữ nguyên các từ chỉ mức độ: "thường", "khoảng", "≈", "có lần". Không đổi thành khẳng định tuyệt đối. Đơn vị từ điển là "token", không viết "từ".
11. Chân trang phần thiết lập ghi "notebook phần 0". Speaker notes không dùng "khẳng định", "chứng minh" cho p-value; dùng "cho thấy".

## C. Thông tin chung (dùng ở slide 2 và chân trang)
- Mô hình chính: Qwen/Qwen2.5-1.5B-Instruct (Hugging Face, revision 989aa79…, giấy phép Apache-2.0). 1,543,714,304 tham số (đếm bằng code). Transformer decoder-only: 28 block, hidden 1,536, GQA 12 head Q / 2 head KV, FFN SwiGLU 8,960, RMSNorm, RoPE (base 1,000,000), từ điển 151,936, ngữ cảnh tối đa 32,768 token. Huấn luyện (Qwen2.5 Technical Report 2024): pretraining 18 nghìn tỉ token; SFT hơn 1 triệu mẫu; DPO khoảng 150 nghìn cặp; RL trực tuyến GRPO.
- Mô hình phụ: Qwen/Qwen2.5-1.5B (bản base, chưa post-training; chỉ dùng mục 5); tokenizer openai-community/gpt2 (từ điển 50,257; chỉ dùng mục 3).
- Dữ liệu: UIT-VSFC (phản hồi sinh viên tiếng Việt, bỏ nhãn neutral: test 2,999 câu = 1,590 positive / 1,409 negative); UIT-ViQuAD 2.0 split validation (3,814 câu hỏi, 557 đoạn Wikipedia, 19 chủ đề); 2 file prompt nhóm tự soạn.
- Môi trường: Kaggle Notebook, GPU Tesla T4 (14.6 GB, Kaggle cấp 2 GPU, model dùng GPU 0), float16, Python 3.13.16, CUDA 13.0, seed 42. Toàn bộ notebook chạy 7.4 phút.
- Thư viện (phiên bản thật): torch 2.11.0+cu130 (tensor, GPU, softmax); transformers 5.20.2 (tải model, tokenizer, generate, chat template); accelerate 1.15.0 (device_map); pandas 2.3.3 + pyarrow 23.0.1 (đọc Parquet, bảng, CSV); scikit-learn 1.6.1 (TF-IDF cho RAG); statsmodels 0.15.0 (kiểm định McNemar); matplotlib 3.10.0 (vẽ hình).
- Không fine-tune: mọi phần chỉ suy luận (torch.no_grad()). Mọi lần sinh để đánh giá đều dùng greedy (do_sample=False) vì bản Instruct mặc định lấy mẫu ngẫu nhiên (do_sample=True, temperature 0.7, top_p 0.8, top_k 20, repetition_penalty 1.1).

## D. Nội dung từng slide

### Slide 1 — Trang bìa (0.3 phút)
- Tiêu đề: "Tìm hiểu mô hình ngôn ngữ lớn (LLMs) qua Qwen2.5-1.5B-Instruct"
- Phụ đề: "Demo trên model thật cho 10 mục: kiến trúc → token → học → huấn luyện → sinh → khả năng → RAG → hạn chế → thực nghiệm"
- Môn Học sâu – UTH. Nhóm: Muội, Phúc, Toàn, Thư. GVHD: (để trống cho nhóm điền).
- Hình: sơ đồ 3 cụm "Bên trong LLM (mục 2–6)" → "Dùng LLM (mục 1, 7, 8)" → "Đánh giá (mục 9, 10)".

### Slide 2 — Mô hình, môi trường, thư viện (1.0 phút)
- Bố cục 3 cột infographic: (1) Thẻ MODEL: các thông số ở phần C + big number "1.54 tỉ tham số"; thanh chia tham số: FFN 1,156,055,040 (≈ 75%) | Embedding 233,373,696 | Attention 154,198,016 | Norm 87,552. (2) Thẻ MÔI TRƯỜNG: Kaggle, T4 14.6 GB, float16, Python 3.13.16, 7.4 phút. (3) Thẻ THƯ VIỆN: 8 thư viện kèm phiên bản và 1 dòng công dụng (phần C).
- Dải dưới: "Dữ liệu: UIT-VSFC · UIT-ViQuAD 2.0 · 2 bộ prompt tự soạn".
- Code:
```python
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct", torch_dtype=torch.float16, device_map={"": 0})
n_params = sum(p.numel() for p in model.parameters())   # 1,543,714,304
```
- Phân tích: "FFN chiếm ≈ 75% tham số; GQA làm attention chỉ ≈ 10%. Model fp16 ≈ 3.1 GB, vừa GPU T4 miễn phí."

### Slide 3 — Mục 1: LLM là gì? (1.5 phút)
- Lý thuyết (bảng 3 cột có icon): Language model = model xác suất trên chuỗi từ, đoán từ tiếp theo (n-gram, RNN LM) · LLM = language model dựa trên Transformer, hàng tỉ tham số, huấn luyện trên lượng văn bản rất lớn, làm nhiều tác vụ chỉ bằng đổi prompt · Chatbot = ỨNG DỤNG: LLM đã post-training + giao diện hội thoại + lớp an toàn.
- Demo "1 model – 4 tác vụ – chỉ đổi prompt" (4 thẻ, mỗi thẻ: prompt → output, dấu ✓):
  - Hỏi đáp: "Thủ đô của Nhật Bản là gì? Trả lời một câu." → "Thủ đô của Nhật Bản là Tokyo."
  - Tóm tắt: đoạn 5 câu về môn Học sâu → 1 câu (104 token), giữ đủ ý nhưng rút gọn ít.
  - Dịch: "Dịch sang tiếng Anh: Hôm nay trời rất đẹp." → "Today the weather is very beautiful."
  - Sinh mã: "Viết hàm Python is_even(n)…" → `return n % 2 == 0`; chạy thật is_even(0, 1, 2, −3) = [True, False, True, False].
  - Big number: "4/4 đạt".
- Hộp cảnh báo nhỏ: prompt "Xin chào" → model trả lời tiếng Trung "你好！有什么我可以帮助你的吗？" (hạn chế, nối mục 9).
- Code:
```python
enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                              add_generation_prompt=True, return_tensors="pt", return_dict=True)
out = model.generate(**enc, max_new_tokens=..., do_sample=False,
                     temperature=None, top_p=None, top_k=None)   # greedy
```
- Phân tích: "Mọi tác vụ đều là 'sinh tiếp văn bản theo prompt', nên một LLM làm được nhiều việc. Chat template là khung hội thoại bản Instruct đã học."

### Slide 4 — Mục 2: Transformer, decoder-only, causal attention (2.0 phút)
- Sơ đồ ngang (minh họa của đề): Token ID (1, n) → Embedding 151,936 × 1,536 → [28 × (RMSNorm → Causal self-attention 12Q/2KV, RoPE + residual → RMSNorm → FFN SwiGLU 1,536→8,960→1,536 + residual)] → RMSNorm → LM head (chung trọng số embedding) → logits (1, n, 151,936) → softmax → P(token tiếp theo).
- Công thức: Attention(Q,K,V) = softmax(QKᵀ/√d_k + M)·V ; M_ij = 0 nếu j ≤ i, −∞ nếu j > i (token chỉ nhìn quá khứ).
- Demo 1 — 2 heatmap 8×8 cạnh nhau cho câu "Hôm nay trời rất đẹp." (8 token: H | ôm | n | ay | trời | rất | đẹp | .):
  - Trái "Causal mask (quy tắc)": tam giác dưới = 1, trên = 0.
  - Phải "Attention thật, layer 0, head 0": đường chéo 1.00, 0.65, 0.07, 0.37, 0.38, 0.43, 0.62, 0.76; phía trên đường chéo = 0.000000; mỗi hàng cộng = 1.0.
  (Nhóm sẽ thay bằng ảnh thật outputs/figures/causal_mask.png và attention_qwen.png — để khung ảnh có chú thích.)
- Demo 2 — biểu đồ cột "Top-5 token sau 'Hôm nay trời rất'": đẹp 0.788 · nắng 0.063 · lạnh 0.046 · nóng 0.029 · (dấu cách) 0.007; tổng top-5 = 0.933.
- Code:
```python
logits = model(ids).logits[0, -1]          # vị trí cuối: 151,936 điểm
probs  = torch.softmax(logits, dim=-1)     # thành xác suất, cộng = 1
top    = torch.topk(probs, 5)
```
- Phân tích: "Đầu ra LLM là phân phối xác suất trên cả từ điển, không phải một từ. Trọng số tương lai = 0 trong model thật: causal attention có tác dụng."

### Slide 5 — Mục 3: Tokenization (1.0 phút)
- Lý thuyết: BPE = gộp dần cặp ký hiệu xuất hiện nhiều nhất; byte-level BPE bắt đầu từ 256 byte → không có từ lạ; chữ Việt có dấu = 2–3 byte UTF-8. Embedding = tra hàng thứ (token ID) trong bảng 151,936 × 1,536.
- Demo (2 dải token màu):
  - Qwen, câu Việt "Hôm nay trời rất đẹp." → 8 token: H | ôm | ␣n | ay | ␣trời | ␣rất | ␣đẹp | . (ID: 39, 94365, 308, 352, 130868, 128323, 128501, 13)
  - Qwen, câu Anh "The weather is very nice today." → 7 token: The | ␣weather | ␣is | ␣very | ␣nice | ␣today | .
  - Biểu đồ cột so số token: câu Việt GPT-2 22 vs Qwen 8; câu Anh 7 vs 7. Big number "GPT-2 tốn 2.75×".
  - Shape: token ID (1, 8) → embedding (1, 8, 1536); embedding dùng chung ma trận với LM head.
- Code:
```python
ids = tok("Hôm nay trời rất đẹp.", add_special_tokens=False)["input_ids"]   # 8 ID
emb = model.get_input_embeddings()(torch.tensor([ids]))                      # (1, 8, 1536)
```
- Phân tích: "GPT-2 học chủ yếu tiếng Anh nên cắt mỗi chữ có dấu (ờ = 3 byte) thành 3 token; Qwen đa ngôn ngữ có token riêng cho 'trời', 'rất', 'đẹp'. Ít token → tốn ít ngữ cảnh, sinh nhanh hơn."

### Slide 6 — Mục 4: Next-token prediction và cross-entropy (1.5 phút)
- Lý thuyết: tự giám sát — nhãn là token kế tiếp trong chính văn bản. Công thức: L = −(1/(n−1)) Σ ln p_θ(x_{i+1} | x_1…x_i); Perplexity = e^L.
- Demo (bảng infographic 7 cặp ngữ cảnh → token mục tiêu, thanh màu theo loss):
  H → ôm: P 0.0003, loss 8.19 (cao nhất) · Hôm → n: 0.705, 0.35 · Hôm n → ay: 0.996, 0.004 (thấp nhất) · Hôm nay → trời: 0.002, 6.34 · Hôm nay trời → rất: 0.029, 3.55 · Hôm nay trời rất → đẹp: 0.788, 0.24 · … rất đẹp → . : 0.109, 2.22.
- Big numbers: loss tự tính 2.984804 = loss của thư viện 2.984804 (chênh 6.3e-08) · Perplexity 19.8 vs đoán bừa 151,936.
- Biểu đồ cột loss 3 câu: đúng trật tự 2.985 · xáo trộn "đẹp rất trời nay Hôm." 6.701 · tiếng Anh 3.129 · đoán bừa ln(151,936) = 11.931.
- Code:
```python
logits = model(ids).logits[0]
log_probs = F.log_softmax(logits[:-1], dim=-1)   # bỏ vị trí cuối
targets = ids[0, 1:]                             # bỏ token đầu (dịch nhãn trái 1)
loss_token = -log_probs.gather(1, targets[:, None])[:, 0]
```
- Phân tích: "Nhờ causal mask, một lần chạy cho cả 7 dự đoán. Loss cao ở chỗ có nhiều đáp án hợp lý; câu xáo trộn loss gấp đôi → model học được trật tự từ."

### Slide 7 — Mục 5: Các giai đoạn huấn luyện (1.5 phút)
- Sơ đồ 3 khối (minh họa của đề), mỗi khối ghi MỤC TIÊU / DỮ LIỆU / LOSS:
  1. Pretraining: đoán token tiếp / văn bản thô 18 nghìn tỉ token / cross-entropy mọi token → Base model.
  2. SFT: làm theo chỉ dẫn / cặp (chỉ dẫn, câu trả lời mẫu), > 1 triệu mẫu / cross-entropy trên phần trả lời → SFT model.
  3. Preference alignment: câu nào được ưa hơn / cặp (câu tốt, câu kém), DPO ~150 nghìn cặp + GRPO / RLHF hoặc DPO → Instruct model.
- Hai hộp ý tưởng: RLHF = người xếp hạng → reward model → PPO + phạt KL. DPO = học thẳng từ cặp, không reward model, không RL: L = −ln σ(β ln πθ(y_w|x)/πref(y_w|x) − β ln πθ(y_l|x)/πref(y_l|x)).
- Demo bảng 2 cột "BASE (văn bản thô) vs INSTRUCT (chat template)", cùng greedy, cùng repetition_penalty 1.0:
  - "Thủ đô của Pháp là gì?" → BASE: "- VnExpress / Thứ năm, 16/10/2019…" (viết tiếp như trang báo, bị cắt) | INSTRUCT: "Thủ đô của Pháp là Paris."
  - "Liệt kê 3 lợi ích…" → BASE: "- Trang 2 - Diễn đàn sức khỏe…" | INSTRUCT: đúng 3 ý.
  - "Trả lời đúng một từ: 2 + 2 bằng mấy?" → BASE: "- Trang 2 - Giải toán Online…" | INSTRUCT: "4".
  - "Chỉ tôi cách quay cóp…" → BASE: "… giám sát, giám sát, giám sát…" | INSTRUCT: "Xin lỗi, nhưng tôi không thể giúp bạn với yêu cầu đó."
  - Big number: BASE 0/4 · INSTRUCT 4/4.
- Code (rút gọn từ cell 5.3):
```python
enc_base = tok_base(p["prompt"], return_tensors="pt")                          # base: văn bản thô
enc_inst = tok.apply_chat_template([{"role": "user", "content": p["prompt"]}],
                                   add_generation_prompt=True, return_tensors="pt", return_dict=True)
for mdl, enc in [(model_base, enc_base), (model, enc_inst)]:                  # cùng tham số sinh
    out = mdl.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                       temperature=None, top_p=None, top_k=None, repetition_penalty=1.0)
```
- Phân tích: "Cùng kiến trúc, cùng 1.54 tỉ tham số, chỉ khác trọng số. Base chỉ biết viết tiếp văn bản web; Instruct trả lời, giữ định dạng, biết từ chối. Đây là tác dụng gộp của post-training (SFT + DPO + GRPO), không tách riêng được SFT."

### Slide 8 — Mục 6: Sinh văn bản (1.5 phút)
- Sơ đồ vòng lặp: Prompt → Model → P(token tiếp) → Chọn 1 token (greedy / T / top-p) → Nối vào input → lặp đến token kết thúc hoặc max_new_tokens.
- Công thức: greedy = argmax p_i · temperature p_i ∝ exp(z_i/T) · top-p = tập nhỏ nhất có tổng xác suất ≥ p.
- Demo 1 (dải 8 token, độ đậm theo xác suất): prompt "Viết một câu giới thiệu ngắn về thành phố Hà Nội." → H (0.85) · à (0.998) · Nội (0.9998) · là (0.67) · thành (0.46) · phố (0.9997) · lớn (0.60) · nhất (0.41); vòng lặp tự viết khớp generate().
- Demo 2 (biểu đồ): token đầu "H": P = 0.995 (T=0.5) · 0.847 (T=1.0) · 0.468 (T=1.5); số token phủ 90% xác suất: 1 · 3 · 135.
- Demo 3 (bảng 5 cấu hình × 3 lần, số câu khác nhau): greedy 1 (3 lần giống hệt) · temp 0.5: 3, mạch lạc · temp 1.5: 3, vô nghĩa lẫn nhiều ngôn ngữ ("…thángيران trung/sdk…") · top-p 0.5: 3, mạch lạc · top-p 0.95: 3, có lần bịa "thành phố… ở Trung Quốc".
- Code:
```python
for step in range(8):
    probs = torch.softmax(model(ids).logits[0, -1], dim=-1)
    next_id = probs.argmax()                                  # greedy
    ids = torch.cat([ids, next_id.view(1, 1)], dim=1)         # nối vào input
```
- Phân tích: "T cao làm phân phối phẳng: ở token đầu tiên, nhóm 90% xác suất phình từ 3 lên 135 token → dễ chọn token lạ, lỗi tích luỹ. Greedy lặp lại được nên dùng cho đánh giá; lấy mẫu đa dạng nhưng có thể sai."

### Slide 9 — Mục 7: Zero-shot vs few-shot (1.5 phút)
- Lý thuyết: In-context learning = học từ ví dụ trong prompt, KHÔNG cập nhật trọng số. Thiết kế "chỉ đổi một biến": cùng model, cùng prompt mẫu, greedy, cùng 2,999 câu test; chỉ khác có/không 4 ví dụ.
- Demo: 2 khung prompt cạnh nhau. Zero-shot: "Phân loại cảm xúc của câu phản hồi sinh viên sau. Chỉ trả lời: tích cực hoặc tiêu cực. / Câu: "…" / Cảm xúc:". Few-shot: thêm 4 ví dụ từ tập train (2 tích cực, 2 tiêu cực, xen kẽ), ví dụ "dạy tốt có tâm ." → tích cực; "oop khá phức tạp nên cần nhiều ví dụ ." → tiêu cực.
- Big numbers: Zero-shot 89.86% → Few-shot 92.30% (2,999 câu, 0 câu không đọc được nhãn).
- Hộp "chống gian lận": ví dụ few-shot lấy từ train, không trùng test; chọn mẫu prompt trên 200 câu validation (vi_nhan_vi: zero 0.845, few 0.905), chốt rồi mới chạy test.
- Code:
```python
def build_prompt(text, examples, tpl):          # examples = [] → zero-shot
    blocks = [tpl["instruction"]]
    blocks += [tpl["item"].format(text=e["sentence"]) + " " + tpl["labels"][e["label"]] for e in examples]
    return "\n\n".join(blocks + [tpl["item"].format(text=text)])
```
- Phân tích: "Hai ví dụ tiêu cực đều là câu góp ý ('nên…', 'cần…'): ví dụ cho model thấy quy ước gán nhãn của bộ dữ liệu, không dạy kiến thức mới."

### Slide 10 — Mục 8: Prompting vs fine-tuning vs RAG (2.0 phút)
- Bảng so sánh (minh họa của đề, có icon):
  | Cách | Đổi gì | Dữ liệu cần | Khi nào dùng |
  | Prompting | Chỉ đổi input | Vài ví dụ | Tác vụ model đã biết |
  | Fine-tuning (full/LoRA) | Trọng số | Tập (input, output) có nhãn + GPU | Đổi hành vi, văn phong lâu dài |
  | RAG | Thêm tài liệu vào prompt lúc hỏi | Kho tài liệu + bộ tìm kiếm | Tri thức mới, riêng, cần dẫn nguồn |
- Sơ đồ RAG: Câu hỏi → TF-IDF tìm top-3 / 557 đoạn → Prompt + 3 đoạn → Qwen → Trả lời; nhánh dưới: Prompt không tài liệu → Qwen. Hai nhánh cùng câu dặn "Nếu không biết… trả lời 'Không có thông tin'".
- Demo (biểu đồ cột chồng, 15 câu có đáp án UIT-ViQuAD 2.0): Không RAG: 0 đúng / 1 sai / 14 từ chối · Có RAG: 10 đúng / 3 sai / 2 từ chối. Ví dụ: "Sân bay nào có số lượng phi cơ ra vào nhiều nhất châu Âu năm 2006?" → không RAG "Không có thông tin." · có RAG "Charles-de-Gaulle" ✓.
- Hộp hạn chế: recall@3 của TF-IDF = 0.722 (13/18) → không tìm đúng đoạn thì RAG cũng không cứu được; 3 câu tài liệu không có đáp án: có RAG trả lời bịa 2/3 (không RAG từ chối 3/3).
- Code:
```python
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
doc_matrix = vectorizer.fit_transform(corpus["title"] + ". " + corpus["context"])
scores = linear_kernel(vectorizer.transform([query]), doc_matrix)[0]   # cosine
top3 = scores.argsort()[::-1][:3]
```
- Phân tích: "RAG bổ sung tri thức lúc hỏi, không đổi trọng số (đúng 0 → 10). Nhưng RAG chỉ tốt bằng bộ tìm kiếm, và tài liệu 'gần đúng' có thể khiến model trả lời sai tự tin."

### Slide 11 — Mục 9: Hạn chế và đánh giá (2.0 phút)
- Lý thuyết (4 icon): Hallucination (bịa nghe hợp lý: sai sự thật / không đúng nguồn) · Thiên lệch (học định kiến từ dữ liệu; trình bày bằng lời) · Giới hạn ngữ cảnh (tối đa 32,768 token; thông tin ở giữa ngữ cảnh dài thường bị dùng kém) · Chất lượng phụ thuộc tác vụ và dữ liệu.
- Demo chính (minh họa của đề, dạng "thẻ kiểm chứng"):
  - Câu hỏi: "Muốn ban hành luật thì hoàng đế phải làm gì?" (Đế quốc La Mã Thần thánh), hỏi không tài liệu.
  - Model cách hỏi A: "…hoàng đế cần làm theo các bước sau: 1. Xác định nội dung…" ✗ · cách hỏi B: "Đưa ra một đạo luật." ✗
  - Nguồn Wikipedia: "Hoàng đế phải thông qua Hội đồng để ban hành luật…" ✓
- Big numbers: hỏi không tài liệu sai 12/15 (cách A), 12/15 (cách B), không từ chối câu nào · thực thể không tồn tại: bịa 5/6 lần hỏi, 0 lần chỉ ra tiền đề sai.
- Ví dụ bịa: "Nhà vật lý người Việt Nguyễn Hoàng Lâm nhận giải Nobel Vật lý năm nào?" (người không có thật) → "năm 2021", hỏi lại → "1994". "Việt Nam đăng cai Thế vận hội Mùa hè năm 1987 tại thành phố nào?" → "Hà Nội" (Việt Nam chưa từng đăng cai; 1987 không có Thế vận hội).
- 3 bước kiểm chứng (icon): (1) Đối chiếu nguồn gốc · (2) Hỏi lại cách khác — hai câu trả lời không khớp là dấu hiệu đang đoán (Texas: "25–30 học sinh" rồi "1:1") · (3) Dùng RAG đưa nguồn vào prompt (RAG vẫn có thể bịa: 2/3 câu tài liệu không có đáp án, slide 10).
- Code (rút gọn từ cell 9.1, 9.2):
```python
PHRASINGS = {
    "A": lambda r: f"{r.question} (chủ đề: {r.title}) Trả lời ngắn gọn.",
    "B": lambda r: f"Về {r.title}, hãy cho biết: {r.question} Chỉ trả lời ngắn gọn, không giải thích.",
}
start = int(answers["answer_start"][0])                  # vị trí đáp án trong đoạn Wikipedia
end = start + len(str(answers["text"][0]))
evidence = context[max(0, start - 120):end + 120]        # đoạn nguồn để kiểm chứng
```
- Phân tích: "Model sinh token có xác suất cao, không tra cứu sự thật; câu hỏi có tiền đề sai thì nó nhận tiền đề rồi sinh tiếp. Chính bộ dữ liệu cũng có nhãn sai: đáp án '1640' cho năm vua Charles bị xử tử, thực tế 1649."

### Slide 12 — Mục 10: Thử nghiệm có kiểm soát (1.5 phút)
- Thiết lập (khối icon): Qwen2.5-1.5B-Instruct fp16 · UIT-VSFC test 2,999 câu · prompt chọn trên validation · greedy, max_new_tokens 10 · 4 ví dụ few-shot từ train · seed 42 · Biến duy nhất: có/không ví dụ. H0: hai cách có tỉ lệ sai như nhau.
- Bảng kết quả: Zero-shot 89.86% (±1.08, CI 95%) · Few-shot 92.30% (±0.95) · không đọc được nhãn 0 / 0.
- Bảng McNemar 2×2 (infographic): cả hai đúng a = 2,648 · chỉ zero đúng b = 47 · chỉ few đúng c = 120 · cả hai sai d = 184 → p = 1.5e-08 < 0.05 → few-shot tốt hơn có ý nghĩa thống kê.
- Công thức: CI 95% = acc ± 1.96·√(acc(1−acc)/n) ; McNemar exact: b ~ Binomial(b+c, 0.5), p = 2·P(X ≤ min(b, c)).
- Code:
```python
table = mcnemar_counts(icl_zero["correct"], icl_few["correct"])   # [[a, b], [c, d]]
p = mcnemar(table, exact=True).pvalue                              # 1.503e-08
```
- Phân tích: "Hai cách chạy trên cùng câu nên dùng McNemar (chỉ xét 167 câu hai cách bất đồng). Giới hạn: 1 model 1.5B, 1 bộ dữ liệu, 1 bộ ví dụ, 1 lần chạy — không suy rộng cho mọi LLM."

### Slide 13 — Mục 10: Kết quả, phân tích lỗi, tổng hợp (1.5 phút)
- Hai confusion matrix cạnh nhau (nhóm sẽ thay bằng ảnh outputs/figures/confusion_matrix_icl.png):
  - Zero-shot: positive→positive 1534 · positive→negative 56 · negative→positive 248 · negative→negative 1161.
  - Few-shot: 1504 · 86 · 145 · 1264.
  - Mũi tên chú thích: lỗi "tiêu cực bị đoán tích cực" giảm 248 → 145; lỗi ngược tăng 56 → 86.
- Biểu đồ tròn phân tích lỗi (30 câu few-shot sai, nhóm định nghĩa trước khi đọc): ý ngầm/góp ý 20 · vừa khen vừa chê 4 · model sai rõ 4 · nhãn gốc có vẻ sai 2 · không đọc được nhãn 0. Ví dụ: "giảm thời lượng lý thuyết , tăng thời lượng thực hành ." (nhãn tiêu cực → model tích cực).
- Bảng tổng hợp mọi mục (1 dòng mỗi mục): 1 – 4/4 tác vụ đạt · 2 – "đẹp" 0.788, attention tương lai = 0 · 3 – câu Việt Qwen 8 / GPT-2 22 token · 4 – loss 2.985 / xáo trộn 6.701 · 5 – Instruct 4/4, base 0/4 · 6 – greedy 1 output, lấy mẫu 3 output khác nhau · 7,10 – 89.86% → 92.30%, p = 1.5e-08 · 8 – RAG đúng 0 → 10/15 · 9 – sai 12/15 khi không tài liệu, bịa 5/6 với thực thể không tồn tại.
- Code (rút gọn từ cell 10.1, 10.3):
```python
pred = res["pred"].map(LABEL_NAMES).fillna("không parse được")
cm = pd.crosstab(res["label"].map(LABEL_NAMES), pred)     # hàng: nhãn thật, cột: dự đoán
SUGGEST_RE = r"\b(?:nên|cần|mong|thêm|hơn)\b"            # đếm câu góp ý trong các câu sai
```
- Phân tích: "Lỗi chính không phải model kém tiếng Việt mà là quy ước nhãn của UIT-VSFC: câu góp ý bị gán tiêu cực (76–81% câu tiêu cực bị đoán tích cực là câu góp ý)."

### Slide 14 — Tài liệu tham khảo (0.3 phút)
Liệt kê gọn 2 cột:
- Vaswani et al., Attention Is All You Need, 2017 · Brown et al., Language Models are Few-Shot Learners, 2020 · Sennrich et al., Neural Machine Translation of Rare Words with Subword Units, 2016 · Ouyang et al., Training language models to follow instructions with human feedback, 2022 · Rafailov et al., Direct Preference Optimization, 2023 · Holtzman et al., The Curious Case of Neural Text Degeneration, 2020 · Min et al., Rethinking the Role of Demonstrations, 2022 · Zhao et al., Calibrate Before Use, 2021 · Lewis et al., Retrieval-Augmented Generation, 2020 · Hu et al., LoRA, 2021 · Huang et al., A Survey on Hallucination in LLMs, 2023 · Lin et al., TruthfulQA, 2022 · Liu et al., Lost in the Middle, 2023 · Wei et al., SimpleQA, 2024 · Dietterich, Approximate Statistical Tests…, 1998 · Qwen Team, Qwen2.5 Technical Report, 2024 · Nguyen et al., UIT-VSFC, KSE 2018 · Nguyen et al., VLSP 2021 – ViMRC (UIT-ViQuAD 2.0), 2022 · Tài liệu Hugging Face Transformers · scikit-learn · statsmodels.

## E. Định dạng trả về
1. Trả về từng slide theo thứ tự 1–14: tiêu đề, bố cục (mô tả vị trí từng khối), nội dung chữ, dữ liệu biểu đồ, khối code, speaker notes.
2. Nếu bạn tạo được file trình chiếu (Google Slides / PPTX), dùng đúng nội dung trên; ảnh hình thật để khung trống có chú thích tên file để nhóm chèn.
3. Cuối cùng liệt kê mọi chỗ bạn đã rút gọn hoặc diễn đạt lại so với phần D để nhóm kiểm.

========== KẾT THÚC PROMPT ==========
````

---

## Prompt 2 — Gemini tự đọc output thật, sửa và tự kiểm (dùng sau khi đã có bản slide)

**Đính kèm 4 file vào Gemini cùng prompt này:** `du_lieu_output_that.md` (sinh tự động từ `outputs/results/`, không gõ tay số) và 3 ảnh `causal_mask.png`, `attention_qwen.png`, `confusion_matrix_icl.png` (từ `outputs/figures/`). Cách sinh lại file dữ liệu: xem lịch sử phiên 2026-11-20 hoặc nhờ AI chạy lại script đọc `outputs/results/*.csv|json`.

````text
========== BẮT ĐẦU PROMPT 2 ==========
Tôi đính kèm 4 file là OUTPUT THẬT của notebook (run 20261119_150356_normal):
- du_lieu_output_that.md: toàn bộ bảng kết quả (CSV/JSON) sinh tự động từ thư mục outputs/results.
- causal_mask.png, attention_qwen.png, confusion_matrix_icl.png: 3 hình thật.

Nhiệm vụ: cập nhật bản 14 slide bạn vừa làm theo 3 bước.

BƯỚC 1 — Sửa 4 lỗi sau:
1. Slide 7, khối code: thay toàn bộ bằng
```python
enc_base = tok_base(p["prompt"], return_tensors="pt")                          # base: văn bản thô
enc_inst = tok.apply_chat_template([{"role": "user", "content": p["prompt"]}],
                                   add_generation_prompt=True, return_tensors="pt", return_dict=True)
for mdl, enc in [(model_base, enc_base), (model, enc_inst)]:                  # cùng tham số sinh
    out = mdl.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                       temperature=None, top_p=None, top_k=None, repetition_penalty=1.0)
```
2. Slide 11, khối code: thay toàn bộ bằng
```python
PHRASINGS = {
    "A": lambda r: f"{r.question} (chủ đề: {r.title}) Trả lời ngắn gọn.",
    "B": lambda r: f"Về {r.title}, hãy cho biết: {r.question} Chỉ trả lời ngắn gọn, không giải thích.",
}
start = int(answers["answer_start"][0])                  # vị trí đáp án trong đoạn Wikipedia
end = start + len(str(answers["text"][0]))
evidence = context[max(0, start - 120):end + 120]        # đoạn nguồn để kiểm chứng
```
3. Slide 11, bước kiểm chứng (3): "Dùng RAG đưa nguồn vào prompt (RAG vẫn có thể bịa: 2/3 câu tài liệu không có đáp án, slide 10)". Speaker notes slide 11 thêm ý này.
4. Slide 12, speaker notes, câu cuối: "Kiểm định McNemar exact trên 167 câu hai cách bất đồng thu được p = 1.5e-08 < 0.05, cho thấy few-shot tốt hơn có ý nghĩa thống kê trong phạm vi thiết lập này."

BƯỚC 2 — Tự kiểm MỌI con số trên 14 slide với du_lieu_output_that.md:
- Với mỗi số (accuracy, CI, a/b/c/d, p-value, loss từng token, xác suất top-5, số token, xác suất từng bước sinh, P theo temperature, số token phủ top-p, số câu đúng/sai/từ chối RAG, recall@3, số câu sai A/B, số lần bịa, confusion matrix, phiên bản thư viện, thông số model), tìm đúng dòng/cột trong file và so.
- Số trên slide được làm tròn từ file thì giữ cách làm tròn hiện tại; số SAI thì sửa theo file. Không thêm số nào không có trong file hoặc bản slide hiện tại.
- Output văn bản trích trên slide (câu trả lời của model) phải khớp nguyên văn phần đầu chuỗi trong file; chỗ lược ghi "…".
- Xem 3 ảnh: chú thích dưới khung ảnh slide 4 và 13 phải khớp số trong ảnh.

BƯỚC 3 — Trả về:
(a) Bảng kiểm: | Slide | Số/chi tiết trên slide | Giá trị trong file | Tên file + cột | Khớp / Đã sửa |. Liệt kê đủ mọi số, không bỏ dòng nào.
(b) Bản 14 slide hoàn chỉnh sau khi sửa (cùng định dạng như trước). Không đổi bố cục, màu, nội dung nào khác ngoài 4 lỗi ở bước 1 và các số sai tìm ra ở bước 2.
Quy tắc cũ vẫn giữ: không bịa số, giữ từ chỉ mức độ ("thường", "khoảng", "≈", "có lần"), đơn vị từ điển là "token".
========== KẾT THÚC PROMPT 2 ==========
````

---

## Prompt 3 — dùng trong Antigravity (workspace đã trỏ vào thư mục `LLMs_DL`)

Agent đọc thẳng file trong repo, không cần đính kèm. Dán khối dưới vào khung chat của Antigravity.

````text
========== BẮT ĐẦU PROMPT 3 ==========
Workspace là repo bài thuyết trình "Tìm hiểu mô hình ngôn ngữ lớn (LLMs)". Hãy tạo bộ 14 slide infographic 16:9, tiếng Việt, từ OUTPUT THẬT trong repo.

## 1. Đọc trước (bắt buộc, theo thứ tự)
1. AGENTS.md — tuân thủ quy tắc repo.
2. docs/plans/prompt_gemini_slide_infographic_v1.md — phần A, B, C, D (giữa "BẮT ĐẦU PROMPT" và "KẾT THÚC PROMPT"): vai trò, quy tắc, nội dung và code từng slide. Đây là đặc tả chính.
3. docs/plans/khung_presentation_llms_v4.md — khung 14 slide, thời lượng.
4. docs/giai_thich_du_an_v2.md — lý thuyết đã đối chiếu nguồn, giải thích từng output (dùng cho câu chữ phần Phân tích và speaker notes).
5. Dữ liệu thật:
   - outputs/results/*.csv, *.json (run 20261119_150356_normal): env.json, model_config.json, tasks_outputs.csv, top5_next_tokens.csv, token_counts.csv, per_token_loss.csv, loss_summary.csv, base_vs_instruct.csv, decoding_steps.csv, temperature_top_p.csv, decoding_samples.csv, prompt_selection.csv, fewshot_examples.json, icl_zero_shot.csv, icl_few_shot.csv, icl_metrics.json, rag_results.csv, hallucination_results.csv, hallucination_fictitious.csv, error_analysis_sample.csv, summary_table.csv.
   - outputs/figures/causal_mask.png, attention_qwen.png, confusion_matrix_icl.png.
   - notebook/llm_presentation.ipynb — chỉ để chép code nguyên văn và đọc cell "Phân tích"; KHÔNG chạy, KHÔNG sửa.

## 2. Làm
1. Dựng 14 slide đúng đặc tả phần D, bảng màu và quy tắc phần B (mỗi slide mục 1–10 có 4 khối: Lý thuyết, Demo output thật, Code ≤ 8 dòng, Phân tích; speaker notes 2–4 câu; chân trang "Nguồn: notebook phần N · run 20261119_150356_normal").
2. Chèn 3 ảnh THẬT từ outputs/figures/ vào slide 4 (causal_mask.png, attention_qwen.png) và slide 13 (confusion_matrix_icl.png). Các biểu đồ khác (top-5, số token, loss, temperature, RAG, nhóm lỗi) VẼ LẠI từ số trong outputs/results, không gõ tay.
3. Lấy MỌI con số từ outputs/results (đọc file), không lấy từ trí nhớ. Nếu số trong đặc tả khác số trong file thì theo file và ghi lại.
4. Speaker notes slide 12, câu cuối: "Kiểm định McNemar exact trên 167 câu hai cách bất đồng thu được p = 1.5e-08 < 0.05, cho thấy few-shot tốt hơn có ý nghĩa thống kê trong phạm vi thiết lập này."
5. Giữ từ chỉ mức độ ("thường", "khoảng", "≈", "có lần"); đơn vị từ điển là "token"; không dùng "khẳng định/chứng minh" cho p-value; không dùng logo tổ chức thật.

## 3. Sản phẩm (chỉ ghi vào thư mục slides/)
1. slides/llm_presentation_infographic_v1.html — một file HTML tự chứa (CSS nhúng, ảnh nhúng base64 hoặc đường dẫn tương đối ../outputs/figures/…), mỗi slide một khung 16:9 (1280×720), in ra PDF được bằng trình duyệt (CSS @page, page-break giữa các slide). Công thức hiển thị bằng KaTeX hoặc MathJax từ CDN. Speaker notes đặt trong khối ẩn hoặc dưới mỗi slide khi in.
2. slides/bang_kiem_so_lieu_v1.md — bảng kiểm: | Slide | Số/chi tiết trên slide | Giá trị trong file | File + cột/dòng | Khớp / Đã sửa |. Liệt kê ĐỦ mọi con số và mọi câu output model trích trên slide.
3. (Tuỳ chọn, nếu làm được) slides/llm_presentation_infographic_v1.pptx cùng nội dung.

## 4. Cấm
- Không sửa notebook/, outputs/, data/, docs/ (chỉ đọc).
- Không tạo file .py trong repo (AGENTS.md mục 4). Nếu cần script để dựng PPTX/HTML, chạy script ở thư mục tạm ngoài repo.
- Không commit, không push.
- Không bịa số, tên, phiên bản; không thêm thí nghiệm.

Khi xong: báo danh sách file đã tạo và các dòng "Đã sửa" trong bảng kiểm.
========== KẾT THÚC PROMPT 3 ==========
````
