# Prompt tạo TỪNG ẢNH infographic — Presentation LLMs (v1)

- **Created**: 2026-11-20 (Toàn, AI hỗ trợ) · Trạng thái: **[đề xuất]**
- **Dùng với:** Gemini / Antigravity (model tạo ảnh). Mỗi khối `Ảnh NN` là **một prompt độc lập** → sinh **một ảnh** 16:9. Dán từng khối, sinh xong ảnh này mới sang ảnh sau.
- **Vì sao mỗi mục tách 2–3 ảnh:** lý thuyết viết đầy đủ thì một ảnh không đủ chỗ cho chữ đọc được. Ảnh A (và B ở mục 2, 5, 10) = lý thuyết + công thức + sơ đồ; ảnh cuối của mục = demo output thật + code + phân tích. Số ảnh = số mục + 2 (Ảnh 03 = mục 1 … Ảnh 12 = mục 10). Tổng **26 ảnh**: 01, 02, 03A–B, 04A–C, 05A–B, 06A–B, 07A–C, 08A–B, 09A–B, 10A–B, 11A–B, 12A–C, 13.
- **Nguồn nội dung:** lý thuyết lấy từ [`../giai_thich_du_an_v2.md`](../giai_thich_du_an_v2.md) (đã đối chiếu bài gốc, §15.1); số liệu lấy từ `outputs/results/` của run `20261119_150356_normal`.
- **Ảnh có hình thật** (Ảnh 04C, 12C): đính kèm file PNG trong `outputs/figures/` làm ảnh tham chiếu. Nếu công cụ không ghép được ảnh thật, để khung trống có nhãn tên file, rồi nhóm dán ảnh thật đè lên.
- **Sau khi sinh:** model tạo ảnh hay viết sai dấu tiếng Việt, sai số, sai ký hiệu toán. **Soát từng ảnh** với chữ trong prompt; ảnh sai thì sinh lại hoặc sửa tay.

---

## Khối phong cách chung (đã chép sẵn vào đầu mỗi prompt)

> PHONG CÁCH: Infographic khoa học, khổ ngang 16:9 (1920×1080), nền trắng, độ tương phản cao để chiếu máy chiếu. Thanh tiêu đề trên cùng màu xanh đậm #1F3A5F chữ trắng. Khối lý thuyết viền xanh #2E86C1 nền #EBF5FB; khối demo/output viền cam #E67E22 nền #FEF5E7; đúng/đạt xanh lá #27AE60; sai/hạn chế đỏ #C0392B; khối code nền xám #F4F6F7 font monospace; công thức toán in rõ như sách giáo khoa. Icon phẳng đơn giản, không logo công ty thật, không ảnh người. Font sans-serif hỗ trợ tiếng Việt có dấu. Chân trang trái: "{NGUỒN}"; chân trang phải: "{TRANG}". VIẾT CHÍNH XÁC TỪNG CHỮ, TỪNG SỐ, TỪNG KÝ HIỆU dưới đây, giữ nguyên dấu tiếng Việt; không thêm chữ, số hay ý nào khác.

---

## Ảnh 01 — Trang bìa

```text
PHONG CÁCH: Infographic khoa học, khổ ngang 16:9 (1920×1080), nền trắng, tương phản cao. Màu chủ đạo xanh đậm #1F3A5F, xanh #2E86C1, cam #E67E22. Icon phẳng, không logo công ty thật, không ảnh người. Font sans-serif hỗ trợ tiếng Việt có dấu. VIẾT CHÍNH XÁC TỪNG CHỮ dưới đây, giữ nguyên dấu.

NỘI DUNG:
- Tiêu đề lớn ở giữa trên (màu #1F3A5F): "Tìm hiểu mô hình ngôn ngữ lớn (LLMs)"
- Dòng thứ hai: "qua Qwen2.5-1.5B-Instruct"
- Phụ đề (màu #2E86C1): "Demo trên model thật cho 10 mục của đề"
- Dải sơ đồ ngang 3 cụm bo góc nối bằng mũi tên:
  [icon bộ não] "Bên trong LLM" – "Mục 2–6: kiến trúc · token · học · huấn luyện · sinh văn bản"
  → [icon hội thoại] "Dùng LLM" – "Mục 1, 7, 8: tác vụ · zero/few-shot · RAG"
  → [icon biểu đồ] "Đánh giá" – "Mục 9, 10: hạn chế · thực nghiệm có kiểm soát"
- Góc dưới trái: "Môn Học sâu – UTH"; "Nhóm: Muội · Phúc · Toàn · Thư"; "GVHD: ……………"
- Chân trang phải: "1"
```

## Ảnh 02 — Mô hình, môi trường, thư viện

```text
PHONG CÁCH: (dùng khối phong cách chung) Chân trang trái: "Nguồn: notebook phần 0 · run 20261119_150356_normal"; phải: "2". VIẾT CHÍNH XÁC TỪNG CHỮ, TỪNG SỐ.

TIÊU ĐỀ: "Mô hình · Môi trường · Thư viện"

BỐ CỤC: 3 thẻ dọc bằng nhau, dưới cùng một dải ngang.

THẺ 1 – MODEL (viền #2E86C1), số lớn nổi bật "1,543,714,304 tham số":
- "Qwen/Qwen2.5-1.5B-Instruct (Hugging Face, Apache-2.0)"
- "Transformer decoder-only · 28 block · hidden 1,536"
- "GQA: 12 head Q / 2 head KV · FFN SwiGLU 8,960"
- "RMSNorm · RoPE (base 1,000,000) · từ điển 151,936 token"
- "Ngữ cảnh tối đa 32,768 token"
- "Huấn luyện: pretraining 18 nghìn tỉ token → SFT > 1 triệu mẫu → DPO ~150 nghìn cặp + GRPO"
- Thanh ngang chia tỉ lệ (stacked bar) có nhãn: "FFN 1,156,055,040 (≈ 75%)" | "Embedding 233,373,696" | "Attention 154,198,016" | "Norm 87,552"
- Dòng nhỏ: "Model phụ: Qwen/Qwen2.5-1.5B (base, mục 5) · tokenizer GPT-2 50,257 token (mục 3)"

THẺ 2 – MÔI TRƯỜNG (viền #E67E22):
- "Kaggle Notebook · GPU Tesla T4 14.6 GB (dùng GPU 0)"
- "float16 · Python 3.13.16 · CUDA 13.0 · seed 42"
- "Chỉ suy luận: torch.no_grad(), không fine-tune"
- "Sinh để đánh giá: greedy (do_sample=False)"
- Số lớn: "7.4 phút" + chú thích "chạy toàn bộ notebook"

THẺ 3 – THƯ VIỆN (viền #1F3A5F), dạng danh sách có icon:
- "torch 2.11.0+cu130 – tensor, GPU, softmax"
- "transformers 5.20.2 – tải model, tokenizer, generate, chat template"
- "accelerate 1.15.0 – device_map"
- "pandas 2.3.3 + pyarrow 23.0.1 – đọc Parquet, bảng, CSV"
- "scikit-learn 1.6.1 – TF-IDF cho RAG"
- "statsmodels 0.15.0 – kiểm định McNemar"
- "matplotlib 3.10.0 – vẽ hình"

DẢI DƯỚI (nền #EBF5FB): "Dữ liệu: UIT-VSFC (test 2,999 câu: 1,590 tích cực / 1,409 tiêu cực) · UIT-ViQuAD 2.0 (3,814 câu hỏi, 557 đoạn Wikipedia) · 2 bộ prompt nhóm tự soạn"

KHỐI CODE nhỏ góc phải dưới (monospace):
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct", torch_dtype=torch.float16, device_map={"": 0})
n_params = sum(p.numel() for p in model.parameters())   # 1,543,714,304
```

---

## Ảnh 03A — Mục 1: LLM là gì? (Lý thuyết)

```text
PHONG CÁCH: (dùng khối phong cách chung) Chân trang trái: "Nguồn: giai_thich_du_an_v2 §3 · Jurafsky & Martin ch.3; Bengio 2003; Brown 2020; Zhao 2023"; phải: "3A". VIẾT CHÍNH XÁC TỪNG CHỮ.

TIÊU ĐỀ: "Mục 1 · LLM là gì? Language model – LLM – Chatbot"

KHỐI TRÊN – "Language model (LM)" (viền #2E86C1):
- "Mô hình gán xác suất cho một chuỗi từ; tương đương: đoán từ tiếp theo."
- Công thức lớn (quy tắc dây chuyền): P(x₁, x₂, …, xₙ) = ∏ᵢ P(xᵢ | x₁, …, xᵢ₋₁)
- "Cách cũ: n-gram – đếm tần suất:" công thức P(wᵢ | wᵢ₋₁) ≈ count(wᵢ₋₁, wᵢ) / count(wᵢ₋₁)
- "Bengio 2003: thay bảng đếm bằng mạng nơ-ron + embedding (neural LM)."

BẢNG 3 CỘT ở giữa (mỗi cột một icon, tiêu đề cột in đậm):
| | Language model | LLM | Chatbot |
| Là gì | Model xác suất trên chuỗi | LM dựa trên Transformer, hàng tỉ tham số, học từ lượng văn bản rất lớn | ỨNG DỤNG xây trên LLM, không phải model |
| Ví dụ | n-gram, RNN LM | Qwen2.5-1.5B (base / Instruct), GPT-3 | ChatGPT, Qwen Chat |
| Làm được | Đoán từ tiếp theo | Nhiều tác vụ chỉ bằng đổi prompt | Hội thoại nhiều lượt với người dùng |

KHỐI DƯỚI TRÁI – "Chatbot = LLM + …" (sơ đồ khối chồng):
"LLM đã post-training (bản Instruct)" + "Giao diện hội thoại" + "Quản lý lịch sử hội thoại" + "Lớp an toàn / kiểm duyệt"

KHỐI DƯỚI PHẢI – "Vì sao một model làm được nhiều tác vụ?":
"Hỏi đáp, tóm tắt, dịch, viết code đều là: văn bản vào → sinh tiếp văn bản ra, có điều kiện theo prompt. Đủ lớn và qua post-training thì chỉ cần đổi prompt là đổi tác vụ (GPT-3, Brown 2020)."
```

## Ảnh 03B — Mục 1: Một model, bốn tác vụ (Demo)

```text
PHONG CÁCH: (dùng khối phong cách chung) Chân trang trái: "Nguồn: notebook phần 1 · run 20261119_150356_normal"; phải: "3B". VIẾT CHÍNH XÁC TỪNG CHỮ, TỪNG SỐ.

TIÊU ĐỀ: "Mục 1 · Demo: 1 model – 4 tác vụ – chỉ đổi prompt"

LƯỚI 2×2 THẺ (viền #E67E22), mỗi thẻ: tên tác vụ, khung "Prompt", mũi tên, khung "Output", dấu tích xanh:
1. "Hỏi đáp" – Prompt: "Thủ đô của Nhật Bản là gì? Trả lời một câu." → Output: "Thủ đô của Nhật Bản là Tokyo." (10 token) ✓
2. "Tóm tắt" – Prompt: "[đoạn 5 câu về môn Học sâu] Tóm tắt đoạn trên trong 1 câu." → Output: "Học sâu là một lĩnh vực học máy sử dụng mạng nơ-ron nhiều lớp…" (1 câu, 104 token; giữ đủ ý nhưng rút gọn ít) ✓
3. "Dịch" – Prompt: "Dịch sang tiếng Anh: Hôm nay trời rất đẹp." → Output: "Today the weather is very beautiful." (8 token) ✓
4. "Sinh mã" – Prompt: "Viết hàm Python is_even(n) trả về True nếu n là số chẵn." → Output: "def is_even(n): return n % 2 == 0" (216 token, kèm ví dụ và giải thích) – chạy thật: is_even(0, 1, 2, −3) = [True, False, True, False] ✓

SỐ LỚN bên phải lưới: "4/4 đạt" (màu #27AE60) + dòng nhỏ "0/4 bị cắt ở max_new_tokens".

HỘP CẢNH BÁO (viền đỏ #C0392B): "Prompt 'Xin chào' (32 token sau chat template) → '你好！有什么我可以帮助你的吗？' – trả lời bằng tiếng Trung. Hạn chế phụ thuộc dữ liệu (xem mục 9)."

KHỐI CODE (monospace):
messages = [{"role": "user", "content": prompt}]
enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                              return_tensors="pt", return_dict=True)
out = model.generate(**enc, max_new_tokens=..., do_sample=False,
                     temperature=None, top_p=None, top_k=None)   # greedy
text = tok.decode(out[0, n_in:], skip_special_tokens=True)       # bỏ phần prompt

KHỐI PHÂN TÍCH (viền trái #1F3A5F):
"Cùng trọng số, cùng greedy, chỉ đổi prompt → đổi tác vụ. Chat template bọc câu hỏi vào khung hội thoại bản Instruct đã học (tự thêm system prompt 'You are Qwen…'), nên model biết phải trả lời và tự dừng bằng <|im_end|>."
```

---

## Ảnh 04A — Mục 2: Transformer, self-attention (Lý thuyết 1/2)

```text
PHONG CÁCH: (dùng khối phong cách chung) Chân trang trái: "Nguồn: Vaswani et al. 2017 §3.2; HF LLM Course"; phải: "4A". VIẾT CHÍNH XÁC TỪNG CHỮ, TỪNG KÝ HIỆU TOÁN.

TIÊU ĐỀ: "Mục 2 · Transformer và self-attention"

KHỐI TRÁI – "Vì sao Transformer thay RNN/LSTM?" (2 cột so sánh có icon):
- RNN/LSTM: "Đọc tuần tự từng token · token thứ 100 chờ 99 bước · khó song song · thông tin xa bị phai"
- Transformer: "Self-attention: mỗi token nhìn trực tiếp mọi token được phép trong 1 bước · tính song song trên GPU"

KHỐI GIỮA – "Self-attention" (viền #2E86C1):
- "Mỗi token x được chiếu thành 3 vector bằng 3 ma trận học được:"
  "q = x·W_Q – token này đang tìm gì" · "k = x·W_K – token này có gì để được tìm" · "v = x·W_V – nội dung nó đóng góp"
- CÔNG THỨC LỚN: Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k ) · V
- Bảng 4 dòng giải thích:
  "Q·Kᵀ → ma trận điểm n×n: token i liên quan token j bao nhiêu"
  "÷ √d_k → giữ phương sai điểm ≈ 1 (tích vô hướng có phương sai d_k); tránh softmax bão hoà, gradient ≈ 0"
  "softmax theo hàng → trọng số dương, mỗi hàng cộng = 1"
  "· V → output = trung bình có trọng số các vector value"
- "Multi-head: nhiều head song song, mỗi head một bộ W_Q, W_K, W_V; nối lại rồi nhân W_O. Bài gốc: d_model = 512, h = 8."

KHỐI PHẢI – "Ba họ Transformer" (bảng):
| Họ | Ví dụ | Attention | Huấn luyện | Dùng cho |
| Encoder-only | BERT | Hai chiều | Che từ rồi đoán lại | Phân loại, trích xuất |
| Decoder-only (tô đậm) | GPT, Llama, Qwen | Một chiều (causal) | Đoán token tiếp theo | Sinh văn bản, chat |
| Encoder-decoder | T5, BART | Encoder 2 chiều + decoder causal | Khôi phục đoạn bị che | Dịch, tóm tắt |
Dòng dưới bảng: "Hầu hết LLM là decoder-only: bài toán huấn luyện (đoán token tiếp) trùng với cách dùng (sinh từng token)."
```

## Ảnh 04B — Mục 2: Decoder-only, causal mask, các khối của Qwen (Lý thuyết 2/2)

```text
PHONG CÁCH: (dùng khối phong cách chung) Chân trang trái: "Nguồn: Vaswani 2017 §3.2.3; Qwen2.5 Technical Report §2; Su 2021; Zhang & Sennrich 2019; Shazeer 2020; Ainslie 2023"; phải: "4B". VIẾT CHÍNH XÁC TỪNG CHỮ, TỪNG KÝ HIỆU.

TIÊU ĐỀ: "Mục 2 · Từ token đầu vào đến xác suất token tiếp theo"

SƠ ĐỒ NGANG LỚN phía trên (minh họa của đề), các hộp nối mũi tên, dưới mỗi hộp ghi shape:
"Token ID" (1, n) → "Embedding 151,936 × 1,536" (1, n, 1536) → [Hộp lớn viền vàng "Decoder block × 28": "RMSNorm → Causal self-attention (12 head Q / 2 head KV, RoPE) + residual" rồi "RMSNorm → FFN SwiGLU 1,536 → 8,960 → 1,536 + residual"] (1, n, 1536) → "RMSNorm cuối" → "LM head (chung trọng số với Embedding)" (1, n, 151,936) → "softmax ở vị trí cuối" → "P(token tiếp theo)"

KHỐI TRÁI DƯỚI – "Causal mask (decoder-only)":
- CÔNG THỨC: Attention = softmax( Q·Kᵀ / √d_k + M ) · V,  M_ij = 0 nếu j ≤ i;  M_ij = −∞ nếu j > i
- "e^(−∞) = 0 → trọng số cho token tương lai đúng bằng 0."
- "Vì sao −∞ mà không đặt điểm = 0? Điểm 0 cho e⁰ = 1, vẫn nhìn trộm tương lai."
- "Vì sao cần: (1) Huấn luyện đưa cả câu vào một lần – không mask thì chép đáp án; (2) Khi sinh, token tương lai chưa có – mask làm lúc train giống lúc dùng; (3) Token cũ không đổi → lưu K, V cũ (KV cache)."

KHỐI PHẢI DƯỚI – bảng "Các khối trong Qwen2.5":
| Khối | Công thức / ý | Vì sao |
| GQA | 12 head Q dùng chung 2 head K/V (6 Q : 1 KV) | Giảm KV cache, ít tham số |
| RoPE | xoay cặp chiều của q, k góc m·θᵢ; θᵢ = base^(−2(i−1)/d); Qwen base = 1,000,000 | q_m·k_n chỉ phụ thuộc khoảng cách → biết thứ tự |
| RMSNorm | x / √(mean(x²) + ε) ⊙ g | Như LayerNorm, không trừ trung bình, rẻ hơn |
| Pre-norm + residual | h = x + Attn(RMSNorm(x)); y = h + FFN(RMSNorm(h)) | Gradient đi thẳng qua 28 block (như ResNet) |
| SwiGLU | W_down( SiLU(W_gate·x) ⊙ W_up·x ), SiLU(x) = x·σ(x) | 3 ma trận; biểu diễn tốt hơn ReLU |
| LM head | logits = h · Eᵀ (E = ma trận embedding) | Dùng chung E, bớt 233 triệu tham số |
```

## Ảnh 04C — Mục 2: Demo attention thật và top-5

```text
PHONG CÁCH: (dùng khối phong cách chung) Chân trang trái: "Nguồn: notebook phần 2 (cell 2.2–2.4) · run 20261119_150356_normal"; phải: "4C". VIẾT CHÍNH XÁC TỪNG SỐ.
ẢNH THAM CHIẾU: đính kèm outputs/figures/causal_mask.png và outputs/figures/attention_qwen.png – đặt NGUYÊN VẸN hai ảnh này vào hai khung bên trái, không vẽ lại.

TIÊU ĐỀ: "Mục 2 · Demo: causal attention trong model thật"

BÊN TRÁI – 2 khung ảnh cạnh nhau, câu "Hôm nay trời rất đẹp." = 8 token: H | ôm | n | ay | trời | rất | đẹp | .
- Khung 1 nhãn "[causal_mask.png] Quy tắc: tam giác dưới = 1 (được nhìn), trên = 0 (bị chặn)"
- Khung 2 nhãn "[attention_qwen.png] Trọng số thật – layer 0, head 0"
- Dưới 2 khung: 3 thẻ số nhỏ: "Phía trên đường chéo = 0.000000" · "Mỗi hàng cộng = 1.0" · "Đường chéo: 1.00 · 0.65 · 0.07 · 0.37 · 0.38 · 0.43 · 0.62 · 0.76"

BÊN PHẢI – biểu đồ cột ngang "Top-5 token sau 'Hôm nay trời rất' (6 token)":
"đẹp" 0.788 (cột dài nhất, màu cam) · "nắng" 0.063 · "lạnh" 0.046 · "nóng" 0.029 · "(dấu cách)" 0.007
Chú thích: "Tổng top-5 = 0.933; 0.067 còn lại chia cho 151,931 token khác."

KHỐI CODE:
logits = model(ids).logits[0, -1]          # vị trí cuối: 151,936 điểm
probs  = torch.softmax(logits, dim=-1)     # thành xác suất, cộng = 1
top    = torch.topk(probs, 5)

KHỐI PHÂN TÍCH:
"Tam giác trên = 0 trong model thật: causal mask có tác dụng thật. Hàng đầu = 1.00 là bắt buộc (chỉ 1 ô được phép). Đây chỉ là 1 trong 28 × 12 = 336 head. Đầu ra của LLM là phân phối xác suất trên cả từ điển; 4 ứng viên đầu đều là tính từ thời tiết → model học cả ngữ pháp lẫn ngữ nghĩa."
```

---

## Ảnh 05A — Mục 3: Tokenization (Lý thuyết)

```text
PHONG CÁCH: (dùng khối phong cách chung) Chân trang trái: "Nguồn: Sennrich 2016; HF Summary of the tokenizers; Qwen2.5 Technical Report §2"; phải: "5A". VIẾT CHÍNH XÁC TỪNG CHỮ.

TIÊU ĐỀ: "Mục 3 · Tokenization: văn bản → token → token ID → embedding"

KHỐI TRÊN – "Ba cách chia văn bản" (bảng):
| Cách | Ưu | Nhược |
| Theo từ | Dễ hiểu | Từ điển khổng lồ, từ lạ thành [UNK] |
| Theo ký tự | Từ điển nhỏ | Chuỗi rất dài, mỗi ký tự ít nghĩa |
| Subword – BPE (tô đậm) | Từ phổ biến = 1 token, từ hiếm = vài mảnh | Cắt theo thống kê, không theo ngữ pháp |

KHỐI GIỮA TRÁI – "Huấn luyện tokenizer BPE" (3 bước đánh số, có vòng lặp):
"① Bắt đầu: mỗi ký tự / byte là một token" → "② Đếm mọi cặp token liền nhau; gộp cặp xuất hiện nhiều nhất thành token mới" → "③ Lặp đến khi đủ kích thước từ điển"
Ví dụ nhỏ: "t · r · ờ · i → tr · ờ · i → trời"

KHỐI GIỮA PHẢI – "Byte-level BPE (GPT-2, Qwen)":
- "Từ điển gốc = 256 giá trị byte → mọi chuỗi đều tokenize được, không có [UNK]"
- "Chữ Việt có dấu chiếm 2–3 byte UTF-8: ô = C3 B4 · ờ = E1 BB 9D · đ = C4 91"
- "GPT-2: 50,257 = 256 byte + 1 token kết thúc + 50,000 lần gộp"
- "Qwen2.5: 151,643 token thường + 22 token điều khiển (<|im_start|>, <|im_end|>…); bảng embedding 151,936 hàng"

KHỐI DƯỚI – luồng ngang:
"Câu" → "Token" → "Token ID (số thứ tự trong từ điển)" → "Embedding = lấy hàng thứ (token ID) của ma trận E ∈ ℝ^(151,936 × 1,536)" → "Vector 1,536 số"
Công thức: e_t = E[t, :] = onehot(t)ᵀ · E
```

## Ảnh 05B — Mục 3: Demo tokenize câu Việt và câu Anh

```text
PHONG CÁCH: (dùng khối phong cách chung) Chân trang trái: "Nguồn: notebook phần 3 · run 20261119_150356_normal"; phải: "5B". VIẾT CHÍNH XÁC TỪNG CHỮ, TỪNG SỐ.

TIÊU ĐỀ: "Mục 3 · Demo: cùng câu, hai tokenizer"

DẢI TOKEN 1 – "Qwen · 'Hôm nay trời rất đẹp.' → 8 token" (mỗi token một ô màu, dưới ghi ID):
[H 39] [ôm 94365] [␣n 308] [ay 352] [␣trời 130868] [␣rất 128323] [␣đẹp 128501] [. 13]

DẢI TOKEN 2 – "Qwen · 'The weather is very nice today.' → 7 token":
[The] [␣weather] [␣is] [␣very] [␣nice] [␣today] [.]

DẢI TOKEN 3 – "GPT-2 · 'Hôm nay trời rất đẹp.' → 22 token" (ô nhỏ, ô "�" màu đỏ nhạt):
[H][ô][m][␣n][ay][␣tr][�][�][�][i][␣r][�][�][�][t][␣�][�][�][�][�][p][.]

BIỂU ĐỒ CỘT ĐÔI bên phải: "Số token" – Câu Việt: GPT-2 22 · Qwen 8 | Câu Anh: GPT-2 7 · Qwen 7. Số lớn: "GPT-2 tốn 2.75× cho câu Việt"
Thẻ nhỏ: "Câu Việt: 21 ký tự = 29 byte UTF-8 (ô 2, ờ 3, ấ 3, đ 2, ẹ 3 byte)"

THẺ SHAPE: "token ID (1, 8) → embedding (1, 8, 1536) · embedding và LM head dùng chung một ma trận (cùng địa chỉ bộ nhớ)" · "Decode ngược ra đúng câu gốc: 4/4"

KHỐI CODE:
ids = tok("Hôm nay trời rất đẹp.", add_special_tokens=False)["input_ids"]   # 8 ID
emb = model.get_input_embeddings()(torch.tensor([ids]))                      # (1, 8, 1536)

KHỐI PHÂN TÍCH:
"GPT-2 học chủ yếu tiếng Anh, chưa gộp byte tiếng Việt: mỗi chữ như 'ờ' thành 3 token 1 byte (decode riêng ra '�'). Qwen đa ngôn ngữ có token riêng cho ' trời', ' rất', ' đẹp', nhưng vẫn cắt 'Hôm' = 'H' + 'ôm' vì BPE gộp theo tần suất, không theo từ điển. Ít token → tốn ít ngữ cảnh, sinh nhanh hơn."
```

---

## Ảnh 06A — Mục 4: Nguyên lý học (Lý thuyết)

```text
PHONG CÁCH: (dùng khối phong cách chung) Chân trang trái: "Nguồn: Slide Ch2; Brown 2020; Qwen2.5 Technical Report; HF Perplexity"; phải: "6A". VIẾT CHÍNH XÁC TỪNG CHỮ, TỪNG KÝ HIỆU.

TIÊU ĐỀ: "Mục 4 · Next-token prediction và cross-entropy"

KHỐI 1 – "Next-token prediction":
- "Cho ngữ cảnh x₁ … xᵢ, đoán xᵢ₊₁ → bài toán phân loại với 151,936 lớp (softmax + cross-entropy như Ch2)."
- "Tự giám sát: nhãn là chính token kế tiếp trong văn bản, không cần người gán → dùng được rất nhiều dữ liệu (GPT-3: 300 tỉ token; Qwen2.5: 18 nghìn tỉ token)."
- Sơ đồ: "[a, b, c, d] → ([a] → b), ([a, b] → c), ([a, b, c] → d)" + "Câu n token → n − 1 bài toán"

KHỐI 2 – "Cross-entropy loss" (công thức lớn):
L = − (1 / (n − 1)) · Σᵢ₌₁ⁿ⁻¹ ln p_θ(xᵢ₊₁ | x₁, …, xᵢ)
Bảng nhỏ: "p = 1 → −ln p = 0 · p = 0.5 → 0.69 · p = 0.1 → 2.30 · p = 0.001 → 6.91"
"Huấn luyện = chỉnh θ bằng gradient descent để giảm L = tăng xác suất của văn bản thật (maximum likelihood)."

KHỐI 3 – "Perplexity":
PPL = e^L  ·  "Đọc: model phân vân như đang chọn đều giữa PPL token."
"Mốc đoán bừa đều: p = 1/V → L = ln V = ln 151,936 = 11.93, PPL = 151,936"

KHỐI 4 – "Một lần chạy cho mọi vị trí":
"Nhờ causal mask, logits[i] chỉ dựa vào token 0…i, nên chính là dự đoán cho token i+1 → so logits[:-1] với ids[1:] ('dịch nhãn sang trái 1 vị trí'). Đây là lý do Transformer huấn luyện song song được."
Hình minh họa: hai dải ô lệch nhau một vị trí, mũi tên chéo.
```

## Ảnh 06B — Mục 4: Demo loss từng token

```text
PHONG CÁCH: (dùng khối phong cách chung) Chân trang trái: "Nguồn: notebook phần 4 · run 20261119_150356_normal"; phải: "6B". VIẾT CHÍNH XÁC TỪNG SỐ.

TIÊU ĐỀ: "Mục 4 · Demo: 'Hôm nay trời rất đẹp.' = 8 token → 7 cặp ngữ cảnh → mục tiêu"

BẢNG CHÍNH (mỗi dòng có thanh ngang dài theo loss, đỏ = cao, xanh = thấp):
| Ngữ cảnh | Token đúng | P | Loss = −ln P |
| H | ôm | 0.0003 | 8.19 (cao nhất) |
| Hôm | n | 0.705 | 0.35 |
| Hôm n | ay | 0.996 | 0.004 (thấp nhất) |
| Hôm nay | trời | 0.002 | 6.34 |
| Hôm nay trời | rất | 0.029 | 3.55 |
| Hôm nay trời rất | đẹp | 0.788 | 0.24 |
| Hôm nay trời rất đẹp | . | 0.109 | 2.22 |

3 SỐ LỚN bên phải:
- "Loss tự tính 2.984804 = loss thư viện 2.984804 (chênh 6.3e-08)"
- "Perplexity 19.8"
- "Đoán bừa: loss 11.931, PPL 151,936"

BIỂU ĐỒ CỘT dưới: "Loss trung bình" – Đúng trật tự 2.985 · Tiếng Anh 3.129 · Xáo trộn 'đẹp rất trời nay Hôm.' 6.701 · Đoán bừa 11.931

KHỐI CODE:
logits = model(ids).logits[0]
log_probs = F.log_softmax(logits[:-1], dim=-1)   # bỏ vị trí cuối
targets = ids[0, 1:]                             # bỏ token đầu (dịch nhãn trái 1)
loss_token = -log_probs.gather(1, targets[:, None])[:, 0]

KHỐI PHÂN TÍCH:
"'ay' sau 'Hôm n' gần chắc chắn (0.996); 'ôm' sau 'H' có hàng nghìn khả năng (8.19). Loss cao ở 'trời' sau 'Hôm nay' vì có nhiều đáp án hợp lý, không phải model kém. Câu xáo trộn loss gấp đôi → model học được trật tự từ. Không train: chỉ tính loss trong torch.no_grad()."
```

---

## Ảnh 07A — Mục 5: Ba giai đoạn huấn luyện (Lý thuyết – sơ đồ của đề)

```text
PHONG CÁCH: (dùng khối phong cách chung) Chân trang trái: "Nguồn: Ouyang 2022; Qwen2.5 Technical Report §3–4"; phải: "7A". VIẾT CHÍNH XÁC TỪNG CHỮ, TỪNG SỐ.

TIÊU ĐỀ: "Mục 5 · Ba giai đoạn huấn luyện: mục tiêu và dữ liệu"

SƠ ĐỒ NGANG 3 KHỐI LỚN nối mũi tên, mỗi khối có 4 dòng MỤC TIÊU / DỮ LIỆU / LOSS / QUY MÔ QWEN2.5, giữa các khối là icon model:
[Văn bản thô] →
KHỐI 1 "Pretraining": Mục tiêu: học ngôn ngữ, kiến thức · Dữ liệu: văn bản thô không nhãn (web, sách, code) · Loss: cross-entropy mọi token · Quy mô: 18 nghìn tỉ token
→ [Base model: biết viết tiếp văn bản] →
KHỐI 2 "SFT – Supervised Fine-Tuning": Mục tiêu: làm theo chỉ dẫn, định dạng hội thoại · Dữ liệu: cặp (chỉ dẫn, câu trả lời mẫu) · Loss: cross-entropy chỉ trên phần trả lời · Quy mô: > 1 triệu mẫu, 2 epoch, dài 32,768 token
→ [SFT model] →
KHỐI 3 "Preference alignment": Mục tiêu: câu nào được ưa hơn (hữu ích, trung thực, vô hại) · Dữ liệu: cặp (câu tốt y_w, câu kém y_l) · Cách: RLHF hoặc DPO · Quy mô: DPO ~150 nghìn cặp + GRPO (8 câu trả lời / truy vấn)
→ [Instruct model: trả lời, giữ định dạng, biết từ chối]

DẢI DƯỚI:
- Công thức SFT: L_SFT = − Σ_(t ∈ trả lời) ln π_θ(y_t | x, y_<t)
- "SFT và alignment đều là fine-tuning (Ch4): bắt đầu từ model đã huấn luyện, cập nhật trọng số trên dữ liệu nhỏ, chuyên biệt."
- "Vì sao cần alignment sau SFT: nhiều câu trả lời đúng nhưng chất lượng khác nhau; người SO SÁNH dễ hơn VIẾT câu hoàn hảo."
```

## Ảnh 07B — Mục 5: RLHF và DPO (Lý thuyết mức ý tưởng)

```text
PHONG CÁCH: (dùng khối phong cách chung) Chân trang trái: "Nguồn: Ouyang et al. 2022 §3; Rafailov et al. 2023 Eq. 2, 3, 7"; phải: "7B". VIẾT CHÍNH XÁC TỪNG KÝ HIỆU TOÁN.

TIÊU ĐỀ: "Mục 5 · Preference alignment: RLHF và DPO (ý tưởng)"

CỘT TRÁI – "RLHF (InstructGPT)" – luồng 3 bước có icon:
① "Người xếp hạng các câu trả lời cho cùng prompt"
② "Huấn luyện reward model r_φ:" L_RM = − ln σ( r_φ(x, y_w) − r_φ(x, y_l) )  ·  "câu tốt phải có điểm cao hơn"
③ "PPO tối ưu model để tăng điểm, phạt KL theo từng token để không trôi xa model SFT:"
   max_θ  E[ r_φ(x, y) ] − β · KL( π_θ ‖ π_ref )
Ghi chú: "InstructGPT: ~13k prompt SFT, ~33k prompt RM, ~31k prompt PPO; model 1.3B được ưa hơn GPT-3 175B."

CỘT PHẢI – "DPO" – một khối:
"Bỏ reward model riêng, bỏ học tăng cường; học thẳng từ cặp (y_w, y_l):"
CÔNG THỨC LỚN: L_DPO = − ln σ( β·ln[π_θ(y_w|x) / π_ref(y_w|x)] − β·ln[π_θ(y_l|x) / π_ref(y_l|x)] )
"Đọc: tăng xác suất câu tốt (so với model tham chiếu) nhiều hơn câu kém. β điều khiển mức bám model tham chiếu."

BẢNG SO SÁNH dưới:
| | RLHF (PPO) | DPO |
| Reward model riêng | Có | Không (xác suất của model là reward ngầm) |
| Học tăng cường | Có | Không |
| Giữ gần model gốc | Phạt KL | Model tham chiếu + β |
| Qwen2.5-Instruct dùng | GRPO (RL trực tuyến) | DPO ~150k cặp (offline) |
```

## Ảnh 07C — Mục 5: Demo base vs Instruct

```text
PHONG CÁCH: (dùng khối phong cách chung) Chân trang trái: "Nguồn: notebook phần 5 · run 20261119_150356_normal"; phải: "7C". VIẾT CHÍNH XÁC TỪNG CHỮ.

TIÊU ĐỀ: "Mục 5 · Demo: cùng prompt, bản base vs bản Instruct"
Dòng phụ: "Cùng kiến trúc, cùng 1,543,714,304 tham số, cùng từ điển – chỉ khác trọng số. Cùng greedy, cùng repetition_penalty 1.0."

BẢNG 3 CỘT (cột BASE viền đỏ, cột INSTRUCT viền xanh lá):
| Prompt | BASE (văn bản thô) | INSTRUCT (chat template) |
| Thủ đô của Pháp là gì? | "- VnExpress / Thứ năm, 16/10/2019, 10:00 (GMT+7) / Thủ đô của Pháp là gì? / Thủ đô của Pháp là Paris…" – bị cắt 64/64 ✗ | "Thủ đô của Pháp là Paris." ✓ |
| Liệt kê 3 lợi ích của việc tập thể dục. | "- Trang 2 - Diễn đàn sức khỏe / Thảo luận trong 'Sức khỏe'…" – bị cắt 160/160 ✗ | "1. Tăng cường sức khỏe tổng thể… 2. Tăng cường sức đề kháng… 3. Tăng cường tinh thần…" ✓ |
| Trả lời đúng một từ: 2 + 2 bằng mấy? | "- Trang 2 - Giải toán Online | Đề thi Toán | L" – bị cắt 16/16 ✗ | "4" ✓ |
| Chỉ tôi cách quay cóp trong phòng thi… | "- Trang 2 - Giải trí… giám sát, giám sát, giám sát…" – bị cắt 128/128 ✗ | "Xin lỗi, nhưng tôi không thể giúp bạn với yêu cầu đó." ✓ |

HAI SỐ LỚN: "BASE 0/4" (đỏ) · "INSTRUCT 4/4" (xanh lá)

KHỐI CODE:
enc_base = tok_base(p["prompt"], return_tensors="pt")                          # base: văn bản thô
enc_inst = tok.apply_chat_template([{"role": "user", "content": p["prompt"]}],
                                   add_generation_prompt=True, return_tensors="pt", return_dict=True)
for mdl, enc in [(model_base, enc_base), (model, enc_inst)]:                  # cùng tham số sinh
    out = mdl.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                       temperature=None, top_p=None, top_k=None, repetition_penalty=1.0)

KHỐI PHÂN TÍCH:
"Base chỉ qua pretraining nên viết tiếp như trang web (câu hỏi trên web thường là tiêu đề bài báo); có 'Paris' (kiến thức) nhưng không trả lời thẳng (hành vi). Base chỉ dừng ở <|endoftext|> nên luôn bị cắt; greedy không phạt lặp gây lặp 'giám sát'. Instruct trả lời, giữ định dạng, biết từ chối. Đây là tác dụng gộp của post-training (SFT + DPO + GRPO) và chat template – không tách riêng được SFT."
```

---

## Ảnh 08A — Mục 6: Sinh văn bản (Lý thuyết)

```text
PHONG CÁCH: (dùng khối phong cách chung) Chân trang trái: "Nguồn: Slide Ch6; Holtzman et al. 2020 §3.1; HF GenerationConfig"; phải: "8A". VIẾT CHÍNH XÁC TỪNG KÝ HIỆU.

TIÊU ĐỀ: "Mục 6 · Sinh văn bản tự hồi quy và các cách chọn token"

SƠ ĐỒ VÒNG LẶP lớn bên trái:
"Prompt" → "Model" → "P(token tiếp theo)" → [hình thoi cam "Chọn 1 token: greedy / temperature / top-p"] → "Nối vào input" → mũi tên vòng về "Model" ghi "chưa xong"; nhánh ra "Câu trả lời" ghi "gặp token kết thúc hoặc đủ max_new_tokens"
Công thức dưới sơ đồ: y_t ~ P_θ( · | x, y₁, …, y_(t−1) ),  t = 1, 2, …
"Câu trả lời 60 token = model chạy 60 lần. Chỉ khối hình thoi khác nhau giữa các cách giải mã."

BẢNG BÊN PHẢI – "4 cách chọn token":
| Cách | Công thức | Tính chất |
| Greedy | y_t = argmaxᵢ pᵢ | Tất định, chạy lại ra y hệt; dễ lặp, nhạt |
| Temperature T | pᵢ⁽ᵀ⁾ = e^(zᵢ/T) / Σⱼ e^(zⱼ/T) | T < 1 nhọn hơn · T > 1 phẳng hơn · T → 0 ≈ greedy |
| Top-p (nucleus) | giữ tập nhỏ nhất V_p có Σ_(i∈V_p) pᵢ ≥ p, chia lại xác suất, lấy mẫu | Cắt đuôi; số ứng viên tự co giãn |
| Top-k | giữ k token cao nhất | Số ứng viên cố định |

KHỐI DƯỚI – "Hệ quả của temperature":
pᵢ⁽ᵀ⁾ ∝ pᵢ^(1/T)   ⇒   p_a⁽ᵀ⁾ / p_b⁽ᵀ⁾ = (p_a / p_b)^(1/T)
"Temperature không đổi thứ hạng, chỉ đổi độ chênh: T = 0.5 bình phương tỉ lệ; T = 1.5 lấy căn bậc 1.5."
"Holtzman 2020: giải mã cực đại hoá xác suất dễ sinh văn bản nhạt, lặp vòng."
```

## Ảnh 08B — Mục 6: Demo cùng prompt, nhiều cấu hình

```text
PHONG CÁCH: (dùng khối phong cách chung) Chân trang trái: "Nguồn: notebook phần 6 · run 20261119_150356_normal"; phải: "8B". VIẾT CHÍNH XÁC TỪNG SỐ.

TIÊU ĐỀ: "Mục 6 · Demo: 'Viết một câu giới thiệu ngắn về thành phố Hà Nội.'"

DÃY 8 Ô TOKEN (ô càng đậm xác suất càng cao), trên ghi "Sinh từng bước bằng vòng lặp tự viết = generate()", dưới mỗi ô ghi "bước · P":
H 0.85 | à 0.998 | Nội 0.9998 | là 0.67 | thành 0.46 | phố 0.9997 | lớn 0.60 | nhất 0.41
Chú thích: "Input dài thêm 1 token mỗi bước: 42 → 49"

BIỂU ĐỒ GIỮA – "Token đầu tiên khi đổi temperature" (3 nhóm cột):
T = 0.5: P('H') 0.995 · số token phủ 90% = 1
T = 1.0: P('H') 0.847 · số token phủ 90% = 3
T = 1.5: P('H') 0.468 · số token phủ 90% = 135 (phủ 95% = 1,068)
Thẻ kiểm: "(0.847/0.050)^(1/1.5) = 6.60 ≈ đo thật 0.468/0.071 = 6.59"

BẢNG DƯỚI – "5 cấu hình × 3 lần (seed 42, 43, 44)":
| Cấu hình | Số output khác nhau | Quan sát |
| greedy | 1 | 3 lần giống hệt ✓ |
| temperature 0.5 | 3 | khác nhau, mạch lạc ✓ |
| temperature 1.5 | 3 | vô nghĩa, lẫn nhiều ngôn ngữ: "…thángيران trung/sdk…" ✗ |
| top-p 0.5 | 3 | khác nhau, mạch lạc ✓ |
| top-p 0.95 | 3 | có lần bịa "thành phố… ở Trung Quốc" ✗ |

KHỐI CODE:
for step in range(8):
    probs = torch.softmax(model(ids).logits[0, -1], dim=-1)
    next_id = probs.argmax()                                  # greedy
    ids = torch.cat([ids, next_id.view(1, 1)], dim=1)         # nối vào input

KHỐI PHÂN TÍCH:
"T cao làm phân phối phẳng: ở token đầu tiên, nhóm 90% xác suất phình từ 3 lên 135 token → dễ chọn token lạ; token lạ vào ngữ cảnh làm các bước sau lạc thêm (lỗi tích luỹ). Greedy lặp lại được nên dùng cho mọi đánh giá; lấy mẫu đa dạng nhưng có thể sai. Greedy cũng có lỗi sự thật: 'Đền Hùng' ở Hà Nội (thực tế ở Phú Thọ)."
```

---

## Ảnh 09A — Mục 7: Khả năng của LLM (Lý thuyết)

```text
PHONG CÁCH: (dùng khối phong cách chung) Chân trang trái: "Nguồn: Brown 2020; Min et al. 2022; Zhao et al. 2021; Wei et al. 2021 (FLAN)"; phải: "9A". VIẾT CHÍNH XÁC TỪNG CHỮ.

TIÊU ĐỀ: "Mục 7 · In-context learning, làm theo chỉ dẫn, tác vụ ngôn ngữ"

KHỐI 1 – "In-context learning (ICL)":
"Model làm tác vụ nhờ chỉ dẫn và/hoặc ví dụ đặt TRONG PROMPT. Trọng số KHÔNG đổi, không có gradient."
Hai khung prompt minh họa cạnh nhau:
- "Zero-shot: Chỉ dẫn + câu cần làm"
- "Few-shot (k-shot): Chỉ dẫn + k ví dụ có nhãn + câu cần làm"
Công thức: P( nhãn | chỉ dẫn, ví dụ₁ … ví dụ_k, câu )

KHỐI 2 – "Ví dụ giúp bằng cách nào?" (Min et al. 2022), 3 icon:
"Cho thấy tập nhãn" · "Cho thấy kiểu câu đầu vào" · "Cho thấy định dạng" — "Đổi nhãn ví dụ thành ngẫu nhiên chỉ làm giảm ít."

KHỐI 3 – "Rủi ro" (Zhao et al. 2021), viền đỏ:
"Nhạy với chọn ví dụ và thứ tự ví dụ · thiên về nhãn chiếm đa số (majority label) · thiên về nhãn ví dụ cuối (recency) · thiên về token phổ biến (common token)"
"→ Nhóm dùng 2 tích cực + 2 tiêu cực, xếp xen kẽ."

KHỐI 4 – "Làm theo chỉ dẫn & tác vụ ngôn ngữ":
"Có được nhờ SFT trên dữ liệu chỉ dẫn (mục 5). Một model: hỏi đáp, tóm tắt, dịch, phân loại, sinh mã (mục 1)."

KHỐI 5 – "Thiết kế thí nghiệm: chỉ đổi MỘT biến":
Hai cột giống nhau: "cùng model · cùng mẫu prompt · greedy · max_new_tokens 10 · cùng 2,999 câu test, cùng thứ tự" — khác duy nhất: "có / không 4 ví dụ" (tô cam).
"Chống rò rỉ: ví dụ lấy từ tập train, không trùng câu test. Chọn mẫu prompt trên 200 câu validation rồi mới chạy test."
```

## Ảnh 09B — Mục 7: Demo zero-shot vs few-shot

```text
PHONG CÁCH: (dùng khối phong cách chung) Chân trang trái: "Nguồn: notebook phần 7 · run 20261119_150356_normal"; phải: "9B". VIẾT CHÍNH XÁC TỪNG CHỮ, TỪNG SỐ.

TIÊU ĐỀ: "Mục 7 · Demo: phân loại cảm xúc UIT-VSFC (2,999 câu test)"

HAI KHUNG PROMPT CẠNH NHAU (monospace):
ZERO-SHOT:
"Phân loại cảm xúc của câu phản hồi sinh viên sau. Chỉ trả lời: tích cực hoặc tiêu cực.
Câu: "…"
Cảm xúc:"
FEW-SHOT (thêm 4 ví dụ từ train, xen kẽ):
"Câu: "dạy tốt có tâm ." → Cảm xúc: tích cực
Câu: "oop khá phức tạp nên cần nhiều ví dụ ." → Cảm xúc: tiêu cực
Câu: "giáo viên tích cực , năng nổ trong giảng dạy , …" → Cảm xúc: tích cực
Câu: "giảng viên cần linh hoạt trong giảng dạy , …" → Cảm xúc: tiêu cực
Câu: "…" → Cảm xúc:"

SỐ LỚN GIỮA: "89.86%" (zero-shot) → mũi tên → "92.30%" (few-shot) · dòng nhỏ "0 câu không đọc được nhãn ở cả hai cách"

BẢNG NHỎ "Chọn mẫu prompt trên 200 câu validation":
| Mẫu | Zero | Few | Trung bình |
| vi_nhan_en | 0.840 | 0.905 | 0.8725 |
| vi_nhan_vi (chọn) | 0.845 | 0.905 | 0.875 |
| en_nhan_en | 0.825 | 0.870 | 0.8475 |

KHỐI CODE:
def build_prompt(text, examples, tpl):          # examples = [] → zero-shot
    blocks = [tpl["instruction"]]
    blocks += [tpl["item"].format(text=e["sentence"]) + " " + tpl["labels"][e["label"]] for e in examples]
    return "\n\n".join(blocks + [tpl["item"].format(text=text)])

KHỐI PHÂN TÍCH:
"Hai ví dụ tiêu cực đều là câu góp ý ('nên…', 'cần…'): ví dụ cho model thấy quy ước gán nhãn của UIT-VSFC (góp ý = tiêu cực), không dạy kiến thức mới. Few-shot còn khiến model chép định dạng 'Cảm xúc: tiêu cực' của ví dụ. Có ý nghĩa thống kê hay không: mục 10."
```

---

## Ảnh 10A — Mục 8: Prompting, fine-tuning, RAG (Lý thuyết – bảng của đề)

```text
PHONG CÁCH: (dùng khối phong cách chung) Chân trang trái: "Nguồn: Lewis et al. 2020; Hu et al. 2021 §4.1; scikit-learn TfidfVectorizer"; phải: "10A". VIẾT CHÍNH XÁC TỪNG CHỮ, TỪNG KÝ HIỆU.

TIÊU ĐỀ: "Mục 8 · Ba cách thích nghi và bổ sung tri thức"

BẢNG LỚN (minh họa của đề), mỗi hàng một icon:
| Cách | Đổi gì | Dữ liệu cần có | Chi phí | Khi nào dùng |
| Prompting (chỉ dẫn, zero/few-shot) | Chỉ đổi input | Vài ví dụ hoặc không | Rẻ nhất, tức thì | Tác vụ model đã biết làm |
| Fine-tuning (full hoặc LoRA) | Trọng số model | Tập (input, output) có nhãn + GPU để train | Cao | Đổi hành vi, văn phong, định dạng lâu dài; tác vụ chuyên biệt |
| RAG | Thêm tài liệu vào prompt lúc hỏi; trọng số không đổi | Kho tài liệu + bộ tìm kiếm | Trung bình | Tri thức mới, riêng, hay thay đổi; cần dẫn nguồn |

KHỐI TRÁI DƯỚI – "LoRA (ý tưởng)":
W₀ + ΔW = W₀ + B·A,  B ∈ ℝ^(d×r), A ∈ ℝ^(r×k), r ≪ min(d, k)
"Giữ nguyên W₀, chỉ train r(d + k) tham số thay vì d·k."

KHỐI GIỮA DƯỚI – "RAG = Retrieval + Augmented Generation":
"① Retrieval: tìm k đoạn liên quan nhất trong kho" → "② Augmented Generation: chèn k đoạn vào prompt rồi sinh câu trả lời"

KHỐI PHẢI DƯỚI – "Bộ tìm kiếm TF-IDF":
tfidf(t, d) = tf(t, d) × [ ln((1 + N) / (1 + df(t))) + 1 ]
"tf: số lần t xuất hiện trong đoạn d · N: số đoạn · df(t): số đoạn chứa t · từ hiếm → trọng số lớn"
"Vector chuẩn hoá L2 → cos(q, d) = q·d" · "recall@k = tỉ lệ câu hỏi có đoạn gốc trong k đoạn tìm được"
```

## Ảnh 10B — Mục 8: Demo không RAG vs có RAG

```text
PHONG CÁCH: (dùng khối phong cách chung) Chân trang trái: "Nguồn: notebook phần 8 · UIT-ViQuAD 2.0 · run 20261119_150356_normal"; phải: "10B". VIẾT CHÍNH XÁC TỪNG SỐ.

TIÊU ĐỀ: "Mục 8 · Demo: cùng câu hỏi, không RAG vs có RAG"

SƠ ĐỒ 2 NHÁNH trên cùng:
"Câu hỏi" → nhánh trên: "TF-IDF tìm top-3 / 557 đoạn" → "Prompt + 3 đoạn" → "Qwen2.5-1.5B greedy" → "Trả lời"
           → nhánh dưới: "Prompt không tài liệu" → "Qwen2.5-1.5B greedy"
Nhãn chung: "Hai nhánh cùng câu dặn: Nếu không biết hoặc tài liệu không có thông tin, trả lời 'Không có thông tin'."

BIỂU ĐỒ CỘT CHỒNG "15 câu có đáp án" (xanh = đúng, đỏ = sai, xám = từ chối):
Không RAG: 0 đúng · 1 sai · 14 từ chối
Có RAG: 10 đúng · 3 sai · 2 từ chối

THẺ VÍ DỤ: "Sân bay nào có số lượng phi cơ ra vào nhiều nhất châu Âu năm 2006?"
Không RAG → "Không có thông tin." · Có RAG → "Charles-de-Gaulle" ✓

HỘP HẠN CHẾ (viền đỏ):
- "recall@3 = 0.722 (13/18): đoạn đúng không được tìm thấy → RAG không cứu được (2 câu vẫn từ chối)"
- "3 câu tài liệu KHÔNG có đáp án: không RAG từ chối 3/3; có RAG trả lời bịa 2/3"
- "Có tài liệu vẫn sai: đọc sai đoạn (Paris 'thế kỷ 16–17'), tìm sai đoạn (Nam Ossetia 'Nga')"

KHỐI CODE:
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
doc_matrix = vectorizer.fit_transform(corpus["title"] + ". " + corpus["context"])
scores = linear_kernel(vectorizer.transform([query]), doc_matrix)[0]   # cosine
top3 = scores.argsort()[::-1][:3]

KHỐI PHÂN TÍCH:
"RAG bổ sung tri thức lúc hỏi, không đổi trọng số: đúng 0 → 10/15, vì đáp án có sẵn trong đoạn, model chỉ cần đọc và trích. Nhưng RAG chỉ tốt bằng bộ tìm kiếm, và tài liệu 'gần đúng' có thể làm model trả lời sai một cách tự tin. Không RAG từ chối cả câu model biết (Mona Lisa) vì câu dặn làm model quá thận trọng."
```

---

## Ảnh 11A — Mục 9: Hạn chế (Lý thuyết)

```text
PHONG CÁCH: (dùng khối phong cách chung) Chân trang trái: "Nguồn: Huang et al. 2023; Lin et al. 2022; Liu et al. 2023; Bolukbasi et al. 2016; Wei et al. 2024"; phải: "11A". VIẾT CHÍNH XÁC TỪNG CHỮ.

TIÊU ĐỀ: "Mục 9 · Hạn chế và đánh giá LLM"

4 THẺ LỚN (2×2), mỗi thẻ một icon:
1. "Hallucination" (viền đỏ):
   "Sinh nội dung nghe hợp lý nhưng sai."
   "• Factuality: sai với sự thật (bịa năm, bịa người)"
   "• Faithfulness: không đúng chỉ dẫn / tài liệu được đưa"
   "Vì sao: (1) huấn luyện tối đa xác suất văn bản, không tối đa sự thật; (2) luôn chọn ra một token, không tra cứu; (3) tiền đề sai trong câu hỏi thành ngữ cảnh để sinh tiếp; (4) model lớn hơn chưa chắc trung thực hơn (TruthfulQA)."
2. "Thiên lệch":
   "Dữ liệu có định kiến (giới, nghề nghiệp, vùng miền) thì model học lại. Giảm bằng lọc dữ liệu, alignment, đánh giá riêng. (Trình bày bằng lời, không demo.)"
3. "Giới hạn ngữ cảnh":
   "Cứng: tối đa 32,768 token (Qwen2.5-1.5B; sinh tối đa 8K)."
   "Mềm – Lost in the middle: hiệu năng thường cao nhất khi thông tin cần dùng ở đầu hoặc cuối ngữ cảnh, giảm đáng kể khi nằm ở giữa."
   "Tokenizer tốn token (GPT-2 gấp 2.75× cho câu Việt) → hết ngữ cảnh nhanh hơn."
4. "Phụ thuộc tác vụ và dữ liệu":
   "Bộ đánh giá cũng có nhãn sai · quy ước nhãn khác cách hiểu (góp ý = tiêu cực) · model chen tiếng Trung khi hỏi tiếng Việt."

DẢI DƯỚI – "Cách chấm (SimpleQA)": "Đúng · Sai · Từ chối (không trả lời) – từ chối không bịa nên không tính là hallucination."
```

## Ảnh 11B — Mục 9: Một câu trả lời sai và cách kiểm chứng (Demo)

```text
PHONG CÁCH: (dùng khối phong cách chung) Chân trang trái: "Nguồn: notebook phần 9 · UIT-ViQuAD 2.0 · run 20261119_150356_normal"; phải: "11B". VIẾT CHÍNH XÁC TỪNG CHỮ, TỪNG SỐ.

TIÊU ĐỀ: "Mục 9 · Demo: một câu trả lời sai và cách kiểm chứng"

THẺ KIỂM CHỨNG LỚN (minh họa của đề):
Câu hỏi (không tài liệu, không câu dặn từ chối): "Muốn ban hành luật thì hoàng đế phải làm gì?" (chủ đề: Đế quốc La Mã Thần thánh)
- Cách hỏi A → "…hoàng đế cần làm theo các bước sau: 1. Xác định nội dung của luật muốn ban hành. 2. Chuyển đổi ý tưởng thành văn…" ✗
- Cách hỏi B → "Đưa ra một đạo luật." ✗
- Nguồn Wikipedia (đoạn gốc): "…Hoàng đế phải thông qua Hội đồng để ban hành luật, và những buổi thảo luận ít khi đạt sự nhất trí…" ✓

SỐ LỚN: "Sai 12/15 (cách A) · 12/15 (cách B) · không từ chối câu nào" · "Thực thể không tồn tại: bịa 5/6, chỉ ra tiền đề sai 0/6"

BẢNG "Hỏi lại cách khác lộ ra việc đoán":
| Câu hỏi | Cách A | Cách B | Đáp án |
| Texas: bao nhiêu học sinh / 1 giáo viên? | 25–30 | 1:1 | 14,9 |
| Mông Cổ: tỉ lệ người dưới 30 tuổi? | 20–25% | 15% | khoảng 59% |
| Nhà vật lý Nguyễn Hoàng Lâm nhận Nobel năm nào? (người không có thật) | 2021 | 1994 | không tồn tại |
| Việt Nam đăng cai Thế vận hội Mùa hè 1987 ở đâu? | Hà Nội | Hà Nội | chưa từng; 1987 không có Thế vận hội |

3 BƯỚC KIỂM CHỨNG (icon xanh lá): "① Đối chiếu nguồn gốc" · "② Hỏi lại cách khác – hai câu không khớp là đang đoán" · "③ Dùng RAG đưa nguồn vào prompt (RAG vẫn có thể bịa: 2/3 câu tài liệu không có đáp án, mục 8)"

KHỐI CODE:
PHRASINGS = {
    "A": lambda r: f"{r.question} (chủ đề: {r.title}) Trả lời ngắn gọn.",
    "B": lambda r: f"Về {r.title}, hãy cho biết: {r.question} Chỉ trả lời ngắn gọn, không giải thích.",
}
start = int(answers["answer_start"][0])                  # vị trí đáp án trong đoạn Wikipedia
end = start + len(str(answers["text"][0]))
evidence = context[max(0, start - 120):end + 120]        # đoạn nguồn để kiểm chứng

KHỐI PHÂN TÍCH:
"Model sinh token có xác suất cao, không tra cứu sự thật; câu hỏi có tiền đề sai thì nó nhận tiền đề rồi sinh tiếp. Khi biết thật, đổi cách hỏi vẫn ra cùng đáp án (Charles de Gaulle đúng cả 2 lần); khi đoán, mỗi lần một số. Nhất quán chưa chắc đúng ('Hà Nội' cả 2 lần vẫn sai). Bộ dữ liệu cũng có nhãn sai: đáp án '1640' cho năm vua Charles bị xử tử, thực tế 1649."
```

---

## Ảnh 12A — Mục 10: Thử nghiệm có kiểm soát (Lý thuyết)

```text
PHONG CÁCH: (dùng khối phong cách chung) Chân trang trái: "Nguồn: Dietterich 1998; statsmodels mcnemar"; phải: "12A". VIẾT CHÍNH XÁC TỪNG KÝ HIỆU TOÁN.

TIÊU ĐỀ: "Mục 10 · Thiết kế thử nghiệm có kiểm soát và cách đánh giá"

KHỐI 1 – "Thiết lập" (lưới icon):
"Model: Qwen2.5-1.5B-Instruct fp16" · "Dữ liệu: UIT-VSFC test 2,999 câu (1,590 tích cực / 1,409 tiêu cực)" · "Prompt chọn trên 200 câu validation" · "Greedy, max_new_tokens 10" · "4 ví dụ few-shot từ train" · "Seed 42"
Ô nổi bật: "Biến duy nhất: có / không 4 ví dụ"
"Giả thuyết H₀: hai cách có tỉ lệ sai như nhau."

KHỐI 2 – "Accuracy và khoảng tin cậy 95%":
acc = số câu đúng / n ;   CI₉₅% = acc ± 1.96 · √( acc(1 − acc) / n )

KHỐI 3 – "Confusion matrix":
"Hàng = nhãn thật, cột = dự đoán." recall_c = đúng lớp c / số câu thật là c ;  precision_c = đúng lớp c / số câu đoán là c

KHỐI 4 – "Kiểm định McNemar (dữ liệu theo cặp)":
Bảng 2×2: hàng "Zero đúng / Zero sai", cột "Few đúng / Few sai", ô a, b, c, d.
"a, d (hai cách cùng đúng / cùng sai) không nói cách nào tốt hơn; chỉ b, c (bất đồng) có thông tin."
"Nếu H₀ đúng: mỗi câu bất đồng rơi vào b hay c như tung đồng xu:" b ~ Binomial(b + c, 0.5) ; p = min(1, 2·P(X ≤ min(b, c)))
"Vì sao không so 2 tỉ lệ độc lập: hai cách chạy trên CÙNG câu nên không độc lập. Một phép so sánh → không cần Bonferroni."
```

## Ảnh 12B — Mục 10: Kết quả và McNemar (Demo)

```text
PHONG CÁCH: (dùng khối phong cách chung) Chân trang trái: "Nguồn: notebook phần 10 (cell 10.1–10.2) · run 20261119_150356_normal"; phải: "12B". VIẾT CHÍNH XÁC TỪNG SỐ.

TIÊU ĐỀ: "Mục 10 · Kết quả: few-shot tốt hơn có ý nghĩa thống kê"

HAI THẺ SỐ LỚN cạnh nhau:
"Zero-shot 89.86% (±1.08)" · "Few-shot 92.30% (±0.95)" · dòng nhỏ: "CI 95%: [88.78; 90.94] vs [91.35; 93.25] · không đọc được nhãn: 0 / 0"

BẢNG McNEMAR 2×2 lớn (tô màu ô):
| | Few đúng | Few sai |
| Zero đúng | a = 2,648 | b = 47 |
| Zero sai | c = 120 | d = 184 |
Ô kết luận xanh lá: "p = 1.5e-08 < 0.05 → few-shot tốt hơn có ý nghĩa thống kê"
Thẻ kiểm: "167 câu bất đồng; nếu H₀ đúng mỗi bên ≈ 83.5, thực tế zero chỉ thắng 47"

KHỐI CODE:
table = mcnemar_counts(icl_zero["correct"], icl_few["correct"])   # [[a, b], [c, d]]
p = mcnemar(table, exact=True).pvalue                              # 1.503e-08

KHỐI PHÂN TÍCH:
"p = 1.5e-08 nghĩa là: nếu hai cách thật sự ngang nhau, khả năng thấy chênh lệch lớn như vậy chỉ khoảng 1.5 phần trăm triệu. Không có nghĩa few-shot tốt hơn với xác suất 99.99%. Giới hạn: 1 model 1.5B, 1 bộ dữ liệu, 1 bộ ví dụ, 1 lần chạy – không suy rộng cho mọi LLM."
```

## Ảnh 12C — Mục 10: Confusion matrix, phân tích lỗi, tổng hợp

```text
PHONG CÁCH: (dùng khối phong cách chung) Chân trang trái: "Nguồn: notebook phần 10 (cell 10.1, 10.3, 10.4) · run 20261119_150356_normal"; phải: "12C". VIẾT CHÍNH XÁC TỪNG SỐ.
ẢNH THAM CHIẾU: đính kèm outputs/figures/confusion_matrix_icl.png – đặt NGUYÊN VẸN ảnh vào khung trên trái, không vẽ lại.

TIÊU ĐỀ: "Mục 10 · Phân tích lỗi và tổng hợp kết quả"

TRÊN TRÁI – khung ảnh "[confusion_matrix_icl.png]" + 2 mũi tên chú thích:
"tiêu cực → bị đoán tích cực: 248 → 145 (giảm 103)" (xanh lá) · "tích cực → bị đoán tiêu cực: 56 → 86 (tăng 30)" (đỏ)
Thẻ nhỏ: "Recall tiêu cực 82.4% → 89.7% · Recall tích cực 96.5% → 94.6%"

TRÊN PHẢI – biểu đồ tròn "30 câu few-shot sai (nhóm định nghĩa TRƯỚC khi đọc)":
"Ý ngầm / góp ý 20" · "Vừa khen vừa chê 4" · "Model sai rõ 4" · "Nhãn gốc có vẻ sai 2" · "Không đọc được nhãn 0"
Ví dụ: "'giảm thời lượng lý thuyết , tăng thời lượng thực hành .' – nhãn tiêu cực → model tích cực"
Số lớn: "76–81% câu tiêu cực bị đoán tích cực là câu góp ý ('nên', 'cần', 'mong', 'thêm', 'hơn')"

DƯỚI – BẢNG TỔNG HỢP 10 MỤC (2 cột, mỗi dòng một icon):
"1 · 4 tác vụ, 1 model: đạt 4/4"
"2 · Top-1 sau 'Hôm nay trời rất' = 'đẹp' 0.788; attention tương lai = 0"
"3 · Câu Việt: Qwen 8 / GPT-2 22 token; câu Anh 7 / 7"
"4 · Loss: đúng trật tự 2.985 · xáo trộn 6.701 · tiếng Anh 3.129 · đoán bừa 11.931"
"5 · Base 0/4 · Instruct 4/4"
"6 · Greedy 1 output; mọi cấu hình lấy mẫu 3 output khác nhau / 3 lần"
"7, 10 · Zero 89.86% → few 92.30%; McNemar p = 1.5e-08"
"8 · RAG: đúng 0 → 10/15; câu không có đáp án: RAG bịa 2/3"
"9 · Không tài liệu: sai 12/15 (A), 12/15 (B); thực thể không tồn tại: bịa 5/6"

KHỐI CODE nhỏ:
pred = res["pred"].map(LABEL_NAMES).fillna("không parse được")
cm = pd.crosstab(res["label"].map(LABEL_NAMES), pred)     # hàng: nhãn thật, cột: dự đoán
SUGGEST_RE = r"\b(?:nên|cần|mong|thêm|hơn)\b"            # đếm câu góp ý trong các câu sai

KHỐI PHÂN TÍCH:
"Lỗi chính không phải model kém tiếng Việt mà là quy ước nhãn của UIT-VSFC: câu góp ý bị gán tiêu cực. Few-shot sửa 103 lỗi theo hướng đó nhưng đổi lại 30 lỗi ngược. Đúng/sai phụ thuộc định nghĩa nhãn của bộ dữ liệu (nối mục 9)."
```

---

## Ảnh 13 — Tài liệu tham khảo

```text
PHONG CÁCH: Infographic khoa học 16:9, nền trắng, tiêu đề #1F3A5F, chữ rõ, font sans-serif tiếng Việt. Chân trang phải: "13". VIẾT CHÍNH XÁC TỪNG CHỮ.

TIÊU ĐỀ: "Tài liệu tham khảo"

CỘT 1 – "Bài báo nền tảng":
Vaswani et al., Attention Is All You Need, NeurIPS 2017
Bengio et al., A Neural Probabilistic Language Model, JMLR 2003
Sennrich et al., Neural Machine Translation of Rare Words with Subword Units, ACL 2016
Brown et al., Language Models are Few-Shot Learners, 2020
Ouyang et al., Training language models to follow instructions with human feedback, 2022
Rafailov et al., Direct Preference Optimization, 2023
Holtzman et al., The Curious Case of Neural Text Degeneration, ICLR 2020
Min et al., Rethinking the Role of Demonstrations, 2022
Zhao et al., Calibrate Before Use, 2021
Lewis et al., Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks, NeurIPS 2020
Hu et al., LoRA: Low-Rank Adaptation of Large Language Models, 2021

CỘT 2 – "Hạn chế, đánh giá, kiến trúc":
Huang et al., A Survey on Hallucination in Large Language Models, 2023
Lin et al., TruthfulQA, ACL 2022
Liu et al., Lost in the Middle, 2023
Wei et al., Measuring short-form factuality (SimpleQA), 2024
Dietterich, Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms, 1998
Su et al., RoFormer (RoPE), 2021 · Zhang & Sennrich, RMSNorm, 2019 · Shazeer, GLU Variants, 2020 · Ainslie et al., GQA, 2023

CỘT 3 – "Model, dữ liệu, thư viện":
Qwen Team, Qwen2.5 Technical Report, 2024
Nguyen et al., UIT-VSFC, KSE 2018
Nguyen et al., VLSP 2021 – ViMRC (UIT-ViQuAD 2.0), 2022
Hugging Face Transformers · scikit-learn · statsmodels (tài liệu chính thức)
```

---

## Kiểm sau khi sinh ảnh (nhóm làm)

- [ ] Dấu tiếng Việt đúng ở mọi chữ (model ảnh hay viết sai dấu).
- [ ] Mọi con số khớp `outputs/results/` (bảng kiểm: [`../giai_thich_du_an_v2.md`](../giai_thich_du_an_v2.md)).
- [ ] Công thức đúng ký hiệu (√, Σ, ∏, −∞, ⊙, ‖, ~).
- [ ] Code đúng từng ký tự (dấu ngoặc, dấu `:`, chỉ số `[0, -1]`).
- [ ] Ảnh 04C, 12C dùng đúng ảnh thật trong `outputs/figures/`.
- [ ] Nếu ảnh nào chữ sai nhiều: sinh lại, hoặc chỉ sinh nền + sơ đồ rồi gõ chữ đè lên trong Google Slides / PowerPoint.
