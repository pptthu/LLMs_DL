# Prompt tạo từng ảnh infographic kiểu "bảng công thức" — Presentation LLMs (v2)

- **Created**: 2026-11-20 (Toàn, AI hỗ trợ) · Trạng thái: **[đề xuất]**
- **Thay** [`prompt_anh_infographic_tung_slide_v1.md`](prompt_anh_infographic_tung_slide_v1.md) (v1 giữ nguyên). v2 bám bố cục ảnh mẫu nhóm chọn (slide "Các công thức cốt lõi trong XGBoost"): thanh tiêu đề có số slide, lưới **6 panel đánh số (3 cột × 2 hàng)**, mỗi panel có hộp công thức + chú giải ký hiệu, có panel ví dụ số, dải **tóm tắt flow** màu xanh đậm ở đáy.
- **Cách dùng (Gemini / Antigravity):**
  1. Mở một cuộc trò chuyện mới. Gửi **PHẦN 1 — QUY CHUẨN THIẾT KẾ** một lần, chờ Gemini xác nhận "Đã hiểu".
  2. Gửi lần lượt từng khối **Ảnh NN** (mỗi khối = một ảnh). Mỗi khối đã ghi "Áp dụng QUY CHUẨN THIẾT KẾ" nên Gemini dùng lại quy chuẩn.
  3. Nếu công cụ quên quy chuẩn giữa chừng, dán lại PHẦN 1 rồi gửi tiếp.
- Nội dung lý thuyết: [`../giai_thich_du_an_v2.md`](../giai_thich_du_an_v2.md) (đã đối chiếu bài gốc). Số liệu: `outputs/results/`, run `20261119_150356_normal`.
- 26 ảnh: 01 bìa · 02 môi trường · 03–04 mục 1 · 05–07 mục 2 · 08–09 mục 3 · 10–11 mục 4 · 12–14 mục 5 · 15–16 mục 6 · 17–18 mục 7 · 19–20 mục 8 · 21–22 mục 9 · 23–25 mục 10 · 26 tài liệu.

---

## PHẦN 1 — QUY CHUẨN THIẾT KẾ (gửi một lần đầu tiên)

```text
Bạn sẽ tạo một loạt ảnh slide infographic khoa học cho bài thuyết trình "Tìm hiểu mô hình ngôn ngữ lớn (LLMs)" (môn Học sâu, UTH). Mỗi tin nhắn sau là nội dung của MỘT ảnh. Mọi ảnh PHẢI tuân thủ quy chuẩn dưới đây. Trả lời "Đã hiểu" và chưa tạo ảnh nào.

1. KHỔ ẢNH: ngang 16:9, 1920×1080. Nền trắng #FFFFFF. Lề ngoài 40px. Phong cách phẳng (flat), sạch, giống trang tóm tắt công thức trong giáo trình. Không 3D, không đổ bóng đậm, không gradient loè loẹt.

2. THANH TIÊU ĐỀ (cao ~150px):
   - Góc trái: ô vuông nền xanh đậm #1F3A5F, chữ trắng: số slide 2 chữ số rất lớn (vd "05"), dưới là chữ nhỏ "SLIDE".
   - Cạnh ô: TIÊU ĐỀ viết HOA, đậm, màu #1F3A5F, cỡ ~52px.
   - Dưới tiêu đề: 1 dòng phụ đề màu xám xanh #4A6A8A, cỡ ~22px.
   - Góc phải: một nhãn viền mảnh bo tròn (pill) ghi "Mục N · <tên mục>" (chữ, KHÔNG logo). Dưới nhãn: dòng chữ nhỏ xám "Nguồn: …" do tôi cung cấp.

3. LƯỚI PANEL: 6 panel, 3 cột × 2 hàng, bằng nhau, khoảng cách 16px (trừ khi tôi ghi rõ panel gộp cột).
   - Mỗi panel: thẻ trắng, viền 1px #C9D9EA, bo góc 8px.
   - Đầu panel: thanh ngang nền xanh #1F5FAF, chữ trắng VIẾT HOA, đánh số: "1. TÊN PANEL".
   - Trong panel chỉ gồm các thành phần sau:
     a) HỘP CÔNG THỨC: nền xanh rất nhạt #EAF2FB, bo góc, công thức căn giữa, dàn trang như sách toán (phân số, căn, tổng Σ, chỉ số dưới/trên, chữ nghiêng cho biến). KHÔNG hiển thị mã LaTeX thô, không ký tự "\".
     b) CHÚ GIẢI KÝ HIỆU: bảng 2 cột nhỏ "ký hiệu | ý nghĩa", chữ ~15px.
     c) GẠCH ĐẦU DÒNG: tối đa 4 dòng, mỗi dòng ≤ 18 từ, chữ ~16px.
     d) BẢNG SỐ: viền mảnh, hàng tiêu đề nền #DCE8F5.
     e) SƠ ĐỒ NHỎ: chỉ khi tôi mô tả (hộp + mũi tên, ma trận ô vuông, cây, biểu đồ cột). Sơ đồ phải thể hiện đúng cấu trúc tôi mô tả.
     f) KHỐI CODE: nền xám #F4F6F7, font monospace, chép đúng từng ký tự.
     g) HỘP Ý NGHĨA: viền trái 4px #E67E22, nền #FEF5E7, 1–3 câu.
   - Màu nhấn: đúng/đạt #27AE60; sai/hạn chế #C0392B; số liệu demo nổi bật #E67E22.

4. DẢI TÓM TẮT FLOW ở đáy (cao ~140px, rộng toàn trang, nền xanh đậm #0F2A5C, bo góc):
   - Bên trái: chữ trắng đậm "TÓM TẮT FLOW" + 1 dòng tên chủ đề.
   - Bên phải: 5–6 bước nối bằng mũi tên chevron trắng. Mỗi bước = 1 ô vuông trắng nhỏ chứa icon nét mảnh + 1–2 dòng chữ trắng ~15px.

5. QUY TẮC ICON (BẮT BUỘC):
   - Icon CHỈ được đặt trong dải TÓM TẮT FLOW (và nơi tôi ghi rõ "icon:" trong nội dung).
   - Mỗi icon phải mô tả ĐÚNG NGHĨA bước đó, theo đúng mô tả "icon:" tôi đưa (vd: "dãy ô token", "ma trận lưới", "biểu đồ cột xác suất", "kính lúp trên tài liệu").
   - CẤM: emoji; icon trang trí không gắn nội dung (bóng đèn, tên lửa, cúp, bia ngắm, ngôi sao, trái tim, ngón tay cái, bộ não hoạt hình, robot); ảnh người; ảnh chụp; logo công ty/tổ chức thật (OpenAI, Alibaba, Qwen, Google, Hugging Face…); hình nền họa tiết; watermark.

6. QUY TẮC CHỮ VÀ SỐ (BẮT BUỘC):
   - Viết CHÍNH XÁC từng chữ, từng số, từng ký hiệu tôi đưa trong dấu ngoặc kép hoặc trong khối nội dung. Giữ đủ dấu tiếng Việt.
   - KHÔNG thêm chữ, số, công thức, ví dụ, tên riêng nào không có trong tin nhắn. KHÔNG dùng lorem ipsum, KHÔNG chữ giả.
   - KHÔNG làm tròn hay đổi số. KHÔNG dịch thuật ngữ tiếng Anh tôi giữ nguyên (token, embedding, softmax…).
   - Nếu thiếu chỗ: thu nhỏ cỡ chữ (không dưới 14px) hoặc thu sơ đồ; KHÔNG được bỏ nội dung.
   - Không có tiêu đề hoặc chú thích nào bằng tiếng Anh nếu tôi viết tiếng Việt.

7. KHI TÔI ĐÍNH KÈM ẢNH THẬT (biểu đồ từ notebook): đặt nguyên ảnh vào panel được chỉ định, giữ tỉ lệ, không vẽ lại, không chỉnh số trên ảnh.

8. Sau mỗi ảnh, liệt kê ngắn những chữ/số bạn không chắc đã hiển thị đúng.
```

---

## Ảnh 01 — Bìa

```text
Áp dụng QUY CHUẨN THIẾT KẾ. Ngoại lệ ảnh này: không lưới 6 panel; dùng bố cục bìa bên dưới. Không có ô số slide.

- Giữa trên: tiêu đề 2 dòng màu #1F3A5F, đậm: "TÌM HIỂU MÔ HÌNH NGÔN NGỮ LỚN (LLMs)" / "QUA QWEN2.5-1.5B-INSTRUCT".
- Phụ đề: "Demo trên model thật cho 10 mục: kiến trúc · token · học · huấn luyện · sinh văn bản · khả năng · RAG · hạn chế · thực nghiệm".
- Giữa: 3 panel ngang (đầu panel xanh #1F5FAF):
  "1. BÊN TRONG LLM" – "Mục 2 Kiến trúc · Mục 3 Tokenization · Mục 4 Nguyên lý học · Mục 5 Giai đoạn huấn luyện · Mục 6 Sinh văn bản"
  "2. DÙNG LLM" – "Mục 1 Giới thiệu · Mục 7 Zero-shot vs few-shot · Mục 8 Prompting, fine-tuning, RAG"
  "3. ĐÁNH GIÁ" – "Mục 9 Hạn chế và kiểm chứng · Mục 10 Thử nghiệm có kiểm soát"
- Dưới trái: "Môn Học sâu – Trường Đại học Giao thông vận tải TP.HCM (UTH)" / "Nhóm: Muội · Phúc · Toàn · Thư" / "GVHD: ……………"
- Dải TÓM TẮT FLOW, tên "Đường đi của bài thuyết trình": 
  bước 1 icon: dãy ô chữ → "Văn bản → token"; bước 2 icon: khối xếp chồng → "Transformer decoder-only"; bước 3 icon: biểu đồ cột → "Xác suất token tiếp theo"; bước 4 icon: mũi tên vòng → "Sinh từng token"; bước 5 icon: tài liệu + kính lúp → "Prompt · RAG"; bước 6 icon: bảng có dấu tích/dấu chéo → "Đánh giá có kiểm soát".
```

## Ảnh 02 — Mô hình, môi trường, thư viện

```text
Áp dụng QUY CHUẨN THIẾT KẾ.
Số slide "02". Tiêu đề: "MÔ HÌNH, MÔI TRƯỜNG VÀ THƯ VIỆN". Phụ đề: "Mọi số đọc bằng code từ model.config và env.json, không chép tay". Nhãn: "Phần 0 · Setup". Nguồn: "notebook phần 0 · run 20261119_150356_normal".

PANEL 1. MODEL CHÍNH
- Số lớn #E67E22: "1,543,714,304 tham số"
- "Qwen/Qwen2.5-1.5B-Instruct · Hugging Face · Apache-2.0"
- Bảng: "Kiến trúc | Transformer decoder-only" · "Số block | 28" · "hidden_size | 1,536" · "Head Q / KV | 12 / 2 (GQA)" · "FFN | SwiGLU, 8,960" · "Chuẩn hoá · vị trí | RMSNorm · RoPE (base 1,000,000)" · "Từ điển | 151,936" · "Ngữ cảnh tối đa | 32,768 token"

PANEL 2. 1.5 TỈ THAM SỐ NẰM Ở ĐÂU
- Sơ đồ: thanh ngang chia tỉ lệ 4 đoạn có nhãn: "FFN 1,156,055,040 (≈ 75%)" · "Embedding 233,373,696" · "Attention 154,198,016" · "Norm 87,552"
- Hộp công thức (3 dòng):
  "Embedding = V × d = 151,936 × 1,536"
  "FFN = 3 × d × 8,960 × 28"
  "Attention/block = (d·d + d) + 2·(d·256 + 256) + d·d"
- Chú giải: "V | kích thước từ điển" · "d | hidden_size = 1,536" · "256 | 2 head KV × 128"

PANEL 3. MODEL ĐÃ ĐƯỢC HUẤN LUYỆN THẾ NÀO
- Sơ đồ 3 hộp nối mũi tên: "Pretraining: 18 nghìn tỉ token" → "SFT: > 1 triệu mẫu" → "DPO ~150 nghìn cặp + GRPO"
- Gạch đầu dòng: "Nguồn: Qwen2.5 Technical Report (2024)" · "Model phụ: Qwen/Qwen2.5-1.5B (bản base, chỉ dùng ở mục 5)" · "Tokenizer phụ: GPT-2, 50,257 token (chỉ dùng ở mục 3)"

PANEL 4. MÔI TRƯỜNG CHẠY
- Bảng: "Nền tảng | Kaggle Notebook" · "GPU | Tesla T4, 14.6 GB (dùng GPU 0)" · "Kiểu số | float16" · "Python · CUDA | 3.13.16 · 13.0" · "Seed | 42" · "Thời gian chạy toàn notebook | 7.4 phút"
- Hộp ý nghĩa: "Chỉ suy luận (torch.no_grad()), không fine-tune. Mọi lần sinh để đánh giá dùng greedy vì bản Instruct mặc định lấy mẫu ngẫu nhiên (temperature 0.7, top_p 0.8, top_k 20)."

PANEL 5. THƯ VIỆN
- Bảng "thư viện | phiên bản | dùng để":
  "torch | 2.11.0+cu130 | tensor, GPU, softmax"
  "transformers | 5.20.2 | tải model, tokenizer, generate, chat template"
  "accelerate | 1.15.0 | device_map"
  "pandas + pyarrow | 2.3.3 + 23.0.1 | đọc Parquet, bảng, CSV"
  "scikit-learn | 1.6.1 | TF-IDF cho RAG"
  "statsmodels | 0.15.0 | kiểm định McNemar"
  "matplotlib | 3.10.0 | vẽ hình"

PANEL 6. DỮ LIỆU VÀ CODE TẢI MODEL
- Gạch đầu dòng: "UIT-VSFC: test 2,999 câu (1,590 tích cực / 1,409 tiêu cực) – mục 7, 10" · "UIT-ViQuAD 2.0: 3,814 câu hỏi, 557 đoạn Wikipedia – mục 8, 9" · "2 bộ prompt nhóm tự soạn – mục 1, 5"
- Khối code:
  model = AutoModelForCausalLM.from_pretrained(
      "Qwen/Qwen2.5-1.5B-Instruct", torch_dtype=torch.float16, device_map={"": 0})
  n_params = sum(p.numel() for p in model.parameters())   # 1,543,714,304

DẢI TÓM TẮT FLOW – "Quy trình chạy notebook":
icon: bánh răng → "Cấu hình SMOKE, SEED" · icon: mũi tên tải xuống → "Tải model + tokenizer" · icon: bảng dữ liệu → "Tải UIT-VSFC, UIT-ViQuAD" · icon: dãy 10 ô → "Chạy 10 phần" · icon: thư mục → "Ghi outputs/ + log"
```

---

## Ảnh 03 — Mục 1 · Lý thuyết

```text
Áp dụng QUY CHUẨN THIẾT KẾ.
Số "03". Tiêu đề: "LANGUAGE MODEL, LLM VÀ CHATBOT". Phụ đề: "Từ mô hình xác suất trên chuỗi từ đến ứng dụng hội thoại". Nhãn: "Mục 1 · Giới thiệu". Nguồn: "Jurafsky & Martin ch.3; Bengio 2003; Brown 2020; Zhao 2023".

PANEL 1. LANGUAGE MODEL LÀ GÌ
- Gạch: "Gán xác suất cho một chuỗi từ; tương đương đoán từ tiếp theo."
- Hộp công thức: P(x₁, x₂, …, xₙ) = ∏ᵢ₌₁ⁿ P(xᵢ | x₁, …, xᵢ₋₁)
- Chú giải: "xᵢ | từ / token thứ i" · "P(xᵢ | x₁…xᵢ₋₁) | xác suất từ tiếp theo khi biết các từ trước" · "∏ | tích (quy tắc dây chuyền)"

PANEL 2. CÁCH CŨ: N-GRAM
- Hộp công thức: P(wᵢ | wᵢ₋₁) ≈ count(wᵢ₋₁, wᵢ) / count(wᵢ₋₁)
- Gạch: "Chỉ nhìn n − 1 từ trước, ước lượng bằng đếm tần suất." · "Cụm chưa gặp trong dữ liệu → xác suất 0."
- Hộp ý nghĩa: "Bengio 2003 thay bảng đếm bằng mạng nơ-ron + embedding: neural language model."

PANEL 3. LLM LÀ GÌ
- Gạch: "Language model dựa trên Transformer." · "Hàng tỉ tham số trở lên (model nhóm dùng: 1.54 tỉ)." · "Huấn luyện trên lượng văn bản rất lớn (Qwen2.5: 18 nghìn tỉ token)." · "Đủ lớn → làm nhiều tác vụ chỉ bằng đổi prompt (GPT-3, Brown 2020)."

PANEL 4. CHATBOT LÀ ỨNG DỤNG, KHÔNG PHẢI MODEL
- Sơ đồ: 4 hộp chồng lên nhau, có dấu "+": "LLM đã post-training (bản Instruct)" + "Giao diện hội thoại" + "Quản lý lịch sử hội thoại" + "Lớp an toàn / kiểm duyệt" = "Chatbot (vd ChatGPT)"

PANEL 5. BẢNG PHÂN BIỆT
- Bảng 4 cột: " | Language model | LLM | Chatbot"
  "Là gì | Model xác suất trên chuỗi | LM Transformer cỡ lớn | Ứng dụng xây trên LLM"
  "Ví dụ | n-gram, RNN LM | Qwen2.5-1.5B, GPT-3 | ChatGPT, Qwen Chat"
  "Làm được | Đoán từ tiếp theo | Nhiều tác vụ qua prompt | Hội thoại nhiều lượt"

PANEL 6. VÌ SAO MỘT MODEL LÀM NHIỀU TÁC VỤ
- Sơ đồ: 4 hộp nhỏ "Hỏi đáp", "Tóm tắt", "Dịch", "Sinh mã" cùng mũi tên vào 1 hộp "văn bản vào → sinh tiếp văn bản ra"
- Hộp ý nghĩa: "Mọi tác vụ đều là sinh tiếp văn bản có điều kiện theo prompt. Bản Instruct còn học chat template: khung hội thoại để biết đến lượt mình trả lời."

DẢI TÓM TẮT FLOW – "Từ LM đến chatbot":
icon: bảng đếm → "n-gram: đếm tần suất" · icon: mạng nút → "Neural LM" · icon: khối xếp chồng → "Transformer, hàng tỉ tham số = LLM" · icon: hộp thoại có dấu tích → "Post-training: làm theo chỉ dẫn" · icon: khung chat → "Chatbot = LLM + giao diện + an toàn"
```

## Ảnh 04 — Mục 1 · Demo

```text
Áp dụng QUY CHUẨN THIẾT KẾ.
Số "04". Tiêu đề: "MỘT MODEL, BỐN TÁC VỤ – CHỈ ĐỔI PROMPT". Phụ đề: "Qwen2.5-1.5B-Instruct, greedy, chat template; chấm theo luật viết trước khi chạy". Nhãn: "Mục 1 · Demo". Nguồn: "notebook phần 1 · run 20261119_150356_normal".

PANEL 1. HỎI ĐÁP
- Hộp "Prompt": "Thủ đô của Nhật Bản là gì? Trả lời một câu."
- Hộp "Output" (viền xanh lá): "Thủ đô của Nhật Bản là Tokyo." · "10 token · đạt (có 'Tokyo')"

PANEL 2. TÓM TẮT
- Hộp "Prompt": "[Đoạn 5 câu về môn Học sâu] Tóm tắt đoạn trên trong 1 câu."
- Hộp "Output": "Học sâu là một lĩnh vực học máy sử dụng mạng nơ-ron nhiều lớp để tự học biểu diễn từ dữ liệu, bao gồm…" · "1 câu, 104 token · đạt (chấm tay): giữ đủ ý nhưng rút gọn ít"

PANEL 3. DỊCH
- Hộp "Prompt": "Dịch sang tiếng Anh: Hôm nay trời rất đẹp."
- Hộp "Output": "Today the weather is very beautiful." · "8 token · đạt"

PANEL 4. SINH MÃ VÀ CHẠY THẬT
- Hộp "Prompt": "Viết hàm Python is_even(n) trả về True nếu n là số chẵn."
- Khối code (output của model):
  def is_even(n):
      return n % 2 == 0
- Bảng chạy thật: "n | 0 | 1 | 2 | −3" / "is_even(n) | True | False | True | False" · "đạt"

PANEL 5. CODE SINH VĂN BẢN
- Khối code:
  messages = [{"role": "user", "content": prompt}]
  enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                return_tensors="pt", return_dict=True)
  out = model.generate(**enc, max_new_tokens=..., do_sample=False,
                       temperature=None, top_p=None, top_k=None)   # greedy
  text = tok.decode(out[0, n_in:], skip_special_tokens=True)       # bỏ phần prompt
- Chú giải: "apply_chat_template | bọc prompt vào khung hội thoại, thêm system prompt" · "do_sample=False | greedy: luôn chọn token xác suất cao nhất" · "out[0, n_in:] | bỏ phần prompt, giữ phần model sinh"

PANEL 6. KẾT QUẢ VÀ HẠN CHẾ
- Số lớn #27AE60: "4/4 đạt" · dòng nhỏ: "0/4 bị cắt ở max_new_tokens"
- Hộp viền đỏ: "Prompt 'Xin chào' (32 token sau chat template) → '你好！有什么我可以帮助你的吗？' – trả lời bằng tiếng Trung."
- Hộp ý nghĩa: "Cùng trọng số, chỉ đổi prompt là đổi tác vụ. Chen tiếng Trung là hạn chế phụ thuộc dữ liệu huấn luyện (mục 9)."

DẢI TÓM TẮT FLOW – "Một lần gọi model":
icon: ô văn bản → "Prompt người dùng" · icon: khung có nhãn system/user → "Chat template" · icon: biểu đồ cột với cột cao nhất tô đậm → "Greedy từng token" · icon: kéo cắt → "Bỏ phần prompt" · icon: bảng có dấu tích → "Chấm theo luật: 4/4"
```

---

## Ảnh 05 — Mục 2 · Self-attention và causal mask

```text
Áp dụng QUY CHUẨN THIẾT KẾ.
Số "05". Tiêu đề: "SELF-ATTENTION VÀ CAUSAL MASK". Phụ đề: "Cơ chế giúp mỗi token lấy thông tin từ các token trước nó". Nhãn: "Mục 2 · Kiến trúc". Nguồn: "Vaswani et al. 2017 §3.2; Ainslie et al. 2023".

PANEL 1. VÌ SAO TRANSFORMER THAY RNN
- Bảng 2 cột: "RNN / LSTM | Transformer"
  "Đọc tuần tự từng token | Mỗi token nhìn trực tiếp mọi token được phép"
  "Token thứ 100 chờ 99 bước | 1 bước, tính song song trên GPU"
  "Thông tin xa bị phai | Khoảng cách không làm yếu kết nối"

PANEL 2. CÔNG THỨC SELF-ATTENTION
- Hộp công thức lớn: Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k ) · V
- Hộp công thức nhỏ: Q = X·W_Q,  K = X·W_K,  V = X·W_V
- Chú giải: "Q (query) | token này đang tìm gì" · "K (key) | token này có gì để được tìm" · "V (value) | nội dung token đóng góp" · "d_k | số chiều của key (Qwen: 128)"

PANEL 3. TỪNG BƯỚC CỦA CÔNG THỨC
- Bảng "bước | ý nghĩa":
  "Q·Kᵀ | ma trận điểm n × n: token i liên quan token j bao nhiêu"
  "÷ √d_k | tích vô hướng có phương sai d_k; chia để phương sai ≈ 1, tránh softmax bão hoà"
  "softmax theo hàng | trọng số dương, mỗi hàng cộng = 1"
  "· V | output = trung bình có trọng số các vector value"

PANEL 4. CAUSAL MASK
- Hộp công thức: Attention = softmax( Q·Kᵀ / √d_k + M ) · V
  M_ij = 0 nếu j ≤ i ;  M_ij = −∞ nếu j > i
- Sơ đồ: ma trận 5×5 ô vuông, tam giác dưới (gồm đường chéo) tô xanh ghi "0", tam giác trên tô xám ghi "−∞"; trục dọc "token đang xét (query)", trục ngang "token được nhìn (key)"
- Gạch: "e^(−∞) = 0 → trọng số tương lai đúng bằng 0." · "Đặt điểm = 0 thì e⁰ = 1, vẫn nhìn trộm tương lai."

PANEL 5. VÌ SAO CẦN CAUSAL MASK
- Gạch đánh số:
  "1. Huấn luyện đưa cả câu vào một lần; không mask thì vị trí 'trời' thấy luôn 'rất' → chép đáp án."
  "2. Khi sinh, token tương lai chưa tồn tại → mask làm lúc train giống lúc dùng."
  "3. Token cũ không phụ thuộc token mới → lưu lại K, V đã tính (KV cache)."

PANEL 6. MULTI-HEAD VÀ GQA
- Gạch: "Multi-head: nhiều head chạy song song, mỗi head một bộ W_Q, W_K, W_V; nối lại rồi nhân W_O." · "Bài gốc: d_model = 512, h = 8 head."
- Sơ đồ: 12 ô nhỏ "Q" chia 2 nhóm 6 ô, mỗi nhóm nối vào 1 cặp ô "K, V"
- Hộp ý nghĩa: "GQA (Qwen2.5-1.5B): 12 head Q dùng chung 2 head K/V, mỗi 6 head Q chung 1 cặp K, V → KV cache nhỏ hơn 6 lần."

DẢI TÓM TẮT FLOW – "Một lớp attention":
icon: 3 mũi tên tách từ 1 hộp → "Chiếu X thành Q, K, V" · icon: lưới ma trận → "Điểm Q·Kᵀ / √d_k" · icon: ma trận tam giác → "Cộng causal mask M" · icon: thanh chia tỉ lệ → "softmax theo hàng" · icon: các mũi tên hội tụ → "Trung bình có trọng số của V"
```

## Ảnh 06 — Mục 2 · Kiến trúc Qwen2.5 từ token đến xác suất

```text
Áp dụng QUY CHUẨN THIẾT KẾ. Ngoại lệ: PANEL 1 gộp cả 3 cột hàng trên (rộng toàn trang); hàng dưới vẫn 3 panel (đánh số 2, 3, 4) — tổng 4 panel.
Số "06". Tiêu đề: "TỪ TOKEN ĐẦU VÀO ĐẾN XÁC SUẤT TOKEN TIẾP THEO". Phụ đề: "Kiến trúc decoder-only của Qwen2.5-1.5B – số lấy từ model.config". Nhãn: "Mục 2 · Kiến trúc". Nguồn: "Qwen2.5 Technical Report §2; Su 2021; Zhang & Sennrich 2019; Shazeer 2020".

PANEL 1. SƠ ĐỒ LUỒNG (minh họa của đề) – rộng toàn trang
- Chuỗi hộp nối mũi tên từ trái sang phải, dưới mỗi hộp ghi shape tensor:
  "Token ID" (1, n) → "Embedding 151,936 × 1,536" (1, n, 1536) → [hộp lớn viền đậm "Decoder block × 28", bên trong 2 hộp dọc: "RMSNorm → Causal self-attention (12 head Q / 2 head KV, RoPE) → + residual" và "RMSNorm → FFN SwiGLU 1,536 → 8,960 → 1,536 → + residual"] (1, n, 1536) → "RMSNorm cuối" → "LM head (dùng chung ma trận với Embedding)" (1, n, 151,936) → "softmax tại vị trí cuối" → "P(token tiếp theo)"
- Dòng chú thích dưới: "Ví dụ 'Hôm nay trời rất' = 6 token: (1, 6) → (1, 6, 1536) → 28 block giữ nguyên shape → logits (1, 6, 151,936) → lấy vị trí cuối → 151,936 xác suất cộng = 1"

PANEL 2. RoPE VÀ RMSNorm
- Hộp công thức: RoPE: xoay mỗi cặp chiều của q, k một góc m·θᵢ,  θᵢ = base^(−2(i−1)/d)
- Chú giải: "m | vị trí token" · "base | 10,000 (bài gốc); 1,000,000 (Qwen2.5)" · "kết quả | q_m·k_n chỉ phụ thuộc khoảng cách giữa hai vị trí"
- Hộp công thức: RMSNorm(x) = x / √( (1/d)·Σᵢ xᵢ² + ε ) ⊙ g
- Gạch: "Như LayerNorm nhưng không trừ trung bình → rẻ hơn."

PANEL 3. RESIDUAL VÀ FFN SwiGLU
- Hộp công thức: h = x + Attn(RMSNorm(x)) ;  y = h + FFN(RMSNorm(h))
- Hộp công thức: FFN(x) = W_down( SiLU(W_gate·x) ⊙ W_up·x ),  SiLU(x) = x·σ(x)
- Chú giải: "⊙ | nhân từng phần tử" · "σ | hàm sigmoid" · "W_gate, W_up, W_down | 3 ma trận của FFN"
- Gạch: "Residual giống ResNet: gradient đi thẳng qua 28 block."

PANEL 4. LM HEAD VÀ ĐẦU RA
- Hộp công thức: logits = h_cuối · Eᵀ ;  P(token) = softmax(logits)
- Chú giải: "E | ma trận embedding 151,936 × 1,536 (dùng chung)" · "h_cuối | vector 1,536 chiều ở vị trí cuối"
- Hộp ý nghĩa: "Đầu ra của LLM không phải một từ mà là phân phối xác suất trên cả 151,936 token. Dùng chung E bớt 233 triệu tham số."

DẢI TÓM TẮT FLOW – "Đường đi của dữ liệu":
icon: dãy ô số → "Token ID" · icon: bảng tra hàng → "Embedding: tra hàng" · icon: 28 khối xếp chồng → "28 decoder block" · icon: ma trận nhân vector → "LM head: h·Eᵀ" · icon: biểu đồ cột xác suất → "softmax → P(token tiếp)"
```

## Ảnh 07 — Mục 2 · Demo trên model thật

```text
Áp dụng QUY CHUẨN THIẾT KẾ. Đính kèm 2 ảnh thật: causal_mask.png, attention_qwen.png (trong outputs/figures/).
Số "07". Tiêu đề: "CAUSAL ATTENTION TRONG MODEL THẬT". Phụ đề: "Câu 'Hôm nay trời rất đẹp.' = 8 token: H | ôm | n | ay | trời | rất | đẹp | ." Nhãn: "Mục 2 · Demo". Nguồn: "notebook phần 2 (cell 2.1–2.4) · run 20261119_150356_normal".

PANEL 1. CAUSAL MASK (QUY TẮC)
- Đặt ảnh thật causal_mask.png.
- Gạch: "1 = được nhìn (chính nó và token trước); 0 = tương lai bị chặn." · "Số ô 1 = n(n + 1)/2 = 36."

PANEL 2. TRỌNG SỐ ATTENTION THẬT (LAYER 0, HEAD 0)
- Đặt ảnh thật attention_qwen.png.
- Gạch: "Phía trên đường chéo = 0.000000" · "Mỗi hàng cộng = 1.0"

PANEL 3. ĐỌC HÌNH ATTENTION
- Bảng "quan sát | vì sao":
  "Tam giác trên = 0 | causal mask có tác dụng trong model thật"
  "Hàng 'H' = 1.00 | chỉ 1 ô được phép, softmax 1 phần tử luôn = 1"
  "Đường chéo 0.38 → 0.76 cuối câu | head này giữ nhiều thông tin của chính token"
  "'đẹp' nhìn 'rất' 0.23 | layer thấp hay nhìn token liền trước"
- Hộp viền đỏ: "Chỉ là 1 trong 28 × 12 = 336 head; không dùng 1 head để giải thích model."

PANEL 4. TOP-5 TOKEN TIẾP THEO
- Biểu đồ cột ngang "Sau 'Hôm nay trời rất' (6 token)": "đẹp 0.788" (cột cam dài nhất) · "nắng 0.063" · "lạnh 0.046" · "nóng 0.029" · "(dấu cách) 0.007"
- Dòng: "Tổng top-5 = 0.933; 0.067 còn lại chia cho 151,931 token khác."

PANEL 5. CODE
- Khối code:
  logits = model(ids).logits[0, -1]          # vị trí cuối: 151,936 điểm
  probs  = torch.softmax(logits, dim=-1)     # thành xác suất, cộng = 1
  top    = torch.topk(probs, 5)
  attn = out_eager.attentions[0][0, 0]       # layer 0, batch 0, head 0
- Chú giải: "attn_implementation='eager' | bản tính attention trả được trọng số (SDPA mặc định không trả)"

PANEL 6. SỐ THAM SỐ ĐẾM BẰNG CODE
- Bảng: "Embedding | 233,373,696" · "Attention | 154,198,016" · "FFN | 1,156,055,040" · "Norm | 87,552" · "Tổng | 1,543,714,304"
- Hộp ý nghĩa: "Cả 4 nhóm khớp công thức tính tay từ config. FFN ≈ 75% tham số. Bốn ứng viên top-5 đều là tính từ thời tiết: model học cả ngữ pháp lẫn ngữ nghĩa."

DẢI TÓM TẮT FLOW – "Kiểm chứng trên model thật":
icon: ma trận tam giác → "Vẽ causal mask" · icon: ma trận màu nhiệt → "Lấy attention thật (eager)" · icon: dấu bằng 0 → "Tương lai = 0.000000" · icon: biểu đồ cột → "Top-5: 'đẹp' 0.788" · icon: bảng số → "Đếm tham số khớp config"
```

---

## Ảnh 08 — Mục 3 · Lý thuyết tokenization

```text
Áp dụng QUY CHUẨN THIẾT KẾ.
Số "08". Tiêu đề: "TOKENIZATION: VĂN BẢN → TOKEN → TOKEN ID → EMBEDDING". Phụ đề: "Byte-level BPE và bảng embedding của Qwen2.5". Nhãn: "Mục 3 · Tokenization". Nguồn: "Sennrich 2016; HF Summary of the tokenizers; Qwen2.5 Technical Report §2".

PANEL 1. BA CÁCH CHIA VĂN BẢN
- Bảng "cách | ưu | nhược":
  "Theo từ | dễ hiểu | từ điển khổng lồ, từ lạ thành [UNK]"
  "Theo ký tự | từ điển nhỏ | chuỗi rất dài, mỗi ký tự ít nghĩa"
  "Subword – BPE | từ phổ biến = 1 token, từ hiếm = vài mảnh | cắt theo thống kê, không theo ngữ pháp" (dòng này tô nhấn)

PANEL 2. HUẤN LUYỆN TOKENIZER BPE
- Gạch đánh số: "1. Bắt đầu: mỗi ký tự / byte là một token." · "2. Đếm mọi cặp token liền nhau; gộp cặp xuất hiện nhiều nhất thành token mới." · "3. Lặp đến khi đủ kích thước từ điển."
- Sơ đồ ví dụ 3 hàng: "t · r · ờ · i" → "tr · ờ · i" → "trời"

PANEL 3. BYTE-LEVEL BPE
- Gạch: "Từ điển gốc = 256 giá trị byte → mọi chuỗi đều tokenize được, không có [UNK]." · "GPT-2: 50,257 = 256 byte + 1 token kết thúc + 50,000 lần gộp."
- Bảng UTF-8: "chữ | byte" · "ô | C3 B4 (2 byte)" · "ờ | E1 BB 9D (3 byte)" · "đ | C4 91 (2 byte)" · "ẹ | E1 BA B9 (3 byte)"

PANEL 4. TOKEN ID VÀ TỪ ĐIỂN QWEN
- Bảng: "tok.vocab_size | 151,643 | token thường" · "len(tok) | 151,665 | + 22 token điều khiển (<|im_start|>, <|im_end|>…)" · "model.config.vocab_size | 151,936 | số hàng bảng embedding"
- Gạch: "Token ID = số thứ tự của token trong từ điển (vd ' đẹp' = 128501)."

PANEL 5. EMBEDDING
- Hộp công thức: e_t = E[t, :] = onehot(t)ᵀ · E ,  E ∈ ℝ^(151,936 × 1,536)
- Chú giải: "t | token ID" · "E | bảng embedding học được" · "e_t | vector 1,536 số của token t"
- Gạch: "Lấy embedding = tra hàng thứ t (như nn.Embedding ở Ch6)."

PANEL 6. VÌ SAO SỐ TOKEN QUAN TRỌNG
- Gạch: "Ngữ cảnh tối đa tính bằng token (32,768) → ít token chứa được nhiều nội dung hơn." · "Mỗi token là một bước sinh → ít token sinh nhanh hơn." · "API tính tiền theo token."
- Hộp ý nghĩa: "Tokenizer học từ dữ liệu ngôn ngữ nào thì cắt ngôn ngữ đó gọn hơn."

DẢI TÓM TẮT FLOW – "Từ chữ đến vector":
icon: dòng chữ → "Câu văn bản" · icon: dãy byte 0/1 → "Byte UTF-8" · icon: các ô ghép lại → "Gộp BPE thành token" · icon: dãy số → "Token ID" · icon: bảng có 1 hàng tô đậm → "Tra hàng → embedding 1,536 chiều"
```

## Ảnh 09 — Mục 3 · Demo tokenize câu Việt và câu Anh

```text
Áp dụng QUY CHUẨN THIẾT KẾ.
Số "09". Tiêu đề: "CÙNG MỘT CÂU, HAI TOKENIZER". Phụ đề: "Qwen2.5 (đa ngôn ngữ) so với GPT-2 (chủ yếu tiếng Anh)". Nhãn: "Mục 3 · Demo". Nguồn: "notebook phần 3 · run 20261119_150356_normal".

PANEL 1. QWEN – CÂU TIẾNG VIỆT (8 TOKEN)
- Dãy ô token, dưới mỗi ô là ID: "H 39 | ôm 94365 | ␣n 308 | ay 352 | ␣trời 130868 | ␣rất 128323 | ␣đẹp 128501 | . 13"
- Gạch: "'␣' = dấu cách nằm trong token."

PANEL 2. QWEN – CÂU TIẾNG ANH (7 TOKEN)
- Dãy ô: "The | ␣weather | ␣is | ␣very | ␣nice | ␣today | ."
- Gạch: "Mỗi từ tiếng Anh = 1 token."

PANEL 3. GPT-2 – CÂU TIẾNG VIỆT (22 TOKEN)
- Dãy 22 ô nhỏ (ô "�" tô đỏ nhạt): "H | ô | m | ␣n | ay | ␣tr | � | � | � | i | ␣r | � | � | � | t | ␣� | � | � | � | � | p | ."
- Gạch: "'�' = token chỉ chứa 1 byte của chữ có dấu, decode riêng không thành ký tự."

PANEL 4. SO SỐ TOKEN
- Biểu đồ cột đôi: "Câu Việt: GPT-2 22 · Qwen 8" ; "Câu Anh: GPT-2 7 · Qwen 7"
- Số lớn #E67E22: "GPT-2 tốn 2.75× cho câu Việt"
- Dòng: "Câu Việt: 21 ký tự = 29 byte UTF-8"

PANEL 5. CODE VÀ SHAPE
- Khối code:
  ids = tok("Hôm nay trời rất đẹp.", add_special_tokens=False)["input_ids"]   # 8 ID
  emb = model.get_input_embeddings()(torch.tensor([ids]))                      # (1, 8, 1536)
- Sơ đồ: "(1, 8)" → "(1, 8, 1536)"
- Gạch: "Embedding và LM head cùng địa chỉ bộ nhớ → dùng chung một ma trận." · "Decode ngược ra đúng câu gốc: 4/4."

PANEL 6. VÌ SAO RA NHƯ VẬY
- Bảng "quan sát | vì sao":
  "GPT-2 cắt 'ờ' thành 3 token | chưa từng gộp 3 byte E1 BB 9D (học chủ yếu tiếng Anh)"
  "GPT-2 gộp được 'ô' | cặp byte C3 B4 có trong văn bản Pháp, Bồ Đào Nha…"
  "Qwen có token ' trời', ' rất', ' đẹp' | dữ liệu đa ngôn ngữ, các từ này đủ phổ biến"
  "Qwen vẫn cắt 'Hôm' = 'H' + 'ôm' | BPE gộp theo tần suất, không theo từ điển tiếng Việt"

DẢI TÓM TẮT FLOW – "Kết quả":
icon: dòng chữ → "Cùng 1 câu Việt" · icon: dãy 22 ô nhỏ → "GPT-2: 22 token" · icon: dãy 8 ô → "Qwen: 8 token" · icon: bảng tra hàng → "Embedding (1, 8, 1536)" · icon: mũi tên vòng về chữ → "Decode đúng 4/4"
```

---

## Ảnh 10 — Mục 4 · Lý thuyết nguyên lý học

```text
Áp dụng QUY CHUẨN THIẾT KẾ.
Số "10". Tiêu đề: "NEXT-TOKEN PREDICTION VÀ CROSS-ENTROPY LOSS". Phụ đề: "LLM học bằng cách đoán token tiếp theo trên văn bản không cần gán nhãn". Nhãn: "Mục 4 · Nguyên lý học". Nguồn: "Slide Ch2; Brown 2020; Qwen2.5 Technical Report; HF Perplexity".

PANEL 1. BÀI TOÁN NEXT-TOKEN PREDICTION
- Gạch: "Cho ngữ cảnh x₁ … xᵢ, đoán xᵢ₊₁." · "Là bài toán phân loại với 151,936 lớp (softmax + cross-entropy như Ch2)."
- Sơ đồ: "[a, b, c, d] → ([a] → b), ([a, b] → c), ([a, b, c] → d)" · "câu n token → n − 1 bài toán"

PANEL 2. TỰ GIÁM SÁT VÀ DỮ LIỆU
- Gạch: "Nhãn là chính token kế tiếp trong văn bản – không cần người gán nhãn." · "Dùng được rất nhiều văn bản: web, sách, code."
- Bảng: "GPT-3 | 300 tỉ token" · "Qwen2.5 | 18 nghìn tỉ token"

PANEL 3. CROSS-ENTROPY LOSS
- Hộp công thức lớn: L = − (1 / (n − 1)) · Σᵢ₌₁ⁿ⁻¹ ln p_θ(xᵢ₊₁ | x₁, …, xᵢ)
- Chú giải: "p_θ | xác suất model (tham số θ) gán cho token đúng" · "n | số token của câu" · "ln | logarit tự nhiên"
- Bảng nhỏ: "p | 1 | 0.5 | 0.1 | 0.001" / "−ln p | 0 | 0.69 | 2.30 | 6.91"

PANEL 4. HUẤN LUYỆN LÀ GIẢM LOSS
- Hộp công thức: θ ← θ − η · ∇_θ L
- Chú giải: "η | learning rate" · "∇_θ L | gradient của loss theo tham số"
- Hộp ý nghĩa: "Giảm L = tăng xác suất của văn bản thật (maximum likelihood, theo quy tắc dây chuyền). Notebook KHÔNG train: chỉ tính L của model đã train."

PANEL 5. PERPLEXITY
- Hộp công thức: PPL = e^L
- Gạch: "Đọc: model phân vân như đang chọn đều giữa PPL token."
- Hộp công thức: Đoán bừa đều: p = 1/V ⇒ L = ln V = ln 151,936 = 11.93 ⇒ PPL = 151,936

PANEL 6. MỘT LẦN CHẠY CHO MỌI VỊ TRÍ
- Sơ đồ: dải trên "logits: vị trí 0 1 2 … n−2 (bỏ vị trí cuối)", dải dưới "nhãn: token 1 2 3 … n−1 (bỏ token đầu)", mũi tên chéo nối từng cặp, nhãn "dịch nhãn sang trái 1 vị trí"
- Hộp ý nghĩa: "Nhờ causal mask, logits[i] chỉ dựa vào token 0…i nên chính là dự đoán cho token i+1. Một lần chạy model cho cả n − 1 dự đoán."

DẢI TÓM TẮT FLOW – "Một bước học":
icon: dòng chữ → "Văn bản thô" · icon: các cặp ô ghép → "Tạo cặp ngữ cảnh → mục tiêu" · icon: biểu đồ cột → "Model cho P(token)" · icon: hàm −ln → "Loss = −ln P(token đúng)" · icon: mũi tên đi xuống dốc → "Gradient descent giảm loss"
```

## Ảnh 11 — Mục 4 · Demo loss từng token

```text
Áp dụng QUY CHUẨN THIẾT KẾ.
Số "11". Tiêu đề: "LOSS TỪNG TOKEN TRÊN CÂU THẬT". Phụ đề: "'Hôm nay trời rất đẹp.' = 8 token → 7 cặp ngữ cảnh → token mục tiêu". Nhãn: "Mục 4 · Demo". Nguồn: "notebook phần 4 · run 20261119_150356_normal".

PANEL 1. 7 CẶP VÀ LOSS (ví dụ minh họa) – gộp 2 hàng (cao gấp đôi)
- Bảng "ngữ cảnh | token đúng | P | loss = −ln P", thêm cột thanh ngang theo loss (đỏ = cao, xanh = thấp):
  "H | ôm | 0.0003 | 8.19"
  "Hôm | n | 0.705 | 0.35"
  "Hôm n | ay | 0.996 | 0.004"
  "Hôm nay | trời | 0.002 | 6.34"
  "Hôm nay trời | rất | 0.029 | 3.55"
  "Hôm nay trời rất | đẹp | 0.788 | 0.24"
  "Hôm nay trời rất đẹp | . | 0.109 | 2.22"
- Dòng tính: "L = (8.19 + 0.35 + 0.004 + 6.34 + 3.55 + 0.24 + 2.22) / 7 = 2.985"

PANEL 2. KHỚP THƯ VIỆN
- Bảng: "Loss tự tính | 2.984804" · "outputs.loss của transformers | 2.984804" · "Chênh lệch | 6.3e-08"
- Gạch: "Truyền labels = input_ids, thư viện tự dịch nhãn bên trong."

PANEL 3. PERPLEXITY
- Số lớn: "PPL = e^2.985 = 19.8"
- So sánh: "Đoán bừa: loss 11.931 · PPL 151,936"

PANEL 4. SO 3 CÂU
- Biểu đồ cột: "Đúng trật tự 2.985" · "Tiếng Anh 3.129" · "Xáo trộn 'đẹp rất trời nay Hôm.' 6.701" · "Đoán bừa 11.931"

PANEL 5. CODE
- Khối code:
  logits = model(ids).logits[0]
  log_probs = F.log_softmax(logits[:-1], dim=-1)   # bỏ vị trí cuối
  targets = ids[0, 1:]                             # bỏ token đầu (dịch nhãn trái 1)
  loss_token = -log_probs.gather(1, targets[:, None])[:, 0]
- Chú giải: "gather | ở hàng i lấy đúng cột của token mục tiêu" · "torch.no_grad() | chỉ tính, không train"

PANEL 6. VÌ SAO RA NHƯ VẬY
- Bảng "quan sát | vì sao":
  "'ôm' sau 'H' loss 8.19 | chỉ biết 'H' → hàng nghìn khả năng"
  "'ay' sau 'Hôm n' loss 0.004 | 'Hôm n…' gần như chắc chắn là 'Hôm nay'"
  "'trời' sau 'Hôm nay' loss 6.34 | nhiều đáp án hợp lý, không phải model kém"
  "Xáo trộn loss gấp đôi | model học trật tự từ, không chỉ từ vựng"

DẢI TÓM TẮT FLOW – "Tính loss trên 1 câu":
icon: dãy 8 ô → "8 token" · icon: các cặp ô → "7 cặp" · icon: biểu đồ cột → "1 lần chạy → 7 phân phối" · icon: hàm −ln → "Loss từng token" · icon: dấu bằng → "Trung bình 2.985 = thư viện"
```

---

## Ảnh 12 — Mục 5 · Ba giai đoạn huấn luyện

```text
Áp dụng QUY CHUẨN THIẾT KẾ. Ngoại lệ: PANEL 1 gộp cả 3 cột hàng trên; hàng dưới 3 panel (2, 3, 4).
Số "12". Tiêu đề: "BA GIAI ĐOẠN HUẤN LUYỆN: MỤC TIÊU VÀ DỮ LIỆU". Phụ đề: "Pretraining → SFT → Preference alignment (số liệu của Qwen2.5)". Nhãn: "Mục 5 · Giai đoạn huấn luyện". Nguồn: "Ouyang et al. 2022; Qwen2.5 Technical Report §3–4".

PANEL 1. SƠ ĐỒ 3 GIAI ĐOẠN (minh họa của đề) – rộng toàn trang
- 3 hộp lớn nối mũi tên, giữa các hộp là hộp nhỏ tên model; mỗi hộp lớn có 4 dòng có nhãn:
  "Văn bản thô" →
  HỘP "1. PRETRAINING": "Mục tiêu: học ngôn ngữ, kiến thức" / "Dữ liệu: văn bản không nhãn (web, sách, code)" / "Loss: cross-entropy mọi token" / "Qwen2.5: 18 nghìn tỉ token"
  → "Base model – biết viết tiếp văn bản" →
  HỘP "2. SFT": "Mục tiêu: làm theo chỉ dẫn, định dạng hội thoại" / "Dữ liệu: cặp (chỉ dẫn, câu trả lời mẫu)" / "Loss: cross-entropy chỉ trên phần trả lời" / "Qwen2.5: > 1 triệu mẫu, 2 epoch, dài 32,768 token"
  → "SFT model" →
  HỘP "3. PREFERENCE ALIGNMENT": "Mục tiêu: câu nào được ưa hơn (hữu ích, trung thực, vô hại)" / "Dữ liệu: cặp (câu tốt y_w, câu kém y_l)" / "Cách: RLHF hoặc DPO" / "Qwen2.5: DPO ~150 nghìn cặp + GRPO"
  → "Instruct model – trả lời, giữ định dạng, biết từ chối"

PANEL 2. LOSS CỦA SFT
- Hộp công thức: L_SFT = − Σ_(t ∈ trả lời) ln π_θ(y_t | x, y_<t)
- Chú giải: "x | prompt (chỉ dẫn)" · "y_t | token thứ t của câu trả lời mẫu" · "π_θ | model đang huấn luyện"
- Gạch: "Cùng cross-entropy như pretraining; chỉ khác dữ liệu và chỉ tính trên phần trả lời."

PANEL 3. VÌ SAO CẦN ALIGNMENT SAU SFT
- Gạch: "Một câu hỏi có nhiều câu trả lời đúng nhưng chất lượng khác nhau." · "Người SO SÁNH hai câu dễ hơn VIẾT một câu hoàn hảo." · "Dữ liệu là cặp: câu được chọn y_w, câu bị loại y_l."

PANEL 4. LIÊN HỆ CH4: FINE-TUNING
- Hộp ý nghĩa: "SFT và alignment đều là fine-tuning: bắt đầu từ model đã huấn luyện, cập nhật trọng số trên dữ liệu nhỏ, chuyên biệt hơn. Base và Instruct cùng kiến trúc, cùng số tham số – chỉ khác trọng số."

DẢI TÓM TẮT FLOW – "Vòng đời của một LLM":
icon: chồng tài liệu → "Văn bản thô" · icon: khối model trống → "Pretraining → Base" · icon: cặp hỏi–đáp → "SFT trên chỉ dẫn" · icon: hai câu trả lời có dấu tích/dấu chéo → "Học từ cặp ưu tiên" · icon: khung chat → "Instruct model"
```

## Ảnh 13 — Mục 5 · RLHF và DPO

```text
Áp dụng QUY CHUẨN THIẾT KẾ.
Số "13". Tiêu đề: "PREFERENCE ALIGNMENT: RLHF VÀ DPO". Phụ đề: "Hai cách dạy model câu trả lời nào được ưa hơn – ở mức ý tưởng". Nhãn: "Mục 5 · Giai đoạn huấn luyện". Nguồn: "Ouyang et al. 2022 §3; Rafailov et al. 2023 Eq. 2, 3, 7".

PANEL 1. RLHF – BƯỚC 1, 2: REWARD MODEL
- Gạch: "Người xếp hạng các câu trả lời cho cùng prompt."
- Hộp công thức: L_RM = − ln σ( r_φ(x, y_w) − r_φ(x, y_l) )
- Chú giải: "r_φ | reward model chấm điểm câu trả lời" · "y_w, y_l | câu được chọn, câu bị loại" · "σ | sigmoid"
- Gạch: "Câu tốt phải có điểm cao hơn câu kém."

PANEL 2. RLHF – BƯỚC 3: PPO + PHẠT KL
- Hộp công thức: max_θ  E[ r_φ(x, y) ] − β · KL( π_θ ‖ π_ref )
- Chú giải: "π_θ | model đang tối ưu" · "π_ref | model SFT ban đầu" · "β | mức phạt khi trôi xa π_ref"
- Gạch: "Phạt KL theo từng token để model không 'lách' reward model."

PANEL 3. INSTRUCTGPT (VÍ DỤ SỐ)
- Bảng: "Prompt SFT | ~13 nghìn" · "Prompt reward model | ~33 nghìn" · "Prompt PPO | ~31 nghìn"
- Hộp ý nghĩa: "Model InstructGPT 1.3 tỉ tham số được người dùng ưa hơn GPT-3 175 tỉ tham số."

PANEL 4. DPO
- Gạch: "Bỏ reward model riêng, bỏ học tăng cường; học thẳng từ cặp (y_w, y_l)."
- Hộp công thức lớn: L_DPO = − ln σ( β·ln[ π_θ(y_w|x) / π_ref(y_w|x) ] − β·ln[ π_θ(y_l|x) / π_ref(y_l|x) ] )
- Gạch: "Tăng xác suất câu tốt (so với model tham chiếu) nhiều hơn câu kém."

PANEL 5. SO SÁNH RLHF VÀ DPO
- Bảng " | RLHF (PPO) | DPO":
  "Reward model riêng | Có | Không (xác suất của model là reward ngầm)"
  "Học tăng cường | Có | Không"
  "Giữ gần model gốc | Phạt KL | Model tham chiếu + β"
  "Độ phức tạp | Cao | Thấp hơn, ổn định hơn"

PANEL 6. QWEN2.5-INSTRUCT DÙNG GÌ
- Gạch: "Offline: DPO ~150 nghìn cặp, 1 epoch." · "Online: GRPO – reward model chấm theo trung thực, hữu ích, ngắn gọn, liên quan, vô hại, giảm thiên lệch; lấy 8 câu trả lời mỗi truy vấn."
- Hộp ý nghĩa: "Model nhóm dùng đã qua cả 3 giai đoạn; khác biệt base / Instruct ở demo là tác dụng gộp của SFT + DPO + GRPO."

DẢI TÓM TẮT FLOW – "RLHF vs DPO":
icon: danh sách xếp hạng 1-2-3 → "Người xếp hạng" · icon: đồng hồ đo điểm → "Reward model r_φ" · icon: mũi tên vòng lặp → "PPO + phạt KL" · icon: cặp câu có dấu tích/dấu chéo → "DPO: học thẳng từ cặp" · icon: khung chat → "Model căn chỉnh"
```

## Ảnh 14 — Mục 5 · Demo base vs Instruct

```text
Áp dụng QUY CHUẨN THIẾT KẾ. Ngoại lệ: PANEL 1 gộp 2 cột trái hàng trên và hàng dưới (bảng lớn); bên phải 2 panel (2, 3) chồng dọc; tổng 3 panel.
Số "14". Tiêu đề: "CÙNG PROMPT: BẢN BASE VÀ BẢN INSTRUCT". Phụ đề: "Cùng kiến trúc, cùng 1,543,714,304 tham số, cùng từ điển – chỉ khác trọng số; cùng greedy, repetition_penalty 1.0". Nhãn: "Mục 5 · Demo". Nguồn: "notebook phần 5 · run 20261119_150356_normal".

PANEL 1. KẾT QUẢ 4 PROMPT (bảng lớn, cột BASE viền đỏ, cột INSTRUCT viền xanh lá)
| Prompt | BASE (văn bản thô) | INSTRUCT (chat template) |
| Thủ đô của Pháp là gì? | "- VnExpress / Thứ năm, 16/10/2019, 10:00 (GMT+7) / Thủ đô của Pháp là gì? / Thủ đô của Pháp là Paris…" – bị cắt 64/64 ✗ | "Thủ đô của Pháp là Paris." ✓ |
| Liệt kê 3 lợi ích của việc tập thể dục. | "- Trang 2 - Diễn đàn sức khỏe / Thảo luận trong 'Sức khỏe'…" – bị cắt 160/160 ✗ | "1. Tăng cường sức khỏe tổng thể… 2. Tăng cường sức đề kháng… 3. Tăng cường tinh thần…" ✓ |
| Trả lời đúng một từ: 2 + 2 bằng mấy? | "- Trang 2 - Giải toán Online | Đề thi Toán | L" – bị cắt 16/16 ✗ | "4" ✓ |
| Chỉ tôi cách quay cóp trong phòng thi… | "- Trang 2 - Giải trí… giám sát, giám sát, giám sát…" – bị cắt 128/128 ✗ | "Xin lỗi, nhưng tôi không thể giúp bạn với yêu cầu đó." ✓ |
Hai số lớn dưới bảng: "BASE 0/4" (đỏ) · "INSTRUCT 4/4" (xanh lá)

PANEL 2. CODE
- Khối code:
  enc_base = tok_base(p["prompt"], return_tensors="pt")                          # base: văn bản thô
  enc_inst = tok.apply_chat_template([{"role": "user", "content": p["prompt"]}],
                                     add_generation_prompt=True, return_tensors="pt", return_dict=True)
  for mdl, enc in [(model_base, enc_base), (model, enc_inst)]:                  # cùng tham số sinh
      out = mdl.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                         temperature=None, top_p=None, top_k=None, repetition_penalty=1.0)

PANEL 3. VÌ SAO RA NHƯ VẬY
- Bảng "quan sát | vì sao":
  "Base viết tiếp như trang báo | pretraining chỉ dạy viết tiếp văn bản web; câu hỏi trên web thường là tiêu đề"
  "Base có 'Paris' nhưng không đạt | kiến thức từ pretraining, hành vi trả lời từ post-training"
  "Base luôn bị cắt | chỉ dừng ở <|endoftext|>; Instruct dừng ở <|im_end|>"
  "Base lặp 'giám sát' | greedy không phạt lặp dễ rơi vào vòng lặp"
- Hộp viền đỏ: "Không tách riêng được tác dụng của SFT: Instruct đã qua SFT + DPO + GRPO và dùng chat template."

DẢI TÓM TẮT FLOW – "Thiết kế so sánh công bằng":
icon: hai khối giống hệt → "Cùng kiến trúc, cùng tham số" · icon: bánh răng → "Cùng greedy, rep. penalty 1.0" · icon: dòng chữ thô / khung chat → "Base: văn bản thô · Instruct: chat template" · icon: bảng luật → "Chấm theo luật viết trước" · icon: hai số 0/4 và 4/4 → "Base 0/4 · Instruct 4/4"
```

---

## Ảnh 15 — Mục 6 · Lý thuyết sinh văn bản

```text
Áp dụng QUY CHUẨN THIẾT KẾ.
Số "15". Tiêu đề: "SINH VĂN BẢN TỰ HỒI QUY VÀ CÁC CÁCH CHỌN TOKEN". Phụ đề: "Greedy, temperature, top-p: cùng một phân phối, khác cách chọn". Nhãn: "Mục 6 · Sinh văn bản". Nguồn: "Slide Ch6; Holtzman et al. 2020 §3.1; HF GenerationConfig".

PANEL 1. VÒNG LẶP TỰ HỒI QUY
- Sơ đồ vòng: "Prompt" → "Model" → "P(token tiếp theo)" → hình thoi "Chọn 1 token" → "Nối vào input" → mũi tên quay về "Model" (nhãn "chưa xong"); nhánh ra "Câu trả lời" (nhãn "gặp token kết thúc hoặc đủ max_new_tokens")
- Hộp công thức: y_t ~ P_θ( · | x, y₁, …, y_(t−1) ),  t = 1, 2, …

PANEL 2. GREEDY
- Hộp công thức: y_t = argmaxᵢ pᵢ
- Gạch: "Mỗi bước lấy token xác suất cao nhất." · "Tất định: chạy lại ra y hệt → dùng cho đánh giá." · "Dễ lặp, nhạt (Holtzman 2020)."

PANEL 3. TEMPERATURE
- Hộp công thức: pᵢ⁽ᵀ⁾ = e^(zᵢ/T) / Σⱼ e^(zⱼ/T)
- Chú giải: "zᵢ | logit của token i" · "T | nhiệt độ"
- Gạch: "T < 1: phân phối nhọn hơn (chắc chắn)." · "T > 1: phẳng hơn (liều, token hiếm có cơ hội)." · "T → 0 ≈ greedy; T = 1 = softmax thường."

PANEL 4. HỆ QUẢ CỦA TEMPERATURE
- Hộp công thức: pᵢ⁽ᵀ⁾ ∝ pᵢ^(1/T)  ⇒  p_a⁽ᵀ⁾ / p_b⁽ᵀ⁾ = (p_a / p_b)^(1/T)
- Hộp ý nghĩa: "Temperature không đổi thứ hạng token, chỉ đổi độ chênh: T = 0.5 bình phương tỉ lệ, T = 1.5 lấy căn bậc 1.5."

PANEL 5. TOP-P (NUCLEUS)
- Hộp công thức: V_p = tập nhỏ nhất sao cho Σ_(i ∈ V_p) pᵢ ≥ p
- Gạch: "Xếp token theo xác suất giảm dần, giữ V_p, chia lại xác suất, lấy mẫu trong V_p." · "Số ứng viên tự co giãn theo độ chắc chắn của model."
- Sơ đồ: biểu đồ cột giảm dần, vùng cột đầu tô xanh ghi "V_p (tổng ≥ p)", đuôi tô xám ghi "bị cắt"

PANEL 6. TOP-K VÀ CẤU HÌNH TRONG NOTEBOOK
- Gạch: "Top-k: giữ k token cao nhất (số ứng viên cố định)."
- Bảng "cấu hình | tham số": "greedy | do_sample=False" · "temp_0.5 / temp_1.5 | T = 0.5 / 1.5, top_p = 1.0" · "top_p_0.5 / top_p_0.95 | T = 1.0, p = 0.5 / 0.95" · "mọi cấu hình | top_k = 0 (tắt), repetition_penalty = 1.0"

DẢI TÓM TẮT FLOW – "Một bước sinh":
icon: khối model → "Chạy model" · icon: dãy số → "Logits ở vị trí cuối" · icon: biểu đồ cột → "Softmax (÷ T)" · icon: phễu lọc → "Cắt top-p / top-k" · icon: mũi tên nối vào dãy ô → "Chọn 1 token, nối vào input"
```

## Ảnh 16 — Mục 6 · Demo cùng prompt, nhiều cấu hình

```text
Áp dụng QUY CHUẨN THIẾT KẾ.
Số "16". Tiêu đề: "CÙNG MỘT PROMPT, CÁC CẤU HÌNH SINH KHÁC NHAU". Phụ đề: "Prompt: 'Viết một câu giới thiệu ngắn về thành phố Hà Nội.' (42 token sau chat template)". Nhãn: "Mục 6 · Demo". Nguồn: "notebook phần 6 · run 20261119_150356_normal".

PANEL 1. SINH TỪNG BƯỚC (GREEDY)
- Dãy 8 ô token, ô càng đậm xác suất càng cao, dưới ghi P:
  "H 0.85 | à 0.998 | Nội 0.9998 | là 0.67 | thành 0.46 | phố 0.9997 | lớn 0.60 | nhất 0.41"
- Gạch: "Input dài thêm 1 token mỗi bước: 42 → 49." · "Vòng lặp tự viết ra đúng các token của generate()."

PANEL 2. TEMPERATURE THAY ĐỔI PHÂN PHỐI (TOKEN ĐẦU TIÊN)
- Bảng "T | P('H') | số token phủ 90% | phủ 95%":
  "0.5 | 0.995 | 1 | 1"
  "1.0 | 0.847 | 3 | 7"
  "1.5 | 0.468 | 135 | 1,068"
- Biểu đồ cột nhỏ P('H') theo T.

PANEL 3. KIỂM CÔNG THỨC (ví dụ minh họa ngắn)
- Dòng tính: "Ở T = 1: p_H / p_Vi = 0.847 / 0.050 = 16.9"
- Dòng tính: "T = 0.5: 16.9² ≈ 287 → P('H') = 0.995"
- Dòng tính: "T = 1.5: 16.9^(1/1.5) ≈ 6.60 ; đo thật 0.468 / 0.071 = 6.59"
- Hộp ý nghĩa: "Số đo thật khớp công thức pᵢ⁽ᵀ⁾ ∝ pᵢ^(1/T)."

PANEL 4. 5 CẤU HÌNH × 3 LẦN (SEED 42, 43, 44)
- Bảng "cấu hình | số output khác nhau | quan sát":
  "greedy | 1 | 3 lần giống hệt ✓"
  "temperature 0.5 | 3 | khác nhau, mạch lạc ✓"
  "temperature 1.5 | 3 | vô nghĩa, lẫn nhiều ngôn ngữ: '…thángيران trung/sdk…' ✗"
  "top-p 0.5 | 3 | khác nhau, mạch lạc ✓"
  "top-p 0.95 | 3 | có lần bịa '…thành phố… ở Trung Quốc' ✗"

PANEL 5. CODE
- Khối code:
  for step in range(8):
      probs = torch.softmax(model(ids).logits[0, -1], dim=-1)
      next_id = probs.argmax()                                  # greedy
      ids = torch.cat([ids, next_id.view(1, 1)], dim=1)         # nối vào input
- Chú giải: "logits[0, -1] | điểm cho token tiếp theo" · "torch.cat | input dài thêm 1"

PANEL 6. VÌ SAO RA NHƯ VẬY
- Gạch: "T = 1.5: ở token đầu tiên nhóm 90% xác suất phình từ 3 lên 135 token → dễ chọn token lạ." · "Token lạ vào ngữ cảnh làm các bước sau lạc thêm (lỗi tích luỹ)." · "Greedy lặp lại được nhưng vẫn có lỗi sự thật: 'Đền Hùng' ở Hà Nội (thực tế ở Phú Thọ)."
- Hộp ý nghĩa: "Đánh giá dùng greedy để lặp lại được; lấy mẫu cho đa dạng nhưng có thể sai."

DẢI TÓM TẮT FLOW – "Kết quả":
icon: dãy ô → "8 bước greedy khớp generate()" · icon: biểu đồ nhọn và phẳng → "T đổi độ nhọn phân phối" · icon: phép tính → "Khớp công thức p^(1/T)" · icon: 3 trang giống nhau → "Greedy: 1 output" · icon: 3 trang khác nhau → "Lấy mẫu: 3 output"
```

---

## Ảnh 17 — Mục 7 · Lý thuyết khả năng của LLM

```text
Áp dụng QUY CHUẨN THIẾT KẾ.
Số "17". Tiêu đề: "IN-CONTEXT LEARNING VÀ LÀM THEO CHỈ DẪN". Phụ đề: "Học từ ví dụ trong prompt mà không cập nhật trọng số". Nhãn: "Mục 7 · Khả năng". Nguồn: "Brown 2020; Min et al. 2022; Zhao et al. 2021; Wei et al. 2021".

PANEL 1. IN-CONTEXT LEARNING LÀ GÌ
- Gạch: "Model làm tác vụ nhờ chỉ dẫn và/hoặc ví dụ đặt trong prompt." · "Trọng số không đổi, không có gradient (khác fine-tuning)."
- Hộp công thức: P( nhãn | chỉ dẫn, ví dụ₁, …, ví dụ_k, câu )

PANEL 2. ZERO-SHOT VÀ FEW-SHOT
- Hai khung prompt cạnh nhau:
  "ZERO-SHOT: Chỉ dẫn → Câu cần làm → Nhãn?"
  "FEW-SHOT: Chỉ dẫn → Ví dụ 1 + nhãn → … → Ví dụ k + nhãn → Câu cần làm → Nhãn?"

PANEL 3. VÍ DỤ GIÚP BẰNG CÁCH NÀO
- Gạch (Min et al. 2022): "Cho thấy tập nhãn." · "Cho thấy kiểu câu đầu vào." · "Cho thấy định dạng." · "Đổi nhãn ví dụ thành ngẫu nhiên chỉ làm giảm ít."

PANEL 4. RỦI RO KHI CHỌN VÍ DỤ
- Bảng "thiên lệch | nghĩa" (Zhao et al. 2021):
  "Majority label | thiên về nhãn chiếm đa số trong ví dụ"
  "Recency | thiên về nhãn của ví dụ cuối"
  "Common token | thiên về token phổ biến"
- Hộp ý nghĩa: "Nhóm dùng 2 tích cực + 2 tiêu cực, xếp xen kẽ."

PANEL 5. LÀM THEO CHỈ DẪN
- Gạch: "Có được nhờ SFT trên dữ liệu chỉ dẫn (FLAN, InstructGPT – mục 5)." · "Một model xử lý nhiều tác vụ ngôn ngữ: hỏi đáp, tóm tắt, dịch, phân loại, sinh mã (mục 1)."

PANEL 6. THIẾT KẾ THÍ NGHIỆM: CHỈ ĐỔI MỘT BIẾN
- Bảng "giữ giống nhau | thay đổi":
  "cùng model · cùng mẫu prompt · greedy · max_new_tokens 10 · cùng 2,999 câu test, cùng thứ tự | có / không 4 ví dụ"
- Gạch: "Chống rò rỉ: ví dụ lấy từ train, không trùng câu test." · "Chọn mẫu prompt trên 200 câu validation rồi mới chạy test."

DẢI TÓM TẮT FLOW – "Thí nghiệm zero vs few-shot":
icon: bảng dữ liệu → "UIT-VSFC 2 nhãn" · icon: tập nhỏ tách ra → "Chọn prompt trên validation" · icon: 4 thẻ → "4 ví dụ từ train" · icon: hai khung cạnh nhau → "Chạy zero và few trên cùng câu" · icon: thước đo → "So accuracy (mục 10)"
```

## Ảnh 18 — Mục 7 · Demo zero-shot vs few-shot

```text
Áp dụng QUY CHUẨN THIẾT KẾ.
Số "18". Tiêu đề: "ZERO-SHOT VS FEW-SHOT TRÊN CÙNG TÁC VỤ". Phụ đề: "Phân loại cảm xúc phản hồi sinh viên UIT-VSFC – 2,999 câu test (1,590 tích cực / 1,409 tiêu cực)". Nhãn: "Mục 7 · Demo". Nguồn: "notebook phần 7 · run 20261119_150356_normal".

PANEL 1. PROMPT ZERO-SHOT
- Khối monospace: "Phân loại cảm xúc của câu phản hồi sinh viên sau. Chỉ trả lời: tích cực hoặc tiêu cực." / "Câu: "…"" / "Cảm xúc:"

PANEL 2. PROMPT FEW-SHOT (THÊM 4 VÍ DỤ TỪ TRAIN)
- Khối monospace:
  "Câu: "dạy tốt có tâm ." → Cảm xúc: tích cực"
  "Câu: "oop khá phức tạp nên cần nhiều ví dụ ." → Cảm xúc: tiêu cực"
  "Câu: "giáo viên tích cực , năng nổ trong giảng dạy , …" → Cảm xúc: tích cực"
  "Câu: "giảng viên cần linh hoạt trong giảng dạy , …" → Cảm xúc: tiêu cực"
  "Câu: "…" → Cảm xúc:"

PANEL 3. KẾT QUẢ TRÊN TEST
- Hai số lớn: "Zero-shot 89.86%" → "Few-shot 92.30%"
- Dòng: "0 câu không đọc được nhãn ở cả hai cách"

PANEL 4. CHỌN MẪU PROMPT TRÊN 200 CÂU VALIDATION
- Bảng "mẫu | zero | few | trung bình":
  "vi_nhan_en | 0.840 | 0.905 | 0.8725"
  "vi_nhan_vi (chọn) | 0.845 | 0.905 | 0.875"
  "en_nhan_en | 0.825 | 0.870 | 0.8475"

PANEL 5. CODE
- Khối code:
  def build_prompt(text, examples, tpl):          # examples = [] → zero-shot
      blocks = [tpl["instruction"]]
      blocks += [tpl["item"].format(text=e["sentence"]) + " " + tpl["labels"][e["label"]] for e in examples]
      return "\n\n".join(blocks + [tpl["item"].format(text=text)])
- Chú giải: "padding_side = 'left' | chạy 16 câu một lô, đệm trái để model sinh ngay sau prompt"

PANEL 6. VÌ SAO FEW-SHOT TỐT HƠN
- Gạch: "Hai ví dụ tiêu cực đều là câu góp ý ('nên…', 'cần…')." · "UIT-VSFC quy ước góp ý = tiêu cực; model đọc giọng lịch sự nên hay đoán tích cực." · "Ví dụ cho thấy quy ước gán nhãn, không dạy kiến thức mới." · "Model chép định dạng 'Cảm xúc: tiêu cực' của ví dụ."
- Hộp ý nghĩa: "Có ý nghĩa thống kê hay không: kiểm định McNemar ở mục 10."

DẢI TÓM TẮT FLOW – "Kết quả":
icon: khung prompt trống → "Zero-shot 89.86%" · icon: khung có 4 thẻ → "Few-shot 92.30%" · icon: tập validation → "Prompt chọn trên validation" · icon: thẻ có chữ 'nên/cần' → "Ví dụ dạy quy ước nhãn" · icon: cân → "Kiểm định ở mục 10"
```

---

## Ảnh 19 — Mục 8 · Prompting, fine-tuning, RAG

```text
Áp dụng QUY CHUẨN THIẾT KẾ. Ngoại lệ: PANEL 1 gộp cả 3 cột hàng trên; hàng dưới 3 panel (2, 3, 4).
Số "19". Tiêu đề: "BA CÁCH THÍCH NGHI VÀ BỔ SUNG TRI THỨC". Phụ đề: "Prompting · Fine-tuning · Retrieval-Augmented Generation (RAG)". Nhãn: "Mục 8 · Thích nghi". Nguồn: "Lewis et al. 2020; Hu et al. 2021 §4.1; scikit-learn TfidfVectorizer".

PANEL 1. BẢNG SO SÁNH (minh họa của đề) – rộng toàn trang
| Cách | Đổi gì | Dữ liệu cần có | Chi phí | Khi nào dùng |
| Prompting (chỉ dẫn, zero/few-shot) | Chỉ đổi input | Vài ví dụ hoặc không | Rẻ nhất, tức thì | Tác vụ model đã biết làm |
| Fine-tuning (full hoặc LoRA) | Trọng số model | Tập (input, output) có nhãn + GPU để train | Cao | Đổi hành vi, văn phong, định dạng lâu dài; tác vụ chuyên biệt |
| RAG | Thêm tài liệu vào prompt lúc hỏi; trọng số không đổi | Kho tài liệu + bộ tìm kiếm | Trung bình | Tri thức mới, riêng, hay thay đổi; cần dẫn nguồn |

PANEL 2. LoRA (Ý TƯỞNG)
- Hộp công thức: W₀ + ΔW = W₀ + B·A ,  B ∈ ℝ^(d×r), A ∈ ℝ^(r×k), r ≪ min(d, k)
- Chú giải: "W₀ | trọng số gốc, giữ nguyên" · "B, A | 2 ma trận hạng thấp được train" · "r | hạng (rank)"
- Gạch: "Số tham số cần train r(d + k) thay vì d·k."

PANEL 3. RAG = RETRIEVAL + GENERATION
- Sơ đồ: "Câu hỏi" → "Bộ tìm kiếm: top-k đoạn từ kho" → "Prompt = chỉ dẫn + k đoạn + câu hỏi" → "LLM" → "Trả lời có nguồn"
- Gạch: "Không đổi trọng số; tri thức đưa vào lúc hỏi." · "Chỉ tốt bằng bộ tìm kiếm."

PANEL 4. TF-IDF VÀ RECALL@K
- Hộp công thức: tfidf(t, d) = tf(t, d) × [ ln( (1 + N) / (1 + df(t)) ) + 1 ]
- Chú giải: "tf(t, d) | số lần t có trong đoạn d" · "N | số đoạn trong kho" · "df(t) | số đoạn chứa t"
- Hộp công thức: cos(q, d) = q·d / (‖q‖·‖d‖) = q·d (vector đã chuẩn hoá L2)
- Gạch: "recall@k = tỉ lệ câu hỏi có đoạn gốc trong k đoạn tìm được."

DẢI TÓM TẮT FLOW – "Chọn cách nào":
icon: ô văn bản → "Model đã biết làm → Prompting" · icon: bánh răng trên khối model → "Cần đổi hành vi lâu dài → Fine-tuning" · icon: tài liệu + kính lúp → "Cần tri thức mới, dẫn nguồn → RAG" · icon: bảng số → "TF-IDF + cosine" · icon: hộp chữ có trích nguồn → "Trả lời dựa trên tài liệu"
```

## Ảnh 20 — Mục 8 · Demo không RAG vs có RAG

```text
Áp dụng QUY CHUẨN THIẾT KẾ.
Số "20". Tiêu đề: "CÙNG CÂU HỎI: KHÔNG RAG VÀ CÓ RAG". Phụ đề: "UIT-ViQuAD 2.0 (Wikipedia tiếng Việt): 15 câu có đáp án + 3 câu tài liệu không có đáp án; kho 557 đoạn". Nhãn: "Mục 8 · Demo". Nguồn: "notebook phần 8 · run 20261119_150356_normal".

PANEL 1. THIẾT KẾ
- Sơ đồ 2 nhánh: "Câu hỏi" → nhánh trên "TF-IDF top-3 / 557 đoạn" → "Prompt + 3 đoạn" → "Qwen2.5-1.5B greedy"; nhánh dưới "Prompt không tài liệu" → "Qwen2.5-1.5B greedy"
- Hộp ý nghĩa: "Hai nhánh cùng câu dặn: 'Nếu không biết hoặc tài liệu không có thông tin, trả lời Không có thông tin'. Chỉ khác có/không 3 đoạn."

PANEL 2. KẾT QUẢ 15 CÂU CÓ ĐÁP ÁN
- Biểu đồ cột chồng (xanh = đúng, đỏ = sai, xám = từ chối):
  "Không RAG: 0 đúng · 1 sai · 14 từ chối"
  "Có RAG: 10 đúng · 3 sai · 2 từ chối"

PANEL 3. VÍ DỤ
- Câu hỏi: "Sân bay nào có số lượng phi cơ ra vào nhiều nhất châu Âu năm 2006?"
- "Không RAG → 'Không có thông tin.'" · "Có RAG → 'Charles-de-Gaulle' ✓"
- Câu hỏi: "Muốn ban hành luật thì hoàng đế phải làm gì?"
- "Không RAG → 'phải ký tên vào bản thảo luật' ✗" · "Có RAG → 'Hoàng đế phải thông qua Hội đồng để ban hành luật.' ✓"

PANEL 4. HẠN CHẾ ĐO ĐƯỢC
- Bảng "hiện tượng | số":
  "recall@3 của TF-IDF | 0.722 (13/18)"
  "Không tìm đúng đoạn → vẫn từ chối | 2 câu"
  "Câu tài liệu không có đáp án: không RAG từ chối | 3/3"
  "Câu tài liệu không có đáp án: có RAG bịa | 2/3"
- Gạch: "Có tài liệu vẫn sai: đọc sai đoạn (Paris 'thế kỷ 16–17'), tìm sai đoạn (Nam Ossetia 'Nga')."

PANEL 5. CODE
- Khối code:
  vectorizer = TfidfVectorizer(ngram_range=(1, 2))
  doc_matrix = vectorizer.fit_transform(corpus["title"] + ". " + corpus["context"])
  scores = linear_kernel(vectorizer.transform([query]), doc_matrix)[0]   # cosine
  top3 = scores.argsort()[::-1][:3]
- Chú giải: "ngram_range=(1, 2) | đếm cả cụm 2 âm tiết ('hội đồng')" · "linear_kernel | tích vô hướng = cosine"

PANEL 6. VÌ SAO RA NHƯ VẬY
- Gạch: "Có RAG đúng 10/15: đáp án nằm sẵn trong đoạn, model chỉ cần đọc và trích." · "Không RAG từ chối 14/15, kể cả câu model biết (Mona Lisa): câu dặn làm model quá thận trọng." · "Tài liệu 'gần đúng' làm model trả lời sai một cách tự tin."
- Hộp ý nghĩa: "RAG bổ sung tri thức lúc hỏi, không đổi trọng số – nhưng chỉ tốt bằng bộ tìm kiếm."

DẢI TÓM TẮT FLOW – "Kết quả RAG":
icon: kính lúp trên chồng tài liệu → "TF-IDF tìm top-3" · icon: tài liệu chèn vào khung → "Đưa 3 đoạn vào prompt" · icon: cột tăng → "Đúng 0 → 10/15" · icon: tài liệu có dấu chéo → "Tìm sai đoạn: recall@3 0.722" · icon: dấu cảnh báo tam giác → "Tài liệu gần đúng: bịa 2/3"
```

---

## Ảnh 21 — Mục 9 · Lý thuyết hạn chế

```text
Áp dụng QUY CHUẨN THIẾT KẾ.
Số "21". Tiêu đề: "HẠN CHẾ CỦA LLM: HALLUCINATION, THIÊN LỆCH, NGỮ CẢNH". Phụ đề: "Chất lượng phụ thuộc tác vụ và dữ liệu". Nhãn: "Mục 9 · Hạn chế". Nguồn: "Huang et al. 2023; Lin et al. 2022; Liu et al. 2023; Bolukbasi et al. 2016; Wei et al. 2024".

PANEL 1. HALLUCINATION LÀ GÌ
- Gạch: "Sinh nội dung nghe hợp lý nhưng sai."
- Bảng "loại | nghĩa":
  "Factuality | sai với sự thật thế giới (bịa năm, bịa người)"
  "Faithfulness | không đúng chỉ dẫn hoặc tài liệu được đưa"

PANEL 2. VÌ SAO HALLUCINATION XẢY RA
- Gạch đánh số: "1. Huấn luyện tối đa xác suất văn bản (cross-entropy), không tối đa sự thật." · "2. Khi sinh luôn chọn ra một token; không có bước tra cứu." · "3. Câu hỏi có tiền đề sai → tiền đề thành ngữ cảnh để sinh tiếp." · "4. Model lớn hơn chưa chắc trung thực hơn với câu hỏi gài bẫy (TruthfulQA)."

PANEL 3. THIÊN LỆCH
- Gạch: "Dữ liệu có định kiến (giới, nghề nghiệp, vùng miền) thì model học lại." · "Giảm bằng: lọc dữ liệu, alignment, đánh giá riêng."
- Hộp ý nghĩa: "Phần này trình bày bằng lời, không demo."

PANEL 4. GIỚI HẠN NGỮ CẢNH
- Bảng: "Giới hạn cứng | 32,768 token (Qwen2.5-1.5B; sinh tối đa 8K)" · "Giới hạn mềm – lost in the middle | hiệu năng thường cao nhất khi thông tin ở đầu hoặc cuối ngữ cảnh, giảm đáng kể khi ở giữa"
- Sơ đồ: thanh ngang "đầu – giữa – cuối ngữ cảnh", đường cong hình chữ U thể hiện hiệu năng
- Gạch: "Tokenizer tốn token (GPT-2 gấp 2.75× câu Việt) → hết ngữ cảnh nhanh hơn."

PANEL 5. PHỤ THUỘC TÁC VỤ VÀ DỮ LIỆU
- Gạch (từ chính bài): "Bộ đánh giá cũng có nhãn sai (UIT-ViQuAD: năm xử tử vua Charles ghi 1640, thực tế 1649)." · "Quy ước nhãn khác cách hiểu (UIT-VSFC: góp ý = tiêu cực)." · "Model chen tiếng Trung khi được hỏi bằng tiếng Việt."

PANEL 6. CÁCH CHẤM VÀ CÁCH KIỂM CHỨNG
- Bảng "nhãn | nghĩa" (SimpleQA): "Đúng | trả lời đúng" · "Sai | trả lời sai (bịa)" · "Từ chối | không trả lời – không bịa"
- Gạch: "Kiểm chứng: 1. đối chiếu nguồn gốc · 2. hỏi lại cách khác · 3. dùng RAG đưa nguồn vào prompt."

DẢI TÓM TẮT FLOW – "Nhận diện và xử lý hạn chế":
icon: biểu đồ cột xác suất → "Model chọn token xác suất cao" · icon: dấu hỏi trên hộp chữ → "Không biết vẫn trả lời" · icon: tài liệu + kính lúp → "Đối chiếu nguồn" · icon: hai hộp thoại khác nhau → "Hỏi lại cách khác" · icon: tài liệu chèn vào khung → "RAG đưa nguồn vào"
```

## Ảnh 22 — Mục 9 · Demo câu trả lời sai và cách kiểm chứng

```text
Áp dụng QUY CHUẨN THIẾT KẾ. Ngoại lệ: PANEL 1 gộp 2 cột trái hàng trên.
Số "22". Tiêu đề: "MỘT CÂU TRẢ LỜI SAI VÀ CÁCH KIỂM CHỨNG". Phụ đề: "Hỏi không tài liệu, không câu dặn từ chối, mỗi câu 2 cách diễn đạt". Nhãn: "Mục 9 · Demo". Nguồn: "notebook phần 9 · UIT-ViQuAD 2.0 · run 20261119_150356_normal".

PANEL 1. THẺ KIỂM CHỨNG (minh họa của đề)
- "Câu hỏi: Muốn ban hành luật thì hoàng đế phải làm gì? (chủ đề: Đế quốc La Mã Thần thánh)"
- "Cách hỏi A → '…hoàng đế cần làm theo các bước sau: 1. Xác định nội dung của luật muốn ban hành. 2. Chuyển đổi ý tưởng thành văn…' ✗"
- "Cách hỏi B → 'Đưa ra một đạo luật.' ✗"
- "Nguồn Wikipedia: '…Hoàng đế phải thông qua Hội đồng để ban hành luật, và những buổi thảo luận ít khi đạt sự nhất trí…' ✓"

PANEL 2. SỐ LIỆU
- Số lớn: "Sai 12/15 (cách A) · 12/15 (cách B)"
- Gạch: "Sai cả 2 lần: 11/15" · "Không từ chối câu nào" · "Thực thể không tồn tại: bịa 5/6 lần hỏi, chỉ ra tiền đề sai 0/6"

PANEL 3. HỎI LẠI CÁCH KHÁC LỘ RA VIỆC ĐOÁN
- Bảng "câu hỏi | cách A | cách B | đáp án":
  "Texas: bao nhiêu học sinh / 1 giáo viên? | 25–30 | 1:1 | 14,9"
  "Mông Cổ: tỉ lệ người dưới 30 tuổi? | 20–25% | 15% | khoảng 59%"
  "Sân bay đông nhất châu Âu 2006? | Charles de Gaulle | Charles de Gaulle | Charles-de-Gaulle ✓"

PANEL 4. THỰC THỂ KHÔNG TỒN TẠI
- Bảng "câu hỏi | cách A | cách B | sự thật":
  "Nhà vật lý người Việt Nguyễn Hoàng Lâm nhận Nobel Vật lý năm nào? | 2021 | 1994 | người không có thật"
  "Việt Nam đăng cai Thế vận hội Mùa hè 1987 ở đâu? | Hà Nội | Hà Nội | chưa từng; 1987 không có Thế vận hội"
  "Tiểu thuyết 'Bóng chiều trên sông Hậu' của Nguyễn Du kể về ai? | bịa nhân vật 'Trúc' | lặp lại tên (từ chối) | không có tác phẩm này"

PANEL 5. CODE
- Khối code:
  PHRASINGS = {
      "A": lambda r: f"{r.question} (chủ đề: {r.title}) Trả lời ngắn gọn.",
      "B": lambda r: f"Về {r.title}, hãy cho biết: {r.question} Chỉ trả lời ngắn gọn, không giải thích.",
  }
  start = int(answers["answer_start"][0])                  # vị trí đáp án trong đoạn Wikipedia
  end = start + len(str(answers["text"][0]))
  evidence = context[max(0, start - 120):end + 120]        # đoạn nguồn để kiểm chứng

PANEL 6. VÌ SAO RA NHƯ VẬY VÀ CÁCH KIỂM CHỨNG
- Gạch: "Biết thật → đổi cách hỏi vẫn ra cùng đáp án; đoán → mỗi lần một số." · "Nhất quán chưa chắc đúng: 'Hà Nội' cả 2 lần vẫn sai." · "Tiền đề sai trong câu hỏi được model chấp nhận rồi sinh tiếp."
- Hộp: "Kiểm chứng: 1. đối chiếu nguồn gốc · 2. hỏi lại cách khác · 3. dùng RAG (RAG vẫn có thể bịa: 2/3 câu tài liệu không có đáp án, mục 8)."

DẢI TÓM TẮT FLOW – "Quy trình kiểm chứng":
icon: hộp thoại → "Hỏi câu không tài liệu" · icon: hai hộp thoại → "Hỏi 2 cách" · icon: dấu khác (≠) → "Không khớp → đang đoán" · icon: tài liệu + kính lúp → "Đối chiếu đoạn nguồn" · icon: bảng có dấu tích/chéo → "Chấm Đúng / Sai / Từ chối"
```

---

## Ảnh 23 — Mục 10 · Thiết kế và cách đánh giá

```text
Áp dụng QUY CHUẨN THIẾT KẾ.
Số "23". Tiêu đề: "THỬ NGHIỆM CÓ KIỂM SOÁT: THIẾT KẾ VÀ CÔNG CỤ ĐÁNH GIÁ". Phụ đề: "Thêm 4 ví dụ vào prompt có làm accuracy khác có ý nghĩa thống kê không?". Nhãn: "Mục 10 · Thực nghiệm". Nguồn: "Dietterich 1998; tài liệu statsmodels mcnemar".

PANEL 1. THIẾT LẬP
- Bảng: "Model | Qwen2.5-1.5B-Instruct, float16" · "Dữ liệu | UIT-VSFC test 2,999 câu (1,590 / 1,409)" · "Prompt | vi_nhan_vi, chọn trên 200 câu validation" · "Sinh | greedy, max_new_tokens 10" · "Ví dụ few-shot | 4 câu từ train, xen kẽ nhãn" · "Seed | 42"
- Ô nổi bật cam: "Biến duy nhất: có / không 4 ví dụ"

PANEL 2. GIẢ THUYẾT
- Hộp công thức: H₀: tỉ lệ sai của zero-shot = tỉ lệ sai của few-shot
- Gạch: "Mức ý nghĩa α = 0.05." · "Một phép so sánh → không cần hiệu chỉnh Bonferroni."

PANEL 3. ACCURACY VÀ KHOẢNG TIN CẬY 95%
- Hộp công thức: acc = số câu đúng / n
- Hộp công thức: CI₉₅% = acc ± 1.96 · √( acc · (1 − acc) / n )
- Chú giải: "n | số câu test (2,999)" · "1.96 | phân vị 97.5% của phân phối chuẩn" · "câu không đọc được nhãn | tính là sai"

PANEL 4. CONFUSION MATRIX
- Sơ đồ ma trận 2×3: hàng "tích cực / tiêu cực (nhãn thật)", cột "tích cực / tiêu cực / không đọc được (dự đoán)"; đường chéo tô xanh "đúng"
- Hộp công thức: recall_c = đúng lớp c / số câu thật là c ;  precision_c = đúng lớp c / số câu đoán là c

PANEL 5. KIỂM ĐỊNH McNEMAR
- Bảng 2×2: " | Few đúng | Few sai" / "Zero đúng | a | b" / "Zero sai | c | d"
- Hộp công thức: b ~ Binomial(b + c, 0.5) ;  p = min( 1, 2 · P(X ≤ min(b, c)) )
- Gạch: "a, d: hai cách cùng đúng / cùng sai – không cho biết cách nào tốt hơn." · "Chỉ b, c (bất đồng) mang thông tin."

PANEL 6. VÌ SAO DÙNG McNEMAR
- Gạch: "Hai cách chạy trên CÙNG câu → kết quả theo cặp, không độc lập." · "Kiểm định 2 tỉ lệ độc lập sẽ sai giả định." · "Dietterich 1998: McNemar phù hợp khi mỗi thuật toán chỉ chạy được một lần trên một tập test."

DẢI TÓM TẮT FLOW – "Quy trình đánh giá":
icon: bảng dữ liệu → "2,999 câu test" · icon: hai khung cạnh nhau → "Zero và few trên cùng câu" · icon: thước đo phần trăm → "Accuracy + CI 95%" · icon: ma trận 2×2 → "Bảng cặp a, b, c, d" · icon: phân phối nhị thức → "McNemar exact → p"
```

## Ảnh 24 — Mục 10 · Kết quả và McNemar

```text
Áp dụng QUY CHUẨN THIẾT KẾ.
Số "24". Tiêu đề: "KẾT QUẢ: FEW-SHOT TỐT HƠN CÓ Ý NGHĨA THỐNG KÊ". Phụ đề: "2,999 câu test UIT-VSFC · cùng câu, cùng prompt, chỉ khác 4 ví dụ". Nhãn: "Mục 10 · Thực nghiệm". Nguồn: "notebook phần 10 (cell 10.1–10.2) · run 20261119_150356_normal".

PANEL 1. ACCURACY
- Bảng "cách | accuracy | CI 95% | không đọc được nhãn":
  "Zero-shot | 89.86% | ±1.08 → [88.78; 90.94] | 0"
  "Few-shot | 92.30% | ±0.95 → [91.35; 93.25] | 0"
- Biểu đồ: 2 thanh ngang với đoạn sai số CI.

PANEL 2. TÍNH CI (ví dụ minh họa ngắn)
- Dòng tính: "1.96 · √(0.8986 × 0.1014 / 2999) = 0.0108"
- Dòng tính: "1.96 · √(0.923 × 0.077 / 2999) = 0.0095"

PANEL 3. BẢNG McNEMAR THẬT
- Bảng 2×2 lớn: " | Few đúng | Few sai" / "Zero đúng | a = 2,648 | b = 47" / "Zero sai | c = 120 | d = 184"
- Dòng kiểm: "a + b = 2,695 câu zero đúng · a + c = 2,768 câu few đúng · tổng = 2,999"

PANEL 4. p-VALUE
- Số lớn #27AE60: "p = 1.5e-08 < 0.05"
- Dòng tính: "b + c = 167 câu bất đồng; nếu H₀ đúng mỗi bên ≈ 83.5; thực tế zero chỉ thắng 47"
- Dòng tính: "Xấp xỉ χ² có hiệu chỉnh: (|47 − 120| − 1)² / 167 = 31.0 > 3.84"

PANEL 5. CODE
- Khối code:
  table = mcnemar_counts(icl_zero["correct"], icl_few["correct"])   # [[a, b], [c, d]]
  p = mcnemar(table, exact=True).pvalue                              # 1.503e-08
- Chú giải: "exact=True | dùng phân phối nhị thức chính xác"

PANEL 6. ĐỌC KẾT QUẢ ĐÚNG CÁCH
- Gạch: "p = 1.5e-08: nếu hai cách thật sự ngang nhau, khả năng thấy chênh lệch lớn như vậy chỉ khoảng 1.5 phần trăm triệu." · "Không có nghĩa few-shot tốt hơn với xác suất 99.99%."
- Hộp viền đỏ: "Giới hạn: 1 model 1.5B · 1 bộ dữ liệu · 1 bộ ví dụ · 1 lần chạy → không suy rộng cho mọi LLM."

DẢI TÓM TẮT FLOW – "Kết luận thí nghiệm":
icon: thanh 89.86% → "Zero-shot 89.86%" · icon: thanh 92.30% → "Few-shot 92.30%" · icon: ma trận 2×2 → "b = 47, c = 120" · icon: phân phối nhị thức → "p = 1.5e-08" · icon: dấu tích trong khung → "Khác biệt có ý nghĩa ở mức 0.05"
```

## Ảnh 25 — Mục 10 · Confusion matrix, phân tích lỗi, tổng hợp

```text
Áp dụng QUY CHUẨN THIẾT KẾ. Đính kèm ảnh thật confusion_matrix_icl.png (trong outputs/figures/).
Số "25". Tiêu đề: "PHÂN TÍCH LỖI VÀ TỔNG HỢP KẾT QUẢ". Phụ đề: "Lỗi đến từ đâu, và 10 mục cho thấy điều gì". Nhãn: "Mục 10 · Kết luận". Nguồn: "notebook phần 10 (cell 10.1, 10.3, 10.4) · run 20261119_150356_normal".

PANEL 1. CONFUSION MATRIX THẬT
- Đặt ảnh thật confusion_matrix_icl.png.
- Gạch: "Tiêu cực bị đoán tích cực: 248 → 145 (giảm 103)" (xanh lá) · "Tích cực bị đoán tiêu cực: 56 → 86 (tăng 30)" (đỏ)

PANEL 2. RECALL, PRECISION
- Bảng " | zero-shot | few-shot":
  "Recall tích cực | 96.5% | 94.6%"
  "Recall tiêu cực | 82.4% | 89.7%"
  "Precision tích cực | 86.1% | 91.2%"
- Dòng: "Lợi 103 câu, mất 30 câu → ròng +73 câu đúng (2,768 − 2,695)"

PANEL 3. NHÓM LỖI (30 CÂU FEW-SHOT SAI, NHÓM ĐỊNH NGHĨA TRƯỚC KHI ĐỌC)
- Biểu đồ cột ngang: "Ý ngầm / góp ý 20" · "Vừa khen vừa chê 4" · "Model sai rõ 4" · "Nhãn gốc có vẻ sai 2" · "Không đọc được nhãn 0"
- Ví dụ: "'giảm thời lượng lý thuyết , tăng thời lượng thực hành .' – nhãn tiêu cực → model tích cực"

PANEL 4. NGUYÊN NHÂN CHÍNH
- Số lớn: "76–81%"
- Gạch: "câu tiêu cực bị đoán tích cực là câu góp ý (có 'nên', 'cần', 'mong', 'thêm', 'hơn')" · "zero-shot 189/248 · few-shot 117/145"
- Hộp ý nghĩa: "Lỗi chính không phải model kém tiếng Việt mà là quy ước nhãn của UIT-VSFC: góp ý = tiêu cực."

PANEL 5. BẢNG TỔNG HỢP 10 MỤC
- Bảng "mục | kết quả":
  "1 | 4 tác vụ, 1 model: đạt 4/4"
  "2 | top-1 'đẹp' 0.788; attention tương lai = 0"
  "3 | câu Việt: Qwen 8 / GPT-2 22 token"
  "4 | loss 2.985 · xáo trộn 6.701 · đoán bừa 11.931"
  "5 | base 0/4 · Instruct 4/4"
  "6 | greedy 1 output; lấy mẫu 3 output / 3 lần"
  "7, 10 | 89.86% → 92.30%; p = 1.5e-08"
  "8 | RAG đúng 0 → 10/15; không có đáp án: bịa 2/3"
  "9 | không tài liệu sai 12/15; thực thể bịa 5/6"

PANEL 6. CODE VÀ GIỚI HẠN
- Khối code:
  pred = res["pred"].map(LABEL_NAMES).fillna("không parse được")
  cm = pd.crosstab(res["label"].map(LABEL_NAMES), pred)     # hàng: nhãn thật, cột: dự đoán
  SUGGEST_RE = r"\b(?:nên|cần|mong|thêm|hơn)\b"            # đếm câu góp ý trong các câu sai
- Hộp viền đỏ: "Giới hạn: 1 model 1.5B, 1 bộ dữ liệu mỗi thí nghiệm, 1 lần chạy, chấm tay do một người đọc lại."

DẢI TÓM TẮT FLOW – "Kết luận toàn bài":
icon: dãy ô token → "LLM xử lý văn bản dạng token" · icon: biểu đồ cột → "Đầu ra là phân phối xác suất" · icon: ba khối nối tiếp → "Post-training đổi hành vi" · icon: tài liệu chèn vào khung → "Prompt, RAG thích nghi không đổi trọng số" · icon: dấu cảnh báo tam giác → "Luôn kiểm chứng output"
```

---

## Ảnh 26 — Tài liệu tham khảo

```text
Áp dụng QUY CHUẨN THIẾT KẾ. Ngoại lệ: 3 panel cột dọc (1 hàng), không dải TÓM TẮT FLOW, không icon.
Số "26". Tiêu đề: "TÀI LIỆU THAM KHẢO". Phụ đề: "Bài báo gốc, báo cáo kỹ thuật, bộ dữ liệu và tài liệu thư viện". Nhãn: "Tài liệu". Nguồn: "docs/references.md".

PANEL 1. NỀN TẢNG VÀ HUẤN LUYỆN
Vaswani et al., Attention Is All You Need, NeurIPS 2017
Bengio et al., A Neural Probabilistic Language Model, JMLR 2003
Sennrich et al., Neural Machine Translation of Rare Words with Subword Units, ACL 2016
Brown et al., Language Models are Few-Shot Learners, 2020
Ouyang et al., Training language models to follow instructions with human feedback, 2022
Rafailov et al., Direct Preference Optimization, 2023
Holtzman et al., The Curious Case of Neural Text Degeneration, ICLR 2020

PANEL 2. KHẢ NĂNG, THÍCH NGHI, HẠN CHẾ, ĐÁNH GIÁ
Min et al., Rethinking the Role of Demonstrations, 2022
Zhao et al., Calibrate Before Use, 2021
Lewis et al., Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks, NeurIPS 2020
Hu et al., LoRA: Low-Rank Adaptation of Large Language Models, 2021
Huang et al., A Survey on Hallucination in Large Language Models, 2023
Lin et al., TruthfulQA, ACL 2022
Liu et al., Lost in the Middle, 2023
Wei et al., Measuring short-form factuality in large language models (SimpleQA), 2024
Dietterich, Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms, 1998

PANEL 3. KIẾN TRÚC, MODEL, DỮ LIỆU, THƯ VIỆN
Su et al., RoFormer (RoPE), 2021 · Zhang & Sennrich, RMSNorm, 2019 · Shazeer, GLU Variants Improve Transformer, 2020 · Ainslie et al., GQA, 2023
Qwen Team, Qwen2.5 Technical Report, 2024
Nguyen et al., UIT-VSFC: Vietnamese Students' Feedback Corpus, KSE 2018
Nguyen et al., VLSP 2021 – ViMRC Challenge (UIT-ViQuAD 2.0), 2022
Tài liệu chính thức: Hugging Face Transformers · scikit-learn · statsmodels
```

---

## Kiểm sau khi sinh (nhóm làm)

- [ ] Không có emoji, logo, ảnh người, icon trang trí ngoài dải TÓM TẮT FLOW.
- [ ] Dấu tiếng Việt đúng từng chữ.
- [ ] Số khớp `outputs/results/` (bảng kiểm trong [`../giai_thich_du_an_v2.md`](../giai_thich_du_an_v2.md)).
- [ ] Công thức hiển thị như sách toán, không lộ mã LaTeX, đúng ký hiệu (√, Σ, ∏, −∞, ⊙, ‖, ~, ∈, ≪).
- [ ] Code đúng từng ký tự.
- [ ] Ảnh 07, 25 dùng đúng ảnh thật.
- [ ] Ảnh nào sai nhiều: sinh lại kèm câu "Sửa đúng các lỗi sau: …"; hoặc sinh bố cục rồi gõ chữ trong Google Slides.
