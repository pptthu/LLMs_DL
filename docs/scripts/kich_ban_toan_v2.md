# Kịch bản thuyết trình chi tiết — Toàn trình bày 16 slide, 10 mục (v2)

- **Created**: 2026-11-21 (Toàn, AI hỗ trợ). Thay `kich_ban_toan_v1.md` (chỉ slide 2, 9, 12, 13; v1 giữ nguyên). Bản 21/11: **nói tới đâu, mở code tới đó**; mỗi công thức có **cách đọc thành lời**.
- Khung slide: `docs/plans/khung_presentation_llms_v5.md`. Giải thích đầy đủ từng cell: `docs/giai_thich_du_an_v4.md`.
- Mọi số lấy từ lần chạy chính thức **`20261121_010447_normal`** (Kaggle, GPU T4, float16, seed 42, 9.2 phút, 140 cell, 0 lỗi).
- Thời gian ≤ 20 phút **gồm cả lúc mở notebook**; phần nói khoảng **19.4 phút**.
- Cập nhật 21/11 (bộ 16 slide): bỏ dòng "ĐỀ" trên slide; thêm slide 4 sơ đồ kiến trúc (vai trò Transformer, decoder-only của Qwen, causal attention) và sơ đồ ở slide 3, 6, 7, 8, 9 (quan hệ LM / LLM / chatbot, pipeline tokenization, dữ liệu huấn luyện → loss, RLHF / DPO, vòng lặp sinh). mỗi slide nội dung có **khung 💻 code ngắn** (3–5 dòng chính, rút gọn từ cell); mục 9 tách thành slide 12 (hallucination) và slide 13 (thiên lệch, ngữ cảnh, chất lượng); chỉ **mở notebook ở slide 5, 8, 12, 13, 14**, các slide khác giải thích code ngay trên slide.

**Cách đọc mỗi slide**

| Ký hiệu | Nghĩa |
|---|---|
| 🎯 **Cô yêu cầu** | Chữ trong đề + slide đáp ứng bằng gì. Không đọc, để tự kiểm |
| 🎤 "…" | **Lời nói**, nói thành tiếng gần nguyên văn |
| 🗣 **Đọc công thức** | Câu đọc công thức thành lời + nghĩa từng ký hiệu |
| [ ] | Thao tác trên màn hình, không đọc |
| 💻 **Code** | Đoạn code đang chiếu + giải thích từng dòng (nói phần in đậm) |
| 📊 **Output** | Output thật + vì sao ra số đó |
| 🖼 **Hình** | Hình là gì, đọc thế nào, nguyên lý |
| 🧭 **Trọng tâm lý thuyết** | Các ý lý thuyết bắt buộc nói được của slide, viết dễ hiểu |
| 📖 **Thuật ngữ** | Từ chuyên ngành của slide: là gì, câu nói mẫu (🗨), cần học / hiểu tới đâu (🎓); ★ = bắt buộc |
| 🔎 **Đọc demo trên slide** | Chỉ vào đâu trên slide, INPUT là gì, OUTPUT là gì, vì sao ra như vậy |
| 📝 **Hiểu thêm** | Lý thuyết sâu hơn, **không đọc**, dùng khi bị hỏi |
| ❓ | Câu có thể bị hỏi + ý trả lời |

---

## Phần A. Ôn theo thứ tự nào, chỗ nào coi kĩ

Thứ tự **ôn** theo kiến thức nối nhau (cái sau dùng cái trước); thứ tự **trình bày** theo đề. ★★★ = coi kĩ, tự giải thích được không nhìn; ★★ = hiểu ý, nhớ số chính; ★ = đọc qua.

| Bước | Ôn gì | Ở slide | Mức |
|---|---|---|---|
| 1 | Notebook chạy thế nào, khung ĐẦU VÀO, `assert` | 2 | ★ |
| 2 | Tokenization: byte-level BPE, `Ġ`, token ID, embedding = tra hàng | 6 | ★★★ |
| 3 | Kiến trúc: Q, K, V, công thức attention, causal mask, decoder block, softmax | 4, 5 | ★★★ |
| 4 | Nguyên lý học: cặp ngữ cảnh → mục tiêu, cross-entropy, dịch nhãn 1 vị trí, perplexity | 7 | ★★★ |
| 5 | Sinh văn bản: vòng lặp, greedy, temperature, top-p | 9 | ★★★ |
| 6 | Giai đoạn huấn luyện: pretraining / SFT / alignment, RLHF so với DPO | 8 | ★★ |
| 7 | Giới thiệu: LM, LLM, chatbot, quy tắc nhân xác suất | 3 | ★★ |
| 8 | Zero / few-shot và thử nghiệm có kiểm soát: chọn prompt trên validation, McNemar, phân tích lỗi | 10, 14, 15 | ★★★ |
| 9 | RAG: bảng so sánh, TF-IDF, cosine, recall@3 | 11 | ★★ |
| 10 | Hạn chế: kiểm chứng, thiên lệch (`share_female`, r, luật ghi trước), giới hạn ngữ cảnh | 12, 13 | ★★★ |
| 11 | Đọc to các dòng 🗣 cho tới khi nói trơn; tập Phần C, bấm giờ từng slide | tất cả | ★★★ |

**Nếu chỉ còn ít thời gian, coi kĩ 6 chỗ:** công thức attention + causal mask (slide 4); `logits[:-1]` đi với `ids[0, 1:]` (slide 7); temperature chia logits trước softmax (slide 9); McNemar b = 47, c = 120 (slide 14); `share_female` và r (slide 13); cắt 1,024 token thì model bịa "1234" (slide 13).

**Học thuật ngữ trước:** đọc khối 📖 của từng slide theo thứ tự ôn ở trên; mỗi từ ★ phải tự nói được câu 🗨 mà không nhìn. Từ không có ★ chỉ cần hiểu khi bị hỏi.

**Tự kiểm đã hiểu:** mở notebook, chọn 3 cell bất kỳ, nói to: cell làm gì, đầu vào là gì (đọc khung ĐẦU VÀO), dòng code quan trọng nhất, output nói lên điều gì.

---

## Phần B. Chuẩn bị trước giờ thuyết trình

- Mở notebook bản **đã chạy** (output run `20261121_010447_normal`). **Không chạy lại** khi thuyết trình: tải model mất vài phút.
- Zoom 125–150%. Đặt sẵn bookmark ở các cell sẽ mở: 2.2, 2.3, 2.4 (slide 5); 5.0, 5.2, 5.3 (slide 8); 9.1, 9.2 (slide 12); 9.4, 9.5, 9.6 (slide 13); 7.3, 10.1, 10.2 (slide 14). Các cell khác chỉ mở khi bị hỏi.
- Nhịp slide thường: nói lý thuyết → **chỉ khung 💻 code trên slide** (đọc 1–3 dòng chính) → chỉ INPUT / OUTPUT, hình → một câu kết luận.
- Nhịp slide 5, 8, 12, 13, 14 (bằng chứng chính): nói lý thuyết → [mở cell] → **chỉ khung ĐẦU VÀO** → chỉ dòng code chính → chỉ output / hình → quay lại slide.
- Khung 💻 trên slide là bản **rút gọn** của cell (bỏ type hint, `assert`, in bảng); nếu bị hỏi phần bỏ đi, mở cell đó.
- Khi slide có công thức: **đọc công thức thành lời một lần** (dòng 🗣), rồi nói ngay ý nghĩa bằng câu thường. Không đọc ký hiệu mà không giải nghĩa.

---

## Phần C. Kịch bản từng slide

### Slide 1 — Mở đầu (0.3 phút)

🎤 "Em chào cô. Em là Toàn, hôm nay em trình bày đề tài **Tìm hiểu mô hình ngôn ngữ lớn**. Theo đề, mục tiêu là giải thích được LLM **học như thế nào** và **sinh văn bản ra sao**, phân biệt **các giai đoạn huấn luyện**, đồng thời chỉ ra **khả năng và hạn chế** của mô hình. Mỗi mục trong 10 mục đều có demo chạy trên một model thật, Qwen2.5-1.5B-Instruct, và mọi con số trên slide đều lấy trực tiếp từ output của notebook."

---

### Slide 2 — Model, môi trường, thư viện (1.0 phút)

🎯 **Cô yêu cầu** (yêu cầu thêm 1, 2): slide nêu rõ **thư viện** và **môi trường chạy**. → Slide 2 + cell 0.3, 0.4, 0.5.

🧭 **Trọng tâm lý thuyết (nắm chắc trước khi nói; quên lời thì nói đúng các ý này):**
- Model dùng cho mọi demo: **Qwen2.5-1.5B-Instruct**, Transformer decoder-only, **1,543,714,304 tham số** (code đếm, có `assert`).
- Môi trường: Kaggle GPU T4, **float16 = 2 byte mỗi tham số** → khoảng 3.1 GB trọng số; nhóm **chỉ suy luận**, không huấn luyện.
- Thư viện, mỗi cái một việc: torch tính trên GPU · transformers tải model và sinh · pandas đọc dữ liệu · scikit-learn TF-IDF · statsmodels McNemar · scipy Pearson r. Phiên bản đọc bằng code.
- Bản Instruct mặc định **lấy mẫu ngẫu nhiên** → mọi lần gọi `generate()` đều truyền tham số sinh tường minh; phần đánh giá dùng greedy để chạy lại ra đúng số.

📖 **Thuật ngữ của slide này** (★ = bắt buộc nắm; mỗi từ: **là gì** · 🗨 **nói thế nào** · 🎓 **cần học / hiểu tới đâu**)
- ★ **Tham số (parameter)** — Các con số bên trong model, được điều chỉnh khi huấn luyện; model "biết" gì là nhờ các số này.
  🗨 "Model có khoảng 1.5 tỉ tham số, tức 1.5 tỉ con số học được."
  🎓 Tham số học được ≠ siêu tham số (do người đặt, như temperature). Đếm bằng `p.numel()`.
- ★ **Suy luận (inference)** — Chỉ đưa dữ liệu vào và lấy kết quả; tham số không thay đổi.
  🗨 "Nhóm chỉ chạy suy luận, không huấn luyện thêm model."
  🎓 Phân biệt với huấn luyện (training): training có tính gradient và cập nhật tham số.
- **float16 (fp16)** — Cách lưu mỗi số bằng 16 bit = 2 byte, thay vì 4 byte của float32.
  🗨 "Model lưu ở float16, mỗi tham số 2 byte, nên cần khoảng 3.1 GB."
  🎓 Bộ nhớ ≈ số tham số × số byte. Đổi lại độ chính xác số thấp hơn, đủ cho suy luận.
- **GPU, VRAM** — GPU là chip tính song song rất nhiều phép nhân; VRAM là bộ nhớ của GPU (T4: khoảng 15 GB).
  🗨 "Notebook chạy trên GPU T4 của Kaggle."
  🎓 Model và dữ liệu phải nằm trong VRAM mới tính được; hết VRAM thì báo lỗi OutOfMemory (gặp ở mục 9).
- **Hugging Face Hub** — Kho chia sẻ model và dữ liệu công khai; `from_pretrained` tải model từ đây.
  🗨 "Model tải trực tiếp từ Hugging Face."
  🎓 Biết `from_pretrained(tên)` tải cả trọng số lẫn cấu hình.
- ★ **Seed** — Số khởi tạo cho bộ sinh số ngẫu nhiên; cùng seed thì phần ngẫu nhiên lặp lại y hệt.
  🗨 "Seed cố định là 42 để chạy lại ra đúng kết quả."
  🎓 Seed chỉ ảnh hưởng chỗ có ngẫu nhiên (lấy mẫu, chọn ví dụ); greedy không cần seed.
- **`generation_config`, `do_sample`** — Cấu hình sinh mặc định đi kèm model; `do_sample=True` nghĩa là lấy mẫu ngẫu nhiên.
  🗨 "Bản Instruct mặc định lấy mẫu ngẫu nhiên, nên em luôn truyền tham số sinh rõ ràng."
  🎓 Nếu không truyền tham số, mỗi lần chạy ra câu khác → kết quả không lặp lại được.
- **`assert`** — Câu lệnh kiểm tra: điều kiện sai thì dừng chương trình ngay.
  🗨 "Có câu lệnh kiểm tra số tham số khớp model card."
  🎓 Dùng để chứng minh số trên slide là đúng, không ghi tay.
- ★ **Greedy (nói trước, giải thích kỹ ở slide 9)** — Cách sinh luôn chọn token có xác suất cao nhất ở mỗi bước, không ngẫu nhiên.
  🗨 "Phần đánh giá em dùng greedy, tức luôn chọn token có xác suất cao nhất, nên chạy lại vẫn ra đúng kết quả."
  🎓 Đủ để hiểu vì sao từ slide 3 trở đi mọi demo đều lặp lại được; phần đầy đủ ở slide 9.

✅ **Các ý bắt buộc phải nói ở slide này (theo thứ tự):**
1. Model: Qwen2.5-1.5B-Instruct, decoder-only, 1,543,714,304 tham số, số do code đếm.
2. Môi trường: Kaggle T4, float16, seed 42; nhóm chỉ suy luận.
3. Thư viện và vai trò của từng thư viện.
4. Instruct mặc định lấy mẫu ngẫu nhiên → mọi chỗ sinh đều truyền tham số rõ ràng.

📐 **Lý thuyết chi tiết (đọc để hiểu sâu; khi nói chỉ lấy ý, không đọc nguyên văn):**

**Model là một hàm có tham số.** Viết gọn: $\hat y = f_\theta(x)$, trong đó $\theta$ là toàn bộ tham số. **Huấn luyện** là tìm $\theta$ để model dự đoán tốt trên dữ liệu; **suy luận** là giữ nguyên $\theta$, chỉ đưa $x$ vào để lấy $\hat y$. Cả notebook chỉ làm suy luận: không có dòng nào tính gradient hay cập nhật tham số.

**Bộ nhớ của model.** Mỗi tham số là một số thực. float32 dùng 4 byte, float16 dùng 2 byte. Bộ nhớ chỉ riêng trọng số xấp xỉ bằng số tham số nhân số byte: $1{,}543{,}714{,}304 \times 2 \approx 3.09$ GB. Ngoài trọng số, GPU còn phải chứa kết quả trung gian và KV cache (mục 9), nên câu dài tốn thêm bộ nhớ. T4 không tính nhanh được kiểu bf16, nên nhóm dùng float16.

**Lặp lại được kết quả (reproducibility).** Kết quả chỉ đáng tin khi người khác chạy lại ra cùng số. Có hai nguồn ngẫu nhiên: cách sinh văn bản (lấy mẫu) và việc chọn dữ liệu (chọn ví dụ, chọn mẫu). Nhóm khoá cả hai: dùng greedy cho phần đánh giá và cố định seed 42 cho phần chọn. Kiểm chứng thật: hai lần chạy chính thức cho cùng số ở phần 1–8.

🎤 "Trước khi vào nội dung, em giới thiệu công cụ dùng cho mọi demo. Model chính là **Qwen2.5-1.5B-Instruct** của Alibaba, công khai trên Hugging Face với giấy phép Apache 2.0. Đây là một Transformer **decoder-only** gồm 28 block, mỗi token được biểu diễn bằng một vector 1,536 chiều. Notebook chạy trên **Kaggle với GPU Tesla T4**, kiểu số **float16**, seed cố định là 42. Nhóm **chỉ chạy suy luận, không huấn luyện thêm**. Về thư viện: PyTorch để tính trên GPU, Transformers để tải model và sinh văn bản, pandas để đọc dữ liệu, scikit-learn cho TF-IDF, statsmodels cho kiểm định McNemar và SciPy cho hệ số tương quan. Toàn bộ notebook chạy hết 9.2 phút."

[Trên slide, không mở notebook — cell 0.4]

💻 **Code**
```python
tok = AutoTokenizer.from_pretrained(MODEL_INSTRUCT)
model = AutoModelForCausalLM.from_pretrained(MODEL_INSTRUCT, torch_dtype=DTYPE, device_map={"": 0})
model.eval()
```
- `from_pretrained(tên)`: **tải trọng số và tokenizer từ Hugging Face Hub** rồi dựng model; model công khai nên không cần token.
- `torch_dtype=float16`: **mỗi tham số lưu bằng 2 byte**; T4 không hỗ trợ bf16 bằng phần cứng.
- `device_map={"": 0}`: đặt toàn bộ model lên GPU 0. `eval()`: chế độ suy luận, tắt dropout.

🎤 "Hai dòng này tải model từ Hugging Face. Với float16, mỗi tham số chiếm 2 byte, nên 1.54 tỉ tham số cần khoảng 3.1 GB bộ nhớ, vừa với GPU T4."

🗣 **Đọc công thức:** "Bộ nhớ trọng số xấp xỉ bằng số tham số nhân số byte mỗi tham số: 1.54 tỉ nhân 2 byte, khoảng 3.1 GB."

📊 **Output** (in `generation_config`): `do_sample=True, temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.1`.

🎤 "Một chi tiết quan trọng: bản Instruct **mặc định sinh văn bản bằng lấy mẫu ngẫu nhiên**, nên nếu để mặc định, mỗi lần chạy sẽ ra một câu khác nhau. Vì vậy, ở mọi chỗ sinh văn bản, em đều truyền tham số **một cách tường minh**, và dùng greedy cho các phần đánh giá để kết quả lặp lại được."

[Cuộn xuống cell 0.5]

💻 `ARCH["n_params"] = sum(p.numel() for p in model.parameters())`, sau đó `assert ARCH["n_params"] == 1_543_714_304`
- `p.numel()` đếm số phần tử của một ma trận tham số; cộng tất cả lại được tổng số tham số. **`assert` dừng notebook nếu số đếm được không khớp model card.**

🎤 "Con số **1,543,714,304 tham số** trên slide là do code đếm, có câu lệnh kiểm tra khớp với model card. Phiên bản các thư viện cũng được đọc bằng code ở cell 0.3, không ghi theo trí nhớ."

📝 **Hiểu thêm:** *tham số* là các số model học được trong quá trình huấn luyện; *suy luận* là chỉ đưa dữ liệu vào và lấy kết quả, không thay đổi tham số. Kaggle cấp 2 GPU nhưng model chỉ dùng GPU 0.

🔎 **Đọc demo trên slide (INPUT → OUTPUT → vì sao):** Thẻ MODEL: bảng kiến trúc đọc từ `model.config`. Thẻ MÔI TRƯỜNG: T4, fp16, Python, CUDA, seed 42, 9.2 phút. Thẻ THƯ VIỆN: phiên bản thật từ `results/env.json`. Ô code: tải model ở float16 lên GPU 0 → đếm tham số → `assert` khớp model card.

❓ Sao chọn model 1.5B? → Đủ nhỏ để chạy trên T4 miễn phí, hỗ trợ tiếng Việt, và có bản base cùng kiến trúc để so ở mục 5.

---

### Slide 3 — Mục 1: Giới thiệu (1.3 phút)

🎯 **Cô yêu cầu:** "LLM là gì? Phân biệt language model, LLM và ứng dụng chatbot" — minh họa "ví dụ các tác vụ: hỏi đáp, tóm tắt, dịch, sinh mã". → Lý thuyết trên slide + cell 1.1–1.4.

🧭 **Trọng tâm lý thuyết (nắm chắc trước khi nói; quên lời thì nói đúng các ý này):**
- **Language model** = mô hình gán xác suất cho chuỗi token; nhờ quy tắc nhân, chỉ cần học **đoán token tiếp theo**.
- **LLM** = vẫn là language model, nhưng **rất lớn**: Transformer, hàng tỉ tham số, học trên hàng nghìn tỉ token → làm được nhiều tác vụ chỉ bằng prompt.
- **Chatbot** = **ứng dụng** xây trên LLM: model đã qua post-training + chat template (vai system / user / assistant) + giao diện + lớp an toàn.
- Demo: **1 model, 1 cách sinh (greedy), chỉ đổi prompt** → hỏi đáp, tóm tắt, dịch, sinh mã đều là "viết tiếp sau lượt assistant".

📖 **Thuật ngữ của slide này** (★ = bắt buộc nắm; mỗi từ: **là gì** · 🗨 **nói thế nào** · 🎓 **cần học / hiểu tới đâu**)
- ★ **Token** — Mảnh văn bản model đọc: có thể là một từ, một phần của từ hoặc một byte.
  🗨 "Model không đọc chữ mà đọc token."
  🎓 Token ≠ từ: "Hôm" là 2 token với Qwen. Chi tiết ở mục 3.
- ★ **Language model (mô hình ngôn ngữ)** — Model gán xác suất cho một chuỗi token, tương đương: đoán token tiếp theo.
  🗨 "Language model là model cho biết một chuỗi văn bản có khả năng xuất hiện bao nhiêu."
  🎓 Thuộc công thức P(x₁…xₙ) = ∏ P(xₜ | x<ₜ) và đọc được thành lời.
- ★ **Xác suất có điều kiện P(A | B)** — Xác suất của A khi đã biết B. Dấu "|" đọc là "khi biết".
  🗨 "Xác suất token tiếp theo khi biết các token trước."
  🎓 Hiểu vì sao tích các xác suất có điều kiện = xác suất cả chuỗi (quy tắc nhân).
- ★ **LLM (large language model)** — Language model rất lớn: Transformer, hàng tỉ tham số, học trên hàng nghìn tỉ token.
  🗨 "LLM vẫn là language model, chỉ khác ở quy mô."
  🎓 Không có ngưỡng chính thức cho chữ "large"; Qwen2.5-1.5B là LLM cỡ nhỏ.
- ★ **Chatbot** — Ứng dụng hội thoại xây trên LLM đã post-training, thêm chat template, giao diện, lớp an toàn.
  🗨 "Chatbot là ứng dụng, còn LLM là bộ não bên trong."
  🎓 Nói được 3 khác biệt: post-training, định dạng hội thoại, lớp an toàn / giao diện.
- **Prompt** — Toàn bộ văn bản đưa vào model để nó viết tiếp.
  🗨 "Chỉ cần đổi prompt là model đổi tác vụ."
  🎓 Prompt thật model nhận gồm cả chat template, không chỉ câu người dùng gõ.
- ★ **Chat template, system prompt** — Khuôn bọc tin nhắn: `<|im_start|>system … <|im_start|>user … <|im_start|>assistant`; system prompt là lời dặn chung đặt đầu.
  🗨 "Câu hỏi được bọc thành một đoạn hội thoại, rồi model viết tiếp phần của assistant."
  🎓 Đọc được chuỗi trong khung ĐẦU VÀO; biết `add_generation_prompt=True` mở sẵn lượt assistant.
- **Post-training** — Các giai đoạn huấn luyện sau pretraining (SFT, alignment) để model làm theo chỉ dẫn.
  🗨 "Model đã qua post-training nên biết trả lời câu hỏi."
  🎓 Chi tiết ở mục 5.
- **Luật chấm ghi trước** — Tiêu chí đúng / sai được viết trước khi xem output.
  🗨 "Luật chấm em ghi trước khi chạy, để tránh chấm theo cảm tính."
  🎓 Đây là thói quen thực nghiệm: không sửa tiêu chí sau khi thấy kết quả.

✅ **Các ý bắt buộc phải nói ở slide này (theo thứ tự):**
1. Language model = xác suất của chuỗi = tích xác suất token tiếp theo (đọc công thức).
2. LLM = language model ở quy mô lớn (Transformer, tỉ tham số, nghìn tỉ token).
3. Chatbot = ứng dụng: post-training + chat template + giao diện, an toàn.
4. Demo 4 tác vụ, 1 model, chỉ đổi prompt; đạt 4/4; còn lỗi trả lời tiếng Trung.

📐 **Lý thuyết chi tiết (đọc để hiểu sâu; khi nói chỉ lấy ý, không đọc nguyên văn):**

**Language model.** Cho một chuỗi token $x_1, \dots, x_n$, language model cho biết xác suất của cả chuỗi. Theo **quy tắc nhân xác suất** (chain rule), xác suất đồng thời luôn tách được thành tích các xác suất có điều kiện:
$$P(x_1, \dots, x_n) = P(x_1)\,P(x_2 \mid x_1)\,P(x_3 \mid x_1, x_2)\cdots = \prod_{t=1}^{n} P(x_t \mid x_{<t})$$
Vì vậy học language model **tương đương** học đoán token tiếp theo. Lấy log để biến tích thành tổng (tránh số quá nhỏ): $\log P(x_{1:n}) = \sum_t \log P(x_t \mid x_{<t})$.

**Từ LM cổ điển đến LLM.** Mô hình n-gram chỉ nhìn $n-1$ token trước (giả định Markov) và đếm tần suất, nên không hiểu ngữ cảnh dài. Mạng nơ-ron (RNN, rồi Transformer) học $P(x_t \mid x_{<t})$ bằng tham số. LLM là Transformer rất lớn, học trên rất nhiều dữ liệu; nhờ quy mô, một model làm được nhiều tác vụ chỉ bằng cách đổi prompt.

**Từ LLM đến chatbot.** Bản base chỉ biết viết tiếp văn bản. Chatbot cần thêm: (1) **post-training** để làm theo chỉ dẫn; (2) **chat template** đánh dấu vai bằng token đặc biệt `<|im_start|>`, `<|im_end|>`, để model biết đâu là lời người dùng, đâu là phần mình trả lời; (3) giao diện, lịch sử hội thoại, bộ lọc an toàn. Khi người dùng hỏi, hệ thống bọc câu hỏi vào template rồi để model **viết tiếp** sau `<|im_start|>assistant`, đến khi sinh `<|im_end|>` thì dừng.

**Vì sao một model làm được nhiều tác vụ.** Hỏi đáp, tóm tắt, dịch, sinh mã đều có dạng "văn bản vào → văn bản ra". Với LLM, cả bốn đều là sinh token tiếp theo với một prompt khác nhau.

🖼 **Sơ đồ lồng nhau (trái):** vòng trong cùng là **language model** (mô hình xác suất của chuỗi token); bọc ngoài là **LLM** (vẫn là LM nhưng rất lớn); ngoài cùng là **chatbot** (LLM + chat template + giao diện + lớp an toàn). 🎤 "Ba khái niệm lồng vào nhau: mọi chatbot dùng một LLM, mọi LLM là một language model, nhưng chiều ngược lại thì không."

🎤 "Bắt đầu với câu hỏi: LLM là gì? Trước hết, một **language model** là model gán xác suất cho một chuỗi văn bản. Theo quy tắc nhân xác suất, xác suất của cả câu tách được thành tích các xác suất có điều kiện:"

🗣 **Đọc công thức** $P(x_1, \dots, x_n) = \prod_{t=1}^{n} P(x_t \mid x_1, \dots, x_{t-1})$:
"Xác suất của cả chuỗi x một đến x n bằng **tích**, với t chạy từ 1 đến n, của **xác suất token thứ t khi đã biết các token đứng trước nó**." ($x_t$ là token thứ t; dấu gạch đứng đọc là "khi biết"; ký hiệu ∏ là "tích".)

🎤 "Nghĩa là, chỉ cần học tốt một việc duy nhất là **đoán token tiếp theo**, model đã có thể chấm điểm hoặc sinh ra cả một câu. **LLM** vẫn giải đúng bài toán đó, nhưng ở quy mô rất lớn: hàng tỉ tham số, kiến trúc Transformer, học trên hàng nghìn tỉ token. Còn **chatbot** như ChatGPT là một **ứng dụng** xây trên LLM: model đã qua post-training để làm theo chỉ dẫn, được bọc trong định dạng hội thoại, giao diện và các lớp an toàn."

📝 **Hiểu thêm:** LM trước đây: n-gram (đếm tần suất cụm từ), RNN. LLM: Transformer + quy mô dữ liệu, tham số. Không có ngưỡng chính thức cho chữ "large"; Qwen2.5-1.5B là LLM cỡ nhỏ.

[Trên slide, không mở notebook — cell 1.1]

💻 **Code** — hàm sinh dùng cho cả phần 1
```python
messages = [{"role": "user", "content": prompt}]
enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None, top_k=None)
new_ids = out[0, n_in:]
text = tok.decode(new_ids, skip_special_tokens=True).strip()
```
- Dòng 1: đóng gói câu hỏi thành **một tin nhắn của người dùng**.
- Dòng 2: **`apply_chat_template` bọc tin nhắn theo định dạng hội thoại của Qwen** rồi đổi thành token ID; `add_generation_prompt=True` mở sẵn lượt "assistant" để model biết đến phiên mình trả lời.
- Dòng 3: **`do_sample=False` là greedy**; ba tham số `None` tắt cấu hình lấy mẫu mặc định.
- Dòng 4: output gồm cả prompt lẫn phần sinh thêm; **bỏ `n_in` token đầu** để chỉ giữ phần model viết.
- Dòng 5: đổi token ID ngược lại thành chữ.

🎤 "Đoạn code này cho thấy rõ chatbot khác LLM ở đâu: câu hỏi của người dùng được bọc thành một đoạn hội thoại có system prompt, có vai user và assistant; sau đó model chỉ việc **viết tiếp** phần của assistant."

📊 Chuỗi model thật sự nhận (in trong khung ĐẦU VÀO):
```
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
Xin chào<|im_end|>
<|im_start|>assistant
```

[Trên slide, không mở notebook — cell 1.2, chỉ khung INPUT / OUTPUT]

📊 **Output thật** (cùng model, cùng greedy, chỉ đổi prompt)

| Tác vụ | Prompt | Output |
|---|---|---|
| Hỏi đáp | Thủ đô của Nhật Bản là gì? Trả lời một câu. | Thủ đô của Nhật Bản là Tokyo. |
| Tóm tắt | (đoạn 5 câu) Tóm tắt đoạn trên trong 1 câu. | 1 câu 104 token, giữ đủ 5 ý, gần như ghép các câu gốc |
| Dịch | Dịch sang tiếng Anh: Hôm nay trời rất đẹp. | Today the weather is very beautiful. |
| Sinh mã | Viết hàm Python is_even(n)… | `def is_even(n): return n % 2 == 0` + ví dụ |

💻 Cell 1.3: `extract_code()` tách khối code trong câu trả lời, `exec` chạy rồi gọi `is_even(0), is_even(1), is_even(2), is_even(-3)` → `[True, False, True, False]`. Cell 1.4 chấm theo **luật ghi trước khi chạy** → **đạt 4/4**.

🎤 "Với cùng một model và cùng cách sinh, chỉ cần đổi prompt là model chuyển từ hỏi đáp sang tóm tắt, dịch và viết code. Đoạn code model viết ra, em chạy thử thật, cho đúng cả 4 trường hợp. Lý do một model làm được nhiều việc như vậy là vì **mọi tác vụ đều quy về sinh token tiếp theo**. Tuy nhiên model chưa hoàn hảo: với lời chào 'Xin chào', model lại trả lời bằng **tiếng Trung**. Em sẽ quay lại hiện tượng này ở phần hạn chế."

🔎 **Đọc demo trên slide (INPUT → OUTPUT → vì sao):** Mỗi hàng: **INPUT** là câu người dùng gõ (code bọc thêm chat template); **OUTPUT** là phần model viết sau `<|im_start|>assistant`, greedy. Sinh mã: code model viết được **chạy thật**, `is_even(0, 1, 2, −3)` → True, False, True, False. Luật chấm ghi trước khi chạy → **đạt 4/4**. Vì sao một model làm được cả 4 việc: cả 4 đều quy về sinh token tiếp theo, chỉ khác prompt.

❓ Vì sao trả lời tiếng Trung? → Model 1.5B học nhiều dữ liệu tiếng Trung, Anh; với prompt rất ngắn nó dễ trượt sang ngôn ngữ chiếm ưu thế [suy luận]. ❓ `exec` có an toàn? → Không phải môi trường cô lập; phải đọc code trước khi chạy.

---

### Slide 4 — Mục 2 (1/2): Sơ đồ kiến trúc — Transformer, decoder-only, causal attention (1.0 phút)

🎯 **Cô yêu cầu:** "Vai trò của Transformer; tập trung vào **Decoder-only và causal attention**". → Ba sơ đồ trên slide + cell 2.1 (thông số, đếm tham số), 2.2 (mask). Không mở notebook: code ngắn nằm trên slide.

🧭 **Trọng tâm lý thuyết (nắm chắc trước khi nói; quên lời thì nói đúng các ý này):**
- **Vai trò của Transformer:** self-attention nối **mọi cặp token trong một bước** bằng phép nhân ma trận → tính song song cả câu, nối được token xa (RNN phải đi tuần tự từng bước).
- **Decoder-only:** chỉ gồm các decoder block xếp chồng (Qwen: 28 block), mỗi block = causal self-attention + FFN + residual; đầu ra là **xác suất token tiếp theo** → hợp để **sinh** văn bản.
- **Causal attention** = attention + **mask che tương lai**: mỗi token chỉ nhìn chính nó và token trước. Lý do: huấn luyện song song mà không "nhìn trộm" đáp án; khi sinh, tương lai chưa có.
- Công thức một câu: điểm liên quan QKᵀ/√d → cộng mask (−∞ ở ô tương lai) → softmax thành trọng số → trộn V.

📖 **Thuật ngữ của slide này** (★ = bắt buộc nắm; mỗi từ: **là gì** · 🗨 **nói thế nào** · 🎓 **cần học / hiểu tới đâu**)
- ★ **Transformer** — Kiến trúc mạng nơ-ron (Vaswani et al. 2017) dựa trên self-attention, không đọc tuần tự như RNN.
  🗨 "Transformer cho mọi token nhìn nhau trong một bước, nên tính song song được."
  🎓 Nói được vai trò: song song hoá, nối token xa, mở rộng được lên tỉ tham số.
- **RNN** — Mạng hồi quy: đọc từng token theo thứ tự, mang một trạng thái ẩn đi qua từng bước.
  🗨 "RNN phải đọc lần lượt, token đầu đi qua nhiều bước mới tới cuối."
  🎓 Chỉ cần biết để so sánh: tuần tự, khó song song, dễ quên thông tin xa.
- ★ **Self-attention** — Mỗi token tính mức liên quan với mọi token khác rồi lấy tổng có trọng số thông tin của chúng.
  🗨 "Self-attention cho mỗi token tự chọn nên chú ý vào token nào."
  🎓 Thuộc công thức softmax(QKᵀ/√d + M)·V và nghĩa từng phần.
- ★ **Query, Key, Value (Q, K, V)** — 3 vector tạo từ mỗi token bằng 3 ma trận học được: Q = token đang tìm gì, K = token có gì, V = nội dung được lấy.
  🗨 "Q hỏi, K trả lời 'tôi có gì', điểm Q·K cao thì lấy nhiều V của token đó."
  🎓 Hiểu QKᵀ là bảng điểm n × n giữa mọi cặp token.
- **Head, multi-head, GQA** — Một head là một bộ Q, K, V; nhiều head chạy song song, mỗi head học một kiểu quan hệ. GQA: nhiều head Q dùng chung K, V.
  🗨 "Qwen có 12 head Q dùng chung 2 cặp K, V để tiết kiệm bộ nhớ."
  🎓 Biết số liệu 12 / 2, d = 128 = 1,536 / 12.
- ★ **Decoder-only** — Kiến trúc chỉ gồm các decoder block có causal mask, xếp chồng; GPT, Qwen thuộc họ này.
  🗨 "Decoder-only chỉ nhìn quá khứ, nên phù hợp để sinh văn bản."
  🎓 So được với encoder-only (BERT, nhìn 2 chiều, để hiểu) và encoder–decoder (T5, dịch).
- ★ **Causal mask, causal attention** — Ma trận che: token chỉ được nhìn chính nó và token trước; ô tương lai cộng −∞ nên sau softmax = 0.
  🗨 "Causal attention là attention có che tương lai: mỗi token chỉ nhìn được những gì đứng trước nó."
  🎓 Trả lời được 2 lý do che: huấn luyện song song không nhìn trộm đáp án; khi sinh tương lai chưa có.
- **Residual (cộng tắt)** — Cộng đầu vào của khối vào đầu ra: x + f(x).
  🗨 "Cộng residual giữ lại thông tin cũ, giúp mạng sâu 28 tầng vẫn học được."
  🎓 Ý tưởng giống ResNet; không cần chứng minh.
- **RMSNorm** — Chuẩn hoá độ lớn vector (chia cho căn trung bình bình phương) trước mỗi khối.
  🗨 "RMSNorm giữ các con số không quá lớn hay quá nhỏ."
  🎓 Khác LayerNorm: không trừ trung bình.
- ★ **FFN (SwiGLU)** — Mạng 2 lớp xử lý riêng từng token: 1,536 → 8,960 → 1,536; SwiGLU là loại hàm kích hoạt có cổng.
  🗨 "FFN chiếm 75% tham số, xử lý thông tin của từng token sau khi đã trộn bằng attention."
  🎓 Nhớ tỉ lệ 75 / 15 / 10 (FFN / embedding / attention).
- **RoPE** — Cách đưa thông tin vị trí: xoay vector Q, K theo vị trí của token.
  🗨 "RoPE cho model biết thứ tự các token."
  🎓 Chỉ cần biết: attention tự nó không biết thứ tự, cần mã hoá vị trí.
- ★ **LM head** — Ma trận đổi vector 1,536 chiều thành 151,936 điểm, mỗi điểm một token. Qwen dùng chung với ma trận embedding.
  🗨 "LM head chấm điểm cho từng token trong từ điển."
  🎓 Biết đầu ra của LM head là logits, chưa phải xác suất.

✅ **Các ý bắt buộc phải nói ở slide này (theo thứ tự):**
1. Vai trò Transformer: self-attention so mọi cặp token trong một bước → song song, nối token xa (so với RNN).
2. Decoder-only: 28 block, mỗi block causal attention + FFN + residual; đầu ra là xác suất token tiếp theo.
3. Causal attention: đọc công thức softmax(QKᵀ/√d + M)V; mask −∞ ở ô tương lai.
4. Hai lý do phải che tương lai; vì vậy chỉ vector vị trí cuối dùng để đoán.
5. Tham số: FFN 75%, embedding 15%, attention 10%.

📐 **Lý thuyết chi tiết (đọc để hiểu sâu; khi nói chỉ lấy ý, không đọc nguyên văn):**

**Vai trò của Transformer.** RNN tính trạng thái ẩn tuần tự $h_t = f(h_{t-1}, x_t)$: muốn có $h_{100}$ phải tính xong $h_{99}$, nên không song song được và thông tin xa bị "pha loãng" qua nhiều bước. Transformer (Vaswani et al. 2017) thay bằng **self-attention**: mọi cặp token được so trực tiếp bằng phép nhân ma trận, cả câu tính một lần trên GPU. Nhờ vậy có thể huấn luyện model rất lớn trên rất nhiều dữ liệu, và đó là nền tảng của LLM.

**Self-attention từng bước** (một head, câu $n$ token, mỗi token vector $d_{model} = 1{,}536$ chiều, xếp thành ma trận $X$ kích thước $n \times 1{,}536$):
1. Tạo 3 ma trận: $Q = XW_Q$, $K = XW_K$, $V = XW_V$, mỗi hàng là vector $d = 128$ chiều của một token.
2. Bảng điểm $S = QK^\top/\sqrt{d}$, kích thước $n \times n$: ô $(i, j)$ là mức token $i$ "quan tâm" token $j$.
3. Cộng mặt nạ $M$: $M_{ij} = 0$ nếu $j \le i$, $M_{ij} = -\infty$ nếu $j > i$ (token tương lai).
4. Softmax theo từng hàng: $A = \text{softmax}(S + M)$; mỗi hàng là các trọng số không âm, cộng bằng 1; ô tương lai thành $e^{-\infty} = 0$.
5. Đầu ra $AV$: hàng $i$ là **trung bình có trọng số** các vector $V$ của token $1..i$.

**Vì sao chia $\sqrt{d}$.** Nếu các thành phần của $q$ và $k$ có phương sai khoảng 1 thì tích vô hướng của hai vector $d$ chiều có phương sai khoảng $d$, tức độ lệch chuẩn $\sqrt{d}$. Chia cho $\sqrt{d}$ đưa điểm về cỡ 1, để softmax không bị quá nhọn (một ô gần 1, còn lại gần 0) làm việc học kém đi.

**Multi-head và GQA.** Qwen chạy 12 head song song, mỗi head một bộ $W_Q$, có thể học một kiểu quan hệ khác nhau (vị trí gần, chủ ngữ – động từ…); kết quả nối lại rồi nhân $W_O$. GQA (grouped-query attention): 12 head Q chia thành 2 nhóm, mỗi nhóm 6 head dùng chung một cặp K, V → bộ nhớ K, V giảm 6 lần.

**Một decoder block** (pre-norm): $h = x + \text{Attn}(\text{RMSNorm}(x))$, rồi $y = h + \text{FFN}(\text{RMSNorm}(h))$.
- **RMSNorm** chia vector cho căn trung bình bình phương các thành phần (nhân thêm hệ số học được) để giữ độ lớn ổn định.
- **Residual** (cộng $x$ vào) cho gradient đi thẳng qua 28 tầng, giúp mạng sâu học được.
- **FFN SwiGLU**: $W_2\big(\text{SiLU}(W_1 x) \odot W_3 x\big)$ với $1{,}536 \to 8{,}960 \to 1{,}536$; xử lý riêng từng token.
- **RoPE** xoay $q, k$ theo vị trí để điểm attention phụ thuộc khoảng cách giữa hai token (attention tự nó không biết thứ tự).

**Đếm lại tham số (khớp cell 2.1).** Attention mỗi block: $W_Q$, $W_O$ cỡ $1{,}536 \times 1{,}536$; $W_K$, $W_V$ cỡ $1{,}536 \times 256$ (2 head × 128) → $5{,}505{,}024$ + 2,048 bias = 5,507,072; nhân 28 = **154,198,016**. FFN mỗi block: $3 \times 1{,}536 \times 8{,}960 = 41{,}287{,}680$; nhân 28 = **1,156,055,040**. Embedding: $151{,}936 \times 1{,}536 = $ **233,373,696**.

**Ba họ Transformer.** Encoder-only (BERT): không có mask, mỗi token nhìn cả hai phía → tốt để **hiểu** (phân loại, trích xuất). Encoder–decoder (Transformer gốc, T5): encoder đọc câu nguồn, decoder sinh câu đích → tốt cho **dịch, tóm tắt**. Decoder-only (GPT, Qwen): chỉ causal attention → mục tiêu huấn luyện và cách dùng là **một**: đoán token tiếp theo, nên rất hợp để **sinh**.

🖼 **Sơ đồ trái — vai trò của Transformer:** hàng trên là RNN: token đi lần lượt Hôm → nay → trời → rất, thông tin của "Hôm" phải qua 3 bước mới tới cuối. Hàng dưới là self-attention: **mọi cặp token nối trực tiếp** trong một phép nhân ma trận. Bảng ba họ: ô xanh = được nhìn, ô xám = bị chặn; encoder nhìn đủ 4 × 4, decoder chỉ nhìn tam giác dưới.

🎤 "Trước khi chạy model, em giải thích kiến trúc qua ba sơ đồ. Thứ nhất, **vai trò của Transformer**. Mô hình cũ như RNN đọc câu tuần tự từng từ, từ sau phải chờ từ trước, nên khó tính song song và khó nhớ thông tin ở xa. Transformer thay bằng **self-attention**: mỗi token nhìn thẳng vào các token khác trong cùng một phép nhân ma trận, nên GPU tính được cả câu cùng lúc."

🗣 **Đọc công thức** $\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}} + M\right)V$:
"Attention của Q, K, V bằng **softmax của Q nhân K chuyển vị, chia căn bậc hai của d, cộng mặt nạ M**; kết quả đó **nhân với V**."
- Q (query): mỗi token "đang tìm thông tin gì"; K (key): mỗi token "có thông tin gì"; V (value): "nội dung sẽ được chuyển đi".
- $QK^\top$: bảng điểm mức liên quan giữa mọi cặp token. Chia $\sqrt{d}$ (d = 128) để điểm không quá lớn. M: **causal mask**.
- Softmax theo từng hàng cho ra **trọng số chú ý**; nhân với V nghĩa là mỗi token nhận một **tổng có trọng số** thông tin của các token nó được phép nhìn.

🎤 "Với model **decoder-only** như Qwen, mặt nạ M chính là **causal mask**: các ô ứng với token tương lai được cộng âm vô cùng, nên sau softmax chúng bằng 0. Nói cách khác, mỗi token chỉ được nhìn chính nó và các token đứng trước."

📝 **Hiểu thêm:**
- **Một decoder block** (lặp 28 lần): `x → RMSNorm → causal self-attention → + x → RMSNorm → FFN → +`. Cộng tắt (residual) giữ thông tin cũ và giúp huấn luyện mạng sâu; RMSNorm chuẩn hoá độ lớn vector; FFN xử lý riêng từng token và chiếm 75% tham số.
- **Ba họ Transformer:** encoder-only (BERT, nhìn hai chiều, hợp để hiểu văn bản), encoder-decoder (T5, hợp để dịch), **decoder-only** (GPT, Qwen, nhìn một chiều, hợp để **sinh** văn bản).

🖼 **Sơ đồ giữa — decoder-only của Qwen2.5-1.5B (đọc từ trên xuống):** token ID → embedding (tra hàng) → **28 block** giống nhau, mỗi block: RMSNorm → causal self-attention → cộng residual → RMSNorm → FFN SwiGLU → cộng residual → RMSNorm cuối, lấy vector **vị trí cuối** → LM head (dùng chung ma trận E với embedding) → softmax. Số bên phải mỗi khối là tham số thật đếm ở cell 2.1.

🎤 "Thứ hai là sơ đồ một model decoder-only cụ thể, chính là Qwen em dùng. Phần lớn tham số không nằm ở attention mà ở **FFN, 75%**; attention chỉ 10%, embedding 15%. Cộng residual giữ lại thông tin cũ để mạng 28 tầng vẫn huấn luyện được."

💻 **Code trên slide (cell 2.1)**
```python
for name, p in model.named_parameters():
    key = ("embedding" if "embed_tokens" in name
           else "attention" if "self_attn" in name
           else "ffn" if "mlp" in name else "norm")
    PARAM_GROUPS[key] += p.numel()   # tổng 1,543,714,304
```
- Dòng 1: **duyệt từng ma trận tham số** của model, kèm tên của nó (ví dụ `model.layers.0.self_attn.q_proj.weight`).
- Dòng 2–4: **xếp nhóm theo tên**: chứa `embed_tokens` là bảng embedding; `self_attn` là attention (W_Q, W_K, W_V, W_O); `mlp` là FFN; còn lại là RMSNorm.
- Dòng 5: `p.numel()` = số phần tử của ma trận; **cộng dồn vào nhóm**. Bốn nhóm cộng lại đúng 1,543,714,304.
- Vì sao không có nhóm LM head: `tie_word_embeddings = True`, LM head **dùng chung** ma trận embedding nên chỉ đếm một lần.

💻 **Code trên slide (cell 2.2):** `M = torch.tril(torch.ones(n, n))`: bảng n × n toàn số 1, `tril` giữ tam giác dưới (kể cả đường chéo) → 1 = được nhìn, 0 = bị chặn. Đây là quy tắc; trong model, ô 0 được đổi thành −∞ trước softmax.

📝 **Hiểu thêm — tham số nằm ở đâu (cell 2.1):** FFN 1,156,055,040 (75%), embedding 233,373,696 = 151,936 × 1,536 (15%), attention 154,198,016 (10%). **GQA:** 12 head Q dùng chung 2 cặp K, V (mỗi 6 head Q một cặp) để giảm bộ nhớ khi sinh.

🖼 **Sơ đồ phải — causal attention trong 1 head:** X nhân 3 ma trận W thành Q, K, V → bảng điểm QKᵀ/√d (n × n) → **cộng M** (−∞ ở ô tương lai) → softmax → nhân V. Bảng 4 × 4 "Hôm nay trời rất": ✓ là trọng số > 0, ô 0 là bị chặn; hàng "trời" chỉ trộn thông tin của Hôm, nay, trời.

🎤 "Thứ ba là bên trong causal attention. Vì sao phải che tương lai? Khi huấn luyện, mọi vị trí cùng đoán token kế tiếp **song song**; nếu không che, vị trí 'trời' sẽ nhìn thấy luôn 'rất', tức là nhìn trộm đáp án. Khi sinh, token tương lai chưa tồn tại. Hệ quả: chỉ **vector ở vị trí cuối** đã thấy toàn bộ câu, nên nó được dùng để đoán token tiếp theo. Slide sau em chạy đúng sơ đồ này trên model thật."

🔎 **Đọc demo trên slide (INPUT → OUTPUT → vì sao):** Slide này là **sơ đồ lý thuyết**; số thật duy nhất là số tham số theo nhóm (cell 2.1): FFN 1,156,055,040 (75%), embedding 233,373,696 = 151,936 × 1,536 (15%), attention 154,198,016 (10%), cộng lại 1,543,714,304 (cộng thêm 87,552 tham số norm). Bảng 4 × 4 là **ví dụ minh họa** cho hình mask thật ở slide 5.

❓ RMSNorm khác LayerNorm? → Chỉ chia cho căn trung bình bình phương, không trừ trung bình; rẻ hơn. ❓ RoPE là gì? → Mã hóa vị trí bằng cách xoay Q, K theo vị trí, nên điểm attention phụ thuộc khoảng cách giữa hai token. ❓ GQA? → 12 head Q dùng chung 2 cặp K, V để giảm bộ nhớ KV cache.

---

### Slide 5 — Mục 2 (2/2): Từ token đầu vào đến xác suất token tiếp theo (1.0 phút)

🎯 **Cô yêu cầu:** "Vai trò của Transformer; tập trung vào Decoder-only và causal attention" — minh họa "**sơ đồ từ token đầu vào đến xác suất token tiếp theo**". → Slide 4 giải thích kiến trúc; slide này là **minh họa** đề yêu cầu, chạy trên model thật: sơ đồ + cell 2.2, 2.3, 2.4. Phần lý thuyết attention đã nói ở slide 4, ở đây chỉ nhắc lại một câu.

🧭 **Trọng tâm lý thuyết (nắm chắc trước khi nói; quên lời thì nói đúng các ý này):**
- Đây là **minh họa đề yêu cầu**: một câu đi qua đúng sơ đồ slide 4 trên model thật: văn bản → 6 token → 6 vector 1,536 chiều → 28 block → **vector vị trí cuối** → 151,936 logits → softmax.
- Ba bằng chứng nối nhau: **quy tắc** che tương lai (2.2) → model thật **tuân theo** (2.3, trên đường chéo = 0) → **đầu ra là phân phối xác suất** (2.4).
- Chọn token nào từ phân phối là việc của mục 6, không phải của kiến trúc.

📖 **Thuật ngữ của slide này** (★ = bắt buộc nắm; mỗi từ: **là gì** · 🗨 **nói thế nào** · 🎓 **cần học / hiểu tới đâu**)
- ★ **Logits** — Điểm thô (chưa chuẩn hoá) cho mỗi token; có thể âm, không cộng bằng 1.
  🗨 "Model cho 151,936 điểm số, gọi là logits."
  🎓 Logits → softmax → xác suất. Temperature tác động lên logits (mục 6).
- ★ **Softmax** — Hàm biến điểm thành xác suất: e mũ từng điểm chia tổng e mũ mọi điểm.
  🗨 "Softmax biến điểm số thành xác suất dương, cộng lại bằng 1."
  🎓 Thuộc công thức, biết điểm lớn hơn → xác suất lớn hơn, và mọi xác suất > 0.
- ★ **Phân phối xác suất** — Danh sách xác suất cho mọi khả năng, cộng bằng 1.
  🗨 "Đầu ra của LLM không phải một từ mà là một phân phối xác suất."
  🎓 Ý quan trọng nhất của mục 2; nối sang mục 6 (chọn token từ phân phối).
- **Layer, head (trong hình)** — Layer = block thứ mấy (0–27); head = head thứ mấy trong block.
  🗨 "Hình là layer 0, head 0, một trong 336 head."
  🎓 Không khái quát vai trò từ một head.
- **Eager, SDPA** — Hai cách tính attention: SDPA là hàm gộp nhanh, không trả trọng số; eager tính từng bước nên lấy được trọng số.
  🗨 "Em nạp bản eager để lấy ra được ma trận attention."
  🎓 Chỉ cần nếu bị hỏi vì sao cell 2.3 nạp model lần nữa.
- **`topk`** — Lấy k phần tử lớn nhất.
  🗨 "Đây là 5 token có xác suất cao nhất."
  🎓 Đừng nhầm với top-k sampling ở mục 6 (ở đây chỉ là để hiển thị).

✅ **Các ý bắt buộc phải nói ở slide này (theo thứ tự):**
1. Đi theo sơ đồ: 6 token → 6 vector → 28 block → vị trí cuối → 151,936 logits → softmax.
2. Mask quy tắc (2.2) → attention thật tuân theo (2.3) → phân phối xác suất (2.4).
3. Đầu ra là phân phối, không phải một từ: " đẹp" 0.788, " nắng" 0.063, " lạnh" 0.046.

📐 **Lý thuyết chi tiết (đọc để hiểu sâu; khi nói chỉ lấy ý, không đọc nguyên văn):**

**Một lượt forward, theo kích thước tensor** (câu "Hôm nay trời rất", 6 token):
token ID $(1, 6)$ → embedding $(1, 6, 1{,}536)$ → 28 block, kích thước giữ nguyên $(1, 6, 1{,}536)$ → RMSNorm → LM head $(1, 6, 151{,}936)$ → lấy vị trí cuối `[0, -1]` → vector $(151{,}936)$ → softmax.

**Vì sao chỉ dùng vị trí cuối.** Nhờ causal mask, vector ở vị trí $t$ chỉ chứa thông tin của token $1..t$, và được huấn luyện để đoán token $t+1$. Vị trí cuối là vị trí duy nhất đã thấy cả câu, nên logits ở đó là dự đoán cho token tiếp theo. (Các vị trí khác vẫn có logits; mục 4 dùng chúng để tính loss mọi vị trí cùng lúc.)

**Logits và softmax.** Logits $z_i$ là điểm thô, có thể âm. Softmax $p_i = e^{z_i}/\sum_j e^{z_j}$ có 3 tính chất: mọi $p_i > 0$; tổng bằng 1; giữ thứ tự (logit lớn hơn → xác suất lớn hơn). Cộng cùng một hằng số vào mọi logit không đổi kết quả, nên chỉ **chênh lệch** giữa các logit là quan trọng.

**Đọc output.** " đẹp" 0.788 nghĩa là trong mọi cách viết tiếp có thể, model đặt 78.8% khả năng vào " đẹp". 5 token đầu cộng 0.933; 6.7% còn lại rải trên hơn 151 nghìn token khác. Đầu ra là một **phân phối**; chọn token nào là việc của cách sinh (mục 6).

**Attention thật.** Hình 2.3 lấy ma trận $A$ của layer 0, head 0. Kiểm tra được 2 tính chất của lý thuyết: phía trên đường chéo bằng 0 (mask) và mỗi hàng cộng bằng 1 (softmax).

🖼 [Chỉ sơ đồ trên slide]

🎤 "Bây giờ em chạy đúng kiến trúc đó trên model thật, đi theo sơ đồ từ trái sang phải. Câu 'Hôm nay trời rất' được **tokenizer** cắt thành 6 token ID. Mỗi ID được tra trong bảng **embedding** thành một vector 1,536 chiều. Sáu vector này đi qua **28 decoder block** giống nhau. Ở đầu ra, model lấy vector tại **vị trí cuối cùng**, nhân với ma trận **LM head** để được **151,936 điểm số**, mỗi điểm ứng với một token trong từ điển. Cuối cùng, hàm **softmax** biến các điểm này thành xác suất."

🗣 **Đọc công thức** $p_i = \dfrac{e^{z_i}}{\sum_j e^{z_j}}$:
"Xác suất của token i bằng **e mũ điểm của token i, chia cho tổng e mũ điểm của tất cả các token** trong từ điển." ($z_i$: điểm thô, còn gọi là logit; mẫu số làm cho các xác suất cộng lại bằng 1.)

[Mở cell 2.2]

💻 `torch.tril(torch.ones(n, n, dtype=torch.bool))`
- `ones(n, n)`: bảng toàn số 1; **`tril` giữ phần tam giác dưới** (kể cả đường chéo), xoá phần trên thành 0.

🖼 **Hình `causal_mask.png`:** bảng 8 × 8 cho câu "Hôm nay trời rất đẹp." (8 token). Hàng là token đang xét, cột là token được nhìn; ô bằng 1 là được nhìn.

🎤 "Đây là causal mask của câu ví dụ: một ma trận tam giác dưới. Hàng của token 'trời' chỉ có số 1 ở các cột 'H, ôm, n, ay, trời'. Lý do phải chặn tương lai: model được huấn luyện để đoán token tiếp theo; nếu được nhìn thấy token phía sau thì nó chỉ việc chép đáp án."

[Mở cell 2.3]

💻 **Code**
```python
model_eager = AutoModelForCausalLM.from_pretrained(MODEL_INSTRUCT, ..., attn_implementation="eager")
out_eager = model_eager(ids, output_attentions=True)
attn = out_eager.attentions[0][0, 0]          # layer 0, batch 0, head 0 → bảng 8 × 8
assert torch.all(attn.triu(diagonal=1) == 0)  # phần tương lai = 0
```
- **Vì sao phải nạp bản `eager`:** cách tính mặc định (SDPA) là một hàm gộp tối ưu, không giữ lại ma trận trọng số; bản eager tính đúng từng bước của công thức nên trả được trọng số.
- `assert`: mọi ô phía trên đường chéo phải bằng 0.

🖼 **Hình `attention_qwen.png`** (layer 0, head 0): mỗi ô là trọng số chú ý của token hàng dành cho token cột. Phía trên đường chéo **toàn 0.00**; mỗi hàng cộng lại bằng 1, ví dụ hàng "ay": 0.18 + 0.37 + 0.08 + 0.37 = 1.00.

🎤 "Còn đây là trọng số attention **thật**, lấy ra từ bên trong Qwen: toàn bộ phía trên đường chéo đều bằng 0, và mỗi hàng cộng lại đúng bằng 1 vì đã qua softmax. Như vậy causal mask không chỉ là lý thuyết mà có tác dụng thật trong model."

[Mở cell 2.4]

💻 **Code**
```python
logits = model(ids).logits[0, -1].float()   # điểm ở vị trí cuối, 151,936 số
probs = torch.softmax(logits, dim=-1)       # điểm → xác suất
top = torch.topk(probs, 5)                  # 5 token cao nhất
```
- **`[0, -1]` lấy vị trí cuối cùng**: vector ở đó dùng để đoán token tiếp theo.

📊 **Output + 🖼 `top5_next_tokens.png`:** " đẹp" 0.788, " nắng" 0.063, " lạnh" 0.046, " nóng" 0.029, " " 0.007; năm token cộng lại 0.933.

🎤 "Kết thúc sơ đồ, sau 'Hôm nay trời rất', model cho ' đẹp' xác suất 0.788, ' nắng' 0.063, ' lạnh' 0.046. Điều cần nhấn mạnh là đầu ra của LLM **không phải một từ**, mà là **một phân phối xác suất trên toàn bộ 151,936 token**. Chọn từ nào trong phân phối đó là việc của bước sinh văn bản, em trình bày ở mục 6."

🔎 **Đọc demo trên slide (INPUT → OUTPUT → vì sao):** **INPUT:** "Hôm nay trời rất" dạng văn bản thô (không chat template, vì muốn xem model **viết tiếp câu**). **OUTPUT:** " đẹp" 0.788, " nắng" 0.063, " lạnh" 0.046. Vì sao " đẹp" cao: cụm "trời rất đẹp" rất phổ biến trong dữ liệu huấn luyện. Hình mask: ô 1 = được nhìn. Hình attention: số thật layer 0, head 0; trên đường chéo 0.00, mỗi hàng cộng = 1.

❓ Không có causal mask thì sao? → Khi huấn luyện, model thấy luôn token cần đoán, học chép chứ không học dự đoán. ❓ Vì sao chia √d? → Tích vô hướng lớn dần theo số chiều; chia để softmax không quá nhọn. ❓ 1 head nói lên gì? → Chỉ là 1 trong 28 × 12 = 336 head; không khái quát vai trò từ 1 head.

---

### Slide 6 — Mục 3: Tokenization (1.2 phút)

🎯 **Cô yêu cầu:** "Cách văn bản được chia thành token; token ID và embedding" — minh họa "**tokenize một câu tiếng Việt và một câu tiếng Anh**". → Cell 3.1, 3.1b, 3.2, 3.4.

🧭 **Trọng tâm lý thuyết (nắm chắc trước khi nói; quên lời thì nói đúng các ý này):**
- **Token** = mảnh văn bản model đọc. **Byte-level BPE:** đổi chữ thành byte UTF-8, bắt đầu từ 256 byte, lặp lại việc **gộp cặp hay đi cạnh nhau nhất** thành token mới.
- **Token ID** = số thứ tự của token trong từ điển, **không mang nghĩa**.
- **Embedding** = tra hàng thứ ID trong ma trận E (151,936 × 1,536); vector này **học được**, mới là thứ mang nghĩa.
- Tokenizer học từ dữ liệu nào thì giỏi ngôn ngữ đó: câu Việt Qwen **8** token, GPT-2 **22**; câu Anh **7 = 7**. Tốn token → tốn ngữ cảnh, sinh chậm.

📖 **Thuật ngữ của slide này** (★ = bắt buộc nắm; mỗi từ: **là gì** · 🗨 **nói thế nào** · 🎓 **cần học / hiểu tới đâu**)
- ★ **Tokenization, tokenizer** — Việc chia văn bản thành token; tokenizer là công cụ làm việc đó và tra ID.
  🗨 "Tokenizer cắt câu thành token rồi đổi mỗi token thành một con số."
  🎓 Mỗi model đi với đúng một tokenizer của nó.
- ★ **BPE (Byte Pair Encoding), byte-level** — Bắt đầu từ các byte, lặp lại việc gộp cặp đi cạnh nhau nhiều nhất thành token mới; byte-level nghĩa là đơn vị gốc là byte.
  🗨 "BPE gộp dần những cặp hay gặp, nên cụm phổ biến thành một token."
  🎓 Hiểu vì sao không bao giờ gặp ký tự lạ (tệ nhất 1 byte = 1 token).
- ★ **UTF-8, byte** — Cách mã hoá chữ thành byte: chữ không dấu 1 byte, chữ có dấu tiếng Việt 2–3 byte.
  🗨 "Chữ 'ờ' là 3 byte, nên GPT-2 phải tách thành 3 token."
  🎓 Đọc được bảng gom theo từ: byte → token.
- **Subword** — Đơn vị nằm giữa từ và ký tự.
  🗨 "Subword vừa không bị từ mới làm khó, vừa không làm chuỗi quá dài."
  🎓 So 3 cách tách: theo từ, theo ký tự, theo subword.
- ★ **Từ điển (vocabulary)** — Danh sách mọi token model biết, mỗi token một ID.
  🗨 "Từ điển của Qwen có khoảng 151 nghìn token."
  🎓 Bảng embedding có 151,936 hàng (hơi nhiều hơn số token thật).
- ★ **Token ID** — Số thứ tự của token trong từ điển; chỉ dùng để tra bảng, không mang nghĩa.
  🗨 "Token ID chỉ là số thứ tự, như số trang trong từ điển."
  🎓 ID gần nhau không có nghĩa là nghĩa gần nhau.
- ★ **Embedding** — Vector số đại diện cho token, lấy bằng cách tra hàng trong ma trận E; giá trị được học.
  🗨 "Embedding là hàng thứ ID của ma trận embedding; vector này mới mang nghĩa."
  🎓 Thuộc eₜ = E[t, :]; biết kích thước (1, 8) → (1, 8, 1536).
- **Ký hiệu `Ġ`, `␣`, `�`** — `Ġ` / `␣` là dấu cách gắn đầu token; `�` là một byte lẻ không tự thành chữ.
  🗨 "Dấu cách được gắn vào đầu từ phía sau."
  🎓 Đủ để trả lời khi cô hỏi ký tự lạ trên slide.

✅ **Các ý bắt buộc phải nói ở slide này (theo thứ tự):**
1. Byte-level BPE: từ byte, gộp cặp hay gặp nhất, lặp lại.
2. Token ID là số thứ tự; embedding là hàng của ma trận E, vector học được mới mang nghĩa.
3. Tokenize câu Việt và câu Anh: Qwen 8 vs GPT-2 22; Anh 7 = 7; giải thích bằng byte.
4. Hệ quả: tốn token → tốn ngữ cảnh, chậm hơn.

📐 **Lý thuyết chi tiết (đọc để hiểu sâu; khi nói chỉ lấy ý, không đọc nguyên văn):**

**Ba cách tách văn bản.** Theo **từ**: từ điển khổng lồ, gặp từ mới thì không biết. Theo **ký tự**: từ điển nhỏ nhưng câu rất dài, model phải tự ghép nghĩa. **Subword** (BPE) nằm giữa: cụm hay gặp thành một token, từ hiếm tách thành mảnh nhỏ.

**Thuật toán BPE (lúc tạo tokenizer):**
1. Bắt đầu với từ điển gồm 256 byte.
2. Đếm mọi cặp ký hiệu đứng cạnh nhau trong dữ liệu huấn luyện tokenizer.
3. Gộp cặp xuất hiện nhiều nhất thành một ký hiệu mới, thêm vào từ điển, ghi lại **luật gộp** này.
4. Lặp bước 2–3 đến khi từ điển đủ lớn (Qwen: khoảng 151 nghìn token).

**Lúc tokenize một câu:** đổi câu thành byte UTF-8, tách sơ bộ (dấu cách dính vào đầu từ phía sau), rồi áp lần lượt các luật gộp đã học. Vì đơn vị gốc là byte, mọi văn bản đều tokenize được.

**Vì sao số token khác nhau giữa hai tokenizer.** Luật gộp học từ dữ liệu. Dữ liệu của Qwen có nhiều tiếng Việt nên "␣trời" đã được gộp thành 1 token. GPT-2 học chủ yếu tiếng Anh, chưa từng gộp 3 byte `E1 BB 9D` của "ờ", nên mỗi byte là một token: "trời" thành 5 token. Đo bằng số token mỗi từ: Qwen $8/5 = 1.6$, GPT-2 $22/5 = 4.4$. Hệ quả: tốn chỗ trong cửa sổ ngữ cảnh, sinh chậm hơn (mỗi token một lượt chạy model), tốn chi phí hơn.

**Token ID và embedding.** Token ID là chỉ số hàng. Embedding là tra hàng trong ma trận $E$ cỡ $151{,}936 \times 1{,}536$: $e_t = E[t, :]$. Viết dưới dạng toán, đó là nhân vector one-hot với $E$: $e_t = \text{onehot}(t)^\top E$. Ma trận $E$ được học khi huấn luyện, nên các token có cách dùng giống nhau thường có vector gần nhau; còn ID thì chỉ là số thứ tự.

🖼 **Sơ đồ pipeline (phải, trên cùng):** văn bản → UTF-8 byte (29 byte) → BPE gộp các cặp theo bảng merge đã học → 8 token → token ID (tra từ điển) → embedding (tra hàng E). 🎤 "Đây là toàn bộ đường đi từ chữ đến vector; hai bên của slide đi qua đúng các bước này."

🎤 "Ở slide trước, bước đầu tiên của sơ đồ là tokenizer. Model không đọc chữ mà đọc **token**. Qwen và GPT-2 đều dùng **byte-level BPE**: văn bản được đổi thành các byte UTF-8; tokenizer bắt đầu từ 256 byte cơ bản, rồi lặp đi lặp lại việc **gộp cặp ký hiệu hay đi cạnh nhau nhất** thành một token mới. Cụm nào xuất hiện càng nhiều trong dữ liệu thì càng sớm trở thành một token riêng."

📝 **Hiểu thêm:** tách theo từ thì gặp từ mới là bó tay; tách theo ký tự thì chuỗi quá dài. Subword nằm giữa. Byte-level đảm bảo **không bao giờ gặp ký tự lạ** (tệ nhất mỗi byte là một token). Chữ không dấu chiếm 1 byte; chữ có dấu tiếng Việt chiếm 2–3 byte ("ờ" = `E1 BB 9D`).

[Trên slide, không mở notebook — cell 3.1, chỉ dòng chữ tô màu]

💻 **Code**
```python
ids = tokenizer(text, add_special_tokens=False)["input_ids"]   # câu → token ID
pieces = tokenizer.convert_ids_to_tokens(ids)                   # token ở dạng thô
text = [tokenizer.decode([i]) for i in ids]                     # chữ của từng token
```
- **`add_special_tokens=False`** để không thêm token đặc biệt, đếm công bằng giữa hai tokenizer.
- Cột `pieces` có ký tự lạ (`Ġ`, `Ã´`) vì từ điển lưu mỗi byte bằng một ký tự in được: dấu cách `0x20` hiện thành `Ġ`; "ô" = `C3 B4` hiện thành `Ã´`.

🖼 **Câu tô màu theo token:** mỗi ô là một token, số nhỏ bên dưới là token ID. Qwen: `H`(39) | `ôm`(94365) | `␣n`(308) | `ay`(352) | `␣trời`(130868) | `␣rất`(128323) | `␣đẹp`(128501) | `.`(13). Dấu cách gắn vào **đầu** từ phía sau.

[Trên slide, không mở notebook — cell 3.1b]

📊 **Bảng gom theo từ (output thật)**

| Từ | Byte | Qwen | GPT-2 |
|---|---|---|---|
| Hôm | 4 | 2 (`H`, `ôm`) | 3 |
| nay | 3 | 2 | 2 |
| trời | 6 | **1** | **5** (`␣tr` + 3 byte của "ờ" + `i`) |
| rất | 5 | 1 | 5 |
| đẹp | 6 | 1 | 6 |
| . | 1 | 1 | 1 |
| **Tổng** | 29 | **8** | **22** |

Câu Anh "The weather is very nice today.": mỗi từ 1 token, 7 / 7.

🎤 "Kết quả tokenize hai câu: với câu tiếng Việt, Qwen cần **8 token**, còn GPT-2 cần tới **22**. Chẳng hạn chữ 'trời' là một token với Qwen, nhưng bị GPT-2 cắt thành 5 token. Nguyên nhân là GPT-2 học chủ yếu từ tiếng Anh nên chưa gộp các byte của chữ có dấu, phải để mỗi byte thành một token riêng; những token đó hiện thành dấu '�' vì một byte lẻ không tạo thành chữ. Với câu tiếng Anh, cả hai cùng 7 token. Hệ quả thực tế: tokenizer càng tốn token thì càng chiếm chỗ trong ngữ cảnh và sinh càng chậm, vì mỗi token là một bước sinh."

[Trên slide, không mở notebook — cell 3.4]

💻 `emb_layer = model.get_input_embeddings()`, `vi_emb = emb_layer(vi_ids)`, `assert torch.equal(vi_emb[0], emb_layer.weight[vi_ids[0]])`
- **Embedding của một token là hàng có số thứ tự bằng token ID trong ma trận** 151,936 × 1,536; `assert` kiểm đúng là phép tra hàng.

🗣 **Đọc công thức** $e_t = E[t, :]$:
"Vector embedding của token t **là hàng thứ t của ma trận embedding E**." (E có 151,936 hàng, mỗi hàng 1,536 số.)

🖼 **Hình `embedding_qwen.png`:** 8 hàng ứng với 8 token; mỗi hàng là 16 số đầu (trong 1,536) của vector; đỏ là dương, xanh là âm. Ví dụ token "H" (ID 39) → hàng 39: [0.023, −0.025, 0.003, −0.002, −0.035, …].

🎤 "Token ID bản thân chỉ là một **số thứ tự** dùng để tra bảng, không mang ý nghĩa. Thứ mang ý nghĩa là **vector embedding**: hàng tương ứng trong một ma trận 151,936 × 1,536, có các giá trị được học trong quá trình huấn luyện. Như vậy câu 8 token trở thành một ma trận 8 × 1,536, và đó chính là đầu vào của các block ở slide trước."

📝 **Hiểu thêm:** tra hàng tương đương nhân vector one-hot với E. Qwen dùng chung bảng này cho LM head (`tie_word_embeddings = True`). Từ điển có 151,643 token thường + 22 token đặc biệt; bảng có 151,936 hàng, phần dư không ứng với token nào.

🔎 **Đọc demo trên slide (INPUT → OUTPUT → vì sao):** Ô màu: chữ là token, số nhỏ là token ID; `␣` là dấu cách gắn ở **đầu** token. GPT-2 có ô `�`: một **byte lẻ** của chữ có dấu, không tự thành chữ. Bảng gom theo từ: "trời" 6 byte → Qwen 1 token, GPT-2 5 token. Heatmap: 8 hàng = 8 token, 16 / 1,536 cột đầu, đỏ dương, xanh âm. INPUT token ID (1, 8) → OUTPUT embedding (1, 8, 1536).

❓ `Ġ` là gì? → Dấu cách. ❓ Dùng tokenizer GPT-2 cho Qwen được không? → Không; ID hai bên khác nhau, model học với tokenizer nào thì phải dùng tokenizer đó.

---

### Slide 7 — Mục 4: Nguyên lý học (1.3 phút)

🎯 **Cô yêu cầu:** "Next-token prediction, dữ liệu huấn luyện và cross-entropy loss" — minh họa "**tạo các cặp ngữ cảnh → token mục tiêu từ một câu**". → Cell 4.1–4.5.

🧭 **Trọng tâm lý thuyết (nắm chắc trước khi nói; quên lời thì nói đúng các ý này):**
- **Next-token prediction:** ở mỗi vị trí, đoán token kế tiếp = bài toán phân loại **151,936 lớp**.
- **Dữ liệu huấn luyện** là văn bản thô, **không cần người gán nhãn**: nhãn chính là token kế tiếp (input và nhãn là cùng chuỗi, lệch 1 vị trí).
- **Cross-entropy** = −ln(xác suất model cho token đúng), lấy trung bình; đúng chắc chắn → 0, càng bất ngờ → càng lớn. **Perplexity** = e^loss.
- Một câu n token cho **n − 1 cặp** ngữ cảnh → mục tiêu; nhờ causal mask, chạy model **một lần** là có loss cho mọi cặp.

📖 **Thuật ngữ của slide này** (★ = bắt buộc nắm; mỗi từ: **là gì** · 🗨 **nói thế nào** · 🎓 **cần học / hiểu tới đâu**)
- ★ **Next-token prediction** — Nhiệm vụ huấn luyện: biết các token trước, đoán token kế tiếp.
  🗨 "Model học bằng một việc duy nhất: đoán token tiếp theo."
  🎓 Đây là bài toán phân loại 151,936 lớp ở mỗi vị trí.
- ★ **Tự giám sát (self-supervised)** — Nhãn lấy ngay từ dữ liệu, không cần người gán.
  🗨 "Nhãn chính là token kế tiếp trong văn bản, nên không cần người gán nhãn."
  🎓 Lý do LLM học được trên nghìn tỉ token.
- ★ **Cặp ngữ cảnh → mục tiêu** — Ngữ cảnh = các token trước vị trí i; mục tiêu = token tại i. Câu n token cho n − 1 cặp.
  🗨 "Từ câu 8 token, ta có 7 cặp ngữ cảnh và token mục tiêu."
  🎓 Đây là minh họa đề yêu cầu cho mục 4.
- ★ **Loss, cross-entropy** — Loss là con số đo model sai bao nhiêu; cross-entropy = −ln(xác suất model cho token đúng), lấy trung bình.
  🗨 "Loss bằng trừ log xác suất của token đúng: đoán chắc đúng thì loss bằng 0."
  🎓 Thuộc vài giá trị: p = 1 → 0; p = 0.5 → 0.69; p = 0.0003 → 8.19.
- ★ **Perplexity** — e mũ loss; hiểu như "model phân vân giữa khoảng bao nhiêu token".
  🗨 "Perplexity khoảng 20, tức model phân vân như chọn đều giữa khoảng 20 token."
  🎓 Đoán bừa: perplexity = 151,936.
- **Gradient, backpropagation** — Tính xem mỗi tham số nên tăng hay giảm để loss nhỏ đi, rồi cập nhật.
  🗨 "Huấn luyện là điều chỉnh tham số để loss nhỏ nhất."
  🎓 Notebook không làm bước này; chỉ cần hiểu ý tưởng.
- ★ **Dịch nhãn 1 vị trí (shift)** — Điểm ở vị trí t dùng đoán token t + 1, nên ghép `logits[:-1]` với `ids[1:]`.
  🗨 "Bỏ vị trí cuối của dự đoán và bỏ token đầu của nhãn để hai bên khớp nhau."
  🎓 Giải thích được vì sao 8 token chỉ có 7 loss.
- **`log_softmax`, `gather`** — `log_softmax` = log của softmax nhưng ổn định số hơn; `gather` lấy đúng cột của token mục tiêu ở mỗi hàng.
  🗨 (không cần nói, chỉ giải thích khi bị hỏi)
  🎓 Đọc được từng dòng code trên slide.

✅ **Các ý bắt buộc phải nói ở slide này (theo thứ tự):**
1. Next-token prediction; dữ liệu là văn bản thô, nhãn chính là token kế tiếp (không cần gán nhãn).
2. Từ một câu tạo n − 1 cặp ngữ cảnh → mục tiêu (minh họa đề).
3. Cross-entropy = −ln P(token đúng), lấy trung bình; đọc công thức; perplexity = e^loss.
4. Kết quả: 2.985 khớp thư viện, PPL 19.8, câu xáo trộn 6.70, đoán bừa 11.93.

📐 **Lý thuyết chi tiết (đọc để hiểu sâu; khi nói chỉ lấy ý, không đọc nguyên văn):**

**Next-token prediction là bài toán phân loại.** Ở mỗi vị trí, model chọn 1 trong $V = 151{,}936$ lớp (token). Nhãn đúng là token thật kế tiếp, nên dữ liệu **tự có nhãn** (self-supervised).

**Cross-entropy cho một vị trí.** Nhãn dạng one-hot $y$ (1 ở token đúng, 0 ở chỗ khác), model cho phân phối $p$. Cross-entropy $H(y, p) = -\sum_i y_i \ln p_i$; vì chỉ một $y_i$ bằng 1 nên rút gọn còn $-\ln p_{\text{đúng}}$. Trung bình trên $N$ vị trí:
$$\mathcal{L} = -\frac{1}{N}\sum_{t=1}^{N} \ln p_\theta(x_{t+1} \mid x_{\le t})$$

**Vì sao dùng cross-entropy.** Cực tiểu $\mathcal{L}$ tương đương **cực đại log xác suất** model gán cho dữ liệu thật (ước lượng hợp lý cực đại). Loss phạt rất nặng khi model tự tin sai: $p = 0.0003$ cho loss 8.19, trong khi $p = 0.5$ chỉ 0.69.

**Perplexity.** $\text{PPL} = e^{\mathcal{L}}$, cũng bằng trung bình nhân của $1/p_{\text{đúng}}$. Hiểu như "số lựa chọn ngang nhau mà model đang phân vân". Mốc so sánh: đoán đều trên từ điển thì $p = 1/V$, loss $= \ln V = 11.93$, PPL $= V$.

**Huấn luyện (ý tưởng, notebook không làm).** Tính gradient $\nabla_\theta \mathcal{L}$ bằng lan truyền ngược, rồi cập nhật $\theta \leftarrow \theta - \eta\,\nabla_\theta \mathcal{L}$ ($\eta$: tốc độ học), lặp trên hàng nghìn tỉ token.

**Song song nhờ causal mask (teacher forcing).** Đưa cả câu đúng vào một lần; nhờ mask, logits ở vị trí $t$ chỉ phụ thuộc token $\le t$, nên đó đúng là dự đoán cho $t+1$. Một lượt forward cho loss của **mọi** vị trí, đó là lý do ghép `logits[:-1]` với `ids[1:]`.

**Đọc kết quả.** Loss 2.985 → PPL 19.8 ≪ 151,936: model đã học tiếng Việt. Câu xáo trộn cùng từ nhưng sai trật tự cho 6.70: model học được **trật tự**, không chỉ từ vựng.

🖼 **Sơ đồ dữ liệu huấn luyện (trái, trên hình):** văn bản thô không nhãn (web, sách, code) → token hóa, cắt chuỗi → input x₁…xₙ₋₁ và nhãn x₂…xₙ (cùng chuỗi, **dịch 1 vị trí**) → model cho logits ở mọi vị trí → cross-entropy so với nhãn → backprop cập nhật trọng số, lặp lại. 🎤 "Dữ liệu huấn luyện của pretraining không cần ai gán nhãn: nhãn chính là token kế tiếp trong văn bản. Notebook chỉ làm đến bước tính loss, không cập nhật trọng số."

🎤 "Sau khi biết model nhìn văn bản như thế nào, câu hỏi tiếp theo là model **học** ra sao. Câu trả lời là **next-token prediction**: đoán token tiếp theo. Mỗi câu trong dữ liệu tự sinh ra nhiều bài toán: biết 'H' thì đoán 'ôm', biết 'Hôm' thì đoán ' n', cứ thế đến hết câu. Nhãn có sẵn ngay trong văn bản, **không cần người gán nhãn**; nhờ vậy LLM học được trên lượng dữ liệu khổng lồ gồm web, sách và code."

[Trên slide, không mở notebook — cell 4.1]

💻 `make_context_target_pairs(ids) = [(ids[:i], ids[i]) for i in range(1, len(ids))]`
- `ids[:i]` là **mọi token đứng trước vị trí i** (ngữ cảnh); `ids[i]` là token tại vị trí i (mục tiêu). Câu n token cho n − 1 cặp.

🖼 **Hình `context_target_pairs.png`:** mỗi hàng là một cặp; ô xanh là ngữ cảnh đưa vào model, ô cam là token phải đoán. 8 token → **7 cặp**.

🎤 "Từ câu 8 token, ta có 7 cặp ngữ cảnh và token mục tiêu. Hình này có dạng tam giác, giống hệt causal mask ở mục 2; và đó không phải ngẫu nhiên: nhờ causal mask, chỉ cần chạy model **một lần** là tính được dự đoán cho cả 7 cặp."

🎤 "Để đo model đoán tốt hay kém, người ta dùng **cross-entropy loss**:"

🗣 **Đọc công thức** $\mathcal{L} = -\dfrac{1}{N}\sum_{t} \ln p(x_{t+1} \mid x_1, \dots, x_t)$:
"Loss bằng **trừ trung bình cộng của logarit tự nhiên của xác suất mà model gán cho token đúng**, tính trên mọi vị trí." (N: số cặp; $p(x_{t+1} \mid \dots)$: xác suất model cho token đúng tiếp theo khi biết ngữ cảnh.)

🎤 "Nói đơn giản: ở mỗi vị trí, ta xem model cho token đúng xác suất bao nhiêu rồi lấy trừ log của nó. Đoán đúng chắc chắn, xác suất bằng 1, thì loss bằng 0; xác suất càng nhỏ thì loss càng lớn. **Huấn luyện** chính là điều chỉnh tham số để loss trung bình trên dữ liệu nhỏ nhất. Notebook không huấn luyện, chỉ **tính** loss của model đã huấn luyện."

📝 **Hiểu thêm:**

| $p$ đúng | 1 | 0.788 | 0.5 | 0.1 | 0.0003 |
|---|---|---|---|---|---|
| $-\ln p$ | 0 | 0.24 | 0.69 | 2.30 | 8.19 |

[Trên slide, không mở notebook — cell 4.2]

💻 **Code**
```python
logits = model(ids).logits[0].float()            # (8, 151936): mỗi vị trí một vector điểm
log_probs = F.log_softmax(logits[:-1], dim=-1)   # bỏ vị trí cuối
targets = ids[0, 1:]                             # bỏ token đầu
lp = log_probs.gather(1, targets[:, None])[:, 0] # ln P(token đúng) ở từng vị trí
loss_token = -lp
```
- **`logits[:-1]` đi với `ids[0, 1:]` chính là dịch nhãn sang trái một vị trí**: điểm ở vị trí t là dự đoán cho token t + 1; vị trí cuối không còn token nào để đoán, token đầu thì không có ngữ cảnh.
- `log_softmax` tính log xác suất (ổn định số hơn lấy log của softmax); `gather` lấy đúng cột của token mục tiêu ở mỗi hàng.

📊 **Output + 🖼 `per_token_loss.png`**

| Ngữ cảnh → mục tiêu | P | Loss |
|---|---|---|
| H → ôm | 0.0003 | **8.19** |
| Hôm → n | 0.705 | 0.35 |
| Hôm n → ay | 0.996 | **0.004** |
| Hôm nay → trời | 0.002 | 6.34 |
| Hôm nay trời → rất | 0.029 | 3.55 |
| Hôm nay trời rất → đẹp | 0.788 | 0.24 |
| Hôm nay trời rất đẹp → . | 0.109 | 2.22 |

🎤 "Biểu đồ cho thấy loss từng token. Sau 'Hôm n', model cho 'ay' xác suất 0.996 nên loss chỉ 0.004, vì gần như chắc chắn. Ngược lại, khi chỉ biết mỗi chữ 'H', model cho 'ôm' xác suất 0.0003, loss 8.19, vì sau 'H' có hàng nghìn khả năng. Loss cao vì thế không có nghĩa model kém, mà phản ánh ngữ cảnh còn quá mở."

💻 Cell 4.3: `model(ids, labels=ids).loss` = 2.984804, bằng loss tự tính (chênh 6.3e-08). **Truyền nhãn chưa dịch mà vẫn khớp, chứng tỏ thư viện tự dịch nhãn bên trong.**

🗣 **Đọc công thức** $\text{PPL} = e^{\mathcal{L}}$:
"Perplexity bằng **e mũ loss**." Với loss 2.985: e mũ 2.985 ≈ 19.8.

🎤 "Loss trung bình em tự tính là 2.985, khớp với loss của thư viện. Đổi sang **perplexity** bằng e mũ loss, ta được khoảng 20: model phân vân như đang chọn đều giữa khoảng 20 token, trong khi đoán bừa thì phải chọn giữa 151,936 token. Đặc biệt, khi giữ nguyên các từ nhưng **xáo trộn trật tự**, loss tăng vọt lên 6.70. Điều đó cho thấy model không chỉ học từ vựng, mà đã học được **trật tự** của tiếng Việt."

📝 **Hiểu thêm:** đoán bừa đều: loss $= \ln 151{,}936 = 11.93$, perplexity 151,936. Câu tiếng Anh loss 3.13: không so Việt với Anh như so "độ thông minh" vì khác token, khác số token.

🔎 **Đọc demo trên slide (INPUT → OUTPUT → vì sao):** **INPUT:** câu 8 token. **OUTPUT 1:** 7 cặp (hình tam giác: xanh ngữ cảnh, cam mục tiêu). **OUTPUT 2:** loss từng token, cao nhất 8.19 ở "H" → "ôm" (ngữ cảnh quá mở), thấp nhất 0.004 ở "Hôm n" → "ay" (gần như chắc chắn). Bốn số: loss 2.985 khớp `outputs.loss` · PPL 19.8 · câu xáo trộn 6.701 (model đã học trật tự từ) · đoán bừa 11.93 = ln 151,936.

❓ Sao 8 token chỉ 7 loss? → Token đầu không có ngữ cảnh. ❓ Sao chạy 1 lần mà ra loss mọi vị trí? → Causal mask: vị trí t chỉ nhìn token ≤ t, nên điểm ở t là dự đoán cho t + 1; Transformer tính song song cả 7 vị trí.

---

### Slide 8 — Mục 5: Các giai đoạn huấn luyện (1.5 phút)

🎯 **Cô yêu cầu:** "Pretraining, SFT và preference alignment; giới thiệu RLHF/DPO ở mức ý tưởng" — minh họa "**sơ đồ mục tiêu và dữ liệu của từng giai đoạn**"; góp ý 11/11: mục 5 phải có demo. → Cell 5.0 (sơ đồ) + 5.1–5.4 (demo).

🧭 **Trọng tâm lý thuyết (nắm chắc trước khi nói; quên lời thì nói đúng các ý này):**
- Mỗi giai đoạn nói đủ **3 thứ: mục tiêu · dữ liệu · kết quả**.
- ① **Pretraining:** đoán token tiếp theo trên văn bản thô khổng lồ → **base**, chỉ biết viết tiếp. ② **SFT:** cross-entropy **chỉ trên phần câu trả lời** của cặp (chỉ dẫn, trả lời mẫu) → biết làm theo chỉ dẫn. ③ **Alignment:** dữ liệu (prompt, câu tốt, câu kém) → câu được ưa thích có xác suất cao hơn → **Instruct**.
- **RLHF:** người so sánh → reward model chấm điểm → PPO tối đa điểm, phạt KL để không lệch xa bản SFT. **DPO:** học thẳng từ cặp tốt / kém, **không cần reward model, không cần RL**.
- Demo: base và Instruct **cùng kiến trúc, cùng số tham số, cùng tham số sinh**, chỉ khác trọng số → khác biệt là tác dụng gộp của post-training.

📖 **Thuật ngữ của slide này** (★ = bắt buộc nắm; mỗi từ: **là gì** · 🗨 **nói thế nào** · 🎓 **cần học / hiểu tới đâu**)
- ★ **Pretraining (tiền huấn luyện)** — Giai đoạn 1: next-token prediction trên văn bản thô khổng lồ.
  🗨 "Pretraining dạy model ngôn ngữ và kiến thức, nhưng chỉ biết viết tiếp."
  🎓 Nói đủ mục tiêu · dữ liệu · kết quả.
- ★ **Base model** — Model chỉ qua pretraining.
  🗨 "Bản base thấy câu hỏi thì viết tiếp như một trang web."
  🎓 Giải thích được output base trong demo.
- ★ **Fine-tuning, SFT (supervised fine-tuning)** — Fine-tuning = huấn luyện tiếp model đã có; SFT = fine-tune trên cặp (chỉ dẫn, câu trả lời mẫu), loss chỉ tính trên phần trả lời.
  🗨 "SFT dạy model bắt chước câu trả lời mẫu của con người."
  🎓 Biết SFT vẫn là cross-entropy, chỉ khác dữ liệu và phần được tính loss.
- ★ **Preference alignment** — Giai đoạn 3: dạy model ưu tiên câu người thích hơn câu bị loại.
  🗨 "Alignment làm model hữu ích, an toàn, biết từ chối."
  🎓 Dữ liệu là bộ ba (prompt, câu tốt, câu kém).
- ★ **RLHF** — Reinforcement Learning from Human Feedback: người so sánh câu trả lời → huấn luyện reward model → dùng RL (PPO) tối đa điểm.
  🗨 "RLHF học một model chấm điểm theo sở thích con người, rồi tối ưu model để được điểm cao."
  🎓 Nói được 3 bước; biết có phạt KL.
- **Reward model, PPO** — Reward model: model cho điểm câu trả lời. PPO: thuật toán học tăng cường cập nhật model theo điểm.
  🗨 "Reward model đóng vai giám khảo."
  🎓 Chỉ ở mức ý tưởng (đề ghi 'mức ý tưởng').
- **KL divergence** — Đo hai phân phối khác nhau bao nhiêu; phạt KL giữ model không lệch xa bản SFT.
  🗨 "Có khoản phạt để model không đi quá xa bản ban đầu."
  🎓 Biết lý do: tránh model 'lách luật' để kiếm điểm.
- ★ **DPO** — Direct Preference Optimization: học thẳng từ cặp tốt / kém, không cần reward model, không cần RL.
  🗨 "DPO tăng xác suất câu tốt, giảm câu kém, so với một model tham chiếu."
  🎓 So được RLHF với DPO trong một câu.
- **Instruct model, GRPO** — Instruct = model đã qua post-training. GRPO = một thuật toán RL Qwen dùng sau DPO.
  🗨 "Bản Instruct đã qua SFT, DPO và GRPO."
  🎓 Vì sao chỉ nói 'tác dụng gộp': không tách được từng giai đoạn.
- **`repetition_penalty`** — Hệ số phạt token đã xuất hiện để giảm lặp; 1.0 = không phạt.
  🗨 "Hai bản cùng repetition_penalty bằng 1 để so sánh công bằng."
  🎓 Biết bản Instruct mặc định 1.1, base 1.0.

✅ **Các ý bắt buộc phải nói ở slide này (theo thứ tự):**
1. Mỗi giai đoạn nói đủ mục tiêu · dữ liệu · kết quả (minh họa đề).
2. SFT: cross-entropy chỉ trên phần trả lời (đọc công thức).
3. RLHF: người so sánh → reward model → PPO + phạt KL; DPO: học thẳng từ cặp, không reward model.
4. Demo base 0/4 vs Instruct 4/4, cùng tham số sinh; tác dụng gộp của post-training.

📐 **Lý thuyết chi tiết (đọc để hiểu sâu; khi nói chỉ lấy ý, không đọc nguyên văn):**

**Ba giai đoạn, so theo mục tiêu và dữ liệu:**

| Giai đoạn | Mục tiêu (loss) | Dữ liệu | Kết quả |
|---|---|---|---|
| Pretraining | Cross-entropy trên **mọi** token | Văn bản thô rất lớn (Qwen2.5: ~18 nghìn tỉ token) | Base: biết ngôn ngữ, kiến thức; chỉ viết tiếp |
| SFT | Cross-entropy **chỉ trên token câu trả lời** | Cặp (chỉ dẫn, câu trả lời mẫu) do người viết | Làm theo chỉ dẫn, đúng định dạng chat |
| Preference alignment | Câu được chọn có xác suất cao hơn câu bị loại | Bộ ba (prompt, câu tốt $y_w$, câu kém $y_l$) | Instruct: hữu ích, an toàn, biết từ chối |

**SFT.** Ghép chỉ dẫn và câu trả lời thành một chuỗi, nhưng loss chỉ cộng trên phần trả lời: $\mathcal{L}_{SFT} = -\sum_{t \in \text{trả lời}} \ln \pi_\theta(y_t \mid x, y_{<t})$. Model không bị phạt vì không đoán được câu hỏi; nó chỉ học **cách trả lời**.

**Vì sao cần alignment sau SFT.** SFT chỉ dạy bắt chước một câu mẫu. Với một câu hỏi có nhiều cách trả lời, con người dễ **so sánh** hai câu ("câu A tốt hơn câu B") hơn là tự viết câu hoàn hảo. Alignment học từ các so sánh đó.

**RLHF (3 bước).** (1) Model SFT sinh vài câu trả lời cho một prompt; người chọn câu tốt hơn. (2) Huấn luyện **reward model** $r(x, y)$ sao cho câu được chọn có điểm cao hơn, theo mô hình Bradley–Terry: $P(y_w \succ y_l) = \sigma\big(r(x, y_w) - r(x, y_l)\big)$. (3) Dùng học tăng cường (PPO) tối đa hóa $\mathbb{E}[r(x, y)] - \beta\,\text{KL}(\pi_\theta \,\|\, \pi_{ref})$: được điểm cao nhưng không lệch quá xa model tham chiếu (bản SFT), tránh model "lách" reward model.

**DPO (ý tưởng).** Rafailov et al. (2023) chứng minh từ bài toán RLHF ở trên có thể viết reward theo chính model: $r = \beta \ln \frac{\pi_\theta(y \mid x)}{\pi_{ref}(y \mid x)} + \text{hằng số}$. Thay vào Bradley–Terry được loss chỉ cần cặp (câu tốt, câu kém):
$$\mathcal{L}_{DPO} = -\ln \sigma\Big(\beta \ln \tfrac{\pi_\theta(y_w)}{\pi_{ref}(y_w)} - \beta \ln \tfrac{\pi_\theta(y_l)}{\pi_{ref}(y_l)}\Big)$$
Không cần huấn luyện reward model, không cần RL, chỉ cần một vòng huấn luyện có giám sát.

**Đọc demo.** Hai bản cùng kiến trúc, cùng số tham số, cùng tham số sinh; chỉ khác trọng số. Base viết tiếp như trang web vì pretraining chỉ dạy viết tiếp văn bản trên web; Instruct trả lời đúng, biết từ chối. Không tách được đóng góp riêng của SFT hay DPO vì Qwen không công bố bản trung gian, nên gọi là **tác dụng gộp của post-training**.

🖼 **Sơ đồ RLHF / DPO (phải dưới):** RLHF: model SFT sinh nhiều câu → người chọn câu tốt hơn → huấn luyện **reward model** chấm điểm → PPO tối đa điểm, trừ β·KL để không lệch xa bản SFT. DPO: dùng thẳng cặp (câu tốt, câu kém) → loss DPO tăng xác suất câu tốt, giảm câu kém so với model tham chiếu → không cần reward model, không cần RL. 🎤 "Hai cách cùng mục tiêu: model ưu tiên câu người thích; DPO đơn giản hơn vì bỏ được reward model."

[Mở cell 5.0 hoặc chiếu hình]

🖼 **Hình `training_stages.png`:** 3 ô từ trái sang phải; mỗi ô ghi **mục tiêu**, **dữ liệu**, **kết quả**.

🎤 "Cách học ở mục 4 mới chỉ là giai đoạn đầu. Một LLM như Qwen-Instruct trải qua ba giai đoạn.
Giai đoạn một là **pretraining**: mục tiêu là đoán token tiếp theo, chính là cross-entropy vừa trình bày, trên **văn bản thô rất lớn và không có nhãn**; Qwen2.5 dùng khoảng 18 nghìn tỉ token. Kết quả là **bản base**: model có kiến thức và ngôn ngữ, nhưng chỉ biết **viết tiếp** văn bản.
Giai đoạn hai là **SFT**, tức supervised fine-tuning: dữ liệu là các **cặp chỉ dẫn và câu trả lời mẫu** do con người viết; vẫn dùng cross-entropy nhưng **chỉ tính trên phần câu trả lời**. Kết quả là model biết **làm theo chỉ dẫn**.
Giai đoạn ba là **preference alignment**: dữ liệu là **bộ ba gồm prompt, một câu trả lời tốt và một câu trả lời kém**; mục tiêu là làm cho câu được ưa thích có xác suất cao hơn câu bị loại."

🗣 **Đọc công thức SFT** $\mathcal{L}_{\text{SFT}} = -\sum_{t \in \text{trả lời}} \ln \pi_\theta(y_t \mid x, y_{<t})$:
"Loss SFT bằng **trừ tổng, chỉ trên các token của câu trả lời, của log xác suất model gán cho token trả lời thứ t, khi biết chỉ dẫn x và các token trả lời trước đó**." ($\pi_\theta$: model với tham số θ.)

🎤 "Ở giai đoạn ba có hai cách tiếp cận. **RLHF** gồm hai bước: trước hết huấn luyện một **reward model** để chấm điểm câu trả lời theo sở thích của con người, sau đó dùng học tăng cường để model sinh ra câu được chấm điểm cao, kèm một khoản phạt để model không đi quá xa bản SFT. **DPO** đơn giản hơn: bỏ hẳn reward model, tối ưu trực tiếp trên từng cặp câu tốt và câu kém."

🗣 **Đọc công thức DPO** $\mathcal{L}_{\text{DPO}} = -\ln \sigma\!\left(\beta \ln\frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \beta \ln\frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}\right)$:
"Loss DPO bằng **trừ log của sigmoid, của beta nhân hiệu hai log-tỉ số**: log-tỉ số thứ nhất so xác suất câu tốt $y_w$ của model đang học với model tham chiếu; log-tỉ số thứ hai làm tương tự với câu kém $y_l$." Ý nghĩa: loss nhỏ khi model **tăng xác suất câu tốt và giảm xác suất câu kém** nhiều hơn mức của model tham chiếu; β điều chỉnh độ mạnh.

📝 **Hiểu thêm:** RLHF tối đa hoá $\mathbb{E}[r] - \beta\,\text{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})$ (đọc: "kỳ vọng điểm thưởng trừ beta nhân độ lệch KL giữa model đang học và model tham chiếu"). Qwen2.5-Instruct: SFT hơn 1 triệu mẫu, DPO khoảng 150 nghìn cặp, sau đó RL trực tuyến GRPO.

[Mở cell 5.2]

📊 `generation_config` mặc định: Instruct `do_sample=True`, `repetition_penalty=1.1`; base `do_sample=False`, 1.0.

🎤 "Để minh họa trên model thật, em so hai bản do Qwen công bố: **bản base**, chỉ qua pretraining, và **bản Instruct**, đã qua cả SFT lẫn alignment. Hai bản có cấu hình sinh mặc định khác nhau, nên để so sánh công bằng, em cho cả hai **cùng greedy, cùng `repetition_penalty` bằng 1, cùng số token tối đa**."

[Mở cell 5.3, chỉ khung ĐẦU VÀO]

💻 **Code**
```python
"base":     tok_base(p["prompt"], return_tensors="pt")                         # văn bản thô
"instruct": tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], add_generation_prompt=True, ...)
output, n_out, sec = generate_greedy(mdl, tk, enc, p["max_new_tokens"], REP_PENALTY_PART05)
```
- **Bản base nhận văn bản thô**, đúng cách nó được huấn luyện; **bản Instruct nhận chat template**, đúng cách nó được dùng. Khung ĐẦU VÀO in hai chuỗi này cạnh nhau.

📊 **Output thật**

| Prompt | Base | Instruct |
|---|---|---|
| Thủ đô của Pháp là gì? | viết tiếp như trang báo ("- VnExpress / Thứ năm…") | "Thủ đô của Pháp là Paris." |
| Liệt kê 3 lợi ích của tập thể dục | "- Trang 2 - Diễn đàn sức khỏe…" | đúng 3 ý |
| 2 + 2 bằng mấy (một từ) | "- Trang 2 - Giải toán Online…" | "4" |
| Chỉ cách quay cóp | viết tiếp bài báo, lặp "giám sát" | từ chối |

Chấm theo luật `follow_rule` ghi trước: **base 0/4, Instruct 4/4**.

🎤 "Kết quả rất rõ: với câu 'Thủ đô của Pháp là gì?', bản base **viết tiếp như một trang báo**, vì nó chỉ học cách nối dài văn bản trên web; bản Instruct trả lời thẳng 'Paris', và còn biết **từ chối** yêu cầu hướng dẫn quay cóp. Hai bản có cùng kiến trúc, cùng số tham số, chỉ khác trọng số. Em diễn đạt đây là **tác dụng tổng hợp của post-training**, vì Qwen2.5-Instruct đã qua SFT, DPO và GRPO, nên không tách riêng được phần đóng góp của từng giai đoạn."

🔎 **Đọc demo trên slide (INPUT → OUTPUT → vì sao):** Cột INPUT: 4 prompt giống nhau cho cả hai bản. Cột base (đỏ): **viết tiếp như trang web** ("- VnExpress / Thứ năm…") vì pretraining chỉ dạy viết tiếp. Cột Instruct (xanh): trả lời đúng ý, đúng định dạng, **từ chối** yêu cầu quay cóp. Luật `follow_rule` ghi trước: base 0/4, Instruct 4/4.

❓ Nhóm có fine-tune không? → Không. Cả hai bản do Qwen huấn luyện; nhóm chỉ so sánh. ❓ So sánh có hoàn toàn công bằng? → Còn khác cách đưa prompt (chat template) và token kết thúc; đã ghi rõ.

---

### Slide 9 — Mục 6: Quá trình sinh văn bản (1.5 phút)

🎯 **Cô yêu cầu:** "Autoregressive generation; greedy decoding, temperature và top-p sampling" — minh họa "**cùng một prompt với các cấu hình sinh khác nhau**". → Cell 6.1–6.4. Prompt: "Viết một câu giới thiệu ngắn về thành phố Hà Nội."

🧭 **Trọng tâm lý thuyết (nắm chắc trước khi nói; quên lời thì nói đúng các ý này):**
- **Autoregressive:** model cho phân phối của **1** token → chọn 1 token → **nối vào input** → lặp tới khi gặp token kết thúc hoặc đủ số token.
- **Greedy:** luôn lấy token xác suất cao nhất → chạy lại ra y hệt.
- **Temperature:** chia logits cho T trước softmax; T < 1 phân phối nhọn (an toàn), T > 1 phẳng (đa dạng, dễ sai).
- **Top-p:** chỉ bốc trong **tập nhỏ nhất có tổng xác suất ≥ p**. Các tham số này **không đổi model**, chỉ đổi cách chọn token.

📖 **Thuật ngữ của slide này** (★ = bắt buộc nắm; mỗi từ: **là gì** · 🗨 **nói thế nào** · 🎓 **cần học / hiểu tới đâu**)
- ★ **Autoregressive (tự hồi quy)** — Sinh từng token; token vừa sinh được nối vào input để sinh token kế.
  🗨 "Đầu ra của bước trước thành đầu vào của bước sau."
  🎓 Vẽ lại được vòng lặp: model → chọn → nối → lặp.
- ★ **Decoding (giải mã)** — Cách chọn token từ phân phối ở mỗi bước.
  🗨 "Model giống nhau, chỉ khác cách chọn token."
  🎓 Ba cách: greedy, temperature sampling, top-p sampling.
- ★ **Greedy decoding (giải mã tham lam)** — Ở **mỗi bước luôn chọn token có xác suất cao nhất** (argmax), không ngẫu nhiên, không tính trước các bước sau. Gọi là "tham lam" vì chỉ chọn cái tốt nhất trước mắt.
  🗨 "Greedy, hay giải mã tham lam, ở mỗi bước luôn chọn token có xác suất cao nhất; vì không có ngẫu nhiên nên chạy lại bao nhiêu lần cũng ra cùng một câu."
  🎓 Hiểu 3 điều: ① `do_sample=False` = greedy; ② ưu: lặp lại được → dùng cho mọi phần đánh giá; ③ nhược: câu đơn điệu, dễ lặp, và chọn tốt nhất từng bước chưa chắc ra câu tốt nhất cả câu.
- ★ **Sampling (lấy mẫu)** — Bốc ngẫu nhiên token theo xác suất: token 0.7 được chọn khoảng 70% số lần.
  🗨 "Lấy mẫu cho câu đa dạng, mỗi lần chạy một khác."
  🎓 Cần seed để lặp lại được.
- ★ **Temperature (T)** — Số chia logits trước softmax. T < 1: phân phối nhọn, an toàn; T > 1: phẳng, đa dạng, dễ sai.
  🗨 "Temperature chia điểm cho T trước softmax: T nhỏ thì chắc chắn hơn, T lớn thì liều hơn."
  🎓 Đọc được biểu đồ: T = 1.5 cần 135 token mới phủ 90% xác suất.
- ★ **Top-p (nucleus sampling)** — Chỉ bốc trong tập nhỏ nhất các token có tổng xác suất ≥ p, bỏ phần đuôi hiếm.
  🗨 "Top-p chỉ bốc trong nhóm token đầu bảng, đủ để tổng xác suất đạt p."
  🎓 So với top-k: top-k giữ số token cố định, top-p thay đổi theo độ chắc chắn.
- **EOS (end of sequence)** — Token đặc biệt báo kết thúc câu trả lời.
  🗨 "Model dừng khi sinh token kết thúc hoặc đủ số token tối đa."
  🎓 Biết `max_new_tokens` là giới hạn số token sinh thêm.
- **KV cache** — Lưu sẵn K, V của các token đã đọc để không tính lại ở bước sau.
  🗨 "Nhờ KV cache, sinh mỗi token không phải tính lại cả câu."
  🎓 Chỉ khi bị hỏi; nối với chi phí bộ nhớ ở mục 9.

✅ **Các ý bắt buộc phải nói ở slide này (theo thứ tự):**
1. Autoregressive: chọn 1 token → nối vào → lặp; vòng lặp tự viết khớp generate() 8/8.
2. Greedy = luôn chọn xác suất cao nhất, tất định.
3. Temperature chia logits cho T (đọc công thức); T nhỏ nhọn, T lớn phẳng.
4. Top-p: tập nhỏ nhất có tổng ≥ p; khác top-k.
5. Cùng prompt × 5 cấu hình × 3 lần: greedy lặp lại, T 1.5 vô nghĩa, top-p 0.95 có lần bịa.

📐 **Lý thuyết chi tiết (đọc để hiểu sâu; khi nói chỉ lấy ý, không đọc nguyên văn):**

**Sinh văn bản là bài toán chọn chuỗi.** Model chỉ cho phân phối của một token. Sinh một câu là lặp: tính phân phối → chọn token → nối vào → tính lại. Sinh 100 token cần 100 lượt chạy model (KV cache giữ lại K, V của phần đã đọc nên mỗi lượt chỉ tính thêm token mới).

**Greedy decoding.** Chọn $x_{t+1} = \arg\max_i p_i$. Tất định (chạy lại ra y hệt). Nhược điểm: tham lam từng bước nên có thể bỏ lỡ câu tốt hơn xét cả câu, dễ lặp, câu đơn điệu. (Beam search giữ nhiều nhánh cùng lúc để giảm nhược điểm này; notebook không dùng.)

**Temperature.** Chia logits cho $T$ trước softmax: $p_i^{(T)} = e^{z_i/T}/\sum_j e^{z_j/T}$. Hai cực: $T \to 0$ thì token lớn nhất chiếm gần hết xác suất, giống greedy; $T \to \infty$ thì mọi token gần bằng nhau, như đoán bừa. Ví dụ thật: P("H") là 0.995 ở $T = 0.5$, 0.847 ở $T = 1$, 0.468 ở $T = 1.5$; để phủ 90% xác suất cần 1, 3, rồi 135 token.

**Top-p (nucleus sampling)** (Holtzman et al. 2020), 4 bước: (1) xếp token theo xác suất giảm dần; (2) cộng dồn; (3) giữ tập nhỏ nhất có tổng $\ge p$; (4) chuẩn hóa lại trong tập rồi bốc ngẫu nhiên. Khi model chắc chắn, tập chỉ có vài token; khi phân vân, tập rộng ra. Top-k thì luôn giữ đúng $k$ token, không thích nghi.

**Thứ tự áp dụng trong `generate()`:** logits → chia $T$ → lọc top-k / top-p → softmax → bốc (nếu `do_sample=True`) hoặc argmax (nếu `False`).

**Đọc demo.** Greedy: 3 lần y hệt. $T = 0.5$, top-p 0.5: khác nhau nhưng đúng. $T = 1.5$: phân phối quá phẳng, token hiếm được chọn → vô nghĩa, lẫn ngôn ngữ. Top-p 0.95: tập rộng nên có lần chọn trúng nhánh sai ("ở Trung Quốc"). Kết luận: đánh giá dùng greedy; sáng tác dùng lấy mẫu vừa phải.

🖼 **Sơ đồ vòng lặp sinh (trái trên):** ids → model → logits vị trí cuối → chia T, softmax → top-p giữ tập nhỏ nhất có tổng xác suất ≥ p → chọn token (greedy: lấy lớn nhất; sampling: bốc thăm theo xác suất) → nối vào ids, lặp đến EOS hoặc đủ số token. 🎤 "Ba tham số chỉ can thiệp ở hai ô giữa của vòng lặp; model thì không đổi."

🎤 "Ở mục 2, ta thấy model chỉ cho ra một phân phối xác suất cho **một** token tiếp theo. Vậy làm thế nào để có cả một câu? Câu trả lời là sinh **tự hồi quy**, autoregressive: đưa chuỗi vào model, lấy phân phối token tiếp theo, **chọn một token**, **nối token đó vào cuối chuỗi**, rồi lặp lại. Đầu ra của bước trước trở thành đầu vào của bước sau."

[Trên slide, không mở notebook — cell 6.1]

💻 **Code**
```python
for step in range(1, N_STEPS + 1):
    logits = model(ids).logits[0, -1].float()          # điểm cho token tiếp theo
    probs = torch.softmax(logits, dim=-1)
    p_top, next_id = probs.max(dim=-1)                 # greedy: lấy token xác suất cao nhất
    ids = torch.cat([ids, next_id.view(1, 1)], dim=1)  # nối vào input
```
- **`probs.max` là greedy**; **`torch.cat` là bước nối token vừa chọn vào chuỗi**, input dài thêm một token sau mỗi vòng.

📊 **Output:** "H" 0.847 → "à" 0.998 → " Nội" 0.9998 → " là" 0.674 → " thành" 0.462 → " phố" 0.9997 → " lớn" 0.601 → " nhất" 0.407 = "Hà Nội là thành phố lớn nhất". `assert`: `generate(do_sample=False)` ra **đúng 8 token này**.

🎤 "Vòng lặp em tự viết sinh ra đúng 8 token giống hệt hàm `generate()` của thư viện, nghĩa là `generate()` thực chất chỉ là vòng lặp này được viết sẵn. Điều quyết định câu văn là **cách chọn token** ở mỗi vòng: greedy, temperature hay top-p."

🗣 **Đọc công thức temperature** $p_i^{(T)} = \dfrac{e^{z_i/T}}{\sum_j e^{z_j/T}}$:
"Xác suất của token i ở nhiệt độ T bằng **e mũ điểm của token i chia cho T, chia cho tổng e mũ điểm của mọi token chia cho T**." Tức là **chia mọi điểm cho T trước khi đưa vào softmax**.

🎤 "**Greedy** luôn chọn token có xác suất cao nhất. **Temperature** chia các điểm cho T trước softmax: T nhỏ hơn 1 khuếch đại chênh lệch nên phân phối **nhọn** hơn; T lớn hơn 1 thu hẹp chênh lệch nên phân phối **phẳng** hơn. **Top-p** thì chỉ bốc ngẫu nhiên trong nhóm token có xác suất cao nhất, sao cho tổng xác suất của nhóm vừa đạt p."

🗣 **Đọc định nghĩa top-p:** "Tập top-p là **tập token nhỏ nhất, xếp theo xác suất giảm dần, có tổng xác suất lớn hơn hoặc bằng p**; sau đó chuẩn hoá lại xác suất trong tập và bốc ngẫu nhiên."

[Trên slide, không mở notebook — cell 6.2]

💻 `probs_T = torch.softmax(logits0 / T, dim=-1)`; `nucleus_size(probs, p) = (cumsum(sort(probs)) < p).sum() + 1`

📊 **Output + 🖼 `temperature_distribution.png`** (8 token đầu, mỗi token 3 cột theo T)

| T | P(token "H") | Số token phủ 90% xác suất |
|---|---|---|
| 0.5 | 0.995 | 1 |
| 1.0 | 0.847 | 3 |
| 1.5 | 0.468 | **135** |

🎤 "Biểu đồ minh họa đúng điều đó: ở T bằng 0.5, gần như toàn bộ xác suất dồn vào token 'H'; ở T bằng 1.5, phân phối trải rộng đến mức phải tới 135 token mới phủ được 90% xác suất, nên những token hiếm có cơ hội được chọn."

[Trên slide, không mở notebook — cell 6.3]

💻 `DECODING_CONFIGS`: greedy; temp 0.5; temp 1.5; top-p 0.5; top-p 0.95 (`top_k=0` để tắt top-k mặc định); mỗi cấu hình 3 lần với `torch.manual_seed(SEED + i)`.

📊 **Output:** greedy 3 lần **giống hệt nhau**; temp 0.5 và top-p 0.5: 3 câu khác nhau nhưng đều đúng; temp 1.5: chuỗi **vô nghĩa, lẫn nhiều ngôn ngữ**; top-p 0.95: có lần **bịa** "Hà Nội … ở Trung Quốc".

🎤 "Đây là minh họa đề yêu cầu: cùng một prompt, năm cấu hình, mỗi cấu hình chạy ba lần. Greedy cho cùng một câu cả ba lần; nhiệt độ thấp hoặc top-p nhỏ cho câu đa dạng mà vẫn đúng; nhiệt độ 1.5 cho chuỗi vô nghĩa; top-p 0.95 có lần còn bịa rằng Hà Nội ở Trung Quốc. Vì vậy, các phần đánh giá phía sau em đều dùng greedy để kết quả lặp lại được."

🔎 **Đọc demo trên slide (INPUT → OUTPUT → vì sao):** **INPUT:** "Viết một câu giới thiệu ngắn về thành phố Hà Nội." Ô màu: 8 token greedy sinh lần lượt, số nhỏ là xác suất lúc chọn. Biểu đồ: 3 cột / token theo T = 0.5, 1, 1.5; T càng lớn cột token đầu càng thấp. Bảng: 5 cấu hình × 3 lần; cột "Khác nhau / 3": greedy 1 (y hệt), các cấu hình lấy mẫu 3. T = 1.5 vô nghĩa, top-p 0.95 có lần bịa "Trung Quốc".

❓ Top-p khác top-k? → Top-k giữ số token cố định; top-p giữ số token thay đổi theo mức chắc chắn của model. ❓ Sinh 100 token chạy model mấy lần? → 100 lần (có KV cache để không tính lại phần đã đọc).

---

### Slide 10 — Mục 7: Khả năng của LLM (1.3 phút)

🎯 **Cô yêu cầu:** "In-context learning, làm theo chỉ dẫn, xử lý các tác vụ ngôn ngữ" — minh họa "**so sánh zero-shot và few-shot trên cùng tác vụ**". → Cell 7.1–7.5.

🧭 **Trọng tâm lý thuyết (nắm chắc trước khi nói; quên lời thì nói đúng các ý này):**
- **In-context learning:** model làm tác vụ mới nhờ **ví dụ đặt trong prompt**, **không đổi tham số**.
- **Làm theo chỉ dẫn:** có được nhờ SFT (mục 5); **xử lý tác vụ ngôn ngữ:** phân loại cũng chỉ là sinh ra chữ "tích cực" / "tiêu cực".
- Thí nghiệm **chỉ đổi một biến**: có 0 hay 4 ví dụ; cùng 2,999 câu test, cùng mẫu prompt (chọn trên validation), cùng greedy.

📖 **Thuật ngữ của slide này** (★ = bắt buộc nắm; mỗi từ: **là gì** · 🗨 **nói thế nào** · 🎓 **cần học / hiểu tới đâu**)
- ★ **In-context learning (ICL)** — Model làm tác vụ mới nhờ ví dụ đặt trong prompt, không đổi tham số.
  🗨 "Model học từ ví dụ ngay trong prompt, không cần huấn luyện lại."
  🎓 Nhấn mạnh: tham số không đổi, đây không phải huấn luyện.
- ★ **Zero-shot, few-shot** — Zero-shot: chỉ có chỉ dẫn, 0 ví dụ. Few-shot: thêm vài ví dụ có nhãn (ở đây 4).
  🗨 "Zero-shot không có ví dụ; few-shot thêm bốn ví dụ có nhãn."
  🎓 Hai prompt chỉ khác đúng phần ví dụ.
- ★ **Instruction following (làm theo chỉ dẫn)** — Khả năng hiểu và thực hiện yêu cầu viết bằng ngôn ngữ tự nhiên; có nhờ SFT.
  🗨 "Model làm theo câu dặn 'chỉ trả lời tích cực hoặc tiêu cực'."
  🎓 Nối với mục 5: base không làm được việc này.
- ★ **Tập train / validation / test** — Train: lấy ví dụ few-shot. Validation: chọn mẫu prompt. Test: đo kết quả cuối, chỉ dùng một lần.
  🗨 "Mẫu prompt chọn trên validation, trước khi nhìn tập test."
  🎓 Lý do: tránh chọn theo kết quả test (rò rỉ dữ liệu).
- ★ **Accuracy** — Tỉ lệ câu đoán đúng: số câu đúng / tổng số câu.
  🗨 "Accuracy 89.86% nghĩa là đúng 2,695 trên 2,999 câu."
  🎓 Tính lại được từ số câu.
- **Parse nhãn** — Đọc output chữ của model và đổi thành nhãn; không đọc được thì tính sai.
  🗨 "Không câu nào bị lỗi đọc nhãn."
  🎓 Biết luật: tìm 'tích cực' / 'tiêu cực', thấy cả hai hoặc không thấy → sai.

✅ **Các ý bắt buộc phải nói ở slide này (theo thứ tự):**
1. Ba khả năng: in-context learning (không đổi tham số), làm theo chỉ dẫn (nhờ SFT), xử lý tác vụ (quy về sinh chữ).
2. Zero-shot vs few-shot cùng tác vụ, chỉ khác phần ví dụ (minh họa đề).
3. Train / validation / test tách biệt; prompt chọn trên validation.
4. 89.86% → 92.30%; kiểm định ở slide 14.

📐 **Lý thuyết chi tiết (đọc để hiểu sâu; khi nói chỉ lấy ý, không đọc nguyên văn):**

**In-context learning (ICL).** Model trả lời theo $P(y \mid \text{chỉ dẫn}, \text{ví dụ}_1, \dots, \text{ví dụ}_k, x)$, tức chỉ **thay đổi điều kiện** (prompt), không tính gradient, không đổi tham số. Model chỉ đoán phần viết tiếp hợp lý nhất: sau 4 cặp "Câu: … / Cảm xúc: …", phần hợp lý tiếp theo là một nhãn đúng định dạng.

**Ví dụ giúp gì.** Theo Min et al. (2022), ví dụ cho model thấy **định dạng**, **tập nhãn** và **phân bố đầu vào**; Zhao et al. (2021) cho thấy kết quả nhạy với cách chọn và thứ tự ví dụ (model thiên về nhãn xuất hiện nhiều, xuất hiện cuối). Vì thế nhóm dùng 2 ví dụ tích cực, 2 tiêu cực, xen kẽ.

**Làm theo chỉ dẫn và xử lý tác vụ ngôn ngữ.** Câu dặn "Chỉ trả lời: tích cực hoặc tiêu cực" chỉ có tác dụng với model đã qua SFT (mục 5). Phân loại được đưa về dạng sinh chữ, nên cùng một model xử lý được nhiều tác vụ.

**Thiết kế để so sánh công bằng.** Ba tập tách biệt: ví dụ few-shot lấy từ **train**; mẫu prompt chọn trên **200 câu validation**; kết quả cuối đo trên **2,999 câu test**, chỉ đo một lần. Kiểm tra ví dụ không trùng câu test (chống rò rỉ dữ liệu). Hai cách dùng cùng hàm `build_prompt`, cùng greedy, cùng luật đọc nhãn → khác nhau đúng một biến.

**Accuracy.** Số câu đúng chia tổng số câu: $2{,}695/2{,}999 = 89.86\%$ và $2{,}768/2{,}999 = 92.30\%$. Câu không đọc được nhãn tính là sai (ở đây 0 câu), để không "làm đẹp" kết quả.

🎤 "Mục 7 nói về khả năng của LLM. Thứ nhất là **in-context learning**: model học cách làm một tác vụ chỉ từ các ví dụ đặt ngay trong prompt, **mà không thay đổi bất kỳ tham số nào**. Thứ hai là **làm theo chỉ dẫn**, có được nhờ giai đoạn SFT ở mục 5. Thứ ba là **xử lý nhiều tác vụ ngôn ngữ**: ngay cả phân loại cũng chỉ là sinh ra chữ 'tích cực' hoặc 'tiêu cực'. Để minh họa, em so sánh zero-shot và few-shot trên **cùng một tác vụ**: phân loại cảm xúc 2,999 câu phản hồi của sinh viên trong bộ UIT-VSFC."

[Trên slide, không mở notebook — cell 7.2]

💻 **Code**
```python
def build_prompt(text, examples, tpl):
    blocks = [tpl["instruction"]]
    blocks += [tpl["item"].format(text=e["sentence"]) + " " + tpl["labels"][e["label"]] for e in examples]
    blocks.append(tpl["item"].format(text=text))
    return "\n\n".join(blocks)
```
- **Một hàm dùng cho cả hai cách**: `examples = []` là zero-shot, truyền 4 ví dụ là few-shot, nên hai prompt **chỉ khác nhau ở phần ví dụ**.
- `parse_label`: chuyển chữ thường, bỏ dấu câu, tìm "tích cực" / "tiêu cực"; không tìm thấy hoặc thấy cả hai thì **không đọc được nhãn và tính là sai**.

[Trên slide, không mở notebook — cell 7.5, chỉ khung INPUT]

📊 **Prompt few-shot thật:**
```
Phân loại cảm xúc của câu phản hồi sinh viên sau. Chỉ trả lời: tích cực hoặc tiêu cực.

Câu: "dạy tốt có tâm ."
Cảm xúc: tích cực
Câu: "oop khá phức tạp nên cần nhiều ví dụ ."
Cảm xúc: tiêu cực
… (2 ví dụ nữa) …
Câu: "<câu cần phân loại>"
Cảm xúc:
```

🎤 "Zero-shot chỉ có câu chỉ dẫn và câu cần phân loại; few-shot thêm bốn ví dụ có nhãn, lấy từ tập **huấn luyện** và đã kiểm tra không trùng với câu test. Mẫu prompt được chọn trên tập **validation trước khi nhìn tập test**. Như vậy giữa hai cách chỉ khác đúng một biến: có ví dụ hay không."

📊 **Output:** zero-shot **89.86%** (2,695 / 2,999 câu đúng), few-shot **92.30%** (2,768), không câu nào bị lỗi đọc nhãn. Ví dụ: "cần giảng chậm và kỹ để chúng em hiểu bài ." — nhãn tiêu cực; zero-shot đoán tích cực, few-shot đúng.

🗣 **Đọc công thức** $\text{accuracy} = \dfrac{\text{số câu đúng}}{\text{tổng số câu}}$: "Accuracy bằng số câu đoán đúng chia cho tổng số câu": 2,695 chia 2,999 bằng 89.86%.

🎤 "Kết quả: accuracy tăng từ **89.86% lên 92.30%**. Một ví dụ điển hình là câu góp ý 'cần giảng chậm và kỹ': bộ dữ liệu gán nhãn tiêu cực; zero-shot đoán tích cực, còn few-shot đoán đúng. Mức tăng này có ý nghĩa thống kê hay không, em sẽ kiểm định ở slide 14."

📝 **Hiểu thêm:** in-context learning hoạt động vì model đoán phần tiếp theo hợp lý nhất; sau 4 cặp "Câu… Cảm xúc: …" thì phần tiếp hợp lý là một nhãn theo đúng kiểu đó. Ví dụ cho model thấy định dạng, tập nhãn và **cách gán nhãn của bộ dữ liệu** (Min et al. 2022); kết quả nhạy với cách chọn và thứ tự ví dụ (Zhao et al. 2021).

🔎 **Đọc demo trên slide (INPUT → OUTPUT → vì sao):** Hai khung INPUT: zero-shot (chỉ dẫn + câu cần phân loại) và few-shot (thêm 4 ví dụ có nhãn từ tập train). Bảng OUTPUT thật: nhãn gốc, đoán zero-shot, đoán few-shot cho 3 câu; câu góp ý "cần giảng chậm…" zero-shot sai, few-shot đúng. Bốn số: 2,999 câu · 89.86% · 92.30% · 0 câu không đọc được nhãn.

❓ Đây có phải huấn luyện không? → Không, tham số không đổi. ❓ Sao chọn prompt trên validation? → Để không chọn theo kết quả test.

---

### Slide 11 — Mục 8: Cách thích nghi và bổ sung tri thức (1.5 phút)

🎯 **Cô yêu cầu:** "Phân biệt prompting, fine-tuning và RAG" — minh họa "**bảng so sánh dữ liệu cần có và trường hợp sử dụng**"; góp ý 11/11: mục 8 phải có demo. → Bảng (markdown phần 8) + cell 8.1–8.4.

🧭 **Trọng tâm lý thuyết (nắm chắc trước khi nói; quên lời thì nói đúng các ý này):**
- Ba cách khác nhau ở chỗ **đổi cái gì**: prompting đổi **câu hỏi**; fine-tuning đổi **tham số** (cần dữ liệu có nhãn + GPU); RAG đưa **tài liệu vào prompt lúc hỏi** (cần kho tài liệu + bộ tìm kiếm).
- **RAG = Retrieval → Augmented → Generation:** tìm 3 đoạn liên quan (TF-IDF + cosine) → dán vào prompt → model trả lời dựa trên đoạn đó.
- Demo công bằng: cùng 15 câu hỏi, **cùng câu dặn** "không biết thì trả lời Không có thông tin"; chỉ khác có dán tài liệu hay không.

📖 **Thuật ngữ của slide này** (★ = bắt buộc nắm; mỗi từ: **là gì** · 🗨 **nói thế nào** · 🎓 **cần học / hiểu tới đâu**)
- ★ **Prompting** — Thích nghi bằng cách đổi câu hỏi, chỉ dẫn, ví dụ; không đổi model.
  🗨 "Prompting rẻ nhất, gần như không cần dữ liệu."
  🎓 Đọc được hàng Prompting trong bảng.
- ★ **Fine-tuning** — Huấn luyện tiếp để đổi tham số; cần dữ liệu có nhãn và GPU.
  🗨 "Fine-tuning đổi chính model, phù hợp khi cần đổi hành vi lâu dài."
  🎓 Nhóm không fine-tune (góp ý của cô); chỉ so sánh.
- ★ **RAG (Retrieval-Augmented Generation)** — Tìm tài liệu liên quan → dán vào prompt → model sinh câu trả lời dựa trên đó.
  🗨 "RAG gồm ba bước: truy xuất, bổ sung vào prompt, rồi sinh."
  🎓 Biết RAG không đổi tham số; cập nhật tri thức bằng cách đổi tài liệu.
- **Corpus (kho tài liệu)** — Tập văn bản để tìm kiếm; ở đây 557 đoạn Wikipedia.
  🗨 "Kho tài liệu gồm 557 đoạn văn."
  🎓 —
- ★ **TF-IDF** — Trọng số của một từ trong đoạn = tần suất trong đoạn × độ hiếm trong kho.
  🗨 "Từ càng xuất hiện nhiều trong đoạn và càng hiếm trong kho thì càng quan trọng."
  🎓 Biết vì sao 'là', 'của' có trọng số thấp.
- ★ **Cosine similarity** — Độ giống của hai vector = tích vô hướng / tích độ dài; 1 là cùng hướng, 0 là không có từ chung.
  🗨 "Độ giống giữa câu hỏi và đoạn văn đo bằng cosine."
  🎓 Đọc được công thức thành lời.
- ★ **recall@3** — Tỉ lệ câu hỏi có đoạn chứa đáp án nằm trong 3 đoạn tìm được.
  🗨 "recall@3 bằng 0.722: 13 trên 18 câu tìm đúng đoạn."
  🎓 Phân biệt: recall@3 đo bước tìm kiếm, không đo câu trả lời.

✅ **Các ý bắt buộc phải nói ở slide này (theo thứ tự):**
1. Bảng so sánh: đổi gì · dữ liệu cần · dùng khi (minh họa đề).
2. RAG = Retrieval → Augmented → Generation; TF-IDF + cosine, top-3.
3. Demo: cùng câu dặn; 0/15 → 10/15; recall@3 0.722; 2/3 vẫn bịa khi tài liệu không có đáp án.

📐 **Lý thuyết chi tiết (đọc để hiểu sâu; khi nói chỉ lấy ý, không đọc nguyên văn):**

**Ba cách thích nghi, khác ở chỗ thay đổi cái gì:**
- **Prompting:** đổi đầu vào. Rẻ, nhanh, không cần dữ liệu lớn; giới hạn ở những gì model đã biết.
- **Fine-tuning:** đổi tham số bằng huấn luyện tiếp. Cần dữ liệu có nhãn, GPU; đổi hành vi lâu dài (văn phong, định dạng, lĩnh vực hẹp). Muốn cập nhật kiến thức phải huấn luyện lại.
- **RAG:** đổi **ngữ cảnh lúc hỏi** bằng tài liệu tìm được. Cần kho tài liệu và bộ tìm kiếm; hợp với tri thức mới, riêng, hay thay đổi; trả lời có nguồn để kiểm chứng.

**RAG gồm 3 bước.** (1) **Retrieval:** biến câu hỏi và mọi đoạn thành vector, lấy $k = 3$ đoạn giống nhất. (2) **Augmented:** dán 3 đoạn vào prompt. (3) **Generation:** model trả lời dựa trên đoạn đó.

**TF-IDF.** Trọng số của từ $t$ trong đoạn $d$: $w(t, d) = \text{tf}(t, d) \times \text{idf}(t)$. tf: số lần $t$ xuất hiện trong $d$. idf (cách tính của scikit-learn): $\ln\frac{1+N}{1+\text{df}(t)} + 1$, với $N = 557$ đoạn, df là số đoạn chứa $t$. Từ có mặt khắp nơi ("là", "của") có idf thấp; từ đặc trưng ("Hội đồng") idf cao. `ngram_range=(1, 2)` đếm cả cặp 2 âm tiết ("ban hành") vì tiếng Việt viết tách âm tiết.

**Cosine.** $\cos(q, d) = \frac{q \cdot d}{\|q\|\,\|d\|}$: đo hai vector cùng hướng tới đâu, không phụ thuộc độ dài đoạn. TfidfVectorizer đã chuẩn hóa mỗi vector về độ dài 1, nên tích vô hướng (`linear_kernel`) chính là cosine.

**recall@3.** Tỉ lệ câu hỏi có đoạn chứa đáp án nằm trong 3 đoạn tìm được: $13/18 = 0.722$. Nó đo **bước tìm kiếm**; tìm sai đoạn thì model không thể trả lời đúng.

**Đọc kết quả.** Không RAG: 0/15 đúng, 14 câu từ chối (đúng như câu dặn). Có RAG: 10/15 đúng. Nhưng với 3 câu tài liệu không chứa đáp án, RAG vẫn bịa 2: có tài liệu trong prompt khiến model tự tin trả lời dù tài liệu không đủ. RAG giảm bịa nhưng không loại bỏ.

[Chỉ bảng trên slide]

| Cách | Đổi gì | Dữ liệu cần | Dùng khi |
|---|---|---|---|
| Prompting | Chỉ đổi câu hỏi, chỉ dẫn, ví dụ | Vài ví dụ | Tác vụ model đã biết làm |
| Fine-tuning | Đổi **tham số** model | Tập (input, output) có nhãn, GPU | Cần đổi hành vi, văn phong lâu dài |
| RAG | Dán tài liệu vào prompt lúc hỏi | Kho tài liệu + bộ tìm kiếm | Tri thức mới, riêng, hay thay đổi; cần dẫn nguồn |

🎤 "Khi model chưa làm tốt, có ba cách để thích nghi. **Prompting** chỉ thay đổi câu hỏi, như ta vừa thấy ở slide trước; gần như không cần dữ liệu. **Fine-tuning** thay đổi tham số của model; cần dữ liệu có nhãn và GPU để huấn luyện, phù hợp khi muốn đổi hành vi lâu dài. **RAG**, tức retrieval-augmented generation, gồm ba bước: **truy xuất** vài đoạn tài liệu liên quan, **bổ sung** chúng vào prompt, rồi để model **sinh** câu trả lời dựa trên đó. RAG không đổi tham số, nên muốn cập nhật tri thức chỉ cần đổi tài liệu."

🖼 **Sơ đồ RAG** (markdown phần 8): không RAG: câu hỏi → model → trả lời; có RAG: câu hỏi → tìm 3 đoạn → dán vào prompt → model → trả lời.

[Trên slide, không mở notebook — cell 8.2]

💻 **Code**
```python
vectorizer = TfidfVectorizer(ngram_range=(1, 2))      # đếm cả cặp 2 âm tiết ("thủ đô")
doc_matrix = vectorizer.fit_transform(corpus["title"] + ". " + corpus["context"])
scores = linear_kernel(vectorizer.transform([query]), doc_matrix)[0]   # = cosine
top3 = scores.argsort()[::-1][:3]
```
- Mỗi đoạn văn thành một vector **TF-IDF**; **độ giống giữa câu hỏi và đoạn văn là cosine**; lấy 3 đoạn có điểm cao nhất.

🗣 **Đọc công thức TF-IDF** $w(t, d) = \text{tf}(t, d) \times \text{idf}(t)$, với $\text{idf}(t) = \ln\frac{1 + N}{1 + \text{df}(t)} + 1$:
"Trọng số của từ t trong đoạn d bằng **tần suất của t trong d, nhân với độ hiếm của t**; độ hiếm bằng **log tự nhiên của một cộng N, chia cho một cộng số đoạn chứa t, rồi cộng một**." (N: tổng số đoạn. Từ xuất hiện ở mọi đoạn như "là", "của" có độ hiếm thấp.)

🗣 **Đọc công thức cosine** $\cos(q, d) = \dfrac{q \cdot d}{\|q\|\,\|d\|}$:
"Cosine giữa câu hỏi q và đoạn d bằng **tích vô hướng của hai vector, chia cho tích độ dài của chúng**." Bằng 1 khi cùng hướng, bằng 0 khi không có từ chung.

[Trên slide, không mở notebook — cell 8.3, chỉ khung INPUT: 2 prompt]

🎤 "Dữ liệu là bộ UIT-ViQuAD 2.0: câu hỏi tiếng Việt có sẵn **đáp án và đoạn Wikipedia nguồn**; kho tài liệu gồm 557 đoạn. Hai cách hỏi nhận **cùng một câu dặn**: nếu không biết thì trả lời 'Không có thông tin'. Khác biệt duy nhất: cách có RAG được dán thêm 3 đoạn do TF-IDF tìm ra."

[Trên slide, không mở notebook — cell 8.4]

💻 `judge_qa`: chuẩn hoá chữ; chứa đáp án → **Đúng**; chứa "không có thông tin" → **Từ chối**; còn lại → **Sai**. Đọc lại tay 2 câu, trong đó một câu có đáp án gốc của bộ dữ liệu bị sai.

📊 **Output**

| | Không RAG | Có RAG |
|---|---|---|
| 15 câu có đáp án: Đúng / Từ chối / Sai | **0** / 14 / 1 | **10** / 2 / 3 |
| 3 câu tài liệu không có đáp án: bịa | 0 | **2** |
| recall@3 | — | 0.722 (13/18) |

🗣 **Đọc định nghĩa recall@3:** "recall at 3 bằng **số câu hỏi có đoạn chứa đáp án nằm trong 3 đoạn tìm được, chia cho tổng số câu hỏi**": 13 chia 18 bằng 0.722.

🎤 "Kết quả: không có tài liệu, model từ chối 14 trên 15 câu; có tài liệu, model trả lời đúng **10 trên 15**. Từ đây rút ra ba ý. Một, RAG **bổ sung tri thức ngay lúc hỏi** mà không cần huấn luyện lại. Hai, RAG chỉ tốt bằng bước truy xuất: recall@3 là 0.722, đoạn đúng không được tìm thấy thì model cũng không trả lời được. Ba, với những câu mà tài liệu không chứa đáp án, có RAG lại khiến model **bịa 2 trên 3 câu**: tài liệu 'gần đúng' làm model tự tin trả lời sai."

🔎 **Đọc demo trên slide (INPUT → OUTPUT → vì sao):** **INPUT không RAG:** câu dặn + câu hỏi → OUTPUT "Để ban hành luật, hoàng đế phải ký tên vào bản thảo luật." → **Sai**. **INPUT có RAG:** câu dặn + 3 đoạn TF-IDF tìm được + câu hỏi → OUTPUT "Hoàng đế phải thông qua Hội đồng để ban hành luật." → **Đúng**. Bốn số: không RAG 0/15 đúng (14 câu từ chối) · có RAG 10/15 · recall@3 0.722 · 2/3 câu tài liệu không có đáp án vẫn bị bịa.

❓ Sao không fine-tune? → Đề chỉ yêu cầu phân biệt; cô góp ý không cần fine-tune.

---

### Slide 12 — Mục 9 (1/2): Hallucination và cách kiểm chứng (1.2 phút)

🎯 **Cô yêu cầu:** "Hallucination, thiên lệch, giới hạn ngữ cảnh; chất lượng phụ thuộc tác vụ và dữ liệu" — minh họa "**một câu trả lời sai và cách kiểm chứng**". → Slide 12: cell 9.1–9.3 (minh họa bắt buộc ở 9.2); slide 13: cell 9.4–9.6 (mỗi ý còn lại của nội dung một demo).

🧭 **Trọng tâm lý thuyết (nắm chắc trước khi nói; quên lời thì nói đúng các ý này):**
- **Hallucination** = câu trả lời nghe hợp lý nhưng **sai sự thật**. Nguyên nhân: model chọn token có xác suất cao theo ngữ cảnh, **không tra cứu sự thật**, không biết mình không biết.
- **Cách kiểm chứng:** ① đối chiếu với **nguồn**; ② **hỏi lại theo cách khác**, hai câu mâu thuẫn là dấu hiệu đoán; ③ dùng **RAG**.
- Hỏi về thứ **không tồn tại** → model chấp nhận tiền đề sai và bịa 5/6. Chất lượng còn **phụ thuộc dữ liệu**: nhãn gốc có thể sai, model chen tiếng Trung.

📖 **Thuật ngữ của slide này** (★ = bắt buộc nắm; mỗi từ: **là gì** · 🗨 **nói thế nào** · 🎓 **cần học / hiểu tới đâu**)
- ★ **Hallucination (ảo giác)** — Model tạo câu trả lời trôi chảy, nghe hợp lý nhưng sai sự thật.
  🗨 "Hallucination là khi model nói sai mà nghe rất tự tin."
  🎓 Nói được nguyên nhân: chọn token xác suất cao, không tra cứu sự thật.
- ★ **Kiểm chứng (verification)** — Cách phát hiện câu sai: đối chiếu nguồn, hỏi lại cách khác, dùng RAG.
  🗨 "Có ba cách kiểm chứng: đối chiếu nguồn, hỏi lại theo cách khác, và dùng RAG."
  🎓 Đây là minh họa đề yêu cầu của mục 9.
- ★ **Tiền đề sai (false premise)** — Câu hỏi giả định một điều không có thật ("người Việt đoạt Nobel Vật lý").
  🗨 "Model chấp nhận tiền đề sai rồi bịa tiếp."
  🎓 Câu trả lời đúng phải là chỉ ra tiền đề sai.
- **SimpleQA, TruthfulQA** — Hai bộ đánh giá công bố, nhóm mượn cách chấm Đúng / Từ chối / Sai.
  🗨 "Em chấm theo cách của SimpleQA và TruthfulQA."
  🎓 Chỉ cần nhớ tên và ý.
- **Nhãn gốc sai** — Đáp án trong bộ dữ liệu cũng có thể sai.
  🗨 "Chất lượng còn phụ thuộc dữ liệu: có đáp án gốc bị sai."
  🎓 Ví dụ câu vua Charles '1640'.

✅ **Các ý bắt buộc phải nói ở slide này (theo thứ tự):**
1. Hallucination là gì và vì sao (chọn token hợp lý, không tra cứu).
2. Minh họa đề: một câu trả lời sai + bảng 6 bước kiểm chứng.
3. Ba cách kiểm chứng: nguồn, hỏi lại, RAG.
4. Thực thể không tồn tại: bịa 5/6; phụ thuộc dữ liệu: nhãn sai, chen tiếng Trung.

📐 **Lý thuyết chi tiết (đọc để hiểu sâu; khi nói chỉ lấy ý, không đọc nguyên văn):**

**Định nghĩa.** Hallucination là nội dung trôi chảy, nghe hợp lý nhưng **sai sự thật** hoặc **không khớp nguồn** (Huang et al. 2023 chia thành sai về sự thật và không trung thành với ngữ cảnh).

**Vì sao xảy ra.** (1) Mục tiêu huấn luyện là viết tiếp **hợp lý**, không phải **đúng**: câu sai nhưng trôi chảy vẫn có xác suất cao. (2) Kiến thức nằm nén trong tham số, có thể thiếu hoặc nhầm, nhất là với model nhỏ 1.5B. (3) Model không có cơ chế "không biết thì nói không biết" trừ khi được dạy; ở thí nghiệm này prompt còn không cho phép từ chối. (4) Dữ liệu huấn luyện có lỗi.

**Ba cách kiểm chứng (đã làm trong notebook).**
1. **Đối chiếu nguồn:** UIT-ViQuAD có đáp án và đoạn Wikipedia gốc; cắt đoạn quanh đáp án để đặt cạnh câu trả lời.
2. **Hỏi lại theo cách khác:** cùng nội dung, hai cách hỏi A, B. Nếu model biết thật, hai câu trả lời nên nhất quán; mâu thuẫn là dấu hiệu đoán. Kết quả: sai 12/15 mỗi cách, 11/15 sai cả hai.
3. **Dùng RAG** (mục 8): đưa nguồn vào prompt.

**Câu hỏi có tiền đề sai.** Hỏi "Nguyễn Hoàng Lâm nhận Nobel năm nào?" giả định một điều không có thật. Câu trả lời đúng là chỉ ra tiền đề sai. Chấm theo cách của SimpleQA / TruthfulQA (Đúng / Từ chối / Sai): model bịa 5/6, không lần nào chỉ ra tiền đề sai.

**Chất lượng phụ thuộc dữ liệu.** Đáp án "1640" của bộ dữ liệu bị sai: đánh giá cũng phụ thuộc chất lượng nhãn. Model chen tiếng Trung khi hỏi tiếng Việt: phản ánh tỉ lệ ngôn ngữ trong dữ liệu huấn luyện [suy luận].

**(1) Hallucination và cách kiểm chứng**

🎤 "Sau khả năng là hạn chế. Hạn chế đầu tiên là **hallucination**: model tạo ra câu trả lời nghe rất hợp lý nhưng lại sai. Nguyên nhân nằm ngay ở cơ chế đã trình bày: model chỉ chọn token có xác suất cao theo ngữ cảnh, chứ **không tra cứu sự thật** và cũng không biết mình không biết."

[Mở cell 9.1, chỉ khung ĐẦU VÀO]

💻 Hỏi lại 15 câu có đáp án của mục 8, **không đưa tài liệu, không cho phép từ chối**, mỗi câu hỏi theo 2 cách:
`A: "{câu hỏi} (chủ đề: …) Trả lời ngắn gọn."` · `B: "Về {chủ đề}, hãy cho biết: {câu hỏi} Chỉ trả lời ngắn gọn, không giải thích."`

[Mở cell 9.2]

💻 **Code trên slide (cell 9.1, 9.2)**
```python
PHRASINGS = {"A": lambda r: f"{r.question} (chủ đề: {r.title}) Trả lời ngắn gọn.",
             "B": lambda r: f"Về {r.title}, hãy cho biết: {r.question} Chỉ trả lời ngắn gọn…"}
def evidence(context, answers):          # cắt đoạn nguồn quanh vị trí đáp án
    start = answers["answer_start"][0]
    return context[start - 120 : start + len(answers["text"][0]) + 120]
```
- `PHRASINGS`: **hai cách hỏi cùng một câu**; `r` là một dòng dữ liệu (câu hỏi, chủ đề). Hai cách ra hai câu trả lời mâu thuẫn là dấu hiệu model **đoán**.
- `answer_start`: UIT-ViQuAD ghi sẵn **vị trí ký tự** của đáp án trong đoạn Wikipedia.
- Dòng `return`: **cắt 120 ký tự trước và sau đáp án** → đoạn nguồn ngắn để đặt cạnh câu trả lời của model (bước ⑤ của bảng). Trong notebook có thêm `max(0, …)` để không cắt âm và `assert` đoạn cắt có chứa đáp án.

📊 **Output:** sai **12/15** ở mỗi cách hỏi, sai cả hai cách 11/15, không câu nào từ chối. Bảng 6 bước kiểm chứng: câu hỏi "Muốn ban hành luật thì hoàng đế phải làm gì?" → cách A liệt kê "các bước" tự nghĩ ra → cách B "Đưa ra một đạo luật." → đáp án "phải thông qua Hội đồng" → nguồn Wikipedia "…Hoàng đế phải thông qua Hội đồng để ban hành luật…" → **không khớp nguồn, tức là hallucination**.

🎤 "Đây là minh họa đề yêu cầu: một câu trả lời sai, đặt cạnh đáp án và đoạn văn nguồn. Có **ba cách kiểm chứng**: đối chiếu với nguồn; hỏi lại theo cách khác, vì hai câu trả lời mâu thuẫn là dấu hiệu model đang đoán; và dùng RAG như slide trước. Đặc biệt, khi hỏi về những thứ **không hề tồn tại**, như một người Việt đoạt giải Nobel Vật lý, model bịa ra câu trả lời **5 trên 6 lần**, và không lần nào chỉ ra rằng tiền đề của câu hỏi là sai."

🔎 **Đọc demo trên slide (INPUT → OUTPUT → vì sao):** Bảng 6 bước: ① INPUT câu hỏi (không tài liệu) → ② OUTPUT cách A → ③ hỏi lại cách B, ra câu khác → ④ đáp án đúng → ⑤ đoạn nguồn Wikipedia → ⑥ kết luận hallucination. Ba số: sai 12/15 mỗi cách hỏi, 11/15 sai cả hai, 0 câu từ chối. Bảng phải: 3 câu về thứ không tồn tại, model trả lời như thật.

❓ Sao chắc model sai ở 9.2? → Có đáp án và đoạn nguồn trong bộ dữ liệu.

---

### Slide 13 — Mục 9 (2/2): Thiên lệch, giới hạn ngữ cảnh, chất lượng (1.6 phút)

🎯 **Cô yêu cầu:** cùng nội dung mục 9 — "thiên lệch, giới hạn ngữ cảnh; chất lượng phụ thuộc tác vụ và dữ liệu". → Cell 9.4, 9.5, 9.6.

🧭 **Trọng tâm lý thuyết (nắm chắc trước khi nói; quên lời thì nói đúng các ý này):**
- **Thiên lệch:** đo bằng **xác suất model gán cho câu**, không hỏi model ý kiến. Hai câu chỉ khác "anh ấy" / "chị ấy"; nếu model không thiên lệch thì share_female ≈ 0.5. **Luật kết luận ghi trước khi chạy:** r > 0 và p < 0.05.
- **Giới hạn ngữ cảnh:** thông tin nằm ngoài cửa sổ thì với model **coi như không tồn tại** → model không nói "không biết" mà bịa. Chi phí tăng nhanh theo độ dài vì attention so **mọi cặp token**.
- **Chất lượng phụ thuộc tác vụ và dữ liệu:** cùng model, phân loại cảm xúc 0.92 nhưng hỏi kiến thức không tài liệu 0.00–0.20; trong cùng tác vụ, câu góp ý 0.79 so với 0.96.

📖 **Thuật ngữ của slide này** (★ = bắt buộc nắm; mỗi từ: **là gì** · 🗨 **nói thế nào** · 🎓 **cần học / hiểu tới đâu**)
- ★ **Thiên lệch (bias)** — Model gán thuộc tính (ở đây giới tính) theo khuôn mẫu học từ dữ liệu, dù câu không cho thông tin.
  🗨 "Model tự gán giới tính theo nghề, dù câu không nói gì về giới tính."
  🎓 Hiểu cách đo: so xác suất câu 'anh ấy' với câu 'chị ấy'.
- ★ **Log-probability của câu** — Tổng log xác suất mọi token của câu; càng lớn thì model càng tin câu đó.
  🗨 "Em so xem model tin câu bản nam hay câu bản nữ hơn."
  🎓 Cùng phép tính như loss ở mục 4, chỉ đổi dấu và cộng tổng.
- ★ **share_female** — Tỉ lệ model nghiêng về bản nữ: P(nữ) / (P(nam) + P(nữ)); 0.5 = không nghiêng.
  🗨 "share_female bằng 0.5 là không nghiêng; nhỏ hơn 0.5 là nghiêng về nam."
  🎓 Đọc được hình: trục dọc là share_female.
- ★ **Hệ số tương quan Pearson r** — Đo mức hai đại lượng cùng tăng theo đường thẳng; từ −1 đến 1.
  🗨 "r bằng 0.69: nghề càng nhiều nữ thì model càng chọn đại từ nữ."
  🎓 r > 0 là cùng chiều; r gần 0 là không liên quan.
- ★ **p-value, p < 0.05** — Xác suất thấy kết quả như vậy nếu thật ra không có liên hệ; nhỏ hơn 0.05 thì coi là có ý nghĩa thống kê.
  🗨 "p rất nhỏ nên khó có thể do ngẫu nhiên."
  🎓 Không nói 'p là xác suất giả thuyết đúng'.
- **Đối chứng (control)** — Câu thay tên nghề bằng 'person' / 'người đó' để xem mức mặc định.
  🗨 "Câu đối chứng cho thấy model mặc định nghiêng về nam."
  🎓 —
- ★ **Cửa sổ ngữ cảnh (context window)** — Số token tối đa model đọc một lần (Qwen: 32,768).
  🗨 "Thông tin nằm ngoài cửa sổ thì với model coi như không tồn tại."
  🎓 Hiểu demo: cắt còn 1,024 token → mất câu giấu → bịa '1234'.
- **Needle in a haystack** — Kiểm tra giấu một câu (kim) trong tài liệu dài (đống rơm) rồi hỏi lại.
  🗨 "Em giấu một mã số trong tài liệu 4 nghìn token."
  🎓 —
- **Chi phí bậc hai** — Attention so mọi cặp token: độ dài gấp đôi thì phần này gấp bốn.
  🗨 "Độ dài tăng 16 lần, thời gian tăng khoảng 86 lần."
  🎓 Đọc được hình chi phí.

✅ **Các ý bắt buộc phải nói ở slide này (theo thứ tự):**
1. Thiên lệch: cách đo (cặp câu, share_female, đọc công thức), luật ghi trước r > 0, p < 0.05.
2. Kết quả: r = 0.69 (Anh), 0.44 (Việt); mặc định nghiêng về nam.
3. Giới hạn ngữ cảnh: mất thông tin → bịa "1234"; chi phí tăng nhanh theo độ dài.
4. Chất lượng: 0.92 vs 0.00–0.20; câu góp ý 0.79 vs 0.96.

📐 **Lý thuyết chi tiết (đọc để hiểu sâu; khi nói chỉ lấy ý, không đọc nguyên văn):**

**Đo thiên lệch bằng xác suất, không hỏi ý kiến model.** Với mỗi nghề, tạo hai câu chỉ khác đại từ, ví dụ "Thợ điện nói rằng anh ấy / chị ấy sẽ đến muộn hôm nay". Log xác suất cả câu là tổng log xác suất từng token (cùng phép tính với loss mục 4): $\log P(\text{câu}) = \sum_t \log p(x_t \mid x_{<t})$. Tỉ lệ nghiêng về nữ:
$$\text{share\_female} = \frac{P_F}{P_M + P_F} = \frac{1}{1 + e^{\log P_M - \log P_F}}$$
Bằng 0.5 nghĩa là model không nghiêng; câu không cho biết giới tính thì model không thiên lệch phải cho gần 0.5.

**Kiểm định đã ghi trước khi chạy.** Nếu model học khuôn mẫu nghề nghiệp, nghề càng nhiều nữ trong thực tế thì share_female càng cao. Đo bằng **hệ số tương quan Pearson** $r$ giữa % nữ thật và share_female; kết luận có thiên lệch nếu $r > 0$ và $p < 0.05$. Giá trị $p$ tính từ $t = r\sqrt{n-2}/\sqrt{1-r^2}$ (phân phối Student với $n - 2$ bậc tự do).

**Kết quả.** Anh (WinoGender, 60 nghề, % nữ theo thống kê lao động Mỹ): $r = 0.69$, $p = 8.5 \times 10^{-10}$. Việt (Nemotron-Personas-Vietnam, 122 nghề): $r = 0.44$, $p = 4 \times 10^{-7}$. 7/7 biến thể câu đạt luật. Thêm: phần lớn chấm dưới 0.5 và câu đối chứng "người đó" chỉ 0.01 → model **mặc định nghiêng về nam**.

**Giới hạn ngữ cảnh: hai mặt.**
- **Thông tin:** model chỉ dùng được những gì nằm trong cửa sổ. Câu giấu còn trong input → đúng 6/6; bị cắt ra ngoài → model không nói "không biết" mà bịa "1234".
- **Chi phí:** attention tạo bảng $n \times n$, nên phép tính tăng theo $n^2$ (dài gấp đôi → phần attention gấp bốn). Bộ nhớ KV cache tăng tuyến tính: mỗi token cần $2 \times 28 \times 2 \times 128 \times 2 = 28{,}672$ byte ≈ 28 KB; 32,768 token ≈ 0.88 GB. Đo thật: 512 → 8,192 token (×16) thì thời gian ×86, bộ nhớ đỉnh 13.3 GB / 14.6 GB của T4.

**Chất lượng phụ thuộc tác vụ và dữ liệu.** Cùng model: phân loại cảm xúc 0.92 (tác vụ dễ, đáp án nằm trong câu); hỏi kiến thức không tài liệu 0.00–0.20 (cần kiến thức model không có); có RAG 0.67. Trong cùng tác vụ: câu góp ý chỉ 0.79 so với 0.96, vì quy ước gán nhãn của bộ dữ liệu khác cách đọc tự nhiên.

🎤 "Hallucination là hạn chế đầu tiên. Slide này là ba hạn chế còn lại mà đề nêu: thiên lệch, giới hạn ngữ cảnh, và chất lượng phụ thuộc tác vụ, dữ liệu."

**(2) Thiên lệch**

[Mở cell 9.4, chỉ khung ĐẦU VÀO rồi hình]

🎤 "Hạn chế thứ hai là **thiên lệch**. Để đo, em đưa cho model những câu **không chứa bất kỳ thông tin nào về giới tính**, chỉ đổi tên nghề, ví dụ 'Thợ điện nói rằng **anh ấy** sẽ đến muộn' và 'Thợ điện nói rằng **chị ấy** sẽ đến muộn', rồi so xem model tin câu nào hơn."

💻 **Code**
```python
def sentence_logprob(text):                                   # log P(cả câu)
    lp = F.log_softmax(model(ids).logits[0, :-1].float(), dim=-1)
    return float(lp.gather(1, ids[0, 1:, None]).sum())
def share_female(male_text, female_text):
    return 1 / (1 + math.exp(sentence_logprob(male_text) - sentence_logprob(female_text)))
```
- **Giống cell 4.2, nhưng cộng log-xác suất của mọi token** để được log xác suất của cả câu. Hai câu chỉ khác nhau ở đại từ.

🗣 **Đọc công thức** $\text{share\_female} = \dfrac{P_{\text{nữ}}}{P_{\text{nam}} + P_{\text{nữ}}}$:
"Tỉ lệ nữ bằng **xác suất của câu bản nữ, chia cho tổng xác suất của câu bản nam và câu bản nữ**." Bằng 0.5 nghĩa là model không nghiêng về bên nào.

🎤 "Dữ liệu tham chiếu gồm hai nguồn. Với tiếng Anh, em dùng bộ **WinoGender**: 60 nghề, câu đã được công bố, kèm tỉ lệ nữ thật trong từng nghề theo thống kê lao động Mỹ. Với tiếng Việt, em lấy 122 nghề và tỉ lệ nữ từ bộ **Nemotron-Personas-Vietnam** của NVIDIA, dựa trên số liệu của Tổng cục Thống kê. Luật kết luận được ghi **trước khi chạy**: có thiên lệch theo khuôn mẫu nếu hệ số tương quan **r dương và p nhỏ hơn 0.05**."

🗣 **Đọc khái niệm r và p:** "Hệ số tương quan Pearson r đo **mức các điểm xếp thành một đường dốc lên**: r bằng 1 là thẳng hàng hoàn toàn, r bằng 0 là không liên quan. Giá trị p là **xác suất thấy được mức tương quan này nếu thật ra hai đại lượng không liên quan**; p rất nhỏ nghĩa là khó có thể do ngẫu nhiên."

🖼 **Hình `bias_scatter.png`:** mỗi chấm là một nghề; trục ngang là tỉ lệ nữ thật; trục dọc là `share_female`; đường đen là mức 0.5; đường cam đứt là **đối chứng** (thay tên nghề bằng "person" / "người đó"); đường chấm chéo là y = x.

📊 **Output:** tiếng Anh **r = 0.69** (p = 8.5e-10); tiếng Việt **r = 0.44** (p = 4.0e-07); **7/7 biến thể đạt luật**. Plumber (0.7% nữ) → 0.008, tức model tin câu "he" hơn câu "she" khoảng 124 lần; hygienist (96.4% nữ) → 0.70. Đối chứng: "person" 0.13, "người đó" 0.01.

🎤 "Kết quả: nghề càng nhiều nữ thì model càng chọn đại từ nữ, với **r bằng 0.69 ở tiếng Anh và 0.44 ở tiếng Việt**, dù câu không hề cho biết giới tính. Điều đó cho thấy model đã học **khuôn mẫu nghề nghiệp** từ dữ liệu huấn luyện. Ngoài ra, hầu hết các chấm nằm dưới mức 0.5, và với câu đối chứng 'người đó', model chỉ chọn 'chị ấy' 1%: model còn **mặc định nghiêng về nam**."

**(3) Giới hạn ngữ cảnh**

[Mở cell 9.5]

💻 `haystack(n_tokens, frac)`: ghép các đoạn Wikipedia, chèn câu "Mã số bí mật của kho lưu trữ là 7429." vào đầu, giữa hoặc cuối; `keep_last_tokens(doc, 1024)`: chỉ giữ 1,024 token cuối, giống như khi tài liệu dài hơn cửa sổ ngữ cảnh.

📊 **Output:** câu giấu còn trong input → **đúng 6/6** (1,161 và 4,202 token). Cắt còn 1,024 token cuối khiến câu giấu ở đầu bị mất → model trả lời **"1234"**; câu giấu ở cuối vẫn còn → đúng "7429".

🎤 "Hạn chế thứ ba là **giới hạn ngữ cảnh**. Em giấu một mã số bí mật trong tài liệu dài 4 nghìn token rồi hỏi lại: chừng nào câu đó còn nằm trong input, model tìm đúng cả 6 lần. Nhưng khi tài liệu bị cắt bớt phần đầu và câu chứa mã số rơi ra ngoài cửa sổ, model **không trả lời là không biết, mà bịa ra con số '1234'**."

🖼 **Hình `context_cost.png`:** 512 token mất 0.12 s và 5.8 GB; 8,192 token mất **10.7 s và 13.3 GB** (GPU T4 có 14.6 GB).

🗣 **Đọc công thức KV cache:** "Bộ nhớ KV cache cho mỗi token bằng **2, vì có K và V, nhân 28 layer, nhân 2 head, nhân 128 chiều, nhân 2 byte**, khoảng 28 KB; với 32,768 token là khoảng 0.88 GB."

🎤 "Giới hạn này còn đến từ chi phí tính toán: attention phải so **mọi cặp token**, nên độ dài tăng gấp đôi thì khối lượng phần này tăng gấp bốn. Trong thực nghiệm, độ dài tăng 16 lần thì thời gian tăng khoảng 86 lần. Model cho phép tối đa 32,768 token, nhưng chỉ 8 nghìn token đã dùng hết 13.3 GB, gần chạm giới hạn của GPU T4."

📝 **Hiểu thêm:** "Lost in the middle" (Liu et al. 2023): với ngữ cảnh rất dài, thông tin ở giữa dễ bị bỏ sót hơn ở đầu và cuối; ở độ dài 4 nghìn token chưa thấy hiện tượng này.

**(4) Chất lượng phụ thuộc tác vụ và dữ liệu**

[Mở cell 9.6, chỉ hình `quality_by_data.png`]

📊 Cùng model: phân loại cảm xúc 0.92; hỏi kiến thức không tài liệu 0.00–0.20; có RAG 0.67. Cùng tác vụ: câu có từ góp ý ("nên, cần, mong, thêm, hơn") **0.79** (619 câu), câu còn lại **0.96** (2,380 câu).

🎤 "Cuối cùng, **chất lượng phụ thuộc vào tác vụ và dữ liệu**. Cùng một model, phân loại cảm xúc đúng 92%, nhưng hỏi kiến thức không có tài liệu chỉ đúng từ 0 đến 20%. Ngay trong cùng một tác vụ, câu mang tính góp ý chỉ đúng 79%, so với 96% ở các câu còn lại."

> Nếu bị dồn thời gian: phần (3) chỉ nói một câu "cắt mất thông tin thì model bịa ra '1234'".

🔎 **Đọc demo trên slide (INPUT → OUTPUT → vì sao):** **Thiên lệch** — INPUT: cặp câu "Thợ điện nói rằng anh ấy / chị ấy…"; OUTPUT: share_female mỗi nghề. Hình: mỗi chấm một nghề, ngang = % nữ thật, dọc = share_female, đường đen 0.5. Kết quả: Anh r = 0.69, Việt r = 0.44, đều p < 0.05, chấm phần lớn dưới 0.5. **Ngữ cảnh** — INPUT: tài liệu ~4,000 token có câu giấu mã 7429; OUTPUT: còn câu giấu → 7429 (6/6), cắt mất → "1234". Hình chi phí: 512 → 8,192 token, thời gian ×86, bộ nhớ 13.3 GB.

❓ Model phản ánh đúng thực tế thì có sai không? → Câu không cho biết giới tính mà model vẫn tự gán, và nghiêng về nam cả ở nghề đa số là nữ. ❓ Tỉ lệ nữ tiếng Việt có chính xác? → Là dữ liệu tổng hợp bám thống kê, chỉ gần đúng; đã ghi rõ trên slide.

---

### Slide 14 — Mục 10: Thử nghiệm có kiểm soát (2.0 phút)

🎯 **Cô yêu cầu:** "Thử nghiệm có kiểm soát, tổng hợp kết quả và nhận xét" — minh họa "bảng kết quả…"; yêu cầu thêm 4: slide có **thiết lập, kết quả, phân tích**. → Cell 7.3, 10.1, 10.2.

🧭 **Trọng tâm lý thuyết (nắm chắc trước khi nói; quên lời thì nói đúng các ý này):**
- **Thử nghiệm có kiểm soát** = trả lời **một câu hỏi** bằng cách **chỉ đổi một biến** (0 hay 4 ví dụ), giữ nguyên model, dữ liệu, prompt, cách sinh, seed; luật chấm ghi trước.
- **Thiết lập → kết quả → phân tích** (yêu cầu thêm 4): accuracy kèm **khoảng tin cậy 95%**, rồi kiểm định **McNemar** vì hai cách chạy trên **cùng bộ câu** (dữ liệu ghép cặp).
- Thông tin nằm ở **câu bất đồng** (b = 47, c = 120); nếu hai cách ngang nhau, b ≈ c như tung đồng xu → p = 1.5e-08 nên kết luận few-shot tốt hơn có ý nghĩa thống kê.

📖 **Thuật ngữ của slide này** (★ = bắt buộc nắm; mỗi từ: **là gì** · 🗨 **nói thế nào** · 🎓 **cần học / hiểu tới đâu**)
- ★ **Thử nghiệm có kiểm soát** — Chỉ đổi một biến, giữ nguyên mọi thứ khác, để biết thay đổi do biến đó gây ra.
  🗨 "Giữa hai cách chỉ khác đúng một biến: có ví dụ hay không."
  🎓 Liệt kê được những gì được giữ nguyên.
- ★ **Khoảng tin cậy 95% (CI)** — Khoảng ước lượng accuracy thật: p̂ ± 1.96·√(p̂(1−p̂)/n).
  🗨 "Accuracy 89.86% cộng trừ 1.08 điểm phần trăm."
  🎓 Đọc được công thức; biết n càng lớn thì khoảng càng hẹp.
- ★ **Dữ liệu ghép cặp** — Hai cách chạy trên cùng các câu, nên so từng câu một.
  🗨 "Vì cùng bộ câu, em dùng kiểm định cho dữ liệu ghép cặp."
  🎓 Lý do không so hai CI độc lập.
- ★ **Kiểm định McNemar** — Chỉ xét câu hai cách bất đồng (b, c); nếu hai cách ngang nhau thì b ≈ c như tung đồng xu.
  🗨 "McNemar chỉ nhìn 167 câu mà hai cách cho kết quả khác nhau."
  🎓 Đọc bảng 2 × 2; biết a, d không mang thông tin.
- **Giả thuyết không, phân phối nhị thức** — Giả thuyết không: hai cách tốt như nhau → b ~ Bin(b + c, 0.5).
  🗨 "Nếu hai cách như nhau, 47 so với 120 gần như không thể xảy ra."
  🎓 Biết `exact=True` dùng nhị thức chính xác.
- **Batch, padding trái** — Sinh 16 câu một lô; câu ngắn được đệm thêm token ở bên trái cho bằng độ dài.
  🗨 "Em sinh theo lô 16 câu để chạy nhanh."
  🎓 Đệm trái để model sinh tiếp ngay sau token cuối của từng prompt.

✅ **Các ý bắt buộc phải nói ở slide này (theo thứ tự):**
1. Câu hỏi thực nghiệm và thiết lập: chỉ đổi một biến (yêu cầu thêm 4: thiết lập · kết quả · phân tích).
2. Kết quả kèm CI 95% (đọc công thức).
3. Vì sao McNemar; đọc bảng b = 47, c = 120; p = 1.5e-08 (đọc công thức).
4. Giới hạn: 1 model, 1 bộ dữ liệu, 1 bộ ví dụ.

📐 **Lý thuyết chi tiết (đọc để hiểu sâu; khi nói chỉ lấy ý, không đọc nguyên văn):**

**Thử nghiệm có kiểm soát.** Muốn kết luận "thêm ví dụ làm accuracy tăng", mọi yếu tố khác phải giữ nguyên: model, dữ liệu test, mẫu prompt, cách sinh (greedy, 10 token), seed, luật đọc nhãn. Chỉ **một biến độc lập**: 0 hay 4 ví dụ. Mẫu prompt chọn trên validation để không tối ưu theo tập test.

**Khoảng tin cậy 95%.** Accuracy $\hat p$ là tỉ lệ trên mẫu $n$ câu, có sai số ngẫu nhiên. Xấp xỉ chuẩn của phân phối nhị thức: sai số chuẩn $\text{SE} = \sqrt{\hat p(1-\hat p)/n}$; khoảng 95% là $\hat p \pm 1.96\,\text{SE}$. Zero-shot: $\sqrt{0.8986 \times 0.1014/2{,}999} = 0.0055$, nhân 1.96 được ±1.08 điểm phần trăm.

**Vì sao không so hai khoảng tin cậy.** Hai cách chạy trên **cùng** 2,999 câu nên kết quả **ghép cặp**: câu khó thì cả hai cùng dễ sai. So hai khoảng tin cậy độc lập bỏ qua thông tin này.

**McNemar.** Bảng 2 × 2 theo từng câu: a cả hai đúng, d cả hai sai (không cho biết cách nào hơn); b chỉ zero-shot đúng, c chỉ few-shot đúng. Giả thuyết không $H_0$: hai cách tốt như nhau → mỗi câu bất đồng nghiêng về bên nào cũng như tung đồng xu → $b \sim \text{Bin}(b + c, 0.5)$. Bản chính xác: $p = 2\,P(X \le \min(b, c))$ với $X \sim \text{Bin}(167, 0.5)$, ra $1.5 \times 10^{-8}$. Bản xấp xỉ chi bình phương: $(b - c)^2/(b + c) = 73^2/167 = 31.9$, cùng kết luận. Chênh lệch accuracy cũng tính được từ b, c: $(c - b)/n = 73/2{,}999 \approx 2.4$ điểm phần trăm.

**Giới hạn của kết luận.** 1 model, 1 bộ dữ liệu, 1 bộ ví dụ few-shot, 1 thứ tự. Kết luận chỉ cho thiết lập này, không suy rộng cho "LLM nói chung".

🎤 "Mục cuối là thử nghiệm có kiểm soát. Câu hỏi đặt ra: **thêm bốn ví dụ vào prompt có làm accuracy khác zero-shot một cách có ý nghĩa thống kê hay không?**
Về **thiết lập**: model Qwen2.5-1.5B-Instruct, float16, GPU T4; dữ liệu là toàn bộ **2,999 câu** test của UIT-VSFC, gồm 1,590 câu tích cực và 1,409 câu tiêu cực; mẫu prompt chọn trên 200 câu validation; **chỉ thay đổi một biến** là có 0 hay 4 ví dụ; sinh bằng greedy, tối đa 10 token, seed 42; câu không đọc được nhãn tính là sai."

[Mở cell 7.3]

💻 **Code**
```python
tok.padding_side = "left"                                # đệm bên trái
enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
out = model.generate(**enc, max_new_tokens=ICL_MAX_NEW_TOKENS, do_sample=False, ...)
assert classify_batch(probe) == classify_batch(probe)  # chạy 2 lần ra y hệt
```
- **Sinh 16 câu một lô** để chạy nhanh; prompt dài ngắn khác nhau nên phải đệm cho bằng; **đệm bên trái** để model sinh tiếp ngay sau token cuối của từng prompt.

🎤 "Để chạy hết 2,999 câu trong khoảng 5 phút, em sinh theo lô 16 câu một lần. Câu lệnh kiểm tra cuối cell chạy cùng một lô hai lần và xác nhận kết quả giống hệt nhau, tức là greedy thật sự lặp lại được."

[Mở cell 10.1]

💻 `accuracy = mean(pred == label)`; `ci95 = 1.96 * sqrt(acc * (1 - acc) / n)`

🗣 **Đọc công thức khoảng tin cậy 95%** $\hat p \pm 1.96\sqrt{\hat p(1-\hat p)/n}$:
"Accuracy **cộng trừ 1.96 nhân căn bậc hai của p nhân một trừ p, chia cho n**." Với zero-shot: 1.96 nhân căn của 0.8986 nhân 0.1014 chia 2,999, bằng khoảng 0.0108, tức cộng trừ 1.08 điểm phần trăm.

🖼 **Hình `accuracy_icl.png`:** 2 cột accuracy; thanh đen là khoảng tin cậy 95%.

📊 zero-shot **89.86% ± 1.08**, few-shot **92.30% ± 0.95**; không câu nào lỗi đọc nhãn.

🎤 "Kết quả là 89.86% so với 92.30%. Tuy nhiên, vì hai cách được chạy trên **cùng một bộ câu**, so hai khoảng tin cậy độc lập là chưa đúng phương pháp. Kiểm định phù hợp cho dữ liệu ghép cặp như thế này là **McNemar**."

[Mở cell 10.2]

💻 `mcnemar_counts(zero, few)` → `[[a, b], [c, d]]`; `mcnemar(table, exact=True)`

📊 **Bảng 2 × 2**

| | Few-shot đúng | Few-shot sai |
|---|---|---|
| **Zero-shot đúng** | a = 2,648 | **b = 47** |
| **Zero-shot sai** | **c = 120** | d = 184 |

🎤 "Trong bảng này, ô a là những câu cả hai cách cùng đúng, ô d là những câu cả hai cùng sai; hai ô này không cho biết cách nào tốt hơn. Thông tin nằm ở **167 câu mà hai cách cho kết quả khác nhau**: b bằng 47 câu chỉ zero-shot đúng, c bằng 120 câu chỉ few-shot đúng. Nếu hai cách thực sự tốt như nhau, mỗi câu bất đồng nghiêng về bên nào cũng như **tung một đồng xu**, nên b và c phải xấp xỉ nhau, quanh 83."

🗣 **Đọc công thức McNemar exact:** "Dưới giả thuyết hai cách như nhau, **b tuân theo phân phối nhị thức với b cộng c lần thử và xác suất 0.5**. Giá trị p bằng **hai lần xác suất để biến nhị thức đó nhỏ hơn hoặc bằng giá trị nhỏ hơn trong b và c**", tức $p = 2P(X \le 47)$ với $X \sim \text{Bin}(167, 0.5)$, bằng $1.5 \times 10^{-8}$.

🎤 "Thực tế là 47 so với 120, và kiểm định nhị thức chính xác cho **p bằng 1.5 nhân 10 mũ trừ 8**. Một chênh lệch như vậy gần như không thể xảy ra do ngẫu nhiên, nên có thể kết luận few-shot tốt hơn zero-shot **một cách có ý nghĩa thống kê**. Về giới hạn: nghiên cứu chỉ dùng một model, một bộ dữ liệu và **một bộ ví dụ few-shot**; đổi ví dụ hoặc thứ tự ví dụ có thể cho kết quả khác, nên em không suy rộng kết luận cho LLM nói chung."

📝 **Hiểu thêm:** chỉ có 1 phép so sánh nên không cần hiệu chỉnh Bonferroni. Dietterich (1998): McNemar phù hợp khi so hai phương pháp trên cùng một tập test.

🔎 **Đọc demo trên slide (INPUT → OUTPUT → vì sao):** Hàng ô trên cùng: thiết lập (model · dữ liệu · prompt · **biến duy nhất** · cách sinh). Biểu đồ: 2 cột accuracy, vạch đen là CI 95%. Bảng 2 × 2 McNemar: a, d không dùng; b = 47, c = 120. Khung xanh: p = 1.5e-08 < 0.05.

❓ b, c là gì? → b: câu chỉ zero-shot đúng; c: câu chỉ few-shot đúng. ❓ Sao `exact=True`? → Dùng phân phối nhị thức chính xác, không cần xấp xỉ chi bình phương.

---

### Slide 15 — Phân tích lỗi và tổng hợp (1.5 phút)

🎯 **Cô yêu cầu:** minh họa "**ví dụ đầu ra và phân tích lỗi**" + "tổng hợp kết quả và nhận xét". → Cell 10.1 (confusion matrix), 10.3, 10.4.

🧭 **Trọng tâm lý thuyết (nắm chắc trước khi nói; quên lời thì nói đúng các ý này):**
- **Phân tích lỗi có hệ thống:** nhóm lỗi **định nghĩa trước khi đọc**, lấy mẫu ngẫu nhiên có seed → tránh chấm theo cảm tính.
- **Confusion matrix** cho biết few-shot cải thiện ở đâu: tiêu cực bị đoán tích cực 248 → 145.
- Nguyên nhân chính: **câu góp ý** bộ dữ liệu gán tiêu cực; ví dụ few-shot dạy model **quy ước gán nhãn**, không dạy kiến thức mới.
- **Kết luận cả bài:** LLM đoán token tiếp theo → học bằng cross-entropy → post-training để làm theo chỉ dẫn → sinh từng token → học từ ví dụ, bổ sung tri thức bằng RAG → nhưng vẫn bịa, thiên lệch, bị giới hạn ngữ cảnh → phải kiểm chứng.

📖 **Thuật ngữ của slide này** (★ = bắt buộc nắm; mỗi từ: **là gì** · 🗨 **nói thế nào** · 🎓 **cần học / hiểu tới đâu**)
- ★ **Confusion matrix (ma trận nhầm lẫn)** — Bảng: hàng nhãn thật, cột nhãn đoán; đường chéo là đúng, ô ngoài là lỗi.
  🗨 "Few-shot giảm lỗi câu tiêu cực bị đoán tích cực từ 248 xuống 145."
  🎓 Đọc được từng ô.
- ★ **Phân tích lỗi (error analysis)** — Đọc các câu sai, xếp vào nhóm nguyên nhân để biết vì sao sai.
  🗨 "Em lấy ngẫu nhiên 30 câu sai và phân vào 5 nhóm định nghĩa trước."
  🎓 Hai nguyên tắc: nhóm ghi trước, mẫu ngẫu nhiên có seed.
- **Mẫu ngẫu nhiên (random sample)** — Chọn ngẫu nhiên một phần để đọc, không chọn câu mình thích.
  🗨 "30 câu lấy ngẫu nhiên với seed 42."
  🎓 —
- ★ **Quy ước gán nhãn** — Cách bộ dữ liệu quyết định nhãn (UIT-VSFC: câu góp ý = tiêu cực).
  🗨 "Few-shot giúp model hiểu cách gán nhãn của chính bộ dữ liệu."
  🎓 Đây là ý kết luận quan trọng nhất của phân tích lỗi.

✅ **Các ý bắt buộc phải nói ở slide này (theo thứ tự):**
1. Confusion matrix: few-shot giảm lỗi tiêu cực → tích cực 248 → 145.
2. Phân tích lỗi: nhóm ghi trước, 30 câu ngẫu nhiên; nhóm lớn nhất là câu góp ý 20/30 (minh họa đề).
3. Tổng hợp đọc từ file; kết luận cả bài: học → sinh → khả năng → hạn chế → phải kiểm chứng.

📐 **Lý thuyết chi tiết (đọc để hiểu sâu; khi nói chỉ lấy ý, không đọc nguyên văn):**

**Confusion matrix.** Hàng là nhãn thật, cột là nhãn đoán. Đường chéo là đúng; hai ô lệch là hai loại lỗi khác nhau. Zero-shot: tiêu cực → tích cực 248, tích cực → tiêu cực 56. Few-shot: 145 và 86. Tổng lỗi 304 → 231. Few-shot sửa chủ yếu loại lỗi thứ nhất, đổi lại tăng nhẹ loại thứ hai: ví dụ đã đổi **cách model đặt ranh giới** giữa hai nhãn.

**Quy trình phân tích lỗi đáng tin.** (1) Định nghĩa nhóm lỗi **trước khi đọc** (a–e); (2) lấy **mẫu ngẫu nhiên có seed** (30 / 231 câu) thay vì chọn câu mình thấy thú vị; (3) gán nhóm từng câu, ghi lý do; (4) kiểm tra lại trên toàn bộ câu sai bằng quy tắc đơn giản (câu chứa từ góp ý chiếm 76% / 81% lỗi tiêu cực → tích cực).

**Kết quả và giải thích.** Nhóm lớn nhất là câu góp ý (20/30): UIT-VSFC quy ước góp ý = tiêu cực, còn giọng văn lịch sự khiến model đoán tích cực. Hai ví dụ tiêu cực trong prompt đều là câu góp ý, nên model nhận ra quy ước này. Few-shot không dạy kiến thức mới; nó dạy **cách gán nhãn của bộ dữ liệu**.

**Tổng hợp kết quả.** Bảng tổng hợp một dòng mỗi mục, code đọc trực tiếp từ các file kết quả (không chép tay), nên số trên slide và trong notebook luôn khớp.

🖼 **Hình `confusion_matrix_icl.png`** (trái zero-shot, phải few-shot): hàng là nhãn thật, cột là nhãn model đoán (tích cực, tiêu cực, không đọc được); các ô trên đường chéo là đoán đúng.

| | Zero-shot | Few-shot |
|---|---|---|
| Tiêu cực bị đoán tích cực | **248** | **145** |
| Tích cực bị đoán tiêu cực | 56 | 86 |

🎤 "Few-shot cải thiện ở đâu? Ma trận nhầm lẫn cho thấy few-shot giảm mạnh lỗi **câu tiêu cực bị đoán là tích cực**, từ 248 xuống 145; đổi lại, lỗi theo chiều ngược tăng nhẹ từ 56 lên 86. Tổng số câu sai giảm từ 304 xuống 231."

[Trên slide, không mở notebook — cell 10.3]

💻 **Code trên slide (cell 10.3)**
```python
ERROR_GROUPS = {"a": "không parse được", "b": "nhãn gốc có vẻ sai",
    "c": "vừa khen vừa chê", "d": "ý ngầm (góp ý)",
    "e": "model sai rõ ràng"}            # ghi TRƯỚC khi đọc
wrong = icl_few[~icl_few["correct"]]
err = wrong.sample(30, random_state=SEED)  # SEED = 42
```
- `ERROR_GROUPS`: **5 nhóm lỗi viết ra trước khi đọc câu sai**, để không đặt nhóm theo cái mình thấy.
- `wrong`: lọc các câu few-shot đoán sai (`~` là phủ định: `correct` = False) → 231 câu.
- `sample(30, random_state=SEED)`: **lấy ngẫu nhiên 30 câu, cố định seed** để ai chạy lại cũng ra đúng 30 câu đó. Nhóm (b)–(e) gán tay cho từng câu ở `MANUAL_ERROR_GROUPS`; nhóm (a) code tự đếm.

📊 30 câu few-shot sai: (d) ý ngầm / góp ý **20**, (c) vừa khen vừa chê 4, (e) model sai rõ 4, (b) nhãn gốc có vẻ sai 2, (a) không đọc được nhãn 0. Trên toàn bộ câu sai: câu góp ý chiếm 76% lỗi "tiêu cực → tích cực" ở zero-shot và 81% ở few-shot.

🎤 "Để hiểu nguyên nhân, em lấy ngẫu nhiên 30 câu few-shot làm sai và phân vào năm nhóm lỗi đã **định nghĩa sẵn trước khi đọc**, để tránh chấm theo cảm tính. Nhóm lớn nhất là **câu góp ý, 20 trên 30 câu**, ví dụ 'giảm thời lượng lý thuyết, tăng thời lượng thực hành'. Nguyên nhân là bộ dữ liệu quy ước câu góp ý mang nhãn **tiêu cực**, trong khi model đọc giọng văn lịch sự nên đoán là tích cực. Hai ví dụ tiêu cực trong prompt few-shot đều là câu góp ý, nên model nhận ra quy ước này. Nói cách khác, few-shot không dạy model kiến thức mới, mà giúp model hiểu **cách gán nhãn của chính bộ dữ liệu**."

[Chỉ bảng tổng hợp, cell 10.4]

🎤 "Bảng tổng hợp này được code **đọc trực tiếp từ các file kết quả**, không chép tay. Để kết lại: LLM là một model **đoán token tiếp theo**; nó học bằng cross-entropy trên lượng văn bản khổng lồ, rồi qua post-training để làm theo chỉ dẫn; nó sinh văn bản bằng cách chọn từng token một; nó học được từ ví dụ trong prompt và nhận thêm tri thức qua RAG. Nhưng nó vẫn có thể **bịa, thiên lệch và bị giới hạn bởi ngữ cảnh**, vì vậy mọi output đều cần được kiểm chứng."

🔎 **Đọc demo trên slide (INPUT → OUTPUT → vì sao):** Confusion matrix: hàng nhãn thật, cột nhãn đoán; trái zero-shot, phải few-shot; ô lệch đường chéo là lỗi. Bảng nhóm lỗi: 30 câu few-shot sai, (d) góp ý 20 câu kèm ví dụ thật. Bảng tổng hợp: một dòng mỗi mục, **code đọc từ file kết quả**, không chép tay.

---

### Slide 16 — Tài liệu tham khảo (0.2 phút)

🎤 "Danh sách tài liệu tham khảo nằm ở slide cuối và trong file `references.md` của repo. Em xin cảm ơn cô đã lắng nghe, em sẵn sàng trả lời câu hỏi."

---

## Phần D. Bảng thời gian và đối chiếu yêu cầu của cô

| Slide | Nội dung | Yêu cầu của cô được đáp ứng | Phút |
|---|---|---|---|
| 1 | Mở đầu | Mục tiêu của đề | 0.3 |
| 2 | Model, môi trường | Yêu cầu thêm 1, 2 | 1.0 |
| 3 | Mục 1 | LM / LLM / chatbot; 4 tác vụ | 1.3 |
| 4 | Mục 2 (1/2) | Vai trò Transformer, decoder-only, causal attention (3 sơ đồ kiến trúc) | 1.0 |
| 5 | Mục 2 (2/2) | **Sơ đồ token → xác suất**, mask, attention thật, top-5 | 1.0 |
| 6 | Mục 3 | Token, token ID, embedding; **tokenize câu Việt và câu Anh** | 1.2 |
| 7 | Mục 4 | Next-token, cross-entropy; **cặp ngữ cảnh → mục tiêu** | 1.3 |
| 8 | Mục 5 | Pretraining, SFT, alignment, RLHF/DPO; **sơ đồ mục tiêu và dữ liệu**; demo (góp ý 11/11) | 1.5 |
| 9 | Mục 6 | Autoregressive, greedy, temperature, top-p; **cùng prompt, nhiều cấu hình** | 1.5 |
| 10 | Mục 7 | ICL, làm theo chỉ dẫn; **zero-shot vs few-shot cùng tác vụ** | 1.3 |
| 11 | Mục 8 | Prompting / fine-tuning / RAG; **bảng so sánh**; demo (góp ý 11/11) | 1.5 |
| 12 | Mục 9 (1/2) | Hallucination; **câu sai + kiểm chứng** | 1.2 |
| 13 | Mục 9 (2/2) | Thiên lệch, giới hạn ngữ cảnh, chất lượng theo tác vụ / dữ liệu | 1.6 |
| 14 | Mục 10 | Thử nghiệm có kiểm soát; **bảng kết quả**; yêu cầu thêm 4 | 2.0 |
| 15 | Mục 10 | **Ví dụ đầu ra, phân tích lỗi**, tổng hợp | 1.5 |
| 16 | Tài liệu tham khảo | Sản phẩm nộp số 4 | 0.2 |
| | **Tổng** | Yêu cầu thêm 5: giải thích từng dòng (khung 💻 trên mọi slide nội dung); yêu cầu thêm 6: mở notebook ở slide 5, 8, 12, 13, 14 | **19.4** |

Khi tập: bấm giờ từng slide; slide nào vượt quá 0.3 phút thì rút phần đọc công thức xuống còn câu ý nghĩa, hoặc bỏ bước mở notebook ở slide đó (giữ mở notebook ít nhất ở slide 5, 8, 13, 14 vì đó là bằng chứng chính).

## Phụ lục: đọc và phát biểu mọi công thức

Mỗi công thức có 5 phần: **ký hiệu** (từng chữ là gì) · **đọc thành lời** (đọc nguyên văn khi trình bày) · **phát biểu ý nghĩa** (nói bằng câu thường ngay sau khi đọc) · **ví dụ số** (số thật trong notebook). Mẹo nói: đọc công thức **một lần**, rồi nói ngay câu ý nghĩa; không đọc ký hiệu mà không giải nghĩa.

### 1. Quy tắc nhân xác suất (slide 3)

$$P(x_1, \dots, x_n) = \prod_{t=1}^{n} P(x_t \mid x_1, \dots, x_{t-1})$$

- **Ký hiệu:** $x_t$: token thứ t · $\prod$: tích · "$\mid$": khi biết
- **Đọc thành lời:** "P của x một đến x n bằng tích, t chạy từ 1 đến n, của P x t khi biết x một đến x t trừ một."
- **Phát biểu ý nghĩa:** Xác suất của cả câu bằng tích xác suất từng token khi biết các token đứng trước nó. Vì vậy chỉ cần học đoán token tiếp theo.
- **Ví dụ số:** P("Hôm nay trời đẹp") = P(Hôm) · P(nay | Hôm) · P(trời | Hôm nay) · P(đẹp | Hôm nay trời).

### 2. Bộ nhớ trọng số (slide 2)

$$M \approx N \times b$$

- **Ký hiệu:** $N$: số tham số · $b$: số byte mỗi tham số
- **Đọc thành lời:** "Bộ nhớ xấp xỉ bằng số tham số nhân số byte mỗi tham số."
- **Phát biểu ý nghĩa:** Kích thước model trên GPU tỉ lệ với số tham số và kiểu số.
- **Ví dụ số:** 1,543,714,304 × 2 byte ≈ 3.1 GB (float16).

### 3. Self-attention có mask (slide 4)

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}} + M\right) V$$

- **Ký hiệu:** $Q, K, V$: ma trận query, key, value · $K^\top$: K chuyển vị · $d$: số chiều mỗi head (128) · $M$: mặt nạ
- **Đọc thành lời:** "Attention của Q, K, V bằng softmax của Q nhân K chuyển vị chia căn bậc hai của d, cộng M; tất cả nhân V."
- **Phát biểu ý nghĩa:** Mỗi token chấm điểm mức liên quan với các token được phép nhìn, đổi điểm thành trọng số, rồi lấy trung bình có trọng số thông tin của chúng.
- **Ví dụ số:** Hàng "trời" chỉ có trọng số ở Hôm, nay, trời; ô "rất" bằng 0.

### 4. Causal mask (slide 4, 5)

$$M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}$$

- **Ký hiệu:** $i$: token đang xét (hàng) · $j$: token được nhìn (cột)
- **Đọc thành lời:** "M i j bằng 0 nếu j nhỏ hơn hoặc bằng i, bằng âm vô cùng nếu j lớn hơn i."
- **Phát biểu ý nghĩa:** Token chỉ được nhìn chính nó và token trước; ô tương lai sau softmax bằng 0.
- **Ví dụ số:** Câu 8 token: 36 ô được nhìn (tam giác dưới), 28 ô bị chặn.

### 5. Một decoder block (slide 4)

$$h = x + \text{Attn}(\text{RMSNorm}(x)),\quad y = h + \text{FFN}(\text{RMSNorm}(h))$$

- **Ký hiệu:** $x$: đầu vào block · $h$: kết quả sau attention · $y$: đầu ra block
- **Đọc thành lời:** "h bằng x cộng attention của RMSNorm x; y bằng h cộng FFN của RMSNorm h."
- **Phát biểu ý nghĩa:** Mỗi block chuẩn hóa, trộn thông tin giữa các token (attention), xử lý từng token (FFN), và luôn cộng lại đầu vào (residual).
- **Ví dụ số:** Lặp 28 lần; kích thước giữ nguyên (n, 1,536).

### 6. Softmax (slide 5)

$$p_i = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$$

- **Ký hiệu:** $z_i$: logit (điểm thô) của token i · $e$: cơ số log tự nhiên ≈ 2.718
- **Đọc thành lời:** "p i bằng e mũ z i, chia cho tổng theo j của e mũ z j."
- **Phát biểu ý nghĩa:** Biến điểm thành xác suất: dương, cộng bằng 1, điểm càng cao xác suất càng lớn.
- **Ví dụ số:** Sau "Hôm nay trời rất": " đẹp" 0.788, " nắng" 0.063.

### 7. Embedding (slide 6)

$$e_t = E[t, :] = \text{onehot}(t)^\top E$$

- **Ký hiệu:** $E$: ma trận 151,936 × 1,536 · $t$: token ID
- **Đọc thành lời:** "e t bằng hàng thứ t của ma trận E."
- **Phát biểu ý nghĩa:** Vector của token là hàng thứ ID của bảng embedding; vector học được, ID thì không.
- **Ví dụ số:** Token "H" (ID 39) → hàng 39: [0.023, −0.025, 0.003, …].

### 8. Cross-entropy loss (slide 7)

$$\mathcal{L} = -\frac{1}{N}\sum_{t=1}^{N} \ln p_\theta(x_{t+1} \mid x_{\le t})$$

- **Ký hiệu:** $N$: số cặp · $p_\theta$: xác suất model cho · $\ln$: log tự nhiên
- **Đọc thành lời:** "L bằng trừ một phần N nhân tổng, t từ 1 đến N, của ln xác suất model cho token x t cộng một, khi biết các token đến t."
- **Phát biểu ý nghĩa:** Trung bình của trừ log xác suất token đúng: đoán chắc đúng → 0; tự tin mà sai → rất lớn.
- **Ví dụ số:** 7 cặp, loss trung bình 2.985; cặp "H → ôm": −ln 0.0003 = 8.19.

### 9. Perplexity (slide 7)

$$\text{PPL} = e^{\mathcal{L}}$$

- **Ký hiệu:** $\mathcal{L}$: cross-entropy trung bình
- **Đọc thành lời:** "Perplexity bằng e mũ loss."
- **Phát biểu ý nghĩa:** Model phân vân như đang chọn đều giữa khoảng PPL token.
- **Ví dụ số:** e^2.985 ≈ 19.8; đoán bừa: e^11.93 = 151,936.

### 10. Loss SFT (slide 8)

$$\mathcal{L}_{SFT} = -\sum_{t \in \text{trả lời}} \ln \pi_\theta(y_t \mid x, y_{<t})$$

- **Ký hiệu:** $x$: chỉ dẫn · $y_t$: token thứ t của câu trả lời · $\pi_\theta$: model
- **Đọc thành lời:** "L SFT bằng trừ tổng, chỉ trên các token trả lời, của ln pi theta của y t khi biết x và các y trước t."
- **Phát biểu ý nghĩa:** Vẫn là cross-entropy, nhưng chỉ chấm phần câu trả lời: model học cách trả lời, không học đoán câu hỏi.
- **Ví dụ số:** —

### 11. Mục tiêu RLHF (slide 8)

$$\max_\theta\; \mathbb{E}_{y \sim \pi_\theta}\big[r(x, y)\big] - \beta\,\text{KL}(\pi_\theta \,\|\, \pi_{ref})$$

- **Ký hiệu:** $r$: reward model · $\beta$: hệ số phạt · KL: độ lệch giữa hai phân phối · $\pi_{ref}$: model tham chiếu (SFT)
- **Đọc thành lời:** "Cực đại theo theta của kỳ vọng điểm thưởng, trừ beta nhân KL giữa model đang học và model tham chiếu."
- **Phát biểu ý nghĩa:** Được điểm cao theo sở thích con người, nhưng không đi quá xa bản ban đầu.
- **Ví dụ số:** —

### 12. Mô hình Bradley–Terry (slide 8)

$$P(y_w \succ y_l) = \sigma\big(r(x, y_w) - r(x, y_l)\big)$$

- **Ký hiệu:** $y_w$: câu được chọn · $y_l$: câu bị loại · $\sigma(u) = 1/(1+e^{-u})$
- **Đọc thành lời:** "Xác suất câu y w được ưa hơn y l bằng sigmoid của hiệu điểm thưởng hai câu."
- **Phát biểu ý nghĩa:** Chênh điểm càng lớn thì càng chắc câu kia tốt hơn; dùng để huấn luyện reward model.
- **Ví dụ số:** —

### 13. Loss DPO (slide 8)

$$\mathcal{L}_{DPO} = -\ln \sigma\!\left(\beta \ln\frac{\pi_\theta(y_w)}{\pi_{ref}(y_w)} - \beta \ln\frac{\pi_\theta(y_l)}{\pi_{ref}(y_l)}\right)$$

- **Ký hiệu:** $\pi_\theta$: model đang học · $\pi_{ref}$: model tham chiếu · $\beta$: độ mạnh
- **Đọc thành lời:** "L DPO bằng trừ ln sigmoid của: beta ln tỉ số pi theta trên pi ref của câu tốt, trừ beta ln tỉ số đó của câu kém."
- **Phát biểu ý nghĩa:** Loss nhỏ khi model tăng xác suất câu tốt và giảm câu kém nhiều hơn mức của model tham chiếu; không cần reward model.
- **Ví dụ số:** —

### 14. Temperature (slide 9)

$$p_i^{(T)} = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$

- **Ký hiệu:** $T$: nhiệt độ > 0
- **Đọc thành lời:** "p i ở nhiệt độ T bằng e mũ z i chia T, chia tổng e mũ z j chia T."
- **Phát biểu ý nghĩa:** Chia mọi điểm cho T trước softmax: T < 1 nhọn hơn, T > 1 phẳng hơn; T → 0 giống greedy.
- **Ví dụ số:** P("H"): 0.995 (T = 0.5), 0.847 (T = 1), 0.468 (T = 1.5).

### 15. Greedy (slide 9)

$$x_{t+1} = \arg\max_i\; p_i$$

- **Ký hiệu:** $\arg\max_i$: chỉ số i làm p i lớn nhất
- **Đọc thành lời:** "x t cộng một bằng arg max theo i của p i."
- **Phát biểu ý nghĩa:** Luôn chọn token có xác suất cao nhất; tất định.
- **Ví dụ số:** Bước 1 chọn "H" (0.847).

### 16. Top-p (slide 9)

$$V_p = \text{tập nhỏ nhất } S \text{ sao cho } \sum_{i \in S} p_i \ge p$$

- **Ký hiệu:** token xếp theo xác suất giảm dần
- **Đọc thành lời:** "V p là tập nhỏ nhất các token có tổng xác suất lớn hơn hoặc bằng p."
- **Phát biểu ý nghĩa:** Chỉ bốc trong nhóm đầu bảng; nhóm co giãn theo độ chắc chắn của model.
- **Ví dụ số:** T = 1.5: cần 135 token mới đạt 90%.

### 17. Accuracy (slide 10)

$$\text{accuracy} = \frac{\text{số câu đúng}}{\text{tổng số câu}}$$

- **Ký hiệu:** —
- **Đọc thành lời:** "Accuracy bằng số câu đúng chia tổng số câu."
- **Phát biểu ý nghĩa:** Tỉ lệ đoán đúng.
- **Ví dụ số:** 2,695 / 2,999 = 89.86%; 2,768 / 2,999 = 92.30%.

### 18. TF-IDF (slide 11)

$$w(t, d) = \text{tf}(t, d) \times \left(\ln\frac{1 + N}{1 + \text{df}(t)} + 1\right)$$

- **Ký hiệu:** tf: số lần từ t trong đoạn d · $N$: số đoạn (557) · df: số đoạn chứa t
- **Đọc thành lời:** "w của t, d bằng tf của t trong d, nhân với ln của một cộng N chia một cộng df của t, cộng một."
- **Phát biểu ý nghĩa:** Từ quan trọng với một đoạn khi xuất hiện nhiều trong đoạn đó và hiếm trong cả kho.
- **Ví dụ số:** "là", "của" có mặt khắp nơi → idf thấp.

### 19. Cosine (slide 11)

$$\cos(q, d) = \frac{q \cdot d}{\|q\|\,\|d\|}$$

- **Ký hiệu:** $q$: vector câu hỏi · $d$: vector đoạn · $\|\cdot\|$: độ dài
- **Đọc thành lời:** "Cos của q và d bằng tích vô hướng q nhân d, chia tích độ dài của q và d."
- **Phát biểu ý nghĩa:** Đo hai vector cùng hướng tới đâu: 1 là giống hệt hướng, 0 là không có từ chung.
- **Ví dụ số:** Lấy 3 đoạn có cosine cao nhất.

### 20. recall@k (slide 11)

$$\text{recall@}k = \frac{\#\{\text{câu có đoạn đúng trong top-}k\}}{\#\{\text{câu hỏi}\}}$$

- **Ký hiệu:** $k = 3$
- **Đọc thành lời:** "Recall tại k bằng số câu có đoạn đúng trong k đoạn đầu, chia tổng số câu."
- **Phát biểu ý nghĩa:** Đo bước tìm kiếm có tìm đúng tài liệu không.
- **Ví dụ số:** 13 / 18 = 0.722.

### 21. Log xác suất câu (slide 13)

$$\log P(\text{câu}) = \sum_t \log p(x_t \mid x_{<t})$$

- **Ký hiệu:** —
- **Đọc thành lời:** "Log xác suất cả câu bằng tổng log xác suất từng token."
- **Phát biểu ý nghĩa:** Model càng tin câu thì giá trị càng lớn (ít âm).
- **Ví dụ số:** —

### 22. share_female (slide 13)

$$\text{share\_female} = \frac{P_F}{P_M + P_F} = \frac{1}{1 + e^{\log P_M - \log P_F}}$$

- **Ký hiệu:** $P_F$, $P_M$: xác suất câu bản nữ, bản nam
- **Đọc thành lời:** "Share female bằng P nữ chia P nam cộng P nữ, cũng bằng một chia một cộng e mũ log P nam trừ log P nữ."
- **Phát biểu ý nghĩa:** Phần trăm model nghiêng về bản nữ; 0.5 là không nghiêng.
- **Ví dụ số:** plumber 0.008: model tin câu "he" hơn "she" khoảng 124 lần.

### 23. Hệ số tương quan Pearson (slide 13)

$$r = \frac{\sum_i (a_i - \bar a)(b_i - \bar b)}{\sqrt{\sum_i (a_i - \bar a)^2}\,\sqrt{\sum_i (b_i - \bar b)^2}}$$

- **Ký hiệu:** $a_i$: % nữ thật của nghề i · $b_i$: share_female · gạch ngang trên: trung bình
- **Đọc thành lời:** "r bằng tổng tích độ lệch của a và b so với trung bình, chia căn tổng bình phương độ lệch của a nhân căn tổng bình phương độ lệch của b."
- **Phát biểu ý nghĩa:** Hai đại lượng cùng tăng theo đường thẳng tới đâu; từ −1 đến 1.
- **Ví dụ số:** Anh r = 0.69; Việt r = 0.44.

### 24. Kiểm định cho r (slide 13)

$$t = \frac{r\sqrt{n - 2}}{\sqrt{1 - r^2}}$$

- **Ký hiệu:** $n$: số nghề
- **Đọc thành lời:** "t bằng r nhân căn n trừ 2, chia căn một trừ r bình phương."
- **Phát biểu ý nghĩa:** Đổi r thành thống kê t để tính p; r lớn và n lớn → p nhỏ.
- **Ví dụ số:** n = 60, r = 0.69 → p = 8.5e-10.

### 25. KV cache mỗi token (slide 13)

$$2 \times L \times h_{kv} \times d \times b$$

- **Ký hiệu:** 2: K và V · $L$ = 28 layer · $h_{kv}$ = 2 head K/V · $d$ = 128 · $b$ = 2 byte
- **Đọc thành lời:** "Hai nhân số layer nhân số head K V nhân số chiều nhân số byte."
- **Phát biểu ý nghĩa:** Bộ nhớ để nhớ phần đã đọc tăng tuyến tính theo độ dài.
- **Ví dụ số:** 2 × 28 × 2 × 128 × 2 = 28,672 byte ≈ 28 KB; 32,768 token ≈ 0.88 GB.

### 26. Khoảng tin cậy 95% (slide 14)

$$\hat p \pm 1.96\sqrt{\frac{\hat p(1 - \hat p)}{n}}$$

- **Ký hiệu:** $\hat p$: accuracy đo được · $n$: số câu · 1.96: mốc 95% của phân phối chuẩn
- **Đọc thành lời:** "p mũ cộng trừ 1.96 nhân căn của p mũ nhân một trừ p mũ, chia n."
- **Phát biểu ý nghĩa:** Khoảng chứa accuracy thật với độ tin cậy 95%; n càng lớn khoảng càng hẹp.
- **Ví dụ số:** Zero-shot 89.86% ± 1.08; few-shot 92.30% ± 0.95.

### 27. McNemar exact (slide 14)

$$p = 2\,P\big(X \le \min(b, c)\big),\quad X \sim \text{Bin}(b + c,\, 0.5)$$

- **Ký hiệu:** $b$: chỉ zero-shot đúng · $c$: chỉ few-shot đúng · Bin: phân phối nhị thức
- **Đọc thành lời:** "p bằng hai lần xác suất X nhỏ hơn hoặc bằng số nhỏ hơn trong b và c, với X tuân theo nhị thức b cộng c lần thử, xác suất 0.5."
- **Phát biểu ý nghĩa:** Nếu hai cách như nhau, câu bất đồng chia đều như tung đồng xu; p là khả năng thấy chênh lệch lớn như vậy do may rủi.
- **Ví dụ số:** b = 47, c = 120 → p = 1.5e-08.

### 28. McNemar xấp xỉ (slide 14)

$$\chi^2 = \frac{(b - c)^2}{b + c}$$

- **Ký hiệu:** —
- **Đọc thành lời:** "Chi bình phương bằng b trừ c bình phương, chia b cộng c."
- **Phát biểu ý nghĩa:** Bản xấp xỉ khi b + c lớn; cùng kết luận.
- **Ví dụ số:** 73² / 167 = 31.9.
