# Research kiểm chứng bài Presentation LLMs (v1)

- Người làm: Toàn (AI hỗ trợ tra cứu) | Ngày tra: 2026-11-14
- Phạm vi: kiểm các điểm kỹ thuật mà `plan_presentation_llm_v1.md` và 2 file kế hoạch đang dựa vào (model, dữ liệu, môi trường, API, tài liệu tham khảo, nguồn lý thuyết).
- Cách đọc:
  - **Đã kiểm**: đọc từ trang gốc (model card, docs, arXiv) ngày 2026-11-14.
  - **Chưa chạy**: chưa chạy code để thử. Máy làm research không tải được file từ Hugging Face, nên mọi thứ cần thử ở buổi chạy thử Ngày 1 trên Kaggle.

---

## 1. Bảy phát hiện phải sửa vào plan

| # | Phát hiện | Ảnh hưởng | Sửa thế nào |
|---|---|---|---|
| 1 | `generation_config.json` của **Qwen2.5-1.5B-Instruct** đặt sẵn `do_sample: true`, `temperature: 0.7`, `top_p: 0.8`, `top_k: 20`, `repetition_penalty: 1.1`. Bản **base** đặt `do_sample: false` | Gọi `generate()` mà không truyền tham số thì bản instruct **lấy mẫu ngẫu nhiên, không phải greedy**. Thí nghiệm 500 câu (cần greedy) và mục 6 (so cấu hình) sẽ sai; mục 5 so base với instruct cũng không công bằng | Luôn truyền tường minh: greedy dùng `do_sample=False`. Ở mục 6 đặt rõ `temperature`, `top_p`, `top_k`, `repetition_penalty` cho từng cấu hình. In `model.generation_config` ra notebook để chứng minh. **Chưa chạy** |
| 2 | Qwen2.5 lưu trọng số dạng `bfloat16`. GPU T4 (compute capability 7.5) không hỗ trợ bf16 bằng phần cứng; PyTorch vẫn chạy được nhưng mô phỏng bằng phần mềm nên chậm | `torch_dtype="auto"` (như code mẫu trên model card) sẽ nạp bf16 và chạy chậm trên T4 | Dùng `torch_dtype=torch.float16`, hoặc `torch.float32` nếu fp16 cho output lạ. 1.54 tỉ tham số: fp32 ≈ 6.2 GB, fp16 ≈ 3.1 GB (tự tính), vừa 16 GB của T4. Chọn dtype ở buổi chạy thử. **Chưa chạy** |
| 3 | Trọng số attention: với kiểu attention mặc định (SDPA), `output_attentions=True` không trả trọng số; `transformers` cảnh báo và yêu cầu dùng `eager` | Phần 2 (vẽ attention thật của Qwen) không lấy được trọng số | Phần 2 nạp model với `attn_implementation="eager"`, hoặc nạp riêng một bản eager chỉ cho phần này. **Chưa chạy** |
| 4 | UIT-VSFC trên Hub (`uitnlp/vietnamese_students_feedback`): nhánh `main` chỉ có file script tải dữ liệu. Từ `datasets` 4.0, script loader bị bỏ hẳn, gọi sẽ lỗi `Dataset scripts are no longer supported` | `load_dataset("uitnlp/vietnamese_students_feedback")` nhiều khả năng lỗi trên bản `datasets` mới | Đọc thẳng file Parquet do Hub tự chuyển đổi bằng `pandas.read_parquet(<url>)` (URL ở §3). **Chưa chạy** |
| 5 | UIT-VSFC có **3 nhãn** sentiment (0 negative, 1 neutral, 2 positive), không phải 2 | Phải bỏ nhãn neutral | `load_uit_vsfc` lọc bỏ `sentiment == 1`, đổi 0 → negative, 2 → positive |
| 6 | Bộ dữ liệu đã chia sẵn train / validation / test | Plan đang tự chia | Lấy 500 câu test (250/250) từ **split test**; lấy ví dụ few-shot từ **split train**. Hai tập tách nhau sẵn nên không rò dữ liệu. `make_fewshot_pools` vẫn nên kiểm trùng câu, vì corpus có thể có câu lặp nguyên văn (một model fine-tune trên UIT-VSFC phải gom nhóm câu trùng khi chia lại dữ liệu) |
| 7 | `apply_chat_template(..., tokenize=False)` rồi gọi `tokenizer(text)` sẽ thêm token đặc biệt lần hai | Prompt bị lặp token đặc biệt, output có thể lệch | Hoặc truyền `add_special_tokens=False` khi tokenize, hoặc dùng `apply_chat_template(..., tokenize=True, add_generation_prompt=True, return_tensors="pt")` như docs khuyên |

---

## 2. Model Qwen2.5-1.5B (số liệu cho slide 5)

**Đã kiểm** từ model card và `config.json` trên Hugging Face.

| Thông số | Giá trị | Nguồn |
|---|---|---|
| Số tham số | 1.54B tổng, 1.31B không tính embedding | Model card |
| Số layer | 28 | Model card, `config.json` |
| Attention head | 12 head cho Q, 2 head cho KV (GQA, grouped-query attention) | Model card |
| `hidden_size` | 1536; mỗi head 1536 / 12 = 128 chiều (tự tính) | `config.json` |
| `intermediate_size` (FFN) | 8960 | `config.json` |
| `vocab_size` | 151936 | `config.json` |
| Ngữ cảnh | 32,768 token; instruct sinh tối đa 8,192 token | Model card |
| Kiến trúc | Transformer có RoPE, SwiGLU, RMSNorm, bias ở QKV, embedding vào/ra dùng chung (tied) | Model card |
| `rope_theta` | 1,000,000 | `config.json` |
| Kiểu dữ liệu lưu | bfloat16 | `config.json` |
| Giai đoạn huấn luyện | Base: Pretraining. Instruct: Pretraining & Post-training | Model card |
| Giấy phép | Apache-2.0 | Model card |
| Yêu cầu thư viện | `transformers` ≥ 4.37.0; bản cũ hơn báo `KeyError: 'qwen2'` | Model card |

Thêm cho mục 5:
- Model card của bản base ghi rõ nhóm Qwen **không khuyến nghị dùng bản base để hội thoại**. Đây là câu dẫn tốt cho demo base so với instruct.
- Báo cáo kỹ thuật Qwen2.5 (arXiv 2412.15115): pretraining trên 18 nghìn tỉ token; post-training gồm SFT trên hơn 1 triệu mẫu rồi học tăng cường nhiều giai đoạn. Khớp sơ đồ Pretraining → SFT → alignment.

---

## 3. Dữ liệu UIT-VSFC (mục 7, 10)

**Đã kiểm** từ trang dataset và thống kê của Hugging Face datasets-server.

| Thông tin | Giá trị |
|---|---|
| Bài báo gốc | Nguyen et al., *UIT-VSFC: Vietnamese Students' Feedback Corpus for Sentiment Analysis*, KSE 2018, tr. 19–24, DOI 10.1109/KSE.2018.8573337 |
| Cột | `sentence`, `sentiment` (0 negative, 1 neutral, 2 positive), `topic` (0 lecturer, 1 training_program, 2 facility, 3 others) |
| Kích thước | train 11,426 · validation 1,583 · test 3,166 (tổng 16,175) |
| Nhãn ở split **test** | negative 1,409 · neutral 167 · positive 1,590 → đủ lấy 250/250 |
| Nhãn ở split **train** | negative 5,325 · neutral 458 · positive 5,643 |
| Giấy phép | Trang dataset ghi "unknown" → ghi nguồn và trích bài báo; không commit dữ liệu vào repo (đúng rule hiện có) |
| Chất lượng nhãn | Bài báo báo độ đồng thuận giữa người gán nhãn sentiment trên 91% → còn khoảng 9% bất đồng; dùng làm căn cứ cho ý "nhãn nhiễu" ở mục 9 |

Cách tải đề xuất (**chưa chạy**; Kaggle cần bật Internet):

```python
import pandas as pd

BASE = ("https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback/"
        "resolve/refs%2Fconvert%2Fparquet/default")
test_df = pd.read_parquet(f"{BASE}/test/0000.parquet")     # 3,166 dòng
train_df = pd.read_parquet(f"{BASE}/train/0000.parquet")   # 11,426 dòng
```

Dự phòng nếu link Parquet đổi: tải 1 lần, đưa lên Kaggle Dataset (cùng cách đưa `src/`).

---

## 4. Môi trường Kaggle

| Thông tin | Giá trị | Mức tin cậy |
|---|---|---|
| GPU | P100 hoặc "GPU T4 x2"; mỗi T4 16 GB, **hai bộ nhớ riêng**, không phải một GPU 32 GB | Nhiều nguồn thống nhất |
| Thời gian một phiên | 12 giờ cho CPU/GPU | Bên thứ ba trích docs Kaggle; trang docs gốc không đọc được vì cần JavaScript |
| Hạn mức GPU tuần | Khoảng 30 giờ, xem chính xác trong trang tài khoản | Không có trong docs chính thức; các nguồn khác nhau |
| Xác minh số điện thoại | **Bắt buộc** để bật GPU và Internet | Nhiều nguồn thống nhất |
| bf16 | T4 không hỗ trợ bằng phần cứng → dùng fp16/fp32 (§1 ý 2) | Nhiều nguồn thống nhất |

Việc cần làm Ngày 1 (Phúc): mỗi người xác minh số điện thoại Kaggle; ghi phiên bản `transformers`, `torch`, `datasets` có sẵn trên Kaggle vào `env.json` (chưa biết Kaggle cài sẵn bản nào).

---

## 5. Ghi chú API (cho phần code)

| API | Điều đã kiểm | Ghi vào code |
|---|---|---|
| `Qwen2ForCausalLM(labels=...)` | Truyền `labels` thì model trả `loss` cho next-token prediction; vị trí có nhãn `-100` bị bỏ qua | `learning.py`: dùng `labels=input_ids`. Docs không nói rõ có tự dịch nhãn không → tự tính cross-entropy bằng tay cho 1 câu rồi so với `loss` của model; trùng là đúng (vừa kiểm vừa là demo tốt cho mục 4) |
| `output_attentions=True` | Trả tuple, mỗi layer 1 tensor `(batch, num_heads, seq_len, seq_len)`, là trọng số sau softmax | `architecture.py`: lấy 1 layer, 1 head để vẽ; cần `attn_implementation="eager"` (§1 ý 3) |
| `generate()` | Mặc định là greedy và 20 token mới, **trừ khi** `GenerationConfig` của model đặt khác (Qwen instruct có đặt) | Mọi lời gọi truyền `max_new_tokens`, `do_sample` tường minh |
| `apply_chat_template` | `add_generation_prompt=True` thêm đoạn mở đầu lượt trả lời của assistant | Xem §1 ý 7 |
| `statsmodels.stats.contingency_tables.mcnemar(table, exact=True, correction=True)` | `table` là bảng 2×2. `exact=True` (mặc định) dùng phân phối nhị thức; trả `statistic` và `pvalue` | `mcnemar_pvalue`: bảng `[[cả hai đúng, chỉ A đúng], [chỉ B đúng, cả hai sai]]`; giữ `exact=True` vì số câu bất đồng có thể nhỏ |

---

## 6. Tài liệu tham khảo: kết quả mở từng link

Danh sách lấy từ `ke_hoach_presentation_llms.md` mục 8.

| Tài liệu | Link | Kết quả |
|---|---|---|
| Vaswani et al., *Attention Is All You Need*, 2017 | arxiv.org/abs/1706.03762 | ✔ Đúng |
| Alammar, *The Illustrated Transformer* | jalammar.github.io/illustrated-transformer/ | ✔ Đúng |
| Alammar, *The Illustrated GPT-2* | jalammar.github.io/illustrated-gpt2/ | ✔ Đúng; có giải thích masked self-attention, hợp mục 2 |
| Hugging Face Tokenizers docs | huggingface.co/docs/tokenizers | ✔ Mở được |
| Hugging Face LLM Course | huggingface.co/learn/llm-course | ✔ Mở được (trang chapter1/1) |
| Stanford CS224N | web.stanford.edu/class/cs224n/ | ✔ Mở được; hiện là trang khóa Winter 2026 |
| Brown et al., *Language Models are Few-Shot Learners* (GPT-3), 2020 | arxiv.org/abs/2005.14165 | ✔ Đúng; abstract nói GPT-3 làm tác vụ mà không cập nhật gradient, ví dụ đưa qua văn bản → căn cứ cho ICL |
| Wei et al., *Finetuned Language Models Are Zero-Shot Learners* (FLAN), 2021 | arxiv.org/abs/2109.01652 | ✔ Đúng; instruction tuning làm zero-shot tốt hơn rõ rệt → căn cứ cho "làm theo chỉ dẫn" |
| Ouyang et al., *Training language models to follow instructions with human feedback* (InstructGPT), 2022 | arxiv.org/abs/2203.02155 | ✔ Đúng; 3 bước: SFT → reward model từ xếp hạng của người → học tăng cường |
| Rafailov et al., *Direct Preference Optimization*, 2023 | arxiv.org/abs/2305.18290 | ✔ Đúng; không cần reward model riêng, không cần vòng lặp RL |
| Holtzman et al., *The Curious Case of Neural Text Degeneration*, 2019 | arxiv.org/abs/1904.09751 | ✔ Đúng; nucleus (top-p) sampling cắt phần đuôi xác suất kém tin cậy |
| Huang et al., *A Survey on Hallucination in LLMs*, 2023 | arxiv.org/abs/2311.05232 | ✔ Đúng |
| Karpathy, *Let's build GPT: from scratch, in code, spelled out* | youtube.com/watch?v=kCc8FmEb1nY | ✔ Đúng (kênh Andrej Karpathy, 17/01/2023) |
| nanoGPT | github.com/karpathy/nanoGPT | ⚠ Mở được, nhưng README ghi repo đã cũ và ngừng phát triển, giới thiệu `nanochat` thay thế. Vẫn dùng được để tham khảo; ghi chú điều này |
| Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, 2020 | arxiv.org/abs/2005.11401 | ✔ Đúng |
| Hu et al., *LoRA*, 2021 | arxiv.org/abs/2106.09685 | ✔ Đúng |
| Zhao et al., *A Survey of Large Language Models*, 2023 | arxiv.org/abs/2303.18223 | ✔ Đúng (bản mới nhất v19) |
| Hugging Face Transformers docs | huggingface.co/docs/transformers | ✔ Mở được |
| Hugging Face PEFT docs | huggingface.co/docs/peft | ✔ Mở được |
| Stanford CS324 | stanford-cs324.github.io/winter2022/ | ✔ Mở được |

**Đề xuất thêm** (đã kiểm, chưa có trong danh sách):

| Tài liệu | Dùng cho |
|---|---|
| Qwen Team, *Qwen2.5 Technical Report*, 2024, arxiv.org/abs/2412.15115 | Mục 2, 5: nguồn chính thức cho model nhóm dùng |
| Nguyen et al., *UIT-VSFC*, KSE 2018, DOI 10.1109/KSE.2018.8573337 | Mục 7, 10: nguồn dữ liệu (bắt buộc trích vì dùng dữ liệu) |
| Zhao et al., *Calibrate Before Use*, 2021, arxiv.org/abs/2102.09690 | Mục 7: few-shot rất nhạy với định dạng prompt, việc chọn và thứ tự ví dụ. Căn cứ cho việc chạy 3 bộ few-shot |
| Min et al., *Rethinking the Role of Demonstrations*, 2022, arxiv.org/abs/2202.12837 | Mục 7, Q&A: đổi nhãn ví dụ ngẫu nhiên chỉ giảm nhẹ kết quả; ví dụ giúp chủ yếu nhờ cho thấy không gian nhãn, phân phối câu và định dạng |
| Liu et al., *Lost in the Middle*, 2023, arxiv.org/abs/2307.03172 | Mục 9: model dùng thông tin ở đầu hoặc cuối ngữ cảnh tốt hơn ở giữa |
| Dietterich, *Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms*, Neural Computation 10(7), 1998, DOI 10.1162/089976698300017197 | Mục 10: căn cứ dùng McNemar khi chỉ chạy được một lần |
| Model card Qwen2.5-1.5B và Qwen2.5-1.5B-Instruct (huggingface.co/Qwen) | Mục 2, 5, slide 2 |

---

## 7. Căn cứ lý thuyết cho từng mục

| Mục | Câu nói trên slide | Nguồn |
|---|---|---|
| 2 | Decoder-only dùng masked (causal) self-attention, mỗi vị trí chỉ nhìn hiện tại và quá khứ | Illustrated GPT-2; Vaswani et al. 2017 |
| 5 | Pretraining → SFT → reward model → RL | InstructGPT (Ouyang et al. 2022) |
| 5 | DPO bỏ reward model riêng và vòng lặp RL | Rafailov et al. 2023 |
| 5 | Qwen2.5 instruct qua SFT + RL nhiều giai đoạn; bản base không khuyến nghị để chat | Qwen2.5 report; model card |
| 6 | Top-p cắt đuôi phân phối; greedy dễ lặp khi sinh dài | Holtzman et al. 2019; docs generation của Hugging Face |
| 7 | ICL không cập nhật trọng số | GPT-3 (Brown et al. 2020) |
| 7 | Instruction tuning giúp zero-shot | FLAN (Wei et al. 2021) |
| 7 | Kết quả few-shot nhạy với ví dụ và thứ tự | Zhao et al. 2021 |
| 9 | Hallucination: định nghĩa, phân loại | Huang et al. 2023 |
| 9 | Thông tin ở giữa ngữ cảnh dài dễ bị bỏ sót | Liu et al. 2023 |
| 10 | McNemar cho 2 bộ phân loại chạy trên cùng tập test | Dietterich 1998 |

Gợi ý cho mục 9 (giới hạn ngữ cảnh): `make_long_context(fact, filler, position)` trong plan đã có tham số `position`. Chạy cả 3 vị trí (đầu, giữa, cuối) thay vì chỉ "đầu văn bản" thì kết quả so sánh được với Liu et al. 2023. Đây là **[đề xuất]**: theo rule Lab 1, thêm thí nghiệm phải được nhóm duyệt.

---

## 8. Sửa plan v1 theo kết quả research  [đề xuất]

- `config.py`: thêm `DTYPE = torch.float16` (dự phòng `torch.float32`), `GREEDY = {"do_sample": False, "max_new_tokens": ...}`, URL Parquet của UIT-VSFC.
- `setup.py`: `load_model(name, device, dtype, attn_implementation="sdpa")`; phần 2 gọi với `"eager"`.
- `icl_experiment.py`:
  - `load_uit_vsfc` đọc Parquet, bỏ neutral, đổi nhãn sang chữ;
  - `make_test_split` lấy từ split test;
  - `make_fewshot_pools` lấy từ split train và vẫn kiểm trùng câu;
  - `run_icl` truyền `do_sample=False` tường minh.
- `generation.py`: mỗi cấu hình trong `DECODING_CONFIGS` ghi đủ `do_sample`, `temperature`, `top_p`, `top_k`, `repetition_penalty` để không bị `generation_config` của model chen vào.
- `learning.py`: thêm 1 cell so loss tự tính với `outputs.loss`.
- `references.md`: thêm 7 tài liệu ở §6; ghi chú nanoGPT.

---

## 9. Chưa kiểm được (làm ở buổi chạy thử Ngày 1)

- Tải model và Parquet thật: máy làm research bị chặn mạng tới Hugging Face.
- Phiên bản `transformers`, `torch`, `datasets` cài sẵn trên Kaggle.
- Qwen2.5-1.5B chạy fp16 có ổn định không: so vài output fp16 với fp32.
- Cách tắt hẳn `top_k` và `repetition_penalty` mặc định của bản instruct: in `generation_config` lúc chạy và kiểm output greedy có lặp lại giống nhau qua 2 lần chạy.
- Thời gian chạy 500 câu × 4 lượt (zero-shot + 3 bộ few-shot) trên T4.

---

## Nguồn

- [Qwen/Qwen2.5-1.5B-Instruct — model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) · [config.json](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json) · [generation_config.json](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/generation_config.json)
- [Qwen/Qwen2.5-1.5B — model card](https://huggingface.co/Qwen/Qwen2.5-1.5B) · [generation_config.json](https://huggingface.co/Qwen/Qwen2.5-1.5B/blob/main/generation_config.json)
- [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)
- [uitnlp/vietnamese_students_feedback](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback) · [cây file main](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback/tree/main) · [danh sách Parquet](https://datasets-server.huggingface.co/parquet?dataset=uitnlp/vietnamese_students_feedback) · [thống kê test](https://datasets-server.huggingface.co/statistics?dataset=uitnlp/vietnamese_students_feedback&config=default&split=test) · [thống kê train](https://datasets-server.huggingface.co/statistics?dataset=uitnlp/vietnamese_students_feedback&config=default&split=train)
- [datasets 4.0.0 release notes — bỏ script loader](https://newreleases.io/project/github/huggingface/datasets/release/4.0.0) · [Hugging Face Forum: Dataset scripts are no longer supported](https://discuss.huggingface.co/t/cannot-load-conll2003/169142)
- [Transformers — Qwen2 model doc](https://huggingface.co/docs/transformers/model_doc/qwen2) · [Generation strategies](https://huggingface.co/docs/transformers/generation_strategies) · [Chat templates](https://huggingface.co/docs/transformers/chat_templating)
- [Transformers SDPA: output_attentions cần eager](https://download.hermesedi.duckdns.org/venv_trading/lib/python3.11/site-packages/transformers/integrations/sdpa_attention.py) (mã nguồn thư viện, bản sao công khai)
- [statsmodels mcnemar](https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html) · [Dietterich 1998](https://mlanthology.org/neco/1998/dietterich1998neco-approximate/)
- T4 và bf16: [PyTorch Forums](https://discuss.pytorch.org/t/current-cuda-device-does-not-support-bfloat16-please-switch-dtype-to-float16/201564) · [bf16 trên T4 chạy chậm](https://earino.github.io/applied-deep-learning/site/blog/05_bf16_on_t4.html)
- Kaggle: [fast.ai — phone verification](https://course.fast.ai/Resources/kaggle.html) · [giới hạn phiên 12 giờ (trích docs Kaggle)](https://gpuperhour.com/blog/free-cloud-gpus-and-credits) · [T4 x2 là hai GPU 16 GB riêng](https://aimultiple.com/gpu-cluster)
- arXiv: [1706.03762](https://arxiv.org/abs/1706.03762) · [2005.14165](https://arxiv.org/abs/2005.14165) · [2109.01652](https://arxiv.org/abs/2109.01652) · [2203.02155](https://arxiv.org/abs/2203.02155) · [2305.18290](https://arxiv.org/abs/2305.18290) · [1904.09751](https://arxiv.org/abs/1904.09751) · [2311.05232](https://arxiv.org/abs/2311.05232) · [2005.11401](https://arxiv.org/abs/2005.11401) · [2106.09685](https://arxiv.org/abs/2106.09685) · [2303.18223](https://arxiv.org/abs/2303.18223) · [2102.09690](https://arxiv.org/abs/2102.09690) · [2202.12837](https://arxiv.org/abs/2202.12837) · [2307.03172](https://arxiv.org/abs/2307.03172)
- [nanoGPT](https://github.com/karpathy/nanoGPT) · [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) · [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) · [CS224N](https://web.stanford.edu/class/cs224n/) · [CS324](https://stanford-cs324.github.io/winter2022/) · [Karpathy, Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)
