# Tài liệu tham khảo

- Tổng hợp: Muội (bản đầu do Toàn soạn từ research 14–15/11/2026); cập nhật 21/11 (Toàn, AI hỗ trợ: thêm #33–35 và 2 nguồn dữ liệu, đều đã mở link); 24/11 (thêm #36–38, DeepEval, Qwen2.5-7B-Instruct; đã mở link) | Trạng thái: **[đề xuất]**, chờ Toàn duyệt trước khi đưa lên slide
- Cột "Đã mở": ngày đã mở link kiểm tra. Link chưa mở thì để trống, không đưa lên slide.
- Đối chiếu thêm với tài liệu giảng viên cung cấp trước khi chốt.

## Slide môn học
| Tài liệu | Dùng cho mục |
|---|---|
| Slide Ch1 (s7, s10, s12, s13, s15) — giới thiệu DL, timeline, các loại mạng, NLP | 1, 2 |
| Slide Ch2 (s7, s9, s11–13) — cross-entropy, softmax | 4 |
| Slide Ch3 — tách train / validation / test | 9, 10 |
| Slide Ch4 — fine-tuning; Practice 3 (Hugging Face) | 5, 7, 8 |
| Slide Ch6 (s11, s16, s17) — LSTM + attention, encoder-decoder, biểu diễn chuỗi | 2, 3, 6 |

## Bài báo và sách
| # | Tài liệu | Link | Mục | Đã mở |
|---|---|---|---|---|
| 1 | Vaswani et al., *Attention Is All You Need*, NeurIPS 2017 | https://arxiv.org/abs/1706.03762 | 2 | 14/11 |
| 2 | Bengio et al., *A Neural Probabilistic Language Model*, JMLR 3, 2003 | https://www.jmlr.org/papers/v3/bengio03a.html | 1 | 14/11 |
| 3 | Jurafsky & Martin, *Speech and Language Processing*, ch. 3 (draft 2026) | https://web.stanford.edu/~jurafsky/slp3/3.pdf | 1 | 14/11 |
| 4 | Sennrich et al., *Neural Machine Translation of Rare Words with Subword Units*, ACL 2016 | https://arxiv.org/abs/1508.07909 | 3 | 15/11 |
| 5 | Brown et al., *Language Models are Few-Shot Learners* (GPT-3), 2020 | https://arxiv.org/abs/2005.14165 | 1, 4, 7 | 14/11 |
| 6 | Wei et al., *Finetuned Language Models Are Zero-Shot Learners* (FLAN), 2021 | https://arxiv.org/abs/2109.01652 | 5, 7 | 14/11 |
| 7 | Ouyang et al., *Training language models to follow instructions with human feedback* (InstructGPT), 2022 | https://arxiv.org/abs/2203.02155 | 5 | 15/11 |
| 8 | Rafailov et al., *Direct Preference Optimization*, 2023 | https://arxiv.org/abs/2305.18290 | 5 | 15/11 |
| 9 | Holtzman et al., *The Curious Case of Neural Text Degeneration*, ICLR 2020 | https://arxiv.org/abs/1904.09751 | 6 | 15/11 |
| 10 | Zhao, Wallace, Feng, Klein, Singh, *Calibrate Before Use*, 2021 | https://arxiv.org/abs/2102.09690 | 7 | 14/11 |
| 11 | Min et al., *Rethinking the Role of Demonstrations*, 2022 | https://arxiv.org/abs/2202.12837 | 7 | 14/11 |
| 12 | Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, NeurIPS 2020 | https://arxiv.org/abs/2005.11401 | 8 | 15/11 |
| 13 | Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021 | https://arxiv.org/abs/2106.09685 | 8 | 15/11 |
| 14 | Huang et al., *A Survey on Hallucination in Large Language Models*, 2023 | https://arxiv.org/abs/2311.05232 | 9 | 15/11 |
| 15 | Lin, Hilton, Evans, *TruthfulQA*, ACL 2022 | https://arxiv.org/abs/2109.07958 | 9 | 15/11 |
| 16 | Liu et al., *Lost in the Middle*, 2023 | https://arxiv.org/abs/2307.03172 | 9 | 15/11 |
| 17 | Bolukbasi et al., *Man is to Computer Programmer as Woman is to Homemaker?*, NeurIPS 2016 | https://arxiv.org/abs/1607.06520 | 9 | 15/11 |
| 18 | Zhao et al., *A Survey of Large Language Models*, 2023 | https://arxiv.org/abs/2303.18223 | 1, 7, 8 | 14/11 |
| 19 | Chen et al., *Evaluating Large Language Models Trained on Code* (Codex), 2021 | https://arxiv.org/abs/2107.03374 | 1 | 14/11 |
| 20 | Qwen Team, *Qwen2.5 Technical Report*, 2024 | https://arxiv.org/abs/2412.15115 | 2, 3, 4, 5 | 14/11 |
| 21 | Nguyen et al., *UIT-VSFC: Vietnamese Students' Feedback Corpus for Sentiment Analysis*, KSE 2018, pp. 19–24 | doi:10.1109/KSE.2018.8573337 | 7, 9, 10 | 14/11 (qua trang dataset) |
| 22 | Dietterich, *Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms*, Neural Computation 10(7), 1998 | doi:10.1162/089976698300017197 | 10 | 14/11 |
| 23 | Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, NAACL 2019 | https://arxiv.org/abs/1810.04805 | 2 (so sánh) | |
| 24 | Raffel et al., *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* (T5), JMLR 2020 | https://arxiv.org/abs/1910.10683 | 2 (so sánh) | |
| 25 | Qwen Team, *Qwen2.5-VL Technical Report*, 2025 | https://arxiv.org/abs/2502.13923 | 2 (so sánh) | 16/11 |
| 26 | Rajpurkar et al., *SQuAD: 100,000+ Questions for Machine Comprehension of Text*, EMNLP 2016 | https://arxiv.org/abs/1606.05250 | 8, 9 (chấm EM/F1: bỏ dấu câu và mạo từ; script chính thức còn chuyển chữ thường) | 16/11 |
| 27 | Chen et al., *Benchmarking Large Language Models in Retrieval-Augmented Generation* (RGB), AAAI 2024 | https://arxiv.org/abs/2309.01431 | 8 | 16/11 |
| 28 | Es et al., *Ragas: Automated Evaluation of Retrieval Augmented Generation*, 2023 | https://arxiv.org/abs/2309.15217 | 8 | 16/11 |
| 29 | Wei et al., *Measuring short-form factuality in large language models* (SimpleQA), 2024 | https://arxiv.org/abs/2411.04368 | 9 | 16/11 |
| 30 | Min et al., *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation*, EMNLP 2023 | https://arxiv.org/abs/2305.14251 | 9 | 16/11 |
| 31 | Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*, NeurIPS 2023 | https://arxiv.org/abs/2306.05685 | 8, 9, 10 (LLM làm giám khảo: dùng cho DeepEval 10.3c, giám khảo khác và mạnh hơn model được chấm) | 16/11 |
| 32 | Landis & Koch, *The Measurement of Observer Agreement for Categorical Data*, Biometrics 33(1):159–174, 1977 | https://pubmed.ncbi.nlm.nih.gov/843571/ | 8, 9 (thang kappa) | 16/11 (chỉ mở được abstract; các ngưỡng 0,41/0,61/0,81 lấy từ tài liệu dạy NLP trích lại, chưa đọc bản gốc) |
| 33 | Rudinger, Naradowsky, Leonard, Van Durme, *Gender Bias in Coreference Resolution* (WinoGender), NAACL 2018 | https://github.com/rudinger/winogender-schemas (bài báo: `paper/gender-bias-coreference.pdf`) | 9 (thiên lệch) | 20/11 |
| 34 | Zhao et al., *Gender Bias in Coreference Resolution: Evaluation and Debiasing Methods* (WinoBias), NAACL 2018 | https://github.com/uclanlp/corefBias | 9 (so sánh, không dùng) | 20/11 |
| 35 | Nangia et al., *CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models*, EMNLP 2020 | https://arxiv.org/abs/2010.00133 | 9 (cách đo bằng xác suất câu) | 21/11 |
| 36 | Papineni et al., *Bleu: a Method for Automatic Evaluation of Machine Translation*, ACL 2002 | https://aclanthology.org/P02-1040/ | 10 (BLEU, cell 10.3b) | 24/11 |
| 37 | Lin, *ROUGE: A Package for Automatic Evaluation of Summaries*, Text Summarization Branches Out 2004 | https://aclanthology.org/W04-1013/ | 10 (ROUGE-1/2/L, cell 10.3b) | 24/11 |
| 38 | Post, *A Call for Clarity in Reporting BLEU Scores* (sacreBLEU), WMT 2018 | https://aclanthology.org/W18-6319/ | 10 (cách tính BLEU chuẩn hoá, cell 10.3b) | 24/11 |
| 39 | Liu et al., *G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment*, EMNLP 2023 | https://arxiv.org/abs/2303.16634 | 10 (G-Eval, cell 10.3d) | 24/11 |
| 40 | Li et al., *A Diversity-Promoting Objective Function for Neural Conversation Models* (distinct-n), NAACL 2016 | https://arxiv.org/abs/1510.03055 | 6 (distinct-n, cell 6.5) | 24/11 |
| 41 | Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2, byte-level BPE), 2019 | https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf | 3 | 24/11 |

## Model, dữ liệu, tài liệu thư viện
| Tài liệu | Link | Mục | Đã mở |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct (model card, `config.json`, `generation_config.json`) | https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct | 0, 2, 5, 6 | 15/11 |
| Qwen2.5-1.5B (base) | https://huggingface.co/Qwen/Qwen2.5-1.5B | 5 | 15/11 |
| GPT-2 tokenizer | https://huggingface.co/openai-community/gpt2 | 3 | 15/11 |
| UIT-VSFC trên Hugging Face | https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback | 7, 10 | 14/11 |
| UIT-ViQuAD 2.0 trên Hugging Face (split validation; trang dataset không ghi giấy phép) | https://huggingface.co/datasets/taidng/UIT-ViQuAD2.0 | 8, 9 | 19/11 |
| Nguyen et al., *VLSP 2021 - ViMRC Challenge: Vietnamese Machine Reading Comprehension* (bài báo của UIT-ViQuAD 2.0) | https://arxiv.org/abs/2203.11400 | 8, 9 | 19/11 |
| WinoGender: `data/all_sentences.tsv`, `data/occupations-stats.tsv` (giấy phép MIT) | https://github.com/rudinger/winogender-schemas | 9 | 20/11 |
| NVIDIA, Nemotron-Personas-Vietnam (dữ liệu tổng hợp, CC BY 4.0; dataset card ghi nguồn Tổng cục Thống kê 2024) | https://huggingface.co/datasets/nvidia/Nemotron-Personas-Vietnam | 9 | 20/11 |
| Hugging Face Transformers docs | https://huggingface.co/docs/transformers | Tất cả | 14/11 |
| HF *Summary of the tokenizers* | https://huggingface.co/docs/transformers/tokenizer_summary | 3 | 15/11 |
| HF *Perplexity of fixed-length models* | https://huggingface.co/docs/transformers/perplexity | 4 | 15/11 |
| HF *Generation strategies* | https://huggingface.co/docs/transformers/generation_strategies | 6 | 15/11 |
| HF *Chat templates* | https://huggingface.co/docs/transformers/chat_templating | 1, 7 | 14/11 |
| HF LLM Course, ch. 1 | https://huggingface.co/learn/llm-course | 1, 2 | 14/11 |
| HF PEFT docs | https://huggingface.co/docs/peft | 8 | 14/11 |
| scikit-learn `TfidfVectorizer` | https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html | 8 | 15/11 |
| statsmodels `mcnemar` | https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html | 10 | 14/11 |
| DeepEval (Confident AI, Apache-2.0), bản 4.3.5 | https://github.com/confident-ai/deepeval | 10 (cell 10.3c) | 24/11 |
| DeepEval docs: *Hallucination* (điểm = số context khớp / tổng context; "Higher is better") | https://deepeval.com/docs/metrics-hallucination | 10 | 24/11 |
| Qwen2.5-7B-Instruct (giám khảo DeepEval, Apache-2.0) | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct | 10 | 24/11 |

## Bài giảng, bài giải thích
| Tài liệu | Link | Mục | Đã mở |
|---|---|---|---|
| Alammar, *The Illustrated Transformer* | https://jalammar.github.io/illustrated-transformer/ | 2 | 14/11 |
| Alammar, *The Illustrated GPT-2* | https://jalammar.github.io/illustrated-gpt2/ | 2 | 14/11 |
| Karpathy, *Let's build GPT: from scratch, in code, spelled out* | https://www.youtube.com/watch?v=kCc8FmEb1nY | 4 | 14/11 |
| nanoGPT (README ghi repo đã cũ, giới thiệu nanochat thay thế) | https://github.com/karpathy/nanoGPT | 4 | 14/11 |
| OpenAI, *Introducing ChatGPT*, 30/11/2022 | https://openai.com/blog/chatgpt/ | 1, 5, 9 | 14/11 |
| Stanford CS224N | https://web.stanford.edu/class/cs224n/ | 1–3 | 14/11 |
| Stanford CS324 | https://stanford-cs324.github.io/winter2022/ | Tổng quan | 14/11 |
