# Mục 3 — Tokenization: token, token ID, embedding

- **Created**: 2026-11-28 (Toàn, AI hỗ trợ, Claude Code). Trạng thái **[đề xuất]**, chờ Toàn duyệt.
- **Nội dung:** BPE byte-level, câu Việt và câu Anh qua 2 tokenizer, embedding (cell 3.1–3.5).
- **Số liệu:** run chính thức `20261128_041850_normal` (Kaggle, GPU T4 x2, float16, seed 42, 22.0 phút, 188 cell; G-Eval chấm 20 lần lấy trung bình). Phần 1–9 và 10.1–10.3b trùng từng file kết quả với run `20261127_234527_normal`, `20261125_214650_normal`, `20261121_010447_normal` (chỉ khác cột thời gian, bộ nhớ); riêng 10.3c, 10.3d đổi theo giám khảo (xem mục 10). **VRAM 6.6, 9.5 lấy run `20261127_234527_normal`**: run chính thức đo lệch +2.97 GiB vì model của lần chạy dừng giữa chừng còn trên GPU. Bảng ở `outputs/results/`, hình ở `outputs/figures/`.
- **Ký hiệu nguồn:** **[Slide ChX]** slide môn học; **[Bổ sung]** nguồn có trong `docs/references.md`; **[Tự tính]** tính lại từ số trong notebook; **[Suy luận]** giải thích của nhóm, chưa đo.
- **Code** chép từ `notebook/llm_presentation.ipynb`; dòng ghi log, gọi `show_input()` và vài dòng in phụ được lược (`# ...`). Mở notebook, tìm số cell (ví dụ "# 4.2") để xem nguyên cell.

---

> **Cô yêu cầu → notebook đáp ứng**
> - Nội dung cần trình bày: *Cách văn bản được chia thành token; token ID và embedding*.
> - Yêu cầu minh họa: *Tokenize một câu tiếng Việt và một câu tiếng Anh*.
> - Đáp ứng: Cell 3.1: câu tô màu theo token + bảng token ID, byte; 3.1b gom theo từ; 3.2 đếm 2 câu × 2 tokenizer; 3.4 embedding (bảng tra hàng + heatmap).
> - Góp ý 11/11 và yêu cầu thêm 3: demo trên model thật, output thật, có phân tích.

## Note lý thuyết để hiểu (dùng cho cả phần 3)

**Ba cách chia văn bản** [Bổ sung: HF *Summary of the tokenizers*]:
| Cách | Ưu | Nhược |
|---|---|---|
| Theo từ | Dễ hiểu | Từ điển khổng lồ, từ lạ thành `[UNK]` |
| Theo ký tự | Từ điển nhỏ | Chuỗi rất dài, mỗi ký tự ít nghĩa |
| **Subword (BPE)** | Từ phổ biến = 1 token, từ hiếm = vài mảnh | Mảnh cắt theo thống kê, không theo ngữ pháp |

**BPE** [Bổ sung: Sennrich et al. 2016] — cách huấn luyện tokenizer:
1. Bắt đầu: mỗi ký tự/byte là một token.
2. Đếm mọi cặp token liền nhau trong kho văn bản; **gộp cặp nhiều nhất** thành token mới.
3. Lặp đến khi đủ kích thước từ điển.

**Byte-level BPE** (GPT-2, Qwen) [Bổ sung: HF *Summary of the tokenizers*; Qwen2.5 report §2: "byte-level byte-pair encoding (BBPE) with a vocabulary of 151,643 regular tokens", control tokens "from 3 to 22"]: từ điển gốc là **256 giá trị byte** → mọi chuỗi đều tokenize được, không có `[UNK]`. GPT-2: 50,257 = 256 byte gốc + 1 token kết thúc + 50,000 lần gộp. Chữ có dấu tiếng Việt chiếm **2–3 byte** UTF-8.

**Token ID** = số thứ tự trong từ điển. **Embedding** [Slide Ch6]: $E \in \mathbb{R}^{V \times d}$; token ID $t$ → $e_t = E[t, :]$ (tra hàng) $= \text{onehot}(t)^\top E$.

## Cell 3.1 — Token và token ID

```python
SENT_VI = "Hôm nay trời rất đẹp."
SENT_EN = "The weather is very nice today."
def tokenize_text(tokenizer, text: str) -> pd.DataFrame:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return pd.DataFrame({"position": range(len(ids)), "token_id": ids,
                         "piece": tokenizer.convert_ids_to_tokens(ids),   # token thô
                         "text": [tokenizer.decode([i]) for i in ids]})   # chữ của từng token
assert "".join(tokenize_text(tok, SENT_VI)["text"]) == SENT_VI
```

**Output thật — Qwen, câu Việt (8 token):**
```text
 position  token_id   piece  text
        0        39       H     H
        1     94365     Ã´m    ôm
        2       308      Ġn     n
        3       352      ay    ay
        4    130868 Ġtrá»Ŀi  trời
        5    128323  Ġráº¥t   rất
        6    128501 ĠÄĳáº¹p   đẹp
        7        13       .     .
```
Qwen, câu Anh: 7 token `The | weather | is | very | nice | today | .` (mỗi từ 1 token).
GPT-2, câu Việt: **22 token**: `H | ô | m | n | ay | tr | � | � | � | i | r | � | � | � | t | � | � | � | � | � | p | .`

**Giải thích ngay tại output:**
- **`piece` trông lạ:** tokenizer byte-level đổi mỗi byte thành một ký tự in được. Dấu cách → `Ġ`. "ô" = byte `C3 B4` → hiện `Ã´`. "ờ" = `E1 BB 9D` → `á»Ŀ`. Cột `text` là chữ thật sau decode.
- **Qwen gộp được " trời", " rất", " đẹp" thành 1 token:** dữ liệu huấn luyện tokenizer đa ngôn ngữ, các từ này (kèm dấu cách phía trước) xuất hiện đủ nhiều để BPE gộp hết các byte.
- **Qwen vẫn cắt "Hôm" = "H" + "ôm", "nay" = " n" + "ay":** BPE gộp theo **tần suất trong dữ liệu huấn luyện tokenizer**, không theo từ điển tiếng Việt. [Suy luận] "Hôm" viết hoa đầu câu hiếm hơn " hôm"; " nay" bị cắt có thể do thứ tự gộp của BPE (cặp " n" được gộp sớm với nhiều chữ khác). Muốn kiểm: tokenize " hôm nay" (chữ thường, có dấu cách) và so.
- **GPT-2 ra "�":** GPT-2 học chủ yếu tiếng Anh nên chưa gộp byte của "ờ", "ấ", "ẹ", "đ". Mỗi chữ thành 2–3 token **1 byte**; decode riêng 1 byte không đủ thành ký tự → "�". Ghép lại vẫn ra đúng chữ (cell 3.5).
- **GPT-2 gộp được "ô" (1 token `Ã´`):** [Suy luận] "ô" (byte `C3 B4`) có trong tiếng Pháp, Bồ Đào Nha… nên cặp byte này đủ phổ biến trong dữ liệu GPT-2 để được gộp; còn "ờ", "ấ", "ẹ" gần như chỉ có trong tiếng Việt.

**Phần thêm ở cell 3.1 (bản 20–21/11): câu tô màu theo token và cột byte**

```python
def piece_bytes(piece: str) -> bytes:
    return bytes(BYTE_DECODER[ch] for ch in piece)      # ký tự trong piece → byte gốc ('Ġ' → b' ')

def word_spans(text):                                   # từ / dấu câu và vị trí byte của nó
    return [(m.group(), len(text[:m.start()].encode()), len(text[:m.end()].encode()))
            for m in re.finditer(r"\w+|[^\w\s]", text)]
# tokenize_text(): mỗi token thêm word (từ chứa byte đầu tiên không phải dấu cách), bytes_hex, n_bytes
assert b"".join(piece_bytes(p) for p in pieces) == text.encode("utf-8")
```

**Giải thích code:**
- `bytes_to_unicode()` dựng lại bảng của byte-level BPE: byte in được giữ ký tự Latin-1 của nó, byte không in được (dấu cách `0x20`, `0x9D`…) dời lên ký tự từ U+0100 (`Ġ`, `Ŀ`, `ĳ`…). `BYTE_DECODER` là bảng ngược: ký tự trong `piece` → byte gốc.
- `piece_bytes('Ġtrá»Ŀi')` = `20 74 72 E1 BB 9D 69` = " trời": 1 token chứa 7 byte.
- `word_spans` tìm từng từ / dấu câu bằng regex `\w+|[^\w\s]` và tính vị trí **byte** của nó; token được gán vào từ chứa byte đầu tiên (không phải dấu cách) của token.
- `assert`: ghép byte của mọi token đúng bằng byte của câu gốc → không mất thông tin.
- `show_token_chips` vẽ câu thành các ô màu, mỗi ô một token, số nhỏ bên dưới là token ID; `␣` là dấu cách.

**Output thật (Qwen, câu Việt):**
```text
word  position  token_id  piece        text      bytes_hex              n_bytes
Hôm   0         39        H            'H'       48                     1
Hôm   1         94365     Ã´m          'ôm'      C3 B4 6D               3
nay   2         308       Ġn           ' n'      20 6E                  2
nay   3         352       ay           'ay'      61 79                  2
trời  4         130868    Ġtrá»Ŀi      ' trời'   20 74 72 E1 BB 9D 69   7
rất   5         128323    Ġráº¥t       ' rất'    20 72 E1 BA A5 74      6
đẹp   6         128501    ĠÄĳáº¹p      ' đẹp'    20 C4 91 E1 BA B9 70   7
.     7         13        .            '.'       2E                     1
```

**Đọc output:** "ô" = 2 byte `C3 B4` → hiện `Ã´` trong `piece`; "ờ" = 3 byte `E1 BB 9D` → `á»Ŀ`; "đ" = `C4 91` → `Äĳ`. Cột `text` là chữ đọc được. Cộng cột `n_bytes` = 29 = số byte UTF-8 của câu ✓.

## Cell 3.1b — Gom theo từ: mỗi từ thành mấy token, mấy byte

> **Lý thuyết ở cell này:** số token của một từ phụ thuộc từ điển của tokenizer. Byte-level BPE luôn tách được (tệ nhất là 1 byte = 1 token), nên tokenizer không có sẵn mảnh cho chữ có dấu thì tốn nhiều token.

```python
def tokens_per_word(text):
    tables = {"Qwen": tokenize_text(tok, text), "GPT-2": tokenize_text(gpt2_tok, text)}
    for j, (w, _, _) in enumerate(word_spans(text)):
        for name, t in tables.items():
            sub = t[t["word_idx"] == j]          # các token thuộc từ thứ j
            row[f"token {name}"] = " | ".join(sub["text"]); row[f"số token {name}"] = len(sub)
# assert: cộng số token từng từ = tổng số token của câu
```

**Output thật:**
```text
từ    số byte  token Qwen      số token Qwen  token GPT-2                         số token GPT-2
Hôm   4        'H' | 'ôm'      2              'H' | 'ô' | 'm'                     3
nay   3        ' n' | 'ay'     2              ' n' | 'ay'                         2
trời  6        ' trời'         1              ' tr' | '�' | '�' | '�' | 'i'       5
rất   5        ' rất'          1              ' r' | '�' | '�' | '�' | 't'        5
đẹp   6        ' đẹp'          1              ' �' | '�' | '�' | '�' | '�' | 'p'  6
.     1        '.'             1              '.'                                 1
TỔNG  29                       8                                                  22
```
Câu Anh: mỗi từ 1 token ở cả hai, tổng 7 / 7.

**Giải thích output:**
- "trời" với GPT-2: `' tr'` + 3 token, mỗi token là **1 byte** của "ờ" (`E1`, `BB`, `9D`), + `'i'`. Decode riêng 1 byte không ra chữ → `�`. Ghép 3 byte lại mới ra "ờ".
- "đẹp" với GPT-2: 6 token = `' '`+ byte đầu của "đ" gộp chung, byte thứ 2 của "đ", 3 byte của "ẹ", `'p'`.
- "Hôm" với Qwen vẫn 2 token: tokenizer học theo tần suất, không theo từ điển tiếng Việt [Suy luận: "Hôm" đầu câu viết hoa ít gặp hơn " hôm"].
- Cột "số byte" chỉ tính chữ của từ, không tính dấu cách đứng trước (dấu cách nằm trong token).

## Cell 3.2 — Đếm token 2 câu × 2 tokenizer

```python
rows.append({"sentence": sent, "lang": lang, "tokenizer": name, "n_chars": len(sent),
             "n_bytes": len(sent.encode("utf-8")),  # chữ có dấu tốn 2–3 byte
             "n_tokens": len(ids), ...})
```

**Output thật:**
```text
                       sentence lang tokenizer  n_chars  n_bytes  n_tokens
          Hôm nay trời rất đẹp.   vi      gpt2       21       29        22
          Hôm nay trời rất đẹp.   vi   qwen2.5       21       29         8
The weather is very nice today.   en      gpt2       31       31         7
The weather is very nice today.   en   qwen2.5       31       31         7
```

**Giải thích:**
- **29 byte** [Tự tính]: 21 ký tự; ô (2 byte), ờ (3), ấ (3), đ (2), ẹ (3) → thêm 1 + 2 + 2 + 1 + 2 = 8 → 29. Câu Anh toàn ASCII → 31 ký tự = 31 byte.
- **GPT-2 22 token ≈ gần bằng số byte (29):** gần như cắt từng byte ở chữ có dấu. **Qwen 8 token**. GPT-2 tốn **2.75 lần**.
- Câu Anh 7 = 7: cả hai đủ dữ liệu tiếng Anh.
- **Hệ quả:** ít token hơn → tốn ít chỗ trong ngữ cảnh (giới hạn 32,768, mục 9), sinh nhanh hơn (mỗi token một bước, mục 6), rẻ hơn nếu dùng API (tính tiền theo token).

## Cell 3.3 — Kích thước từ điển

**Output thật:**
```text
gpt2 vocab_size: 50,257
qwen tok.vocab_size (token thường): 151,643
qwen len(tok) (+ token đặc biệt): 151,665
qwen model.config.vocab_size (số hàng embedding): 151,936
```
**Giải thích:** 151,665 − 151,643 = **22 token đặc biệt** (`<|im_start|>`, `<|im_end|>`, `<|endoftext|>`…). Bảng embedding có 151,936 hàng, **dư 271 hàng** không ứng với token nào. [Suy luận] 151,936 = 128 × 1,187: kích thước được làm tròn lên bội số của 128 để phép nhân ma trận trên GPU hiệu quả hơn (thói quen phổ biến; model card không ghi lý do). `assert tok.vocab_size <= len(tok) <= model.config.vocab_size` bảo đảm mọi token ID đều có hàng embedding.

## Cell 3.4 — Embedding: tra hàng theo token ID

```python
emb_layer = model.get_input_embeddings()  # nn.Embedding
vi_ids = torch.tensor([tok(SENT_VI, add_special_tokens=False)["input_ids"]], device=model.device)
with torch.no_grad():
    vi_emb = emb_layer(vi_ids)
same_matrix = model.get_output_embeddings().weight.data_ptr() == emb_layer.weight.data_ptr()
assert torch.equal(vi_emb[0], emb_layer.weight[vi_ids[0]])   # embedding = các hàng của bảng
assert same_matrix == ARCH["tie_word_embeddings"]
```

**Output thật:**
```text
Bảng embedding: (151936, 1536) = 233,373,696 tham số
Token ID của câu: (1, 8) → embedding: (1, 8, 1536)
5 số đầu của vector token đầu tiên 'H': [0.0228, -0.0245, 0.0029, -0.0022, -0.0347]
embedding (151936, 1536) | câu (1, 8) → (1, 8, 1536) | dùng chung với LM head: True
```

**Giải thích:**
- `(1, 8) → (1, 8, 1536)`: mỗi token ID thay bằng một vector 1,536 số. `assert torch.equal(...)` chứng minh **embedding = lấy hàng theo ID**, đúng công thức $e_t = E[t, :]$.
- Các số nhỏ quanh 0 (±0.03): trọng số học được, bản thân từng số không có nghĩa riêng; ý nghĩa nằm ở **vị trí tương đối** giữa các vector.
- `data_ptr()` = địa chỉ bộ nhớ. Bằng nhau → embedding và LM head là **cùng một ma trận** → LM head tính logits $= h E^\top$: token có vector embedding giống trạng thái cuối $h$ thì điểm cao.

**Phần thêm ở cell 3.4:** bảng "token → token_id → lấy hàng số mấy → 5 số đầu của vector" và heatmap `figures/embedding_qwen.png` (8 token × 16 chiều đầu trong 1,536 chiều, màu đỏ dương, xanh âm). Đọc hình: mỗi **hàng** là vector của một token; hai token khác nhau có mẫu màu khác nhau; các số đều nhỏ (khoảng ±0.05) vì embedding được khởi tạo và học ở thang nhỏ. Hình chỉ để thấy "token ID → một hàng số", không đọc nghĩa từ từng chiều.

## Cell 3.5 — Decode ngược
Câu → token ID → `decode` → phải ra đúng câu gốc, cả 2 tokenizer, cả 2 câu: **4/4 đúng**. Chứng minh tokenization **không mất thông tin**, kể cả GPT-2 cắt vụn chữ có dấu.

## Câu hỏi tự kiểm

| Câu hỏi | Ý trả lời |
|---|---|
| `Ġ` là gì? | Dấu cách (byte `0x20`) được thay bằng ký tự in được |
| Token ID có nghĩa không? | Không; chỉ là số thứ tự để tra hàng embedding |
| Dùng tokenizer GPT-2 cho Qwen được không? | Không; token ID hai bên khác nhau, model học với tokenizer nào phải dùng tokenizer đó |
| Embedding là gì? | Một hàng 1,536 số trong bảng 151,936 × 1,536, tra theo token ID, học được khi train |

---
