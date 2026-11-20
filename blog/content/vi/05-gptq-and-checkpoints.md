---
title: Chất lượng 4-bit tốt hơn với GPTQ, nạp checkpoint AWQ / GPTQ thật, và vài cái bẫy
date: 2026-11-20
order: 5
summary: INT4 round-to-nearest mất 2 đến 4 điểm perplexity. Một bản GPTQ lấy lại 47 đến 60 phần trăm, sau một câu chuyện gỡ lỗi về massive activation từng làm nó tệ hơn RTN trên Llama. Checkpoint AWQ, GPTQ (cả act-order) và FP8 thật được nạp và kiểm chứng.
---

Kernel nhanh chỉ là một nửa câu chuyện lượng tử hóa: nửa kia là weight 4 bit mất bao nhiêu chất lượng. Vì vậy thư viện có hai cách có được weight, một quantiser round-to-nearest một dòng và một quantiser có hiệu chỉnh, và nó có thể nạp checkpoint do các công cụ khác tạo ra.

## Round-to-nearest: đường cơ sở

`quantforge.nn.quantize_model(model, "int4", group_size=128)` lượng tử hóa từng `nn.Linear` độc lập: mỗi nhóm 128 hàng đầu vào của một cột ánh xạ giá trị nhỏ nhất và lớn nhất của nó lên 16 mức. Mất 3 giây cho model 0.5B, không cần dữ liệu, và mất một lượng chất lượng nhìn thấy được (1 đến 4 điểm perplexity trên các model dưới đây). Head (`lm_head`) mặc định giữ ở fp16; lượng tử hóa nó sang INT8 là không mất mát (perplexity 24.08 so với 24.09 cho Qwen2.5-0.5B) và nhanh hơn 16 phần trăm, nên các số trong bài này dùng head INT8 ở mọi nơi.

## GPTQ trong thư viện

`quantforge.gptq.quantize_model_gptq` cài đặt GPTQ tuần tự (Frantar và cộng sự) và cho ra đúng các layout mà kernel dùng:

1. các decoder layer chạy trên 128 cửa sổ hiệu chỉnh 512 token; hidden state đi vào layer đầu được bắt bằng một module thế chỗ ghi lại input rồi hủy lượt forward;
2. từng layer một, một forward pre-hook trên mọi Linear tích lũy Hessian của input, `H = 2 X^T X / n`, trong khi layer xử lý activation hiệu chỉnh (với norm của mỗi token bị giới hạn ở 10 lần trung vị, xem cái bẫy bên dưới);
3. mỗi Linear được lượng tử hóa từng cột; sai số của mỗi cột đã làm tròn, nhân theo đường chéo của nghịch đảo thừa số Cholesky của `H`, được đẩy sang các cột chưa lượng tử hóa;
4. scale và zero point của nhóm được tính từ weight *đã cập nhật* ở đầu mỗi nhóm, với scale được làm tròn về f16 **trước** khi lan truyền sai số, để quá trình tối ưu thấy đúng weight mà kernel sẽ dùng;
5. layer đã lượng tử hóa được chạy lại trên tập hiệu chỉnh; output của nó là input của layer kế tiếp (nên các layer sau thấy nhiễu lượng tử hóa của các layer trước).

Hiệu chỉnh mất 49 giây cho Qwen2.5-0.5B và khoảng 150 giây cho 1.5B trên GPU laptop 6 GB. Bộ nhớ bị chặn bởi một layer, các Hessian của nó (lớn nhất, `down_proj`, là `intermediate x intermediate` ở f32) và các activation đã lưu.

{{chart:gptq_ppl}}

| model | fp16 | RTN INT4 | GPTQ INT4 | phần mất mát của RTN được lấy lại |
|---|--:|--:|--:|--:|
| Qwen2.5-0.5B-Instruct | 18.79 | 22.91 | **20.45** | 60 % |
| Qwen2.5-1.5B-Instruct | 12.66 | 14.30 | **13.38** | 56 % |
| Llama-3.2-1B-Instruct | 17.91 | 22.28 | **20.21** | 47 % |
| OLMo-2-0425-1B-Instruct | 17.24 | 19.02 | **18.13** | 50 % |
| Qwen2.5-3B-Instruct (không có mốc fp16: không vừa) | - | 13.13 | **12.06** | - |

Đây là perplexity trên **toàn bộ** tập test Wikitext-2 (564 đến 584 cửa sổ 512 token); hiệu chỉnh dùng 128 cửa sổ lấy từ 20 000 dòng đầu của tập *train*. Tốc độ decode không đổi (cùng layout).

### Một cái bẫy: phiên bản đầu của bảng này sai, và phiên bản thứ hai cũng vậy

Lần đo đầu của tôi chỉ dùng 8 cửa sổ test đầu tiên (4096 token), và trên đó GPTQ thắng round-to-nearest ở mọi model. Đo lại trên toàn tập test cho thấy trên **Llama-3.2-1B GPTQ tệ hơn RTN (24.08 so với 22.28)** và trên Qwen2.5-1.5B nó chỉ lấy lại 24 phần trăm mất mát. Tám cửa sổ đơn giản là quá ít cho một phương pháp mà các lỗi hiếm khi xảy ra. Cuộc điều tra, qua bốn bước:

1. **Loss theo từng cửa sổ** (`benchmarks/ppl_per_window.py`): GPTQ tốt hơn RTN ở 83 phần trăm số cửa sổ (chênh lệch trung vị -0.11 nat), nhưng ở vài cửa sổ nó hỏng hoàn toàn, với loss 9 đến 10 mỗi token thay vì khoảng 3. Một nhúm cửa sổ như vậy chi phối trung bình.
2. **Theo từng layer** (hidden state so với model fp16 trên một cửa sổ tốt và một cửa sổ hỏng): sai số nhỏ cho đến layer 1 rồi bùng nổ ở *một vị trí token* (vị trí 1 trong cửa sổ hỏng) và lan từ đó sang mọi token sau.
3. **Layer nào?** Chỉ quantize `layers.1.mlp.down_proj` bằng RTN (còn lại bằng GPTQ) đưa Llama từ 24.08 xuống 20.26. Layer đó tạo ra *massive activation* của token attention-sink: input của nó chứa một token có norm **gấp 675 lần trung vị** (790 lần ở Qwen2.5-1.5B). Trong chính layer đó sai số trung bình của GPTQ trên các token hiệu chỉnh thấp hơn RTN 2.5 đến 5 lần, đúng như thiết kế, nên kiểm tra sai số trung bình không thể thấy vấn đề.
4. **Vì sao**: token đơn lẻ đó chi phối Hessian `H = 2 X^T X / n`, việc bù sai số được tinh chỉnh theo nó, và các token khác, cũng như các token sink ở vị trí khác, phải trả giá.

Một danh sách layer cố định không phải cách sửa: ép RTN cho các layer massive-activation giúp Llama (20.26) và Qwen-1.5B (13.47) nhưng làm hại Qwen-0.5B (20.83 so với 20.20). Thứ hiệu quả cho cả ba là **giới hạn norm token trong Hessian**: trước khi một token hiệu chỉnh vào `H`, norm của nó bị giới hạn ở một bội số của norm token trung vị. Token massive vẫn tham gia nhưng với trọng số bị chặn.

**Phiên bản thứ hai cũng sai.** Lúc đầu tôi chọn cap 30, và nó sửa được perplexity toàn tập (Llama 24.08 xuống 20.12). Một bài kiểm tra hồi quy chậm mà tôi viết sau đó, trên năm cửa sổ từng bị hỏng, cho thấy **một cửa sổ vẫn hỏng** (cửa sổ 70: loss 5.9 so với 3.0 của fp16), dù perplexity toàn tập không phân biệt được. Một phân tích theo từng cửa sổ trên cả 564 cửa sổ đã chốt lựa chọn:

| cap | ppl toàn tập | số cửa sổ GPTQ tệ hơn RTN | cửa sổ tệ nhất so với RTN |
|---|--:|--:|--:|
| 30 | 20.123 | 2 (1 %) | +2.76 nat |
| **10** | 20.192 | **0** | không có |
| 5 | 20.328 | 0 | không có |

Vì vậy mặc định là **10** (`token_clip=10`): nó tốn khoảng 0.07 điểm perplexity so với 30 và loại bỏ các cửa sổ thảm họa hiếm gặp. Bài học: với một phương pháp mà lỗi hiếm khi xảy ra, giá trị trung bình là tiêu chí nghiệm thu sai; bài kiểm tra quan trọng phải nhìn vào trường hợp xấu nhất.

### Những thứ không giúp (nên mặc định tắt)

* cắt khoảng giá trị theo nhóm (`clip=True`): giúp OLMo và Qwen-1.5B, làm hỏng Llama-3.2-1B (30.9 trên 8 cửa sổ đầu so với 23.4 khi không dùng): sai số làm tròn trong không gian weight mà nó tối thiểu hóa không phải sai số output của layer mà GPTQ tối thiểu hóa;
* act-order (`act_order=True`): một layer có hoán vị đầu vào không thể tham gia việc gộp q/k/v, nên decode tụt từ 226 xuống 169 token/s trên Qwen2.5-0.5B, và perplexity cũng không tốt hơn;
* damping lớn hơn: trước khi có giới hạn token, 0.1 biến Llama thành model có perplexity 58 (giới hạn token làm kết quả ít nhạy với nó hơn nhiều);
* nhiều cửa sổ hiệu chỉnh hơn (256) hoặc dài hơn (2048 token): không có mức tăng nhất quán.

### Độ bền vững đến đâu?

Ba kiểm tra ngoài một con số perplexity duy nhất (toàn bộ tập test Wikitext-2, GPTQ với các giá trị mặc định):

| kiểm tra | Qwen2.5-0.5B | Llama-3.2-1B | Qwen2.5-1.5B |
|---|--:|--:|--:|
| RTN INT4 | 22.91 | 22.28 | 14.30 |
| GPTQ, 3 seed hiệu chỉnh | 20.45 / 20.46 / 20.48 (sd 0.01) | 20.21 / 20.22 / 20.25 (sd 0.02) | 13.38 (một seed) |
| GPTQ, hiệu chỉnh trên bài báo cnn_dailymail thay vì Wikitext | 20.70 | 20.43 | 13.65 |

* **Seed:** độ lệch giữa các cửa sổ hiệu chỉnh ngẫu nhiên là 0.01 đến 0.02 điểm perplexity, thấp hơn nhiều so với khoảng cách tới RTN. Với cap 30 nó là 0.08 đến 0.15: việc chặn ảnh hưởng của token massive cũng làm kết quả ổn định hơn.
* **Miền khác:** hiệu chỉnh trên báo chí và test trên Wikipedia tốn 0.2 đến 0.3 điểm và vẫn thắng RTN từ hai điểm trở lên. Các số Wikitext-trên-Wikitext trong bảng chính là *trong miền*, hơi có lợi cho GPTQ.
* **Model lớn hơn:** Qwen2.5-3B (không có mốc fp16: không vừa card này): RTN 13.13, GPTQ **12.06**, hiệu chỉnh mất 300 giây.

**Một task thay vì độ hợp lý.** Độ chính xác ARC zero-shot của các model GPTQ cạnh fp16 và RTN (cách chấm của lm-evaluation-harness: `acc` chọn đáp án có log-likelihood cao nhất, `acc_norm` chọn cao nhất trên mỗi ký tự), tính bằng phần trăm. Sai số chuẩn: 1.0 điểm với ARC-Easy (2376 câu), 1.4 với ARC-Challenge (1172 câu):

| model | biến thể | ARC-Easy acc | ARC-Easy acc_norm | ARC-Challenge acc | ARC-Challenge acc_norm |
|---|---|--:|--:|--:|--:|
| Qwen2.5-0.5B | fp16 | 65.6 | 59.2 | 30.7 | 33.8 |
| | RTN | 62.5 | 59.3 | 31.7 | 32.3 |
| | GPTQ | 63.3 | 60.5 | 29.4 | 33.0 |
| Llama-3.2-1B | fp16 | 69.0 | 63.8 | 34.6 | 37.9 |
| | RTN | 64.3 | 60.4 | 33.5 | 35.3 |
| | GPTQ | 66.4 | 62.5 | 33.2 | 36.9 |
| OLMo-2-1B | fp16 | 71.8 | 71.0 | 41.0 | 42.0 |
| | RTN | 71.2 | 68.3 | 39.8 | 40.7 |
| | GPTQ | 70.4 | 69.5 | 39.3 | 42.2 |
| Qwen2.5-1.5B | fp16 | 76.8 | 76.0 | 43.5 | 46.7 |
| | RTN | 65.6 | 62.3 | 37.0 | 39.7 |
| | GPTQ | **73.1** | **72.0** | **40.5** | **44.0** |

Điều bất ngờ là **Qwen2.5-1.5B**: round-to-nearest mất 11 điểm trên ARC-Easy (76.8 xuống 65.6) và 6.5 điểm trên ARC-Challenge, nhiều hơn hẳn mức tăng perplexity khiêm tốn của nó (12.66 lên 14.30) gợi ý, và GPTQ lấy lại 7.5 và 3.5 điểm trong số đó. (Với cap 30 trước đây nó lấy lại 9.4 và 5.1: chênh lệch khoảng hai sai số chuẩn, nên tôi không kết luận cap 10 kém hơn trên model này, nhưng dữ liệu cũng không loại trừ điều đó.) Với ba model còn lại, chênh lệch giữa fp16, RTN và GPTQ phần lớn nằm trong sai số chuẩn: ở INT4 các model nhỏ này mất ít trên ARC dù theo cách nào. Một bảng perplexity một mình có thể che giấu một vấn đề chất lượng thật, và cũng có thể thổi phồng một vấn đề.

## Checkpoint AWQ và GPTQ thật

Hầu hết model 4 bit trên Hub là file AWQ hoặc GPTQ. `quantforge.nn.load_quantized_checkpoint(model, "model.safetensors", method="gptq" | "awq")` chuyển các tensor đóng gói int32 sang layout của kernel (`quant.from_gptq`, `quant.from_awq`), hỗ trợ checkpoint GPTQ `desc_act` bằng cách hoán vị hàng offline (hoán vị activation được áp bằng một phép gather nhỏ lúc chạy) và quy ước zero-point `gptq_v2`.

Kiểm chứng với bản sinh đôi dense (cùng weight đã dequantise trong các layer `nn.Linear` thường), Qwen2.5-0.5B-Instruct, 16 x 512 token test:

| checkpoint | perplexity, bản sinh đôi dense | perplexity, QuantForge | greedy decode | tốc độ decode |
|---|--:|--:|--:|--:|
| GPTQ-Int4 (Qwen chính thức) | 26.335 | 26.336 | 59 / 59 token giống hệt | 182 tok/s |
| AWQ (Qwen chính thức) | 26.689 | 26.689 | 59 / 59 token giống hệt | 184 tok/s |
| **TinyLlama-1.1B GPTQ với `desc_act=True`** (TheBloke; cả 154 layer có act-order) | 17.259 | 17.259 | 60 / 60 token giống hệt | 124 tok/s (layer act-order không được gộp) |
| **FP8-dynamic** (RedHatAI, scale theo kênh), Qwen2.5-0.5B | 22.543 | 22.543 (model fp16 gốc: 22.441) | 28 / 59 giống, xem bên dưới | 165 tok/s |

File FP8 chỉ cung cấp *weight* (activation vẫn là fp16); scale theo kênh của nó được mở rộng ra group 128 và 0.55 phần trăm trong số đó là subnormal ở f16. Các lần greedy decode lệch nhau sau 28 token vì hai token gần hòa đổi chỗ dưới phép làm tròn fp16 hơi khác của phép nhân ma trận bên bản sinh đôi; logit khớp 99.4 phần trăm top-1 và perplexity giống hệt đến ba chữ số thập phân, nên đây là chuyện phá thế hòa, không phải lỗi. Checkpoint act-order là bài kiểm tra *thật* đầu tiên của đường hoán vị hàng (trước đó chỉ test bằng file tổng hợp).

### Cái bẫy tốn một giờ: AWQ thay đổi các norm

Lần chạy AWQ đầu tiên in ra perplexity **83,014** cho bản sinh đôi dense trong khi model QuantForge, dựng cùng cách, in 26.7. Một layer đơn lẻ khớp tới từng bit, nên kernel không sao. Nguyên nhân: AWQ nhân các kênh đầu vào của một layer tuyến tính với một hệ số học được `s` để bảo vệ các weight quan trọng, và gộp nghịch đảo của hệ số đó vào phép tính *đứng trước*, thường là weight của RMSNorm. Vì vậy weight norm lưu trong checkpoint AWQ không phải của model gốc. Nạp chỉ các layer tuyến tính đã lượng tử hóa lên trên norm của model gốc cho kết quả rác; loader giờ copy mọi tensor phi tuyến (embedding, norm) tồn tại ở cả checkpoint và model, và README có ghi điều này. GPTQ không làm vậy, đó là lý do GPTQ chạy đúng ngay lần đầu.

## Tái hiện

```bash
# 4 bit có hiệu chỉnh, kèm so sánh với round-to-nearest
python benchmarks/hf_gptq.py --model Qwen/Qwen2.5-0.5B-Instruct \
    --train wikitext2-train.parquet --text wikitext2-test.parquet

# một checkpoint thật so với bản sinh đôi dense
python benchmarks/hf_load_checkpoint.py --base Qwen/Qwen2.5-0.5B-Instruct \
    --ckpt Qwen2.5-0.5B-Instruct-GPTQ-Int4/model.safetensors --method gptq --text wikitext2-test.parquet
```

Mọi thứ ở trên chạy trên một GPU RTX 4050 Laptop 6 GB; model 3B và Phi-3.5 (3.8B) được nạp trên CPU và lượng tử hóa từng layer trên GPU, còn model 7B chạy từ checkpoint bitsandbytes 4 bit trong 4.65 GB.
