---
title: Chất lượng 4-bit tốt hơn với GPTQ, nạp checkpoint AWQ / GPTQ thật, và vài cái bẫy
date: 2026-11-20
order: 5
summary: INT4 round-to-nearest mất 2 đến 4 điểm perplexity. Một bản GPTQ lấy lại 38 đến 66 phần trăm, sau một câu chuyện gỡ lỗi về massive activation từng làm nó tệ hơn RTN trên Llama. Checkpoint AWQ, GPTQ (cả act-order) và FP8 thật được nạp và kiểm chứng.
---

Kernel nhanh chỉ là một nửa câu chuyện lượng tử hóa: nửa kia là weight 4 bit mất bao nhiêu chất lượng. Vì vậy thư viện có hai cách có được weight, một quantiser round-to-nearest một dòng và một quantiser có hiệu chỉnh, và nó có thể nạp checkpoint do các công cụ khác tạo ra.

## Round-to-nearest: đường cơ sở

`quantforge.nn.quantize_model(model, "int4", group_size=128)` lượng tử hóa từng `nn.Linear` độc lập: mỗi nhóm 128 hàng đầu vào của một cột ánh xạ giá trị nhỏ nhất và lớn nhất của nó lên 16 mức. Mất 3 giây cho model 0.5B, không cần dữ liệu, và mất một lượng chất lượng nhìn thấy được (1 đến 4 điểm perplexity trên các model dưới đây). Head (`lm_head`) mặc định giữ ở fp16; lượng tử hóa nó sang INT8 là không mất mát (perplexity 24.08 so với 24.09 cho Qwen2.5-0.5B) và nhanh hơn 16 phần trăm, nên các số trong bài này dùng head INT8 ở mọi nơi.

## GPTQ trong thư viện

`quantforge.gptq.quantize_model_gptq` cài đặt GPTQ tuần tự (Frantar và cộng sự) và cho ra đúng các layout mà kernel dùng:

1. các decoder layer chạy trên 128 cửa sổ hiệu chỉnh 512 token; hidden state đi vào layer đầu được bắt bằng một module thế chỗ ghi lại input rồi hủy lượt forward;
2. từng layer một, một forward pre-hook trên mọi Linear tích lũy Hessian của input, `H = 2 X^T X / n`, trong khi layer xử lý activation hiệu chỉnh (với norm của mỗi token bị giới hạn ở 30 lần trung vị, xem cái bẫy bên dưới);
3. mỗi Linear được lượng tử hóa từng cột; sai số của mỗi cột đã làm tròn, nhân theo đường chéo của nghịch đảo thừa số Cholesky của `H`, được đẩy sang các cột chưa lượng tử hóa;
4. scale và zero point của nhóm được tính từ weight *đã cập nhật* ở đầu mỗi nhóm, với scale được làm tròn về f16 **trước** khi lan truyền sai số, để quá trình tối ưu thấy đúng weight mà kernel sẽ dùng;
5. layer đã lượng tử hóa được chạy lại trên tập hiệu chỉnh; output của nó là input của layer kế tiếp (nên các layer sau thấy nhiễu lượng tử hóa của các layer trước).

Hiệu chỉnh mất 42 giây cho Qwen2.5-0.5B và 121 giây cho 1.5B trên GPU laptop 6 GB. Bộ nhớ bị chặn bởi một layer, các Hessian của nó (lớn nhất, `down_proj`, là `intermediate x intermediate` ở f32) và các activation đã lưu.

{{chart:gptq_ppl}}

| model | fp16 | RTN INT4 | GPTQ INT4 | phần mất mát của RTN được lấy lại |
|---|--:|--:|--:|--:|
| Qwen2.5-0.5B-Instruct | 18.79 | 22.91 | **20.20** | 66 % |
| Qwen2.5-1.5B-Instruct | 12.66 | 14.30 | **13.38** | 56 % |
| Llama-3.2-1B-Instruct | 17.91 | 22.28 | **20.12** | 49 % |
| OLMo-2-0425-1B-Instruct | 17.24 | 19.02 | **18.35** | 38 % |

Đây là perplexity trên **toàn bộ** tập test Wikitext-2 (564 đến 584 cửa sổ 512 token); hiệu chỉnh dùng các cửa sổ của tập *train*. Tốc độ decode không đổi (cùng layout): khoảng 226 token/s cho cả hai biến thể trên Qwen2.5-0.5B.

### Một cái bẫy: phiên bản đầu của bảng này đã sai

Lần đo đầu của tôi chỉ dùng 8 cửa sổ test đầu tiên (4096 token), và trên đó GPTQ thắng round-to-nearest ở mọi model. Đo lại trên toàn tập test cho thấy trên **Llama-3.2-1B GPTQ tệ hơn RTN (24.08 so với 22.28)** và trên Qwen2.5-1.5B nó chỉ lấy lại 24 phần trăm mất mát. Tám cửa sổ đơn giản là quá ít cho một phương pháp mà các lỗi hiếm khi xảy ra. Cuộc điều tra, qua bốn bước:

1. **Loss theo từng cửa sổ** (`benchmarks/ppl_per_window.py`): GPTQ tốt hơn RTN ở 83 phần trăm số cửa sổ (chênh lệch trung vị -0.11 nat), nhưng ở vài cửa sổ nó hỏng hoàn toàn, với loss 9 đến 10 mỗi token thay vì khoảng 3. Một nhúm cửa sổ như vậy chi phối trung bình.
2. **Theo từng layer** (hidden state so với model fp16 trên một cửa sổ tốt và một cửa sổ hỏng): sai số nhỏ cho đến layer 1 rồi bùng nổ ở *một vị trí token* (vị trí 1 trong cửa sổ hỏng) và lan từ đó sang mọi token sau.
3. **Layer nào?** Chỉ quantize `layers.1.mlp.down_proj` bằng RTN (còn lại bằng GPTQ) đưa Llama từ 24.08 xuống 20.26. Layer đó tạo ra *massive activation* của token attention-sink: input của nó chứa một token có norm **gấp 675 lần trung vị** (790 lần ở Qwen2.5-1.5B). Trong chính layer đó sai số trung bình của GPTQ trên các token hiệu chỉnh thấp hơn RTN 2.5 đến 5 lần, đúng như thiết kế, nên kiểm tra sai số trung bình không thể thấy vấn đề.
4. **Vì sao**: token đơn lẻ đó chi phối Hessian `H = 2 X^T X / n`, việc bù sai số được tinh chỉnh theo nó, và các token khác, cũng như các token sink ở vị trí khác, phải trả giá.

Một danh sách layer cố định không phải cách sửa: ép RTN cho các layer massive-activation giúp Llama (20.26) và Qwen-1.5B (13.47) nhưng làm hại Qwen-0.5B (20.83 so với 20.20). Thứ hiệu quả cho cả ba là **giới hạn norm token trong Hessian**: trước khi một token hiệu chỉnh vào `H`, norm của nó bị giới hạn ở 30 lần norm token trung vị. Token massive vẫn tham gia nhưng với trọng số bị chặn. Kết quả: Llama 24.08 xuống **20.12**, Qwen-1.5B 13.91 xuống **13.38**, Qwen-0.5B không đổi (20.21 xuống 20.20). Đây là mặc định (`token_clip=30`).

### Những thứ không giúp (nên mặc định tắt)

* cắt khoảng giá trị theo nhóm (`clip=True`): giúp OLMo và Qwen-1.5B, làm hỏng Llama-3.2-1B (30.9 trên 8 cửa sổ đầu so với 23.4 khi không dùng): sai số làm tròn trong không gian weight mà nó tối thiểu hóa không phải sai số output của layer mà GPTQ tối thiểu hóa;
* act-order (`act_order=True`): một layer có hoán vị đầu vào không thể tham gia việc gộp q/k/v, nên decode tụt từ 226 xuống 169 token/s trên Qwen2.5-0.5B, và perplexity cũng không tốt hơn;
* damping lớn hơn: trước khi có giới hạn token, 0.1 biến Llama thành model có perplexity 58 (giới hạn token làm kết quả ít nhạy với nó hơn nhiều);
* nhiều cửa sổ hiệu chỉnh hơn (256) hoặc dài hơn (2048 token): không có mức tăng nhất quán.

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
