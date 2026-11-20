---
title: Chất lượng 4-bit tốt hơn với GPTQ, nạp checkpoint AWQ / GPTQ thật, và vài cái bẫy
date: 2026-11-20
order: 5
summary: INT4 round-to-nearest mất 1 đến 4 điểm perplexity. Một bản GPTQ trong thư viện lấy lại 17 đến 65 phần trăm, checkpoint AWQ và GPTQ thật được nạp và kiểm chứng, và các thứ không giúp được liệt kê rõ.
---

Kernel nhanh chỉ là một nửa câu chuyện lượng tử hóa: nửa kia là weight 4 bit mất bao nhiêu chất lượng. Vì vậy thư viện có hai cách có được weight, một quantiser round-to-nearest một dòng và một quantiser có hiệu chỉnh, và nó có thể nạp checkpoint do các công cụ khác tạo ra.

## Round-to-nearest: đường cơ sở

`quantforge.nn.quantize_model(model, "int4", group_size=128)` lượng tử hóa từng `nn.Linear` độc lập: mỗi nhóm 128 hàng đầu vào của một cột ánh xạ giá trị nhỏ nhất và lớn nhất của nó lên 16 mức. Mất 3 giây cho model 0.5B, không cần dữ liệu, và mất một lượng chất lượng nhìn thấy được (1 đến 4 điểm perplexity trên các model dưới đây). Head (`lm_head`) mặc định giữ ở fp16; lượng tử hóa nó sang INT8 là không mất mát (perplexity 24.08 so với 24.09 cho Qwen2.5-0.5B) và nhanh hơn 16 phần trăm, nên các số trong bài này dùng head INT8 ở mọi nơi.

## GPTQ trong thư viện

`quantforge.gptq.quantize_model_gptq` cài đặt GPTQ tuần tự (Frantar và cộng sự) và cho ra đúng các layout mà kernel dùng:

1. các decoder layer chạy trên 128 cửa sổ hiệu chỉnh 512 token; hidden state đi vào layer đầu được bắt bằng một module thế chỗ ghi lại input rồi hủy lượt forward;
2. từng layer một, một forward pre-hook trên mọi Linear tích lũy Hessian của input, `H = 2 X^T X / n`, trong khi layer xử lý activation hiệu chỉnh;
3. mỗi Linear được lượng tử hóa từng cột; sai số của mỗi cột đã làm tròn, nhân theo đường chéo của nghịch đảo thừa số Cholesky của `H`, được đẩy sang các cột chưa lượng tử hóa;
4. scale và zero point của nhóm được tính từ weight *đã cập nhật* ở đầu mỗi nhóm, với scale được làm tròn về f16 **trước** khi lan truyền sai số, để quá trình tối ưu thấy đúng weight mà kernel sẽ dùng;
5. layer đã lượng tử hóa được chạy lại trên tập hiệu chỉnh; output của nó là input của layer kế tiếp (nên các layer sau thấy nhiễu lượng tử hóa của các layer trước).

Hiệu chỉnh mất 43 giây cho Qwen2.5-0.5B và 125 giây cho 1.5B trên GPU laptop 6 GB. Bộ nhớ bị chặn bởi một layer, các Hessian của nó (lớn nhất, `down_proj`, là `intermediate x intermediate` ở f32) và các activation đã lưu.

{{chart:gptq_ppl}}

| model | fp16 | RTN INT4 | GPTQ INT4 | phần mất mát của RTN được lấy lại |
|---|--:|--:|--:|--:|
| Qwen2.5-0.5B-Instruct | 19.93 | 24.08 | **21.38** | 65 % |
| Qwen2.5-1.5B-Instruct | 13.48 | 15.31 | **15.00** | 17 % |
| Llama-3.2-1B-Instruct | 20.15 | 24.42 | **23.39** | 24 % |
| OLMo-2-0425-1B-Instruct | 19.32 | 21.31 | **20.62** | 34 % |

Hiệu chỉnh dùng các cửa sổ của tập *train* Wikitext-2; perplexity đo trên 8 x 512 token của tập *test*. Tốc độ decode không đổi (cùng layout): 226 token/s so với 219 cho Qwen2.5-0.5B trong cùng phiên, nằm trong nhiễu của máy này.

**Một cách đọc trung thực**: GPTQ luôn giúp, nhưng mức tăng rất không đều (65 phần trăm mất mát ở model nhỏ nhất, 17 phần trăm ở 1.5B). Tôi chưa có lời giải thích rõ ràng; các model mà mất mát RTN bị chi phối bởi vài kênh ngoại lai có vẻ hưởng lợi nhiều nhất. Với tập test tám cửa sổ 4096 token, chữ số thập phân thứ hai của các perplexity này cũng nhạy với nhiễu. Một nghiên cứu đàng hoàng nên dùng toàn bộ tập test và nhiều seed cho các cửa sổ hiệu chỉnh.

### Những thứ không giúp (nên mặc định tắt)

| biến thể | Qwen 0.5B | Qwen 1.5B | Llama-3.2 1B | OLMo-2 1B |
|---|--:|--:|--:|--:|
| GPTQ, mặc định (damping 0.01, group 128) | 21.38 | 15.00 | 23.39 | 20.62 |
| + cắt khoảng giá trị theo nhóm (`clip=True`) | 21.66 | **14.40** | 30.87 | **19.77** |
| act-order (`act_order=True`) | 21.74 | | | |
| damping 0.1 / 0.03 | 21.54 / 21.61 | | | |
| 256 cửa sổ hiệu chỉnh | 21.44 | | | |
| group size 64 | **21.26** | | | |

Cắt khoảng giá trị (co min/max của mỗi hàng trong nhóm khi điều đó giảm sai số làm tròn của weight) là thứ thú vị nhất: nó cho kết quả tốt nhất trên hai model và tệ nhất trên Llama-3.2-1B, nơi perplexity nhảy từ 23.4 lên 30.9. Tiêu chí nó tối ưu, sai số của weight, không phải sai số output của layer mà GPTQ tối ưu, và hai thứ này bất đồng ở một số layer. Nó vẫn là một tùy chọn. Act-order cũng tốn tốc độ: một layer có hoán vị đầu vào không thể tham gia việc gộp q/k/v, nên decode tụt từ 226 xuống 169 token/s trên Qwen2.5-0.5B.

## Checkpoint AWQ và GPTQ thật

Hầu hết model 4 bit trên Hub là file AWQ hoặc GPTQ. `quantforge.nn.load_quantized_checkpoint(model, "model.safetensors", method="gptq" | "awq")` chuyển các tensor đóng gói int32 sang layout của kernel (`quant.from_gptq`, `quant.from_awq`), hỗ trợ checkpoint GPTQ `desc_act` bằng cách hoán vị hàng offline (hoán vị activation được áp bằng một phép gather nhỏ lúc chạy) và quy ước zero-point `gptq_v2`.

Kiểm chứng với bản sinh đôi dense (cùng weight đã dequantise trong các layer `nn.Linear` thường), Qwen2.5-0.5B-Instruct, 16 x 512 token test:

| checkpoint | perplexity, bản sinh đôi dense | perplexity, QuantForge | greedy decode | tốc độ decode |
|---|--:|--:|--:|--:|
| GPTQ-Int4 (Qwen chính thức) | 26.335 | 26.336 | 59 / 59 token giống hệt | 182 tok/s |
| AWQ (Qwen chính thức) | 26.689 | 26.689 | 59 / 59 token giống hệt | 184 tok/s |

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
