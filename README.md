LongSWE-Bench 128K / SGLang SLO 并发测试
======================================

目的：对用户已经启动的一个 SGLang 服务，以约 20 分钟预算搜索实测满足
P90 TTFT < 10 s 且 mean decode TPS > 15 tokens/s 的最高客户端在途并发。
脚本不启动、停止服务器，也不访问模型/数据下载站点。

一、测试口径

1. “p90”暂解释为 P90 TTFT；不是 P90 请求总时延。
2. mean decode TPS = mean_i[(L_i - 1)/(E2E_i - TTFT_i)]。
   不是系统总 output throughput，也不是 1/mean(TPOT)。
   同时保存 P90 E2E 和 mean E2E TPS 方便核对另一种口径。
3. 输出长度必须显式传 --output-len。示例使用 1024 tokens；四种配置必须一致。
   不同输出长度的结果不能直接比较。
4. max_new_tokens 固定、temperature=0、ignore_eos=true。只测推理性能，不评价补丁正确率。
5. 使用数据集完整 text/prompt 字符串，不额外套聊天模板，不截断、填充成 131072。
   先查看本地 tokenizer 下实际输入长度；128K 是数据集档位，不保证各模型 token 数一样。
6. 默认每请求在最开头加唯一注释，阻止重复回放 100 条数据时完整前缀命中。
   注释会加少量 tokens，对注意力访问也可能有轻微影响。这一口径是无前缀复用的
   改写负载。若业务要求原始数据前缀复用，用 --allow-prefix-reuse，四种配置保持相同。
   不要改掉 HiCache 所需的 radix cache；原始启动方式本来就有不同缓存能力。
7. 每个阶段之前 flush cache；阶段内不 flush，维持 C 个 worker，请求结束立即补齐。
   这是 closed-loop 客户端在途并发，不等于服务器同时 decode 的请求数量。
   原样保留四条启动命令；最大并发搜索上限 64。
8. 同一 GPU 上不同时运行其他负载。各服务完全 ready 后再执行脚本。
   统一 mem-fraction-static、chunked prefill、max-running-requests 等共同设置，
   并记录实际默认值。CUDA Graph 最大 batch size 不是客户端并发限制。

二、安装和检查（均在运行模型的服务器）

    pip install -r requirements.txt

    python longswe_slo.py --inspect-only --output-len 1024 --output-dir unused

默认数据路径：
~/datasets/longswebench/LongSWE_Bench/128K

默认本地 tokenizer：
/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash-0731

自动检测 text/prompt；若二者都有或均无，脚本会打印字段并要求 --prompt-column。
--inspect-only 完全不请求服务器、不写输出文件。
如果本地 tokenizer 确实需要自定义 Python，可添加 --trust-remote-code；不会联网下载。

三、逐个运行

在另一个终端启动第一种服务，等待 ready，然后：

    python longswe_slo.py --label hicache --output-len 1024 --output-dir results/hicache

服务正常关闭后换第二种启动命令，再：

    python longswe_slo.py --label hisparse --output-len 1024 --output-dir results/hisparse

然后依次：

    python longswe_slo.py --label adaptive_hisparse --output-len 1024 --output-dir results/adaptive_hisparse
    python longswe_slo.py --label kvduo --output-len 1024 --output-dir results/kvduo

每个命令只测试当前已启动的服务器。不要同时运行这些命令。
所有命令默认 http://127.0.0.1:32073；客户端通常不需要 sudo。
输出目录必须不存在，以免覆盖旧实验。

四、20 分钟搜索如何工作

准备数据、tokenization 和短请求预热不计入 1200 秒测试预算。
从 C=8 开始，先 8,16,32,64 倍增；遇到 FAIL 后，在已测 PASS/FAIL 之间二分。
如果 C=8 就失败，搜索 4、2、1 等更低值。每个 probe 默认发请求 180 秒，
然后等该阶段所有请求自然结束，再测试下个并发，避免两个阶段混在一起。
默认留至少 300 秒做最高通过并发的确认，并预留最后 90 秒给尾部请求排空。
若边界已经找到，会把更多剩余时间用于确认。

1200 秒是总预算，不是每个候选并发都测 1200 秒。
客户端在预算结束前 90 秒停止发新请求，已经发出的请求完整收尾。
若尾部请求超过预留的 90 秒，实际墙钟可能略超 20 分钟，脚本记录真实耗时；
不强杀尾部请求来人为改善统计。每请求 timeout 默认 300 秒；任何错误/超时
使该阶段 ERROR 并停止本次搜索，不把慢请求静默从指标中剔除。
超时后建议重启当前服务再测，防止服务器仍在处理旧请求。

每阶段至少完成 20 条才下判断。样本不足标为 INSUFFICIENT，不能当作 FAIL。
小并发或很长输出时，20 分钟可能不够搜索和确认：这会显式报告，不伪造最大值。
20 条适合探索，最终应关注确认阶段的样本数；接近阈值时仍应延长和重复测试。

倍增/二分只是一种启发式，因为连续批处理、内存回收和调度使性能不保证严格单调。
输出是“本轮实测通过并复测的候选”，不是全局最大值的数学证明。
如果只有 C=32 PASS、C=48 FAIL，报告区间 [32,48)，不要称 32 是精确最大并发。
如果 C=64 通过，只能说 >=64。若复测失败，结果为 INCONCLUSIVE，不沿用探测阶段
的通过结论冒充稳定结果，也不会把不同并发的请求合在一起计算 P90。

若 leader 要的是“最高并发连续稳定运行 20 分钟”，先搜索，再独立做最终 20 分钟
验证；总计每配置只给 20 分钟，不能同时保证完整搜索和该项长时间验证。

五、结果

每个输出目录：
- config.json：数据指纹、基础输入长度、采样与测试参数。
- requests.jsonl：逐请求 TTFT/E2E/TPS、token 数、缓存命中量、错误；不保存庞大生成文本。
- stages.json：探测阶段检查点。
- summary.json：最终候选、失败边界、确认结果、各阶段统计和真实耗时。

优先查看 summary.json 中 confirmed_concurrency；为 null 时本轮无已确认结论。
smallest_observed_fail 和 confirmed_concurrency 相邻时，可称“本轮边界已细化到相邻整数”；
若中间仍有未测值，必须同时报告区间。

流式 /generate 必须返回 cumulative meta_info.completion_tokens。
TTFT 按客户端收到首个含生成 token 的 SSE chunk 计算，包含客户端 HTTP/网络及服务端排队。
按服务端 token 计数计算 TPS，绝不把 SSE chunk 数当 token 数。
如果 coalesced_first_chunks 不为 0，确认服务端 stream-interval=1 后重测，
否则 chunk 合并会让首 token 时刻测量变粗。

交付验证：脚本通过本地模拟 SSE 服务的边界搜索、流异常和指标计算检查。
尚未连接用户 H100 服务器或执行真实模型压测。
