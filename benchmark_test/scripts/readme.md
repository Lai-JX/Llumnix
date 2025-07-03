+ run_pdd_multi.sh：用数据集（sharegpt）进行测试，不同实例的tp相同，支持pd分离
    + 参数：tp、请求数量、实例总数、模型、分布类型、qps
    + 实例：`./run_pdd_multi.sh 1 2000 4 llama-7b poisson 2`
    + 在Llumnix的实验中，在 4*4 GPU（NVIDIA A10 (24 GB)）的情况下，qps采用的是7.0-8.0，故而将qps设为2，但发现running的请求数经常是个位数，waiting的请求数为0，因此需要增加qps，这是因为L40显存更高，算力也更足。
+ tp_heterogeneity.sh: 不同实例（prefill/decode）的tp不同
    + 参数：prefill_dps、decode_dps、请求数量、模型、分布类型、qps、log_dir_prefix
    + 例：`./tp_heterogeneity.sh "1,1" "2" 2000 llama-7b poisson 4`