#!/bin/bash
ulimit -n 63225
req_num=2000
gputype=A6000-2-formal2 
# gputype=A6000-2-test
prefix="$gputype"
max_migration_concurrencys=(1) 

# qps=(4 4.2 4.4 4.6 4.8 5 5.2 5.4 5.6 5.8 6 6.2 6.4 6.6 6.8 7 7.2 7.4 7.6 7.8 8)

# for max_migration_concurrency in "${max_migration_concurrencys[@]}"; do
#     for QPS in "${qps[@]}"; do
#         echo $max_migration_concurrency, $QPS
#         ./run_base.sh "1,1" "1,1" $req_num llama-7b poisson $QPS "$prefix-concurrency-$max_migration_concurrency-pdd-4" $max_migration_concurrency
#     done
# done

# gputype=A6000-2
# prefix="$gputype"
# prompt_len=128
# response_len=256
# qps=(15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35)
# # qps=(13.6 14 14.4 14.8 15 15.4 15.8 16 16.4 16.8 17 17.4 17.8 18 18.4 18.8 19 19.4 19.8 20)
# for max_migration_concurrency in "${max_migration_concurrencys[@]}"; do
#     for QPS in "${qps[@]}"; do
#         echo $max_migration_concurrency, $QPS
#         ./run_base.sh "1,1" $req_num llama-13b poisson $QPS "$prefix-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len
#     done
# done

gputype=A6000-2
prefix="$gputype"
prompt_len=128
response_len=256
qps=(3.5 4)
# qps=(13.6 14 14.4 14.8 15 15.4 15.8 16 16.4 16.8 17 17.4 17.8 18 18.4 18.8 19 19.4 19.8 20)
for max_migration_concurrency in "${max_migration_concurrencys[@]}"; do
    for QPS in "${qps[@]}"; do
        echo $max_migration_concurrency, $QPS
        ./run_base.sh "1,1,1,1,1,1" "" $req_num llama-13b poisson $QPS "$prefix-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len
        ./run_base.sh "1" "1,1,1,1,1" $req_num llama-13b poisson $QPS "$prefix-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len
        ./run_base.sh "1,1,1,1,1,1,1" "" $req_num llama-13b poisson $QPS "$prefix-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len
        ./run_base.sh "1" "1,1,1,1,1,1" $req_num llama-13b poisson $QPS "$prefix-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len
        ./run_base.sh "1,1,1,1,1,1,1,1" "" $req_num llama-13b poisson $QPS "$prefix-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len
        ./run_base.sh "1" "1,1,1,1,1,1,1" $req_num llama-13b poisson $QPS "$prefix-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len

        ./run_base.sh "1" "2,2,2" $req_num llama-13b poisson $QPS "$prefix-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len
        ./run_base.sh "1" "2,2" $req_num llama-13b poisson $QPS "$prefix-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len

    done
done

qps=(12 14 16 18 20)
# qps=(13.6 14 14.4 14.8 15 15.4 15.8 16 16.4 16.8 17 17.4 17.8 18 18.4 18.8 19 19.4 19.8 20)
for max_migration_concurrency in "${max_migration_concurrencys[@]}"; do
    for QPS in "${qps[@]}"; do
        echo $max_migration_concurrency, $QPS

        ./run_base.sh "1,1,1,1,1,1" "" $req_num llama-7b poisson $QPS "$prefix-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len
        ./run_base.sh "1" "1,1,1,1,1" $req_num llama-7b poisson $QPS "$prefix-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len
        ./run_base.sh "1,1,1,1,1,1,1" "" $req_num llama-7b poisson $QPS "$prefix-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len
        ./run_base.sh "1" "1,1,1,1,1,1" $req_num llama-7b poisson $QPS "$prefix-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len
        ./run_base.sh "1,1,1,1,1,1,1,1" "" $req_num llama-7b poisson $QPS "$prefix-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len
        ./run_base.sh "1" "1,1,1,1,1,1,1" $req_num llama-7b poisson $QPS "$prefix-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len

        ./run_base.sh "1" "2,2,2" $req_num llama-7b poisson $QPS "$prefix-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len
        ./run_base.sh "1" "2,2" $req_num llama-7b poisson $QPS "$prefix-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len
    done
done
