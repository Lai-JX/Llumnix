#!/bin/bash
ulimit -n 63225
req_num=2000
gputype=A6000-2-0916 # multi-port-zmp-parallel-parallel-buffer-main
# gputype=A6000-2-test
max_migration_concurrencys=(1) 
prefix="$gputype"
prompt_len=2016
response_len=32
# 示例：./run_base.sh "1" "1" 100 llama-13b poisson 16 A6000-test-concurrency-4-128-256 4 128 256
# prefix="$gputype-pdd-hetero"
# prompt_len=2016
# response_len=32
# for max_migration_concurrency in "${max_migration_concurrencys[@]}"; do
#     echo $max_migration_concurrency
#     ./run_base.sh "1,1" "2" $req_num llama-7b poisson 4 "$prefix-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len
#     ./run_base.sh "1,1,1,1" "" $req_num llama-7b poisson 4 "$gputype-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len

#     ./run_base.sh "1,1" "2" $req_num llama-13b poisson 2 "$prefix-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len
#     ./run_base.sh "1,1,1,1" "" $req_num llama-13b poisson 2 "$gputype-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len

# done

# prompt_len=2016
# response_len=32
# for max_migration_concurrency in "${max_migration_concurrencys[@]}"; do
#     echo $max_migration_concurrency
#     ./run_base.sh "1,1" "2" $req_num llama-13b poisson 2 "$prefix-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len
#     ./run_base.sh "1,1,1,1" "" $req_num llama-13b poisson 2 "$gputype-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len

# done

# prompt_len=512
# response_len=256
# for max_migration_concurrency in "${max_migration_concurrencys[@]}"; do
#     echo $max_migration_concurrency
#     ./run_base.sh "1,1" "2" $req_num llama-7b poisson 4 "$prefix-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len
#     ./run_base.sh "1,1,1,1" "" $req_num llama-7b poisson 4 "$gputype-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len
#     ./run_base.sh "1" "1,1,1" $req_num llama-7b poisson 4 "$gputype-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len
#     ./run_base.sh "1,1" "1,1" $req_num llama-7b poisson 4 "$gputype-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len

#     ./run_base.sh "1,1" "2" $req_num llama-13b poisson 2 "$prefix-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len
#     ./run_base.sh "1,1,1,1" "" $req_num llama-13b poisson 2 "$gputype-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len

# done

# # prompt_len=1024
# # response_len=1024
# # for max_migration_concurrency in "${max_migration_concurrencys[@]}"; do
# #     echo $max_migration_concurrency
# #     ./run_base.sh "1,1" "2" $req_num llama-7b poisson 4 "$prefix-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len
# #     ./run_base.sh "1,1,1,1" "" $req_num llama-7b poisson 4 "$gputype-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len

# #     ./run_base.sh "1,1" "2" $req_num llama-13b poisson 2 "$prefix-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len
# #     ./run_base.sh "1,1,1,1" "" $req_num llama-13b poisson 2 "$gputype-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len

# # done

# prompt_len=256
# response_len=256
# for max_migration_concurrency in "${max_migration_concurrencys[@]}"; do
#     echo $max_migration_concurrency
#     ./run_base.sh "1,1" "2" $req_num llama-7b poisson 4 "$prefix-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len
#     ./run_base.sh "1,1,1,1" "" $req_num llama-7b poisson 4 "$gputype-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len
#     ./run_base.sh "1" "1,1,1" $req_num llama-7b poisson 4 "$gputype-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len
#     ./run_base.sh "1,1" "1,1" $req_num llama-7b poisson 4 "$gputype-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len

#     ./run_base.sh "1,1" "2" $req_num llama-13b poisson 2 "$prefix-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len
#     ./run_base.sh "1,1,1,1" "" $req_num llama-13b poisson 2 "$gputype-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len

# done

# prompt_len=128
# response_len=256
# for max_migration_concurrency in "${max_migration_concurrencys[@]}"; do
#     echo $max_migration_concurrency
#     ./run_base.sh "1,1" "2" $req_num llama-7b poisson 4 "$prefix-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len
#     ./run_base.sh "1,1,1,1" "" $req_num llama-7b poisson 4 "$gputype-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len
#     ./run_base.sh "1" "1,1,1" $req_num llama-7b poisson 4 "$gputype-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len
#     ./run_base.sh "1,1" "1,1" $req_num llama-7b poisson 4 "$gputype-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len

#     ./run_base.sh "1,1" "2" $req_num llama-13b poisson 2 "$prefix-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len
#     ./run_base.sh "1,1,1,1" "" $req_num llama-13b poisson 2 "$gputype-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len

# done

# max_migration_concurrencys=(2 1 4)   # 24 16 8 4 2 1 # 1 4 8 16
# for max_migration_concurrency in "${max_migration_concurrencys[@]}"; do
#     echo $max_migration_concurrency
#     ./run_base.sh "1,1,1" "" $req_num llama-13b poisson 1 "$prefix-concurrency-$max_migration_concurrency" $max_migration_concurrency
#     ./run_base.sh "1" "2" $req_num llama-13b poisson 1 "$prefix-concurrency-$max_migration_concurrency" $max_migration_concurrency

#     ./run_base.sh "1,1,1" "" $req_num llama-13b poisson 2 "$prefix-concurrency-$max_migration_concurrency" $max_migration_concurrency
#     ./run_base.sh "1" "2" $req_num llama-13b poisson 2 "$prefix-concurrency-$max_migration_concurrency" $max_migration_concurrency

# done

# prompt_len=128
# response_len=256
# for max_migration_concurrency in "${max_migration_concurrencys[@]}"; do
#     echo $max_migration_concurrency
#     ./run_base.sh "1,1,1" "" $req_num llama-13b poisson 1 "$prefix-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len
#     ./run_base.sh "1" "2" $req_num llama-13b poisson 1 "$prefix-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len

#     ./run_base.sh "1,1,1" "" $req_num llama-13b poisson 2 "$prefix-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len
#     ./run_base.sh "1" "2" $req_num llama-13b poisson 2 "$prefix-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len

# done

# prompt_len=256
# response_len=256
# for max_migration_concurrency in "${max_migration_concurrencys[@]}"; do
#     echo $max_migration_concurrency
#     ./run_base.sh "1,1,1" "" $req_num llama-13b poisson 1 "$prefix-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len
#     ./run_base.sh "1" "2" $req_num llama-13b poisson 1 "$prefix-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len

#     ./run_base.sh "1,1,1" "" $req_num llama-13b poisson 2 "$prefix-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len
#     ./run_base.sh "1" "2" $req_num llama-13b poisson 2 "$prefix-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len

# done

# prompt_len=512
# response_len=256
# for max_migration_concurrency in "${max_migration_concurrencys[@]}"; do
#     echo $max_migration_concurrency
#     ./run_base.sh "1,1,1" "" $req_num llama-13b poisson 1 "$prefix-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len
#     ./run_base.sh "1" "2" $req_num llama-13b poisson 1 "$prefix-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len

#     ./run_base.sh "1,1,1" "" $req_num llama-13b poisson 2 "$prefix-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len
#     ./run_base.sh "1" "2" $req_num llama-13b poisson 2 "$prefix-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len

# done

for max_migration_concurrency in "${max_migration_concurrencys[@]}"; do
    echo $max_migration_concurrency
    ./run_base.sh "1" "1,1,1" $req_num llama-13b poisson 3 "$prefix-concurrency-$max_migration_concurrency" $max_migration_concurrency
    ./run_base.sh "1,1" "1,1" $req_num llama-13b poisson 3 "$prefix-concurrency-$max_migration_concurrency" $max_migration_concurrency
    ./run_base.sh "1,1,1" "1" $req_num llama-13b poisson 3 "$prefix-concurrency-$max_migration_concurrency" $max_migration_concurrency
    ./run_base.sh "1" "1,1,1" $req_num llama-13b poisson 5 "$prefix-concurrency-$max_migration_concurrency" $max_migration_concurrency
    ./run_base.sh "1,1" "1,1" $req_num llama-13b poisson 5 "$prefix-concurrency-$max_migration_concurrency" $max_migration_concurrency
    ./run_base.sh "1,1,1" "1" $req_num llama-13b poisson 5 "$prefix-concurrency-$max_migration_concurrency" $max_migration_concurrency
    ./run_base.sh "1" "1,1,1" $req_num llama-13b poisson 6 "$prefix-concurrency-$max_migration_concurrency" $max_migration_concurrency
    ./run_base.sh "1,1" "1,1" $req_num llama-13b poisson 6 "$prefix-concurrency-$max_migration_concurrency" $max_migration_concurrency
    ./run_base.sh "1,1,1" "1" $req_num llama-13b poisson 6 "$prefix-concurrency-$max_migration_concurrency" $max_migration_concurrency
    ./run_base.sh "1" "1,1,1" $req_num llama-13b poisson 7 "$prefix-concurrency-$max_migration_concurrency" $max_migration_concurrency
    ./run_base.sh "1,1" "1,1" $req_num llama-13b poisson 7 "$prefix-concurrency-$max_migration_concurrency" $max_migration_concurrency
    ./run_base.sh "1,1,1" "1" $req_num llama-13b poisson 7 "$prefix-concurrency-$max_migration_concurrency" $max_migration_concurrency
    
done

# for max_migration_concurrency in "${max_migration_concurrencys[@]}"; do
#     echo $max_migration_concurrency
#     ./run_base.sh "4,4" "" $req_num llama-7b poisson 6 "$prefix-concurrency-$max_migration_concurrency" $max_migration_concurrency
#     ./run_base.sh "4" "4" $req_num llama-7b poisson 6 "$prefix-concurrency-$max_migration_concurrency" $max_migration_concurrency

#     ./run_base.sh "4,4" "" $req_num llama-7b poisson 8 "$prefix-concurrency-$max_migration_concurrency" $max_migration_concurrency
#     ./run_base.sh "4" "4" $req_num llama-7b poisson 8 "$prefix-concurrency-$max_migration_concurrency" $max_migration_concurrency

#     ./run_base.sh "4,4" "" $req_num llama-7b poisson 10 "$prefix-concurrency-$max_migration_concurrency" $max_migration_concurrency
#     ./run_base.sh "4" "4" $req_num llama-7b poisson 10 "$prefix-concurrency-$max_migration_concurrency" $max_migration_concurrency

# done

# for max_migration_concurrency in "${max_migration_concurrencys[@]}"; do
#     echo $max_migration_concurrency
#     ./run_base.sh "4,4" "" $req_num llama-13b poisson 6 "$prefix-concurrency-$max_migration_concurrency" $max_migration_concurrency
#     ./run_base.sh "4" "4" $req_num llama-13b poisson 6 "$prefix-concurrency-$max_migration_concurrency" $max_migration_concurrency

#     ./run_base.sh "4,4" "" $req_num llama-13b poisson 8 "$prefix-concurrency-$max_migration_concurrency" $max_migration_concurrency
#     ./run_base.sh "4" "4" $req_num llama-13b poisson 8 "$prefix-concurrency-$max_migration_concurrency" $max_migration_concurrency

#     ./run_base.sh "4,4" "" $req_num llama-13b poisson 10 "$prefix-concurrency-$max_migration_concurrency" $max_migration_concurrency
#     ./run_base.sh "4" "4" $req_num llama-13b poisson 10 "$prefix-concurrency-$max_migration_concurrency" $max_migration_concurrency

# done

# for max_migration_concurrency in "${max_migration_concurrencys[@]}"; do
#     echo $max_migration_concurrency
#     ./run_base.sh "8" "" $req_num llama-7b poisson 6 "$prefix-concurrency-$max_migration_concurrency" $max_migration_concurrency

#     ./run_base.sh "8" "" $req_num llama-7b poisson 8 "$prefix-concurrency-$max_migration_concurrency" $max_migration_concurrency

#     ./run_base.sh "8" "" $req_num llama-7b poisson 10 "$prefix-concurrency-$max_migration_concurrency" $max_migration_concurrency

# done


# for max_migration_concurrency in "${max_migration_concurrencys[@]}"; do
#     echo $max_migration_concurrency
#     ./run_base.sh "8" "" $req_num llama-13b poisson 6 "$prefix-concurrency-$max_migration_concurrency" $max_migration_concurrency

#     ./run_base.sh "8" "" $req_num llama-13b poisson 8 "$prefix-concurrency-$max_migration_concurrency" $max_migration_concurrency

#     ./run_base.sh "8" "" $req_num llama-13b poisson 10 "$prefix-concurrency-$max_migration_concurrency" $max_migration_concurrency

# done

# prompt_len=128
# response_len=256
# for max_migration_concurrency in "${max_migration_concurrencys[@]}"; do
#     echo $max_migration_concurrency
#     ./run_base.sh "1,1,1" "" $req_num llama-7b poisson 4 "$prefix-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len
#     ./run_base.sh "1" "2" $req_num llama-7b poisson 4 "$prefix-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len

# done

# prompt_len=256
# response_len=256
# for max_migration_concurrency in "${max_migration_concurrencys[@]}"; do
#     echo $max_migration_concurrency
#     ./run_base.sh "1,1,1" "" $req_num llama-7b poisson 4 "$prefix-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len
#     ./run_base.sh "1" "2" $req_num llama-7b poisson 4 "$prefix-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len

# done

# prompt_len=512
# response_len=256
# for max_migration_concurrency in "${max_migration_concurrencys[@]}"; do
#     echo $max_migration_concurrency
#     ./run_base.sh "1,1,1" "" $req_num llama-7b poisson 4 "$prefix-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len
#     ./run_base.sh "1" "2" $req_num llama-7b poisson 4 "$prefix-concurrency-$max_migration_concurrency-$prompt_len-$response_len" $max_migration_concurrency $prompt_len $response_len

# done