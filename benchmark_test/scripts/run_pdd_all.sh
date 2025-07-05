#!/bin/bash
req_num=2000
gputype=A6000-2-multi-port-zmp-parallel-main
prefix="$gputype-pdd-hetero"
./run_pdd_multi.sh 1 $req_num 4 llama-7b poisson 6 $gputype
# ./tp_heterogeneity.sh "1,1" "2" $req_num llama-7b poisson 6 $prefix
exit 0
# ./run_pdd_multi.sh 1 $req_num 4 llama-7b poisson 2 $gputype
# ./run_pdd_multi.sh 1 $req_num 4 llama-7b poisson 4 $gputype
# ./run_pdd_multi.sh 1 $req_num 4 llama-7b poisson 6 $gputype
# ./run_pdd_multi.sh 1 $req_num 4 llama-7b poisson 8 $gputype
# ./run_pdd_multi.sh 1 $req_num 4 llama-7b poisson 10 $gputype
# ./run_pdd_multi.sh 1 $req_num 4 llama-7b poisson 12 $gputype

# ./tp_heterogeneity.sh "1,1" "2" $req_num llama-7b poisson 2 $prefix
# ./tp_heterogeneity.sh "1,1" "2" $req_num llama-7b poisson 4 $prefix
# ./tp_heterogeneity.sh "1,1" "2" $req_num llama-7b poisson 6 $prefix
# ./tp_heterogeneity.sh "1,1" "2" $req_num llama-7b poisson 8 $prefix
# ./tp_heterogeneity.sh "1,1" "2" $req_num llama-7b poisson 10 $prefix
# ./tp_heterogeneity.sh "1,1" "2" $req_num llama-7b poisson 12 $prefix

./run_pdd_multi.sh 1 $req_num 4 llama-13b poisson 1 $gputype
./tp_heterogeneity.sh "1,1" "2" $req_num llama-13b poisson 1 $prefix
./run_pdd_multi.sh 1 $req_num 4 llama-13b poisson 8 $gputype
./run_pdd_multi.sh 1 $req_num 4 llama-13b poisson 2 $gputype
./run_pdd_multi.sh 1 $req_num 4 llama-13b poisson 4 $gputype
./run_pdd_multi.sh 1 $req_num 4 llama-13b poisson 6 $gputype
./run_pdd_multi.sh 1 $req_num 4 llama-13b poisson 10 $gputype
./run_pdd_multi.sh 1 $req_num 4 llama-13b poisson 12 $gputype
./tp_heterogeneity.sh "1,1" "2" $req_num llama-13b poisson 8 $prefix
./tp_heterogeneity.sh "1,1" "2" $req_num llama-13b poisson 2 $prefix
./tp_heterogeneity.sh "1,1" "2" $req_num llama-13b poisson 4 $prefix
./tp_heterogeneity.sh "1,1" "2" $req_num llama-13b poisson 6 $prefix
./tp_heterogeneity.sh "1,1" "2" $req_num llama-13b poisson 10 $prefix
./tp_heterogeneity.sh "1,1" "2" $req_num llama-13b poisson 12 $prefix


# ./run_pdd_gen_request.sh 1 1000 2 llama-2-7b poisson 4
# ./run_pdd_gen_request.sh 1 1000 2 llama-2-7b poisson 8
# ./run_pdd_gen_request.sh 1 1000 2 llama-2-7b poisson 1
# ./run_pdd_gen_request.sh 1 1000 2 llama-2-7b poisson 2

# ./run_pdd_gen_request.sh 1 1000 2 llama-2-7b uniform 1
# ./run_pdd_gen_request.sh 1 1000 2 llama-2-7b uniform 2
# ./run_pdd_gen_request.sh 1 1000 2 llama-2-7b uniform 4
# ./run_pdd_gen_request.sh 1 1000 2 llama-2-7b uniform 8

# ./run_pdd_gen_request.sh 1 1000 2 llama-2-7b burst 1
# ./run_pdd_gen_request.sh 1 1000 2 llama-2-7b burst 2
# ./run_pdd_gen_request.sh 1 1000 2 llama-2-7b burst 4
# ./run_pdd_gen_request.sh 1 1000 2 llama-2-7b burst 8

# ./run_pdd_gen_request.sh 1 1000 2 llama-2-7b gamma 1
# ./run_pdd_gen_request.sh 1 1000 2 llama-2-7b gamma 2
# ./run_pdd_gen_request.sh 1 1000 2 llama-2-7b gamma 4
# ./run_pdd_gen_request.sh 1 1000 2 llama-2-7b gamma 8

# ./run_pdd_gen_request.sh 1 1000 2 llama-2-13b poisson 1
# ./run_pdd_gen_request.sh 1 1000 2 llama-2-13b poisson 2
# ./run_pdd_gen_request.sh 1 1000 2 llama-2-13b poisson 4
# ./run_pdd_gen_request.sh 1 1000 2 llama-2-13b poisson 8

# ./run_pdd_gen_request.sh 1 1000 2 llama-2-13b uniform 1
# ./run_pdd_gen_request.sh 1 1000 2 llama-2-13b uniform 2
# ./run_pdd_gen_request.sh 1 1000 2 llama-2-13b uniform 4
# ./run_pdd_gen_request.sh 1 1000 2 llama-2-13b uniform 8

# ./run_pdd_gen_request.sh 1 1000 2 llama-2-13b burst 1
# ./run_pdd_gen_request.sh 1 1000 2 llama-2-13b burst 2
# ./run_pdd_gen_request.sh 1 1000 2 llama-2-13b burst 4
# ./run_pdd_gen_request.sh 1 1000 2 llama-2-13b burst 8

# ./run_pdd_gen_request.sh 1 1000 2 llama-2-13b gamma 1
# ./run_pdd_gen_request.sh 1 1000 2 llama-2-13b gamma 2
# ./run_pdd_gen_request.sh 1 1000 2 llama-2-13b gamma 4
# ./run_pdd_gen_request.sh 1 1000 2 llama-2-13b gamma 8