# Description: Script to run rollout.py
# export CUDA_VISIBLE_DEVICES=0,1
# for model in /home/test/test05/cyl/models/DeepSeek-R1-Distill-Llama-8B 
# do
# python -u rollout.py --data_name gsm8k@train,gsm8k@test,aime_83_24,aime_25  --model_path $model --temperature 0.6 --append_str "<think>" 
# done


# export CUDA_VISIBLE_DEVICES=2,3

# for model in /home/test/test05/cyl/models/DeepSeek-R1-Distill-Qwen-7B 
# do
# python -u rollout.py --data_name math@train,math_500,gsm8k@train,gsm8k@test,aime_83_24,aime_25 --model_path $model --temperature 0.6 --append_str "<think>"
# done

# /home/test/test05/cyl/models/DeepSeek-R1-Distill-Qwen-32B

# export CUDA_VISIBLE_DEVICES=0,1,2,3

# for model in /home/test/test05/cyl/models/DeepSeek-R1-Distill-Qwen-32B 
# do
# python -u rollout.py --data_name math@train,math_500,gsm8k@train,gsm8k@test,aime_83_24 --model_path $model --temperature 0.6 
# done

# export CUDA_VISIBLE_DEVICES=4,5

# for model in /home/test/test05/cyl/models/DeepSeek-R1-Distill-Qwen-7B 
# do
# python -u rollout.py --data_name math@train,math_500,gsm8k@train,gsm8k@test,aime_83_24,aime_25 --model_path $model --temperature 0.6 
# done


# export CUDA_VISIBLE_DEVICES=0,1,2,3

for model in /home/test/test05/cyl/models/DeepSeek-R1-Distill-Llama-8B 
do
python -u rollout.py --data_name aime_83_24,aime_25 --model_path $model --temperature 0.6 
done