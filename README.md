# vLLM Neuron Test

I created a uv venv to get the awscli:

```sh
uv venv
source .venv/bin/activate
uv pip install awscli
```

Then you can proceed to pull the image:

```sh
export IMAGE=421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-28ce3c3:latest
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 421672808698.dkr.ecr.us-east-1.amazonaws.com
docker pull "$IMAGE"
```

Make sure `HF_TOKEN` is set.
Then you can start the server:

```bash
export HF_CACHE=`realpath ~/.cache/huggingface/`
docker run -d --privileged \
--name vllm_neuron \
-v $HF_CACHE:/root/.cache/huggingface \
-v /tmp/vllm_neff_cache:/root/.cache/vllm \
-v /tmp/vllm_cache:/tmp/vllm_cache \
-v /tmp/nki_cache:/tmp/nki_cache \
--env "HF_TOKEN=$HF_TOKEN" \
--env "NEURON_SKIP_EFA_AFFINITY=1" \
-p 8000:8000 \
--ipc=host \
--device /dev/neuron0 --device /dev/neuron1 --device /dev/neuron2 --device /dev/neuron3 \
"$IMAGE" \
sleep infinity
```

We can get into the server:

```bash
docker exec -it vllm_neuron /bin/bash
```

Then, from inside the container:

```bash
vllm serve meta-llama/Llama-3.2-1B-Instruct \
    --max-model-len 10240 \
    --max-num-batched-tokens 2048 \
    --max-num-seqs 4 \
    --tensor-parallel-size 4 \
    --additional-config '{ \
        "neuron_config": { \
        "num_batched_tokens_buckets": [2048], \
        "num_seqs_buckets": [4] \
    } \
}'
```

this gives this error:

```
vllm serve: error: argument --additional-config: Value { \
        "neuron_config": { \
        "num_batched_tokens_buckets": [2048], \
        "num_seqs_buckets": [4] \
    } \
} cannot be converted to <function loads at 0x7f848b64f2e0>.
```

doing the same without the additional-config, and it seems to work. You can then use curl to call the API:

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "messages": [
        {
            "role": "user",
            "content": "Hello"
        }
    ]
    }'
```