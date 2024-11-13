from vllm import LLM, SamplingParams

prompts = [
    "A robot may not injure a human being or, through inaction, allow a human being to come to harm.\n\nThe above is the first of",
    "It is only with the heart that one can see rightly; what is essential is invisible to the eye.\n\n— Antoine de Saint-Exupéry\n",
    "The greatest glory in living lies not in never falling, but in rising every time we fall.\n\n-Nelson Mandela\n\nThe 2019-",
]
N = 1
# Currently, top-p sampling is disabled. `top_p` should be 1.0.
sampling_params = SamplingParams(temperature=0.0,
                                 top_p=1.0,
                                 n=N,
                                 max_tokens=16)

# Set `enforce_eager=True` to avoid ahead-of-time compilation.
# In real workloads, `enforace_eager` should be `False`.
llm = LLM(model="google/gemma-2b",
          enforce_eager=True,
          enable_prefix_caching=True)
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
