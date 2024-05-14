import replicate

prompt = """SEPARATE YOUR ANSWER INTO STEPS LIKE <STEP 1>, <STEP 2> etc. Question: Tina makes $18.00 an hour. If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1/2 your hourly wage. If she works 10 hours every day for 5 days, how much money does she make? 
Answer: """
print(prompt)
input = {
    "top_k": 50,
    "top_p": 0.9,
    "prompt": prompt, 
    "temperature": 0.7,
    "max_new_tokens": 512,
    "prompt_template": "<s>[INST] {prompt} [/INST] "
}


cached_outputs = []
for i in range(10):
    # for event in replicate.stream(
    #     "mistralai/mistral-7b-instruct-v0.2",
    #     input=input
    # ): 
    #     cached_outputs.append(event)
    # print(i)
    output = replicate.run(
  "meta/meta-llama-3-8b-instruct",
  #"mistralai/mistral-7b-instruct-v0.1:5fe0a3d7ac2852264a25279d1dfb798acbc4d49711d126646594e212cb821749",
  input=input, 
)
    cached_outputs.append("".join(output))
    print(i, "".join(output))


# save cached outputs to text file withj current date
from datetime import datetime
now = datetime.now()
current_time = now.strftime("%Y-%m-%d_%H-%M-%S")
current_time = "mistral_output at time " + current_time
with open(f"outputs_{current_time}.txt", "w") as f:
    for output in cached_outputs:
        f.write(f"{output}\n")

