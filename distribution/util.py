import re
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

def get_fewshot_examples(num_examples):
    examples = [
        "Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nA: We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted. So, they must have planted 21 - 15 = 6 trees. The answer is 6",
        "Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\nA: There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. The answer is 5.",
        "Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nA: Leah had 32 chocolates and Leah's sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39.",
        "Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\nA: Jason had 20 lollipops. Since he only has 12 now, he must have given the rest to Denny. The number of lollipops he has given to Denny must have been 20 - 12 = 8 lollipops. The answer is 8.",
        "Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\nA: He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is 9.",
        "Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\nA: There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers. The answer is 29.",
        "Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\nA: Michael initially had 58 balls. He lost 23 on Tuesday, so after that he has 58 - 23 = 35 balls. On Wednesday he lost 2 more so now he has 35 - 2 = 33 balls. The answer is 33.",
        "Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\nA: She bought 5 bagels for $3 each. This means she spent 5 * $3 = $15 on the bagels. She had $23 in the beginning, so now she has $23 - $15 = $8. The answer is 8."
    ]
    return examples[:num_examples]

def get_prompt_message(question, num_fewshot_examples=1):
    # num_fewshot_examples=-1 for direct prompting, 0 for zero-shot CoT
    examples = get_fewshot_examples(num_fewshot_examples)
    prompt = ""
    
    if num_fewshot_examples > 0: 
        for i in range(min(num_fewshot_examples, len(examples))):
            prompt += examples[i] + "\n"
    
    prompt += "Q: " + question + "\nA:"

    if num_fewshot_examples == 0:
        prompt += " Let's think step by step." 
    
    return [{"role": "user", "content": prompt}]

def extract_last_sentence(text):
    # Split the text into sentences based on both "." and "\n"
    # sentences = re.split(r'[.\n]+', text)
    # sentences = [sentence.strip() for sentence in sentences if sentence] # Filter out empty strings and strip whitespace
    # return sentences[-1] if sentences else ""
    sentences = sent_tokenize(text)
    return sentences[-1] if sentences else ""

def extract_last_integer(ak):
    numbers = re.findall(r'\d+', ak)
    return int(numbers[-1]) if numbers else None

def extract_last_number(text):
    # Regex to find numbers, including floats
    numbers = re.findall(r'\b\d+\.?\d*\b', text)
    # Convert the extracted strings to floats or integers
    numbers = [float(num) if '.' in num else int(num) for num in numbers]
    return numbers[-1] if numbers else None

def extract_unique_answers_as_integers(rk_ak_pairs):
    unique_answers = dict()

    for _, ak in rk_ak_pairs:
        # Extract the last integer from the answer
        last_integer = extract_last_integer(ak)
        print(last_integer, " : ", ak)
        assert last_integer is not None, "no ints in a_k"
        unique_answers[last_integer] = 1 + unique_answers.get(last_integer, 0)

    return unique_answers

def remove_last_sentence(text):
    pattern = re.compile(r'(.*)(?=[.?!]\s+|\n+)[^.?!]*[.?!]\s*|\n*', re.DOTALL)
    matches = pattern.findall(text)
    text_without_last_sentence = matches[0] if matches else text

    return text_without_last_sentence