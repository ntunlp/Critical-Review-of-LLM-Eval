from transformers import AutoTokenizer
import os
from datasets import load_dataset

def apply_tokenizer(tokenizer, sentence):
    output = tokenizer(sentence, return_tensors="pt").input_ids
    tokenized_output = tokenizer.convert_ids_to_tokens(output.tolist()[0])
    return tokenized_output

def main():
    access_token = ""
    model_names = [
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Meta-Llama-3-8B",
        "mistralai/Mistral-7B-Instruct-v0.1",
        "01-ai/Yi-1.5-34B",
        "google/gemma-7b",
        "microsoft/phi-2",
        "Qwen/Qwen1.5-7B-Chat"
    ]

    dataset_names = [
        "mmlu",
        "mixeval",
        "mixeval-hard",
        "mmlu-pro"
    ]

    token_percentage = {}
    for model_name in model_names:

        tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
        vocab = tokenizer.vocab
        subset_vocab = {}
        for dataset_name in dataset_names:
            subset_vocab = {}
            if dataset_name in ["mmlu"]:
                path = "mmlu/test"
                for domain in os.listdir(path):
                    domain_file = open(os.path.join(path, domain)).readlines()
                    for example in domain_file:
                        example = example.strip()

                        tokenized_example = apply_tokenizer(tokenizer, example)

                        for token in tokenized_example:
                            if token in subset_vocab:
                                subset_vocab[token] += 1
                            else:
                                subset_vocab[token] = 1
            elif dataset_name in ["mixeval"]:
                dataset = load_dataset("MixEval/MixEval", 'MixEval')
                for sentence in dataset["free_form"]:
                    tokenized_example = apply_tokenizer(tokenizer, sentence["prompt"])

                    for token in tokenized_example:
                        if token in subset_vocab:
                            subset_vocab[token] += 1
                        else:
                            subset_vocab[token] = 1

                for sentence in dataset["multiple_choice"]:
                    tokenized_example = apply_tokenizer(tokenizer, sentence["prompt"])

                    for token in tokenized_example:
                        if token in subset_vocab:
                            subset_vocab[token] += 1
                        else:
                            subset_vocab[token] = 1

            elif dataset_name in ["mixeval-hard"]:
                dataset = load_dataset("MixEval/MixEval", 'MixEval_Hard')
                for sentence in dataset["free_form"]:
                    tokenized_example = apply_tokenizer(tokenizer, sentence["prompt"])

                    for token in tokenized_example:
                        if token in subset_vocab:
                            subset_vocab[token] += 1
                        else:
                            subset_vocab[token] = 1

                for sentence in dataset["multiple_choice"]:
                    tokenized_example = apply_tokenizer(tokenizer, sentence["prompt"])

                    for token in tokenized_example:
                        if token in subset_vocab:
                            subset_vocab[token] += 1
                        else:
                            subset_vocab[token] = 1

            elif dataset_name in ["mmlu-pro"]:
                dataset = load_dataset("TIGER-Lab/MMLU-Pro")
                for sentence in dataset["test"]:
                    tokenized_example = apply_tokenizer(tokenizer, sentence["question"])

                    for token in tokenized_example:
                        if token in subset_vocab:
                            subset_vocab[token] += 1
                        else:
                            subset_vocab[token] = 1

            found_count, all_count = 0, 0
            for key_word_original in vocab.keys():
                if key_word_original in subset_vocab:
                    found_count += 1

                all_count += 1

            token_percentage.setdefault(model_name, {}).setdefault(dataset_name, found_count / all_count)
        print(model_name + "\t" + str(len(vocab)))

if __name__ == "__main__":
    main()