import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, set_seed
import os
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import string
import re
import json
import seaborn as sns
from sklearn.metrics import confusion_matrix
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--modelname", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
parser.add_argument("--datasetname", type=str, default="flux")
args = parser.parse_args()

seed = args.seed 
model_name = args.modelname
dataset_name =args.datasetname

# Setting seeds
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
set_seed(seed)

def get_label_json(data, id_num):
    for d in data:
        if d["id"] == id_num:
            return d["attractive"][2]
    print("Error file can't be found in classfier results")
    return -1

# Implementations from SQuAD and SQuAD2.0 metrics
# https://huggingface.co/spaces/evaluate-metric/squad_v2
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_line(text):
        return text.split('\n')[0]
    def remove_articles(text):
        return re.compile(r"\b(a|an|the)\b", re.UNICODE).sub(" ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(remove_line(s)))))

# Processing and loading data
data = []
data_path = f"../generated_images_{dataset_name}"

with open(f'../result_{dataset_name}.json', 'r') as f:
    json_data = json.load(f)
    
for d in os.listdir(data_path):
    index_id = int(d.split("_")[0].strip())
    label_str = d.split("_")[-1].strip().split(".")[0].strip()
    if label_str == "unattractive":
        label = 0
        data.append({"id": f'{index_id}.{label}', "path": f"{data_path}/{d}", "label": label, "c_label": get_label_json(json_data, d)})
    elif label_str == "normal":
        label = 1
        data.append({"id": f'{index_id}.{label}', "path": f"{data_path}/{d}", "label": label, "c_label": get_label_json(json_data, d)})
    elif label_str == "attractive":
        label = 2
        data.append({"id": f'{index_id}.{label}', "path": f"{data_path}/{d}", "label": label, "c_label": get_label_json(json_data, d)})
    else:
        label = -1
        print(f"Error! Unknown label at {d}")

print(f"Size of data: {len(data)}")

# Loading model
# https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained(model_name)

# Inference
results = {}

progress_bar = tqdm(range(len(data)), desc="Generating")

for d in data:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": d["path"],
                },
                {"type": "text", "text": "This is an image of a human. Measure how likely it is AI-generated on a scale between 1 to 10, with 1 being highly unlikely and 10 being highly likely."},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=5)
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    model_answer = normalize_answer(output_text[0]).strip()
    try:
        results[d["id"]] = (d["label"], d["c_label"], int(model_answer))
    except:
        print(model_answer)
        print(f"Error! Non-int answer was outputted for {d['id']}")
    
    progress_bar.update(1)

print(f"Number of generated results: {len(list(results.values()))}")

# Plotting Graphs

# Plotting Confusion Matrix
# https://stackoverflow.com/questions/77866916/how-can-i-achieve-a-scikit-learn-confusion-matrix-without-extra-columns
y_pred = []
y_test = []
for r in results.values():
    y_test.append(r[0])
    y_pred.append(r[2])

max_label = 10
unique_labels = list(range(max_label))
conf_matrix = np.zeros((len(unique_labels), 3), dtype=int)
for true_label, pred_label in zip(y_pred, y_test):
    conf_matrix[true_label, pred_label] += 1

sns.heatmap(conf_matrix.T, annot=True, fmt="d", cmap="Blues", yticklabels=["Unattractive", "Normal", "Attractive"])

plt.title(f"Qwen3-VL-8B, Scale 1 to {max_label}, {dataset_name}, Correlation Coefficient {round(np.corrcoef(y_pred, y_test)[0][1], 5)}")
plt.xlabel("AI-generated Score")
plt.ylabel("Classification")
plt.savefig("./plots/confusion_matrix.png")
plt.close()

# Plotting results vs SwinFace labels
plt.plot(y_pred, y_test, "o")

plt.title(f"Qwen3-VL-8B, Scale 1 to {max_label}, {dataset_name}, Correlation Coefficient {round(np.corrcoef(y_pred, y_test)[0][1], 5)}")
plt.xlabel("AI-generated Score")
plt.ylabel("Attractiveness Score Classifier")
plt.savefig("./plots/eval_using_swinface_labels.png")
plt.close()