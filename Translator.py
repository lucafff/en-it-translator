import os
os.environ["WANDB_DISABLED"]="true"
import transformers
print(transformers.__version__)

#modello pretrainato
model_checkpoint = "Helsinki-NLP/opus-mt-en-it"

from datasets import load_dataset,load_metric,list_metrics

#dataset = load_dataset("opus100", "en-it")
dataset = load_dataset("iwslt2017", "iwslt2017-it-en")
metric = load_metric("sacrebleu")
#metric2 = load_metric("accuracy")


dataset


import datasets
import random
import pandas as pd

print("\n")
print("Visualization of three examples of translation\n")

for i in range(3):
    print(dataset["train"][random.randint(0,10000)])
    print("\n")

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)


tokenizer(["Hello, this one sentence!", "This is another sentence."])

prefix = ""
max_input_length = 128
max_target_length = 128
source_lang = "en"
target_lang = "it"
def preprocess_function(examples):
    inputs = [prefix + ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation = True)
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length,truncation = True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


preprocess_function(dataset['train'][:2])

tokenized_datasets = dataset.map(preprocess_function, batched=True)

from transformers import Seq2SeqTrainingArguments,DataCollatorForSeq2Seq
batch_size = 16
model_name = model_checkpoint.split("/")[-1]
args = Seq2SeqTrainingArguments(
    output_dir = f"{model_name}-finetuned-{source_lang}-to-{target_lang}",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True    
)


data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)



import numpy as np
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result



from transformers import Seq2SeqTrainer
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)




trainer.train()



import os
for dirname, _, filenames in os.walk('opus-mt-en-it-finetuned-en-to-it'):
    for filename in filenames:
        print(os.path.join(dirname, filename))




trainer.save_model()



from transformers import MarianMTModel, MarianTokenizer
choice = input("\nWrite your phrase you want the model to convert in italian:\n")
src_text = [choice]

model_name = 'opus-mt-en-it-finetuned-en-to-it/checkpoint-14000'
tokenizer = MarianTokenizer.from_pretrained(model_name)
print(tokenizer.supported_language_codes)


model = MarianMTModel.from_pretrained(model_name)
translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True))
print("Traduzione: ")
[tokenizer.decode(t, skip_special_tokens=True) for t in translated]

