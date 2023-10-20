from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from argparse import ArgumentParser
from tqdm.auto import tqdm
from dataset import EmotionDataLoader

import torch
import evaluate


def reform_tokenized_datasets(datasets):
    # Train, val, test
    for i in range(len(datasets)):
        datasets[i] = datasets[i].remove_columns(["text", "__index_level_0__"])
        datasets[i].set_format(type="torch")
    return datasets


def train(args, device, model, optimizer, scheduler, train_dataloader, progress_bar):
    model.train()
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)


@torch.no_grad()
def eval(metric, device, model, dataloader):
    model.eval()
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
        
    return metric.compute()


def main():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="distilbert-base-uncased")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_labels", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        print("Successfully loaded cuda")

    filenames = ["train", "dev", "test"]

    goem_files = [filename + ".tsv" for filename in filenames]
    edfer_files = [filename + ".csv" for filename in filenames]

    loader = EmotionDataLoader()
    goem_dfs = loader.load_GoEM(goem_files[0], goem_files[1], goem_files[2])
    edfer_dfs = loader.load_EDFER(edfer_files[0], edfer_files[1], edfer_files[2])

    id2label = loader.get_id2label()
    label2id = loader.get_label2id()

    train_dataset, val_dataset, test_dataset = loader.combine_dataframes(goem_dfs, edfer_dfs)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    datasets = [tokenized_train, tokenized_val, tokenized_test]
    tokenized_train, tokenized_val, tokenized_test = reform_tokenized_datasets(datasets)
                                                            
    train_dataloader = DataLoader(tokenized_train, batch_size=args.batch_size)
    eval_dataloader = DataLoader(tokenized_val, batch_size=args.batch_size)
    test_dataloader = DataLoader(tokenized_test, batch_size=args.batch_size)

    # Tokenizer & model
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint, 
                                                               num_labels=args.num_labels,
                                                               id2label=id2label,
                                                               label2id=label2id,
                                                               ).to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_training_steps = args.epochs * len(train_dataloader)
    progress_bar = tqdm(range(num_training_steps))

    lr_scheduler = get_scheduler(
        name="linear", 
        optimizer=optimizer, 
        num_warmup_steps=0, 
        num_training_steps=num_training_steps
    )

    metric = evaluate.load("accuracy")

    # Adds a whitespace after the transformers warning 
    print()

    # Training loop
    for epoch in range(args.epochs):
        train(args, device, model, optimizer, lr_scheduler, train_dataloader, progress_bar)

        # Evaluate
        train_result = eval(metric, device, model, train_dataloader)
        val_result = eval(metric, device, model, eval_dataloader)

        print()
        print(f"Epoch: {epoch}")
        print(f"Train data accuracy: {train_result['accuracy']}")
        print(f"Val. data accuracy: {val_result['accuracy']}", end="\n\n")
        path_to_save = f"saved_model\\epoch_{epoch}\\"
        model.save_pretrained(path_to_save, from_pt=True)

    # Final run on unseen test data
    test_result = eval(metric, device, model, test_dataloader)
    print("Done training")
    print(f"Test data accuracy: {test_result['accuracy']}", end="\n\n")
    

if __name__ == "__main__":
    main()