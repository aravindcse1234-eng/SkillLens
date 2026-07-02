import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback,
    DataCollatorForTokenClassification
)
from datasets import Dataset
from src.utils.helpers import get_device, save_json
from src.utils.metrics import compute_ner_metrics
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class NERTrainer:
    def __init__(self, model_name: str = "bert-base-uncased", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or str(get_device())
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=3,
            id2label={0: "O", 1: "B-SKILL", 2: "I-SKILL"},
            label2id={"O": 0, "B-SKILL": 1, "I-SKILL": 2}
        )
        self.model.to(self.device)
        self.trainer = None
        self.training_logs = []

    def prepare_data(self, texts: List[str], tags: List[List[str]],
                     val_split: float = 0.1, test_split: float = 0.1):
        label_list = ["O", "B-SKILL", "I-SKILL"]
        label_to_id = {l: i for i, l in enumerate(label_list)}

        encodings = self.tokenizer(
            texts, truncation=True, padding=True,
            is_split_into_words=False, return_tensors="pt"
        )
        labels = []
        for i, tag_seq in enumerate(tags):
            word_ids = encodings.word_ids(batch_index=i)
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                else:
                    tag = tag_seq[word_idx] if word_idx < len(tag_seq) else "O"
                    label_ids.append(label_to_id.get(tag, 0))
            labels.append(label_ids)

        encodings["labels"] = torch.tensor(labels)
        dataset = Dataset.from_dict({
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": encodings["labels"]
        })

        dataset = dataset.train_test_split(test_size=val_split + test_split)
        if test_split > 0:
            test_size = test_split / (val_split + test_split)
            val_test = dataset["test"].train_test_split(test_size=test_size)
            return dataset["train"], val_test["train"], val_test["test"]
        return dataset["train"], dataset["test"], None

    def train(self, train_dataset, eval_dataset=None, epochs: int = 5,
              batch_size: int = 16, lr: float = 2e-5, weight_decay: float = 0.01):
        logger.info(f"Training {self.model_name} for NER...")
        data_collator = DataCollatorForTokenClassification(self.tokenizer)

        args = TrainingArguments(
            output_dir=f"models/ner/{self.model_name.split('/')[-1]}",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            learning_rate=lr,
            weight_decay=weight_decay,
            logging_dir="logs/ner",
            logging_steps=50,
            evaluation_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            fp16=torch.cuda.is_available(),
            dataloader_pin_memory=False,
        )

        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)] if eval_dataset else []

        self.trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            callbacks=callbacks,
        )

        self.trainer.train()
        self.training_logs = self.trainer.state.log_history
        logger.info("NER training complete")
        return self.training_logs

    def evaluate(self, test_dataset) -> Dict:
        if self.trainer is None:
            return {}
        results = self.trainer.evaluate(test_dataset)
        logger.info(f"Evaluation results: {results}")
        return results

    def save(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        save_json(self.training_logs, f"{path}/training_logs.json")
        logger.info(f"NER model saved to {path}")

    def load(self, path: str):
        self.model = AutoModelForTokenClassification.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model.to(self.device)
        logger.info(f"NER model loaded from {path}")
        return self
