import os
import logging

import torch
from torch.utils.data import DataLoader

from transformers import BertTokenizer
from transformers import (
    AdamW,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

import config
import models
from trainer import Trainer
from utils.nnUtils import manual_seed
from utils.logger import set_logger

from utils.metrics import get_metrics, get_classification_report
from data_loader import Collate, LAWSDataset


logger = logging.getLogger(__name__)


def main(args, tokenizer, device):
    label2id = {}
    id2label = {}

    with open(os.path.join(args.data_dir, "labels.txt"), "r", encoding="utf-8") as fp:
        labels = fp.read().strip().split("\n")

    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label

    args.id2label = id2label
    args.label2id = label2id

    collate = Collate(tokenizer=tokenizer, max_len=args.max_seq_len, tag2id=label2id)

    dataset = LAWSDataset(os.path.join(args.data_dir, "text_data.txt"))

    train_loader = DataLoader(
        dataset[:8000],
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate.collate_fn,
    )

    eval_loader = DataLoader(
        dataset[8000:],
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate.collate_fn,
    )

    model = models.LawClassification.from_pretrained(
        args.bert_dir, num_labels=len(labels)
    )

    num_train_optimization_steps = (
        int(
            len(train_loader) / args.train_batch_size / args.gradient_accumulation_steps
        )
        * args.train_epochs
    )
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    warmup_steps = int(args.warmup_proportion * num_train_optimization_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_train_optimization_steps,
    )

    metrics = get_metrics

    trainer = Trainer(
        args,
        model,
        train_loader,
        eval_loader,
        tokenizer,
        optimizer,
        scheduler,
        device,
        metrics,
    )

    if args.do_train:
        # 训练和验证
        trainer.train()

    # 测试
    if args.do_test:
        logger.info('========进行测试========')
        model1 = models.LawClassification.from_pretrained(args.output_dir)
        trainer.model = model1
        total_loss, test_outputs, test_targets = trainer.test()
        accuracy, micro_f1, macro_f1 = get_metrics(test_outputs, test_targets)
        logger.info(
            "【test】 loss：{:.6f} accuracy：{:.4f} micro_f1：{:.4f} macro_f1：{:.4f}".format(total_loss, accuracy, micro_f1,
                                                                                        macro_f1))
        report = get_classification_report(test_outputs, test_targets, labels)
        logger.info(report)


if __name__ == "__main__":
    args = config.Args().get_parser()
    print(args.seed)
    manual_seed(args.seed)
    set_logger(os.path.join(args.log_dir, "main.log"))
    tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
    device = torch.device("cpu" if args.gpu_ids == "-1" else "cuda:" + args.gpu_ids)
    main(args, tokenizer, device)
    
    
