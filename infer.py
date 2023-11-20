import os
import config
import numpy as np
import torch
from transformers import BertTokenizer

from utils.nnUtils import manual_seed
from utils.logger import set_logger
from utils.decorator import timer

from models import LawClassification

import logging

logger = logging.getLogger(__name__)

class InferClassify:
    def __init__(self, config) -> None:
        self.args = config
        self.id2label, self.label2id = self.get_label()
        self.tokenizer = BertTokenizer.from_pretrained(self.args.bert_dir)
        self.model = LawClassification.from_pretrained(self.args.output_dir)
        self.device = torch.device("cpu" if self.args.gpu_ids == "-1" else "cuda:" + self.args.gpu_ids)
        
        manual_seed(self.args.seed)
        self.model.to(self.device)
        
    def get_label(self):
        label2id = {}
        id2label = {}

        with open(
            os.path.join(self.args.data_dir, "labels.txt"), "r", encoding="utf-8"
        ) as fp:
            labels = fp.read().strip().split("\n")
            logger.info("labels:{}".format(labels))

        for i, label in enumerate(labels):
            label2id[label] = i
            id2label[i] = label
        return id2label, label2id

    
    def cn_text_decode(self, sentence):
        inputs = self.tokenizer.encode_plus(
            text=sentence,
            add_special_tokens=True,
            max_length=self.args.max_seq_len,
            truncation="longest_first",
            padding="max_length",
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"]
    
    @timer
    def predict(self, sentence):
        self.model.eval()
        
        token_ids, attention_masks, _ = self.cn_text_decode(sentence)
        with torch.no_grad():
            outputs = self.model(token_ids.to(self.device), attention_mask=attention_masks.to(self.device)).logits
        outputs = np.argmax(outputs.cpu().detach().numpy(), axis=1).flatten().tolist()
        if len(outputs) != 0:
            outputs = [self.id2label[i] for i in outputs]
            return outputs
        else:
            return "不好意思，我没有识别出来"

if __name__ == "__main__":
    args = config.Args().get_parser()
    set_logger(os.path.join(args.log_dir, "main.log"))
    text = "公诉机关指控：2015年10月的一天中午，被告人张某到本区黄阁镇东里村东塘西街XX号二楼XX房，盗得被害人凌某放在床头的苹果牌Iphone4S（32G）手机1部（价值人民币688元）；2015年12月28日中午13时许，被告人张某到上址三楼XX房，盗得被害人陈某放在梳妆台抽屉里的现金人民币360元；2016年1月10日晚上20时许，被告人张某再次到上址一楼被害人罗某、李某的房间实施盗窃时，因被群众发现被当场抓获并扭送公安机关。公诉机关认为被告人张某有坦白的从轻处罚情节，建议判处其××至一年，并处罚金。"
    logger.info("推理文本：{}".format(text))
    model = InferClassify(args)
    outputs = model.predict(text)
    logger.info("推理结果：{}".format(outputs))
    print(outputs)
