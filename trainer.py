import logging
import torch
import numpy as np

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(
        self, 
        args = None,
        model = None,
        train_loader = None, 
        eval_loader = None, 
        tokenizer = None,
        optimizer = None,
        scheduler= None,
        device = None,
        metrics = None
        ):
        
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.metrics = metrics


    def train(self):
        total_step = len(self.train_loader) * self.args.train_epochs
        global_step = 0
        eval_step = self.args.eval_step
        best_dev_micro_f1 = 0.0
        self.model.to(self.device)
        self.model.train()
        for epoch in range(self.args.train_epochs):
            for train_step, train_data in enumerate(self.train_loader):
                self.model.train()
                token_ids = train_data['token_ids'].to(self.device)
                attention_masks = train_data['attention_masks'].to(self.device)
                token_type_ids = train_data['token_type_ids'].to(self.device)
                labels = train_data['labels'].to(self.device)
                train_outputs = self.model(token_ids, attention_mask=attention_masks, labels=labels)
                loss = train_outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                
                if (train_step + 1) % self.args.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                    
                logger.info(
                    "【train】 epoch：{} step:{}/{} loss：{:.6f}".format(epoch, global_step, total_step, loss.item()))
                
                if global_step % eval_step == 0:
                    dev_loss, dev_outputs, dev_targets = self.dev()
                    accuracy, micro_f1, macro_f1 = self.metrics(dev_outputs, dev_targets)
                    logger.info(
                        "【dev】 loss：{:.6f} accuracy：{:.4f} micro_f1：{:.4f} macro_f1：{:.4f}".format(dev_loss, accuracy,
                                                                                                   micro_f1, macro_f1))
                    if macro_f1 > best_dev_micro_f1:
                        logger.info("------------>保存当前最好的模型")
                        # 保存模型
                        self.model.save_pretrained(self.args.output_dir)
                        

    def dev(self):
        self.model.eval()
        total_loss = 0.0
        dev_outputs = []
        dev_targets = []
        with torch.no_grad():
            for dev_step, dev_data in enumerate(self.eval_loader):
                token_ids = dev_data['token_ids'].to(self.device)
                attention_masks = dev_data['attention_masks'].to(self.device)
                token_type_ids = dev_data['token_type_ids'].to(self.device)
                labels = dev_data['labels'].to(self.device)
                outputs = self.model(token_ids, attention_mask=attention_masks, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                outputs = np.argmax(outputs.logits.cpu().detach().numpy(), axis=1).flatten()
                dev_outputs.extend(outputs.tolist())
                dev_targets.extend(labels.cpu().detach().numpy().tolist())

        return total_loss, dev_outputs, dev_targets
    
    def test(self):
        self.model.eval()
        self.model.to(self.device)
        total_loss = 0.0
        test_outputs = []
        test_targets = []
        with torch.no_grad():
            for test_step, test_data in enumerate(self.eval_loader):
                token_ids = test_data['token_ids'].to(self.device)
                attention_masks = test_data['attention_masks'].to(self.device)
                token_type_ids = test_data['token_type_ids'].to(self.device)
                labels = test_data['labels'].to(self.device)
                outputs = self.model(token_ids, attention_mask=attention_masks, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                outputs = np.argmax(outputs.logits.cpu().detach().numpy(), axis=1).flatten()
                test_outputs.extend(outputs.tolist())
                test_targets.extend(labels.cpu().detach().numpy().tolist())

        return total_loss, test_outputs, test_targets


