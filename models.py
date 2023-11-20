from transformers import BertForSequenceClassification


class LawClassification(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        
    
    # @staticmethod
    # def cn_text_decode(
    #     tokenizer, 
    #     text, 
    #     label= None, 
    #     maxlen=512,
    #     tag2id=None):
        
    #     output = tokenizer.encode_plus(
    #             text=text,
    #             max_length=maxlen,
    #             padding="max_length",
    #             truncation=True
    #         )
        
    #     token_ids = output["input_ids"]
    #     token_type_ids = output["token_type_ids"]
    #     attention_mask = output["attention_mask"]
    #     if label is not None:
    #         label = tag2id[label]
            
    #     return token_ids, token_type_ids, attention_mask, label
    
    # def predict(self, sentence):
    #     self.cn_text_decode()
    #     pass

