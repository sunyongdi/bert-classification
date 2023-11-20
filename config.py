import argparse


class Args:
    @staticmethod
    def parse():
        parser = argparse.ArgumentParser()
        return parser

    @staticmethod
    def initialize(parser):
        # args for path
        parser.add_argument(
            "--model_name",
            default="bert",
            help="the model name",
        )

        parser.add_argument(
            "--output_dir",
            default="./checkpoints/chinese-bert-wwm-ext-classify",
            help="the output dir for model checkpoints",
        )

        parser.add_argument(
            "--bert_dir",
            default="/root/sunyd/pretrained_models/chinese-bert-wwm-ext/",
            help="bert dir for uer",
        )

        parser.add_argument(
            "--data_dir", default="./data/LAW/", help="data dir for uer"
        )

        parser.add_argument("--log_dir", default="./logs/", help="log dir for uer")

        # other args
        parser.add_argument("--num_tags", default=5, type=int, help="number of tags")

        parser.add_argument("--seed", type=int, default=64, help="random seed")

        parser.add_argument(
            "--gpu_ids",
            type=str,
            default="0",
            help='gpu ids to use, -1 for cpu, "0,1" for multi gpu',
        )

        parser.add_argument("--max_seq_len", default=512, type=int)

        parser.add_argument(
            "--train_epochs", default=1, type=int, help="Max training epoch"
        )

        parser.add_argument("--train_batch_size", default=8, type=int)

        parser.add_argument("--eval_batch_size", default=12, type=int)

        parser.add_argument("--eval_step", default=100, type=int, help="多少步验证")

        parser.add_argument(
            "--lr", default=3e-5, type=float, help="learning rate for the bert module"
        )

        parser.add_argument(
            "--do_train", default=True, action="store_true", help="是否训练"
        )

        parser.add_argument("--do_test", default=True, action="store_true", help="是否测试")

        parser.add_argument("--gradient_accumulation_steps", default=1, type=int)

        parser.add_argument("--weight_decay", default=0.01, type=float)

        parser.add_argument("--warmup_proportion", default=0.1, type=float)
        parser.add_argument("--adam_epsilon", default=1e-8, type=float)

        parser.add_argument("--max_grad_norm", default=100.0, type=float)
        return parser

    def get_parser(self):
        parser = self.parse()
        parser = self.initialize(parser)
        return parser.parse_args()
