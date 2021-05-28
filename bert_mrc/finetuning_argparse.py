import argparse


def get_argparse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="data/ccf_mrc_format", help="data dir")
    parser.add_argument("--bert_config_dir", type=str, default="prev_trained_model/chinese_roberta_wwm_ext__base_pytorch", help="bert config dir")
    parser.add_argument("--pretrained_checkpoint", default="", type=str, help="pretrained best_f1_checkpoint path")
    parser.add_argument("--max_length", type=int, default=128, help="max length of dataset")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--workers", type=int, default=0, help="num workers for dataloader")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="warmup steps used for scheduler.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")

    parser.add_argument("--gpus", default="0", type=str,
                        help="gpus")
    parser.add_argument("--accumulate_grad_batches", default=1, type=float,
                        help="accumulate_grad_batches")
    parser.add_argument("--max_epochs", default=10, type=float,
                        help="Emax_epochs")
    parser.add_argument("--seed", default=1029, type=float,
                        help="seed"),
    parser.add_argument("--log_step", default=500, type=float,
                        help="log_step")
    parser.add_argument("--output_dir", default='msra_output', type=str,
                        help="output_dir")

    # Required parameters
    parser.add_argument("--mrc_dropout", type=float, default=0.1,
                        help="mrc dropout rate")
    parser.add_argument("--bert_dropout", type=float, default=0.1,
                        help="bert dropout rate")
    parser.add_argument("--weight_start", type=float, default=1.0)
    parser.add_argument("--weight_end", type=float, default=1.0)
    parser.add_argument("--weight_span", type=float, default=1.0)
    parser.add_argument("--flat", action="store_true", help="is flat ner")
    parser.add_argument("--span_loss_candidates", choices=["all", "pred_and_gold", "gold"],
                        default="all", help="Candidates used to compute span loss")
    parser.add_argument("--chinese", action="store_true",
                        help="is chinese dataset")
    parser.add_argument("--loss_type", choices=["bce", "dice"], default="bce",
                        help="loss type")
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw",
                        help="loss type")
    parser.add_argument("--dice_smooth", type=float, default=1e-8,
                        help="smooth value of dice loss")
    parser.add_argument("--final_div_factor", type=float, default=1e4,
                        help="final div factor of linear decay scheduler")

    return parser
