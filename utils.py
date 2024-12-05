import argparse
from termcolor import colored

def parse_args():
    parser = argparse.ArgumentParser(description="fmri tokenizer")

    parser.add_argument('-m', "--model", type=str, help="rnn, lstm, gru, markov, mamba", default='rnn')
    parser.add_argument('-l', "--layer", type=int, help="model layer", default=2)
    parser.add_argument('-H', "--hidden", type=int, help="model hidden size", default=32)
    parser.add_argument("--epochs", type=int, default=15000)

    parser.add_argument("--state_token", '-st', type=int, help="state token number", default=32)
    parser.add_argument("--trans_token", '-tt', type=int, help="trans token number", default=32)


    parser.add_argument("--resume", action='store_true', default=False)
    parser.add_argument("--resume_checkpoint", type=str, default='')

    parser.add_argument("--log_name", type=str,default='test')
    parser.add_argument("--check_name", type=str,default='test')

    args = parser.parse_args()

    return args

class log_util(object):
    def __init__(self, name, writer, is_train=True):
        self.writer = writer
        self.name = name
        self.count = 0
        self.avg = 0.
        self.total = 0.
        self.is_train = is_train

    def update(self, total_val, count):
        self.count += float(count)
        self.total += float(total_val) * float(count)

    def summary(self, epoch_counter):
        self.avg = self.total / self.count
    
        self.print(epoch_counter)
        self.log(epoch_counter)
        self.reset()

    def reset(self):
        self.total = 0.
        self.count = 0.
        self.avg = 0.

    def print(self, epoch_counter):
        if self.is_train is True:
            print(f"[{epoch_counter}]: {self.name}: {self.avg:.4f}")
        else:
            print(colored(f"[{epoch_counter}]: {self.name}: {self.avg:.4f}", "yellow", attrs=["bold"]))

    def log(self, epoch_counter):
        self.writer.add_scalar(f'{self.name}', self.avg, global_step=epoch_counter)