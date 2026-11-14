# coding : utf-8
# Author : Anonymous
# 日志
import logging
import sys
import time
import numpy as np
import platform
import glob
import os
from tqdm import tqdm

class Logger:
    def __init__(self, args):
        self.exper_detail = f"Dataset : {args.dataset.upper()}, Model : {args.model}, Train_size : {args.train_size}"
        if args.model != 'ml':
            self.exper_detail += f", Bs : {args.bs}, Rank : {args.rank}"
        # 运行细节保留在终端；批次摘要由契约写入 run.log。
        self.logger = logging.getLogger(f'llgat.{id(self)}')
        self.logger.addHandler(logging.NullHandler())
        self.logger.propagate = False
        args.log = self
        self.args = args
        self.prepare_the_experiment()


    def prepare_the_experiment(self):
        self.logger.info('```python')
        self.log(self.format_and_sort_args_dict(self.args.__dict__))

    # 日志记录
    def log(self, string):
        import time
        if string[0] == '\n':
            print('\n', end='')
            string = string[1:]
        final_string = time.strftime('|%Y-%m-%d %H:%M:%S| ', time.localtime(time.time())) + string
        self.only_print(string)
        self.logger.info(final_string)

    def __call__(self, string):
        self.log(string)

    def only_print(self, string):
        import time
        if string[0] == '\n':
            print('\n', end='')
            string = string[1:]
        green_string = f"\033[1;38;2;151;200;129m{time.strftime('|%Y-%m-%d %H:%M:%S| ', time.localtime(time.time()))}\033[0m" + f"\033[1m{string}\033[0m"
        tqdm.write(green_string, file=sys.stdout)
        sys.stdout.flush()

    def show_epoch_error(self, runId, epoch, monitor, epoch_loss, result_error, train_time):
        values = ' '.join(f'{name}={value:.4f}' for name, value in result_error.items())
        self.only_print(f"Round={runId + 1}/{self.args.rounds} Epoch={epoch + 1}/{self.args.epochs} Loss={epoch_loss:.4f} valid_NMAE={monitor.val_loss_min:.4f}@{monitor.best_epoch} {values} time={sum(train_time):.1f}s")

    def show_test_error(self, runId, monitor, results, sum_time):
        values = ' '.join(f'{name}={value:.4f}' for name, value in results.items())
        self(f'Round={runId + 1} BestEpoch={monitor.best_epoch} {values} Training_time={sum_time:.1f}s')
        print()

    def format_and_sort_args_dict(self, args_dict, items_per_line=4):
        # Sort the dictionary by keys
        sorted_items = sorted(args_dict.items())
        formatted_str = ''
        for i in range(0, len(sorted_items), items_per_line):
            line_items = sorted_items[i:i + items_per_line]
            line_str = ', '.join([f"'{key}': {value}" for key, value in line_items])
            formatted_str += f"     {line_str},\n"
        return f"{{\n{formatted_str}}}".strip(',\n')


    def clear_the_useless_logs(self):
        def delete_small_log_files(directory):
            # 获取所有.log文件
            log_files = glob.glob(os.path.join(directory, '*.md'))
            number_lines = 23

            # 遍历所有的.log文件
            for file_path in log_files:
                # 使用with语句安全地打开文件
                with open(file_path, 'r') as file:
                    lines = file.readlines()

                # 检查行数并删除文件
                if len(lines) < number_lines:
                    os.remove(file_path)  # 删除文件
                    print(f"Deleted '{file_path}' as it had less than {number_lines} lines.")


        root_directory = f'./results/'
        for dirpath, dirnames, filenames in os.walk(root_directory):
            if 'log' in dirpath:
                delete_small_log_files(dirpath)

    def delete_empty_directories(self, dir_path):
        import os
        # 检查目录是否存在
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            # 遍历目录中的所有文件和子目录，从最底层开始
            for root, dirs, files in os.walk(dir_path, topdown=False):
                # 先删除空的子目录
                for name in dirs:
                    dir_to_remove = os.path.join(root, name)
                    # 如果目录是空的，则删除它
                    if not os.listdir(dir_to_remove):  # 判断目录是否为空
                        os.rmdir(dir_to_remove)
                        print(f"Directory {dir_to_remove} has been deleted.")
                # 检查当前目录是否也是空的，如果是则删除它
                if not os.listdir(root):  # 判断当前根目录是否为空
                    os.rmdir(root)
                    print(f"Directory {root} has been deleted.")
        else:
            print(f"Directory {dir_path} does not exist.")


    # 邮件发送日志
    def send_email(self, subject, body, receiver_email="zengyuxiang@hnu.edu.cn"):
        if self.args.debug:
            return
        import yagmail
        import os
        import pickle

        if isinstance(body, dict):
            metrics = body
            body = []
            # Add experiment results
            body.append('*' * 10 + 'Experiment Results:' + '*' * 10)
            for key in metrics:
                body.append(f'{key}: {np.mean(metrics[key]):.4f} ± {np.std(metrics[key]):.4f}')

            # Add experiment success message
            body.append('*' * 10 + 'Experiment Success' + '*' * 10)

            # Add metrics for each round
            for i in range(self.args.rounds):
                round_metrics = f"Round {i + 1} : "
                for key in metrics:
                    round_metrics += f"{key}: {metrics[key][i]:.4f} "
                body.append(round_metrics)


            # Add formatted args dictionary
            body.append(self.format_and_sort_args_dict(self.args.__dict__, 3))
        else:
            temp = body
            body = [self.format_and_sort_args_dict(self.args.__dict__, 3), temp]

        # Get email credentials
        email_code_address = os.path.expanduser('~') + '/qq_smtp_info.pickle'
        with open(email_code_address, 'rb') as f:
            all_info = pickle.load(f)
        sender_email = all_info['email']
        sender_password = all_info['password']

        # Initialize the SMTP server
        yag = yagmail.SMTP(user=sender_email, password=sender_password, host='smtp.qq.com')

        # Prepare the contents
        contents = body

        # Send the email
        yag.send(
            to=receiver_email,
            subject=subject,
            contents=contents
        )
        print("Email sent successfully!")


    def end_the_experiment(self):
        self.logger.info('```')
