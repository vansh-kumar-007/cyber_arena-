# utils/logger.py
# This file keeps a log of everything that happens during training.
# Like a battle diary — every important event gets recorded.

import datetime

class Logger:
    def __init__(self, print_logs=True):
        self.logs = []
        self.print_logs = print_logs  # Set to False to silence output

    def log(self, episode, step, message):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] Episode {episode} | Step {step} | {message}"
        self.logs.append(entry)

        if self.print_logs:
            print(entry)

    def save(self, filename="battle_log.txt"):
        with open(filename, "w") as f:
            for line in self.logs:
                f.write(line + "\n")
        print(f"Log saved to {filename}")

    def get_logs(self):
        return self.logs