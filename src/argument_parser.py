import argparse

class ArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Process a picture.")
        self._add_arguments()

    def _add_arguments(self):
        self.parser.add_argument("--num-pictures-to-process", type=str, required=True)

    def parse(self):
        # Parse the command-line arguments
        return self.parser.parse_args()
