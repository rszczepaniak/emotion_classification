import argparse

class ArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Process a picture.")
        self._add_arguments()

    def _add_arguments(self):
        self.parser.add_argument("--num-pictures-to-process", type=str, required=True)
        self.parser.add_argument('--haar', action='store_true', help='run the haar cascade face detector')
        self.parser.add_argument('--pretrained', type=str, default='checkpoint/model_weights/weights_epoch_75.pth.tar'
                            , help='load weights')
        self.parser.add_argument('--head_pose', action='store_true', help='visualization of head pose euler angles')
        self.parser.add_argument('--path', type=str, default='', help='path to video to test')
        self.parser.add_argument('--image', action='store_true', help='specify if you test image or not')

    def parse(self):
        # Parse the command-line arguments
        return self.parser.parse_args()
