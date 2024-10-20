"""Main entry point of program. Here, a configs file is parsed and the trainer is instantiated and run."""

import argparse

from senn.trainer import init_trainer


def main():
    """
    Entry point to the trainer.
    Binds together the config and the Trainer class
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/mnist_lambda1e-3_seed29_cuda.json", help='experiment config file')
    args = parser.parse_args()

    trainer = init_trainer(args.config)
    trainer.run()
    trainer.finalize()


if __name__ == "__main__":
    main()
