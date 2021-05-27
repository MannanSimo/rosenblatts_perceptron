"""Script perfrom training of Rosenblatt's perceptron based on Iris data."""

from loguru import logger

from cli_application import CliApplication


def main():
    """Encapsulate script's main workflow."""
    logger.add("file_{time}.log", format="{time} {level} {message}", level='INFO')
    app = CliApplication(show=True, logger=logger)
    app.run()


if __name__ == "__main__":
    main()
