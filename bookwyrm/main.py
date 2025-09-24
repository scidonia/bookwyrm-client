"""Main entry point for the BookWyrm client."""

from .cli import main as cli_main


def main():
    """Main function for the BookWyrm client CLI."""
    cli_main()


if __name__ == "__main__":
    main()
