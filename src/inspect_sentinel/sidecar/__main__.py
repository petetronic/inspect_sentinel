import argparse

from ._serve import add_arguments, handler_from_arguments, serve


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m inspect_sentinel.sidecar",
        description="Run a sentinel in a sidecar, beside a model proxy.",
    )
    add_arguments(parser)
    arguments = parser.parse_args()
    serve(handler_from_arguments(arguments), arguments)


if __name__ == "__main__":
    main()
