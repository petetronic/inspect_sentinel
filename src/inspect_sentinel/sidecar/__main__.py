from ._serve import handler_from_arguments, parse_arguments, serve


def main() -> None:
    arguments = parse_arguments()
    serve(handler_from_arguments(arguments), arguments)


if __name__ == "__main__":
    main()
