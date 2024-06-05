import argparse
import sys
from importlib import metadata
from typing import NoReturn

from Granny.Interfaces.UI.GrannyCLI import GrannyCLI
from Granny.Interfaces.UI.GrannyPyQt import GrannyPyQt


class GrannyParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        """
        Overrides argparse error message
        """
        sys.stderr.write("error: %s\n" % message)
        self.print_help()
        sys.exit(2)

    def print_help(self, file=None) -> None:  # type: ignore
        """
        Overrides argparse print_help() with a customized help message
        """
        self._print_message(self.format_help(), file)


def run():
    """
    Launces Granny.

    The user specifies the interface to use. If no interface is provided then
    the default interface is used.
    """
    parser = GrannyParser(
        description="Welcome to Granny!",
        conflict_handler="resolve",
    )
    sub_parser = parser.add_subparsers(
        metavar="interface",
        title="commands",
        description="The following commands are available.",
        dest="cmd",
        required=True,
    )
    parser.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="show this help message and exit",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"Granny {metadata.version('granny')}",
    )

    # Now calls the proper interface class.
    cli = GrannyCLI(parser)
    cli.configureParser(sub_parser)
    cli.run()
