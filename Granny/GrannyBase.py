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
        # Get the default help message
        help_message = self.format_help()

        # Remove the program description if present
        if self.description:
            help_message = help_message.replace(self.description + "\n", "")

        # Print the modified help message
        self._print_message(help_message, file)


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
    parser.add_argument(
        "-i",
        "--interface",
        dest="interface",
        type=str,
        nargs="?",
        required=False,
        default="cli",
        choices=["cli", "gui"],
        help="Indicates the user interface to use, either the command-line (cli) or the graphical interface (gui).",
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
    namespace, _ = parser.parse_known_args()
    interface = namespace.interface

    # Now calls the proper interface class.
    if interface == "cli":
        GrannyCLI(parser).run()
    elif interface == "gui":
        GrannyPyQt(parser).run()
    else:
        print("Error: unknown interface")
        exit(1)
