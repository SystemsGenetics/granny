import argparse
import sys
from importlib import metadata
from typing import NoReturn

from Granny.Interfaces.UI.GrannyCLI import GrannyCLI
from Granny.Interfaces.UI.GrannyPyQt import GrannyPyQt


class GrannyArgParser(argparse.ArgumentParser):
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
    Launches Granny.

    The user specifies the interface to use. If no interface is provided then
    the default interface is used.
    """
    parser = GrannyArgParser(
        description="Welcome to Granny!",
        add_help=False
    )
    parser.add_argument(
        "-i",
        dest="interface",
        type=str,
        required=True,    
        choices=["cli", "gui"],
        help="Indicates the user interface to use, either the command-line (cli) or the graphical interface (gui).",
    )
    parser.add_argument(
        "-v",
        action="version",
        version=f"Granny {metadata.version('granny')}",
    )
    
    # Parse known arguments so we can get the interface.
    namespace, _ = parser.parse_known_args()

    # Now create the proper interface class and add its
    # list of arguments.
    interface = namespace.interface
    iface = None
    if interface == "cli":
        iface = GrannyCLI(parser)
    elif interface == "gui":
        iface = GrannyPyQt(parser)

    # Add the interface arguments.
    iface.addProgramArgs()

    # Run the program.
    iface.run()
