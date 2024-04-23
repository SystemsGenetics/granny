import argparse

from Granny.Interfaces.UI.GrannyCLI import GrannyCLI
from Granny.Interfaces.UI.GrannyPyQt import GrannyPyQt


def run():
    """
    Launces Granny.

    The user specifies the interface to use. If no interface is provided then
    the default interface is used.
    """
    parser = argparse.ArgumentParser(description="Welcome to Granny!")
    parser.add_argument(
        "-i",
        "--interface",
        dest="interface",
        type=str,
        nargs=1,
        required=False,
        default="cli",
        choices=["cli", "gui"],
        help="Indicates the user interface to use, either the command-line (cli) or the graphical interface (gui).",
    )
    namespace, _ = parser.parse_known_args()
    interface = namespace.interface[0]

    # Now calls the proper interface class.
    if interface == "cli":
        GrannyCLI(parser).run()
    elif interface == "gui":
        GrannyPyQt(parser).run()
    else:
        print("Error: unknown interface")
        exit(1)
