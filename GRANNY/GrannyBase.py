import argparse

from GRANNY.Interfaces.UI.GrannyCLI import GrannyCLI
from GRANNY.Interfaces.UI.GrannyPyQt import GrannyPyQt


def run():
    """
    Launces Granny.

    The user specifies the interface to use. If no interface is provided then
    the default interface is used.
    """
    parser = argparse.ArgumentParser(description="Welcome to Granny")
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
    namespace, extra = parser.parse_known_args()

    # The parse_known_args function sometimes reurns a list for the
    # the argument and sometimes a scalar. This is just to check both.
    interface = ""
    if isinstance(namespace.interface, list):
        interface = namespace.interface[0]
    else:
        interface = namespace.interface

    # Now call the proper interface class.
    if interface == "cli":
        GrannyCLI(parser).run()
    elif interface == "gui":
        GrannyPyQt(parser).run()
    else:
        print("Error: unknown interface")
        exit(1)
