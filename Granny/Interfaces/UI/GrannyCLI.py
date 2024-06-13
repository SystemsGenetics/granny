from argparse import ArgumentParser, _SubParsersAction  # type: ignore

from Granny.Analyses.Analysis import Analysis
from Granny.Analyses.BlushColor import BlushColor
from Granny.Analyses.PeelColor import PeelColor
from Granny.Analyses.Segmentation import Segmentation
from Granny.Analyses.StarchArea import StarchArea
from Granny.Analyses.SuperficialScald import SuperficialScald
from Granny.Interfaces.UI.GrannyUI import GrannyUI


class GrannyCLI(GrannyUI):
    def __init__(self, parser):  # type:ignore
        """
        {@inheritdoc}
        """
        GrannyUI.__init__(self, parser)
        self.analysis: str = ""

    def checkArgs(self) -> bool:
        """
        Checks the incoming command-line arguments.

        Ensures that the command-line arguments are appropriate for the
        image collection and analyses to be performed.  When arguments
        do not pass a test, then error messages are printed to STDERR
        and False is returned.

        @return bool
          True if all argument are good, FALSE otherwsie.
        """
        analyses = Analysis.__subclasses__()
        for aclass in analyses:
            if self.analysis == aclass.__analysis_name__:
                return True
        return False

    def run(self):
        """
        {@inheritdoc}
        """
        # Get the input arguments.
        self.addProgramArgs()
        program_args, _ = self.parser.parse_known_args()
        self.analysis = program_args.analysis

        # Checks the incoming arguments for errors, if all is okay then collect the arguments.
        if not self.checkArgs():
            self.parser.print_help()
            exit(1)

        # Iterates through all of the available analysis classes.
        # Finds then one whose machine name matches the argument
        # provided by the user and run the performAnalysis() function.
        analyses = Analysis.__subclasses__()
        for aclass in analyses:
            if self.analysis == aclass.__analysis_name__:
                # Instantiates the analysis class with the Image list
                analysis = aclass()

                # Extracts from the analysis the list of parameters needed to be set up,
                # then adds CLI's argument for each parameter
                self.addAnalysisArgs(analysis=analysis)

                # Performs the analysis with a newly updated set of parameters provided by the user
                analysis.performAnalysis()

    def addProgramArgs(self) -> None:
        """
        Parses the following command-line arguments: analysis, image directory, metadata directory,
        and result directory. These parameters are required to run the program.
        """
        self.parser.add_argument(
            "-a",
            "--analysis",
            dest="analysis",
            type=str,
            nargs="?",
            required=True,
            choices=["segmentation", "blush", "color", "scald", "starch"],
            help="Chooses an analysis you want to perform.",
        )

    def addAnalysisArgs(self, analysis: Analysis) -> None:
        """
        Parses the command-line arguments for the analysis's parameters.

        These parameters are not required to run the program, but if there is no value provided by
        the user, the value is set to the default value by the analysis class.
        """
        # calls analysis.getValues() for additional parameters of the analysis.
        params = analysis.getValues()
        # checks the list of parameters
        if len(params) > 0:
            # calls argparse to parse analysis arguments specified by the user
            for param in params.values():
                self.parser.add_argument(
                    f"--{param.getLabel()}",
                    type=param.getType(),  # type: ignore
                    required=False if param.isSet() else True,
                    help=param.getHelp(),
                )
            analysis_args, _ = self.parser.parse_known_args()
            args_dict = analysis_args.__dict__
            # @todo: validate the arguments before setting the value.
            # resets the parameter list in the analysis to update new parameter's values
            # from the user
            analysis.setValue({})
            # loops through the parameter list to update new values using setValue()
            for param in params.values():
                # if the user provide a value
                arg_value = args_dict.get(param.getLabel())
                if arg_value is not None:
                    print(
                        f"\t{param.getLabel():<{25}}: (user) {arg_value}",
                    )
                    param.setValue(arg_value)
                # if the user doesn't provide a value
                else:
                    print(
                        f"\t{param.getLabel():<{25}}: (default) {param.getValue()}",
                    )
                    param.setValue(param.getValue())
                analysis.addValue(param)
