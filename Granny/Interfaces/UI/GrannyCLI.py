import argparse
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

    def run(self):
        """
        {@inheritdoc}
        """
        # Get the input arguments.
        program_args, _ = self.parser.parse_known_args()
        self.analysis = program_args.analysis

        # Iterates through all of the available analysis classes.
        # Finds then one whose machine name matches the argument
        # provided by the user and run the performAnalysis() function.
        analyses = Analysis.__subclasses__()
        for aclass in analyses:
            if self.analysis == aclass.__analysis_name__:
                # Instantiates the analysis class with the Image list
                analysis = aclass()

                # Extracts from the analysis the list of parameters needed to
                # be set up, then adds CLI's argument for each parameter
                self._setAnalysisParams(analysis=analysis)

                # Performs the analysis with a newly updated set of parameters
                # provided by the user.
                analysis.performAnalysis()

    def addProgramArgs(self) -> None:
        """
        Adds to the argparser the set of arguments required by the CLI.

        This will include the arguments of the analyses as well if 
        the user specified the type of analysis.
        """

        iface_grp = self.parser.add_argument_group("CLI interface args")
        iface_grp.add_argument(
            "--analysis",
            dest="analysis",
            type=str,
            required=True,
            choices=["segmentation", "blush", "color", "scald", "starch"],
            help="Indicates the analysis to run.",
        )

        # Parse the existing arguments to see if an analysis was provided.
        program_args, _ = self.parser.parse_known_args()

        # Iterates through all of the available analysis classes.
        # Finds the one whose machine name matches the argument
        # provided by the user and run the performAnalysis() function.
        analysis_name = program_args.analysis
        analyses = Analysis.__subclasses__()
        analysis = None     
        for aclass in analyses:
            if analysis_name == aclass.__analysis_name__:
                analysis = aclass()
        if analysis is None:
            return
        
        # Create an argument group for this analysis.
        analysis_grp = self.parser.add_argument_group("{} args".format(analysis_name))

        # Iterate through the parameters for the analysis requested,
        # and add each one as an argument.
        params = analysis.getInParams()
        for param in params.values():
            analysis_grp.add_argument(
                f"--{param.getLabel()}",
                type=param.getType(),  
                required=param.getIsRequired(),
                help=param.getHelp(),
            )

    def _setAnalysisParams(self, analysis: Analysis) -> None:

        """
        Sets the analysis's parameters using the user provided arguments.
        """

        analysis_args, _ = self.parser.parse_known_args()
        args_dict = analysis_args.__dict__

        # loops through the parameter list to set values.        
        params = analysis.getInParams()
        analysis.resetInParams()
        for param in params.values():
            # If the user provides a value.
            arg_value = args_dict.get(param.getLabel())
            if arg_value is not None:
                print(f"\t{param.getLabel():<{25}}: (user) {arg_value}")
                param.setValue(arg_value)
            # If the user doesn't provide a value
            else:
                print(f"\t{param.getLabel():<{25}}: (default) {param.getValue()}")
                param.setValue(param.getValue())
            analysis.addInParam(param)
