from Granny.Analyses.Analysis import Analysis
from collections import defaultdict, deque


class Scheduler(object):

    def __init__(self):
        """
        Initialize the Scheduler object.
        """
        # Dictionary to hold analyses with their IDs.
        self.analyses = {}

        # Graph to hold dependencies.
        self.graph = defaultdict(list)

        # Dictionary to track in-degrees, or the number of dependencies
        # that the an analysis.
        self.in_degree = defaultdict(int)

    def add_analysis(self, analysis: Analysis, dependencies=None):
        """
        Adds an analysis to the scheduler.

        Args:
            analysis (Analysis): The analysis to be added.
            dependencies (list): A list of dependencies (other Analysis objects).
        """
        dependencies = dependencies or []

        # Get the ID of the analysis object and store it.
        analysis_id = id(analysis)  
        self.analyses[analysis_id] = analysis
        
        # For each dependency of the analysis, store the ID of its parent 
        # in the graph.
        for dependency in dependencies:
            dependency_id = id(dependency)
            if dependency_id in self.analyses:

                # Store the parent of the dependency.
                self.graph[dependency_id].append(analysis_id)

                # Increment the in-degree for the analysis (# of deps).
                self.in_degree[analysis_id] += 1

    def schedule(self):
        """
        Schedule the analyses in the correct order based on dependencies.

        Returns:
            list: A list of analysis IDs in the order they should be run.

        Raises:
            ValueError: If there is a cycle in the dependencies.
        """
        # Use a queue to manage analyses with no dependencies.
        queue = deque([analysis_id for analysis_id in self.analyses if self.in_degree[analysis_id] == 0])
        order = []

        while queue:
            current = queue.popleft()
            order.append(current)
            for neighbor in self.graph[current]:
                self.in_degree[neighbor] -= 1
                if self.in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(self.analyses):
            raise ValueError("There is a cycle in the dependencies.")
        
        return order
    
    def run(self):
        """
        Run the analyses in the scheduled order.

        Returns:
            dict: A dictionary containing the results of all analyses.
        """
        order = self.schedule()
        for analysis_id in order:
            analysis: Analysis = self.analyses[analysis_id]
            analysis.performAnalysis()

            # @todo get the return values from the analysis and feed those 
            # into the dependency.            
    