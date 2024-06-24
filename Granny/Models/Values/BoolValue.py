from Granny.Models.Values.Value import Value


class BoolValue(Value):
    """
    Class for a boolean value.
    """

    def __init__(self, name: str, label: str, help: str):
        """
        {@inheritdoc}
        """
        super().__init__(name, label, help)
        self.type = bool

    def validate(self) -> bool:
        """
        {@inheritdoc}
        """
        return True
