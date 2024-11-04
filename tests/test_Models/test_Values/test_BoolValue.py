from Granny.Models.Values.BoolValue import BoolValue


def test_validate():
    value_1 = BoolValue("name", "label", "help")
    assert value_1.validate() is True