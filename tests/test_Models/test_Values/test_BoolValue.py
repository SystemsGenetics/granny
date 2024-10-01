from Granny.Models.Values.BoolValue import BoolValue


def test_validate():
    value_1 = BoolValue("name1", "label1", "help1")
    value_1.setValue(True)
    assert not value_1.validate("string")
    assert value_1.validate(True)
