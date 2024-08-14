from Granny.Models.Values.StringValue import StringValue


def test_getsetValidValues():
    value_1 = StringValue("name1", "label1", "help1")
    value_1.setValidValues(["string 1", "string 2"])
    value_1.setValue("string 1")
    assert value_1.getValue() == "string 1"
    assert value_1.is_set

    value_2 = StringValue("name2", "label2", "help2")
    value_2.setValidValues(["string 1", "string 3"])
    value_2.setValue("string 2")
    assert value_2.getValue() is None
    assert value_2.is_set


def test_validate():
    value_1 = StringValue("name1", "label1", "help1")
    value_1.setValidValues(["string 1", "string 2"])
    assert value_1.validate("string 1") is True
    assert value_1.validate("string 3") is False
    assert value_1.validate(123) is False

    value_2 = StringValue("name2", "label2", "help2")
    assert value_2.validate("any string") is True
    assert value_2.validate(456) is False
