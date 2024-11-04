from Granny.Models.Values.StringValue import StringValue


def test_getsetValidValues():
    # Checks if the valid values are set and get and if a valid string value can be set 
    # as the value.
    value_1 = StringValue("name1", "label1", "help1")
    value_1.setValidValues(["string 1", "string 2"])
    value_1.setValue("string 1")
    assert value_1.getValue() == "string 1"
    assert value_1.getValidValues() == ["string 1", "string 2"]
    assert value_1.is_set

    # Checks if the valid values are set and get and if non valid string value can be set 
    # as the value.
    value_2 = StringValue("name2", "label2", "help2")
    value_2.setValidValues(["string 1", "string 3"])
    value_2.setValue("string 2")
    assert value_2.getValue() is None
    assert value_2.is_set

    # Checks if the valid values are set and get and if valid int value can be set 
    # as the value.
    value_3 = StringValue("name3", "label3", "help3")
    value_3.setValidValues([1,2,3])
    value_3.setValue(2)
    assert value_3.getValue() is None
    assert value_3.is_set

    # Checks if the valid values are set and get and if valid float value can be set 
    # as the value.
    value_4 = StringValue("name4", "label4", "help4")
    value_4.setValidValues([1.1,2.2,3.3])
    value_4.setValue(2.2)
    assert value_4.getValue() is None
    assert value_4.is_set


def test_validate():

    # Checks if the value that is validated is one of the setValidValues.
    value_1 = StringValue("name1", "label1", "help1")
    value_1.setValidValues(["string 1", "string 2"])
    assert value_1.validate("string 1") is True
    assert value_1.validate("string 3") is False
    assert value_1.validate(123) is False

    # Checks what happens when there are not any set valid values.
    value_2 = StringValue("name2", "label2", "help2")
    assert value_2.validate("any string") is True
    assert value_2.validate(456) is False
    assert value_2.validate(1.1) is False

    # Checks what happens when ints are set as valid values and what happens if
    # you validate ints in the valid set and not in the valid set.
    value_3 = StringValue("name3", "label3", "help3")
    value_3.setValidValues([1,2,3])
    assert value_3.validate(1) is False
    assert value_3.validate(4) is False

    # Checks what happens when floats are set as valid values and what happens if
    # you validate float in the valid set and not in the valid set.
    value_4 = StringValue("name3", "label3", "help3")
    value_4.setValidValues([1.1,2.2,3.3])
    assert value_4.validate(1.1) is False
    assert value_4.validate(4) is False