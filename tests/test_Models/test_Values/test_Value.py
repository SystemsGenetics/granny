from Granny.Models.Values.Value import Value
from typing import Any


class testObject(Value):
    def validate(self, value: Any) -> bool:
        return super().validate(value)


def test_getName():
    # Tests if the name is properly set.
    value_1 = testObject("name1", "label1", "help1")
    assert value_1.getName() == "name1"


def test_getLabel():
    # Tests if the label is properly set.
    value_1 = testObject("name1", "label1", "help1")
    assert value_1.getLabel() == "label1"


def test_getHelp():
    # Tests if the help is properly set.
    value_1 = testObject("name1", "label1", "help1")
    assert value_1.getHelp() == "help1"


def test_getsetValue():
    # Checks if a string value is properly set and get.
    value_1 = testObject("name1", "label1", "help1")
    value_1.setValue("a string")
    assert value_1.getValue() == "a string"
    assert value_1.is_set

    # Checks if a int value is properly set and get.
    value_2 = testObject("name2", "label2", "help2")
    value_2.setValue(1)
    assert value_2.getValue() == 1
    assert value_2.is_set

    # Checks if a float value is properly set and get.
    value_3 = testObject("name3", "label3", "help3")
    value_3.setValue(1.0)
    assert value_3.getValue() == 1.0
    assert value_3.is_set


def test_getsetType():
    # Checks if the string type set matches the type get.
    value_1 = testObject("name1", "label1", "help1")
    value_1.setValue("")
    assert value_1.getType() == str

    # Checks if the int type set matches the type get.
    value_2 = testObject("name2", "label2", "help2")
    value_2.setValue(1)
    assert value_2.getType() == int

    # Checks if the float type set matches the type get.
    value_3 = testObject("name3", "label3", "help3")
    value_3.setValue(1.1)
    assert value_3.getType() == float


def test_isRequired():
    # Checks if the setIsRequired properly returns the set value.
    value_1 = testObject("name1", "label1", "help1")
    value_1.setIsRequired(True)
    assert value_1.getIsRequired()

    # Checks if the setIsRequired properly returns the set value.
    value_2 = testObject("name2", "label2", "help2")
    value_2.setIsRequired(False)
    assert not value_2.getIsRequired()
