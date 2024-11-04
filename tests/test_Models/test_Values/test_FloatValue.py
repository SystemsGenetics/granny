from Granny.Models.Values.FloatValue import FloatValue


def test_set_get_valid_values():
    # Checks if floats, ints, and strings can be set, get and validated properly.
    float_value = FloatValue("test", "label", "help")
    valid_values = [1.1, 2.2, 3.3]
    float_value.setValidValues(valid_values)
    assert float_value.getValidValues() == valid_values

    float_value_2 = FloatValue("test2", "label2", "help2")
    valid_values_2 = [1,2,3]
    float_value_2.setValidValues(valid_values_2)
    assert float_value_2.getValidValues() == valid_values_2

    float_value_3 = FloatValue("test3", "label3", "help3")
    valid_values_2 = ["string1", "string2", "string3"]
    float_value_2.setValidValues(valid_values_2)
    assert float_value_2.getValidValues() == valid_values_2

def test_set_min_max():
    # Checks if floats, ints, and strings can be instanctiated as the min and
    # max values.
    float_value = FloatValue("test", "label", "help")
    float_value.setMin(1.1)
    float_value.setMax(10.1)
    assert float_value.min_value == 1.1
    assert float_value.max_value == 10.1
    assert not float_value.max_value == 11
    assert not float_value.min_value == 1.01
    assert not float_value.min_value == "string"


def test_validate_valid_values():
    float_value = FloatValue("test", "label", "help")
    float_value.setValidValues([1.1, 2.2, 3.3])
    assert float_value.validate(1.1)
    assert not float_value.validate(1)
    assert not float_value.validate(4.4)
    assert not float_value.validate("string")

    float_value = FloatValue("test1", "label1", "help1")
    float_value.setValidValues([1,2,3])
    assert not float_value.validate(1)
    assert not float_value.validate(4)
    assert not float_value.validate(1.1)
    assert not float_value.validate("string")

    float_value = FloatValue("test2", "label2", "help2")
    float_value.setValidValues(["string1","string2","string3"])
    assert not float_value.validate("string1")
    assert not float_value.validate(1)
    assert not float_value.validate(1.1)
    assert not float_value.validate("string")

def test_validate_min_max():
    float_value = FloatValue("test", "label", "help")
    float_value.setMin(1.1)
    float_value.setMax(10.1)
    assert float_value.validate(5.5)
    assert not float_value.validate(0.0)
    assert not float_value.validate(11.1)
    assert not float_value.validate("string")

    float_value1 = FloatValue("test1", "label1", "help1")
    float_value1.setMin(1)
    float_value1.setMax(10)
    assert not float_value1.validate(2)
    assert not float_value1.validate(10.1)
    assert not float_value1.validate(11)
    assert not float_value1.validate("string")


def test_validate_type():
    float_value = FloatValue("test", "label", "help")
    assert not float_value.validate("string")
    assert not float_value.validate(1)
    assert float_value.validate(1.1)
