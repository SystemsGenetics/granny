from Granny.Models.Values.IntValue import IntValue


def test_set_get_valid_values():
    int_value = IntValue("test", "label", "help")
    valid_values = [1, 2, 3]
    int_value.setValidValues(valid_values)
    assert int_value.getValidValues() == valid_values


def test_set_min_max():
    int_value = IntValue("test", "label", "help")
    int_value.setMin(1)
    int_value.setMax(10)
    assert int_value.min_value == 1
    assert int_value.max_value == 10


def test_validate_valid_values():
    int_value = IntValue("test", "label", "help")
    int_value.setValidValues([1, 2, 3])
    assert int_value.validate(1)
    assert not int_value.validate(4)


def test_validate_min_max():
    int_value = IntValue("test", "label", "help")
    int_value.setMin(1)
    int_value.setMax(10)
    assert int_value.validate(5)
    assert not int_value.validate(0)
    assert not int_value.validate(11)


def test_validate_type():
    int_value = IntValue("test", "label", "help")
    try:
        int_value.validate("string")
    except TypeError:
        assert True
    assert not int_value.validate(1.5)
    assert int_value.validate(1)
