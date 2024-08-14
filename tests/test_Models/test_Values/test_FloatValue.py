from Granny.Models.Values.FloatValue import FloatValue


def test_set_get_valid_values():
    float_value = FloatValue("test", "label", "help")
    valid_values = [1.1, 2.2, 3.3]
    float_value.setValidValues(valid_values)
    assert float_value.getValidValues() == valid_values


def test_set_min_max():
    float_value = FloatValue("test", "label", "help")
    float_value.setMin(1.1)
    float_value.setMax(10.1)
    assert float_value.min_value == 1.1
    assert float_value.max_value == 10.1


def test_validate_valid_values():
    float_value = FloatValue("test", "label", "help")
    float_value.setValidValues([1.1, 2.2, 3.3])
    assert float_value.validate(1.1)
    assert not float_value.validate(4.4)


def test_validate_min_max():
    float_value = FloatValue("test", "label", "help")
    float_value.setMin(1.1)
    float_value.setMax(10.1)
    assert float_value.validate(5.5)
    assert not float_value.validate(0.0)
    assert not float_value.validate(11.1)


def test_validate_type():
    float_value = FloatValue("test", "label", "help")
    assert not float_value.validate("string")
    assert not float_value.validate(1)
    assert float_value.validate(1.1)
