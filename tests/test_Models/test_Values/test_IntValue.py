from Granny.Models.Values.IntValue import IntValue


def test_set_get_valid_values():
    # Checks what happens when string and floats are set as valid values and 
    # what happens when you try to get these values.
    int_value = IntValue("test", "label", "help")
    int_value_1 = IntValue("test", "label", "help")
    int_value_2 = IntValue("test", "label", "help")
    valid_values = [1, 2, 3]
    valid_values_1 = ["string1" , "string2", "string3"]
    valid_values_2 = [1.1, 2.2, 3.3]
    int_value.setValidValues(valid_values)
    int_value_1.setValidValues(valid_values_1)
    int_value_2.setValidValues(valid_values_2)
    assert int_value.getValidValues() == valid_values
    try:
        int_value_1.getValidValues() == valid_values_1
    except TypeError:
        assert True
    try: 
        int_value_2.getValidValues() == valid_values_2
    except TypeError:
        assert True 

# For setting the min and max values for the intValue class, it allows Strings
# Doubles be to set. I do not know if this is an oversight or if it matters.
def test_set_min_max():
    # Checks what happens when string and floats are set as max and min values and 
    # what happens when you try to change these values. 
    int_value = IntValue("test", "label", "help")
    int_value_1 = IntValue("test", "label", "help")
    int_value.setMin(1)
    int_value.setMax(10)
    int_value_1.setMin("string")
    int_value_1.setMax(10.1)
    assert int_value.min_value == 1
    assert int_value.max_value == 10
    try:
        int_value_1.min_value == 'string'
    except TypeError:
        assert True
    try:
        int_value_1.max_value == 10.1
    except TypeError:
        assert True
    


def test_validate_valid_values():
    # Checks if the validate function handles non valid values properly.
    int_value = IntValue("test", "label", "help")
    int_value.setValidValues([1, 2, 3])
    assert int_value.validate(1)
    assert not int_value.validate(4)
    assert not int_value.validate("string")
    assert not int_value.validate(1.1)


def test_validate_min_max():
    # checks if 
    int_value = IntValue("test", "label", "help")
    int_value.setMin(1)
    int_value.setMax(10)
    assert int_value.validate(5)
    assert not int_value.validate(0)
    assert not int_value.validate(11)
    try:
        int_value.validate("string")
    except TypeError:
        assert True
    assert not int_value.validate(1.1)
   
    assert not int_value.validate(10.1)
 


def test_validate_type():
    int_value = IntValue("test", "label", "help")
    try:
        int_value.validate("string")
    except TypeError:
        assert True
    assert not int_value.validate(1.5)
    assert int_value.validate(1)
