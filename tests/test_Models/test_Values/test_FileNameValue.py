from Granny.Models.Values.FileNameValue import FileNameValue 

def test_get_validate_value():
    # Checks if a valid file path/name can be "get" and "validated" properly.
    value_1 = FileNameValue("name1", "string1", "help1")
    file = "tests/test_Models/test_Values/test_FileNameValue.py"
    value_1.value = file
    assert value_1.getValue() == file
    assert value_1.validate()

    # Checks if a non valid file path/name be "get" and "validated" properly.
    value_2 = FileNameValue("name2", "string2", "help2")
    file2 = "tests/test_Models/test"
    value_2.value = file2
    assert value_2.getValue() != file
    assert not value_2.validate()