from Granny.Models.Values.FileDirValue import FileDirValue
import uuid


def test_set_validate_value():
    # Tests for exisiting Dir file name. 
    FileDirValue_1 = FileDirValue("name1", "label1", "help1")
    FileDirValue_1.setValue("Granny/Analyses")
    assert FileDirValue_1.validate()

    # Tests if a new Dir is created using the file path entered. 
    # Generates a random string and appends it to the file name.
    # This is for testing and making sure we know where it is from.
    FileDirValue_2 = FileDirValue("name2", "label2", "help2")
    filename = str(uuid.uuid4())
    FileDirValue_2.setValue("test-results/FileDirValue_test_" + filename)
    assert FileDirValue_2.validate()