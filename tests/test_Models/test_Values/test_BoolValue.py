from Granny.Models.Values.BoolValue import BoolValue


def test_validate():
    value_1 = BoolValue("name", "label", "help")
    assert value_1.validate() is True


def test_name_label_help():
    v = BoolValue("myname", "mylabel", "myhelp")
    assert v.getName() == "myname"
    assert v.getLabel() == "mylabel"
    assert v.getHelp() == "myhelp"


def test_set_get_true():
    v = BoolValue("b", "b", "b")
    v.setValue(True)
    assert v.getValue() is True


def test_set_get_false():
    v = BoolValue("b", "b", "b")
    v.setValue(False)
    assert v.getValue() is False


def test_type_is_bool():
    v = BoolValue("b", "b", "b")
    assert v.type is bool


def test_overwrite_value():
    v = BoolValue("b", "b", "b")
    v.setValue(True)
    v.setValue(False)
    assert v.getValue() is False


def test_validate_with_value():
    v = BoolValue("b", "b", "b")
    assert v.validate(True) is True
    assert v.validate(False) is True


def test_set_is_required():
    v = BoolValue("b", "b", "b")
    v.setIsRequired(True)
    assert v.getIsRequired() is True


def test_set_is_required_false():
    v = BoolValue("b", "b", "b")
    v.setIsRequired(False)
    assert v.getIsRequired() is False