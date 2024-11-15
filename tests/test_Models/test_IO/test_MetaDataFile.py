from Granny.Models.IO.MetaDataFile import MetaDataFile

def test_load_save():
    List = list[1,2,3,4]
    File = MetaDataFile("test-results/results")
    File.save(List)
    assert File.load() == []
   