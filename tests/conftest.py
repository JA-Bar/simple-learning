# port of Robert Collins' test scenarios to check different cases easily (only on classes)
def pytest_generate_tests(metafunc):
    if metafunc.cls:
        idlist = []
        argvalues = []
        for scenario in metafunc.cls.scenarios:
            idlist.append(scenario[0])
            items = scenario[1].items()
            argnames = [x[0] for x in items]
            argvalues.append([x[1] for x in items])
        metafunc.parametrize(argnames, argvalues, ids=idlist, scope="class")
