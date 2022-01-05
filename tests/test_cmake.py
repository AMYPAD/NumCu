import numcu as nc


def test_includes():
    assert nc.include_path.is_dir()
    assert {i.name for i in nc.include_path.iterdir()} == {'elemwise.h', 'numcu.h'}


def test_cmake_prefix():
    assert nc.cmake_prefix.is_dir()
    assert {i.name
            for i in nc.cmake_prefix.iterdir()} == {
                f'AMYPADnumcu{i}.cmake'
                for i in ('Config', 'ConfigVersion', 'Targets', 'Targets-release')}
