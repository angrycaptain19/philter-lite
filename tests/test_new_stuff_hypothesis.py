import re

import pytest
from hypothesis import given
from hypothesis.strategies import text

import philter_lite.filters.filter_db as filter_db

old_and_new_regex_pairs = []

for section, regex_dict in filter_db.regex_db.items():
    for filter_name, regex in regex_dict.items():
        new_regex = filter_db.regex_new_db[section][filter_name]
        old_and_new_regex_pairs.append((regex, new_regex))

@pytest.mark.parametrize("old_regex,new_regex", old_and_new_regex_pairs)
@given(a_string=text())
def test(old_regex, new_regex, a_string):
    old_c = re.compile(old_regex)
    new_c = re.compile(new_regex)
    assert old_c.match(a_string) == new_c.match(a_string)
