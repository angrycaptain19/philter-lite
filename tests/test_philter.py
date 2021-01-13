import os

import pytest

from philter_lite import philter


def test_filter_from_dict():
    filter_dict = {
        "title": "test_city",
        "type": "regex",
        "keyword": "addresses.city",
        "exclude": "test_ex",
        "notes": "test_notes",
    }

    filter = philter.filter_from_dict(filter_dict)

    assert filter.type == "regex"
    assert filter.title == "test_city"
    assert filter.data is not None
    assert filter.exclude == "test_ex"
    assert isinstance(filter, philter.RegexFilter)

    filter_dict = {
        "title": "Find Names 1",
        "type": "regex_context",
        "exclude": True,
        "context": "right",
        "context_filter": "Firstnames Blacklist",
        "keyword": "regex_context.names_regex_context1",
    }

    filter = philter.filter_from_dict(filter_dict)

    assert filter.type == "regex_context"
    assert filter.title == "Find Names 1"
    assert filter.data is not None
    assert filter.exclude is True
    assert filter.context == "right"
    assert filter.context_filter == "Firstnames Blacklist"
    assert isinstance(filter, philter.RegexContextFilter)

    filter_dict = {
        "title": "Whitelist 1",
        "type": "set",
        "exclude": False,
        "keyword": "nonames",
        "pos": [],
    }

    filter = philter.filter_from_dict(filter_dict)

    assert filter.type == "set"
    assert filter.title == "Whitelist 1"
    assert filter.data is not None
    assert filter.exclude is False
    assert filter.pos == []
    assert isinstance(filter, philter.SetFilter)

    filter_dict = {
        "title": "POS MATCHER",
        "type": "pos_matcher",
        "exclude": False,
        "pos": ["CD",],
    }

    filter = philter.filter_from_dict(filter_dict)

    assert filter.type == "pos_matcher"
    assert filter.title == "POS MATCHER"
    assert filter.exclude is False
    assert filter.pos == ["CD"]
    assert isinstance(filter, philter.PosFilter)


def test_find_phi():
    test_report = """
Record date: 2069-04-07

 
 
 
Mr. Villegas is seen today.  I have not seen him since November. 
About three weeks ago he stopped his Prednisone on his own because
he was gaining weight.  He does feel that his shoulders are
definitely improved.  It is unclear what he is actually taking, but
I think based on the color of his pills and the timing of the
medication that he continues taking his Atenolol for hypertension
and 1 Hydroxychloroquine tablet.  He is concerned because of the
relatively recent onset of difficulties turning his head to the
right.  When he does this, he will note that he feels as though he
is going to pass out although this has not actually happened.  This
only occurs when he turns to the right and not to the left.  He has
no visual changes otherwise and denies any headache or other
cranial complaints.  
 
On examination today, BP 120/80.  He has no bruits over the
carotid.  He has no tenderness in this region either.  He has good
peripheral pulses at the arms.  His joint examination is much
improved with better ROM of the shoulders and no peripheral joint
synovitis.  
 
Clinical Impression:
 
#1:  Inflammatory arthritis - possibly RA - with response noted to
Hydroxychloroquine along with Prednisone.  He has stopped the
Prednisone, and I would not restart it yet.  
 
#2:  New onset of symptoms suspicious for right-sided carotid
disease.  Will arrange for carotid ultrasound studies.  Patient
advised to call me if he develops any worsening symptoms.  He has
been taking 1 aspirin per day prophylaxis long-term, and I stressed
that he continue to do so.  He will follow-up with me shortly after
the ultrasound study.
 
 
 
Xzavian G. Tavares, M.D.
XGT:holmes
 
DD: 04/07/69
DT: 04/15/69
DV: 04/07/69
          Approved but not reviewed by Attending Provider         
"""
    patterns = philter.build_filters(
        "/Users/torme/code/philter-ucsf/philter_lite/configs/philter_delta.toml"
    )
    results = philter.find_phi(test_report, patterns)

    for entry in results:
        print(
            f"{entry.start}, {entry.stop}, {entry.filter.exclude}, {entry.filter.title}, {test_report[entry.start:entry.stop]}"
        )
    # print(results)
