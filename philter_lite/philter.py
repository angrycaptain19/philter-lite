import os
import re
import subprocess
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Pattern

import nltk
import toml
from chardet.universaldetector import UniversalDetector
from nltk.tag.stanford import StanfordNERTagger

from philter_lite.coordinate_map import CoordinateMap
from philter_lite.filters import filter_db


@dataclass(frozen=True)
class Filter:
    title: str
    type: str
    exclude: bool
    phi_type: Optional[str]


@dataclass(frozen=True)
class SetFilter(Filter):
    pos: List[str]
    data: Pattern[str]


@dataclass(frozen=True)
class RegexFilter(Filter):
    data: Pattern[str]


@dataclass(frozen=True)
class RegexContextFilter(Filter):
    context: str
    context_filter: str
    data: Pattern[str]


@dataclass(frozen=True)
class PosFilter(Filter):
    pos: List[str]


@dataclass(frozen=True)
class NerFilter(Filter):
    pos: Optional[List[str]]


@dataclass(frozen=True)
class PhiEntry:
    start: int
    stop: int
    word: str
    phi_type: str


@dataclass(frozen=True)
class NonPhiEntry:
    start: int
    stop: int
    word: str
    filepath: str


@dataclass(frozen=True)
class DataTracker:
    text: str
    phi: List[PhiEntry]
    non_phi: List[NonPhiEntry]


@dataclass(frozen=True)
class CoordinateEntry:
    start: int
    stop: int
    filter: Filter


def precompile(regex: str):
    """ precompiles our regex to speed up pattern matching"""
    # NOTE: this is not thread safe! but we want to print a more detailed warning message
    with warnings.catch_warnings():
        warnings.simplefilter(
            action="error", category=FutureWarning
        )  # in order to print a detailed message
        try:
            re_compiled = re.compile(regex)
        except FutureWarning as warn:
            warnings.simplefilter(action="ignore", category=FutureWarning)
            re_compiled = re.compile(regex)  # assign nevertheless
    return re_compiled


def filter_from_dict(
    filter_dict,
    regex_db=filter_db.regex_db,
    regex_context_db=filter_db.regex_context_db,
    set_db=filter_db.set_db,
):
    known_pattern_types = {
        "regex",
        "set",
        "regex_context",
        "stanford_ner",
        "pos_matcher",
        "match_all",
    }

    filter_type = filter_dict["type"]

    if filter_type not in known_pattern_types:
        raise Exception("Pattern type is unknown", filter_type)

    if filter_type == "set":
        set_keyword = filter_dict["keyword"]
        data = _nested_get(set_db, set_keyword.split("."))
        return SetFilter(
            title=filter_dict["title"],
            type=filter_type,
            exclude=filter_dict["exclude"],
            data=data,
            pos=filter_dict["pos"],
            phi_type=filter_dict.get("phi_type", None),
        )
    elif filter_type == "regex":
        regex_keyword = filter_dict["keyword"]
        regex = _nested_get(regex_db, regex_keyword.split("."))
        data = precompile(regex)
        return RegexFilter(
            title=filter_dict["title"],
            type=filter_type,
            exclude=filter_dict["exclude"],
            data=data,
            phi_type=filter_dict.get("phi_type", None),
        )

    elif filter_type == "regex_context":
        regex_keyword = filter_dict["keyword"]
        regex = _nested_get(regex_context_db, regex_keyword.split("."))
        data = precompile(regex)

        return RegexContextFilter(
            title=filter_dict["title"],
            type=filter_type,
            exclude=filter_dict["exclude"],
            context=filter_dict["context"],
            context_filter=filter_dict["context_filter"],
            data=data,
            phi_type=filter_dict.get("phi_type", None),
        )
    elif filter_type == "pos_matcher":
        return PosFilter(
            title=filter_dict["title"],
            type=filter_type,
            exclude=filter_dict["exclude"],
            pos=filter_dict["pos"],
            phi_type=filter_dict.get("phi_type", None),
        )
    else:
        return Filter(
            title=filter_dict["title"],
            type=filter_type,
            exclude=filter_dict["exclude"],
            phi_type=filter_dict.get("phi_type", None),
        )


def build_filters(filter_path) -> List[Filter]:
    if not os.path.exists(filter_path):
        raise Exception("Filepath does not exist", filter_path)
    with open(filter_path, "r") as fil_file:
        return [filter_from_dict(x) for x in toml.loads(fil_file.read())["filters"]]


def build_ner_tagger(
    classifier, tagger_jar, download: bool = True
) -> StanfordNERTagger:
    if not os.path.exists(classifier) and not download:
        raise Exception(
            "Filepath does not exist", classifier,
        )
    else:
        # download the ner data
        process = subprocess.Popen(
            "cd generate_dataset && ./download_ner.sh".split(), stdout=subprocess.PIPE,
        )
        process.communicate()

    if not os.path.exists(tagger_jar):
        raise Exception("Filepath does not exist", tagger_jar)

    return StanfordNERTagger(classifier, tagger_jar)


def get_pos(cleaned):
    return nltk.pos_tag(cleaned)


def get_clean(text, pre_process=r"[^a-zA-Z0-9]"):
    # Use pre-process to split sentence by spaces AND symbols, while preserving spaces in the split list
    lst = re.split(r"(\s+)", text)
    cleaned = []
    for item in lst:
        if len(item) > 0:
            if not item.isspace():
                split_item = re.split(r"(\s+)", re.sub(pre_process, " ", item))
                for elem in split_item:
                    if len(elem) > 0:
                        cleaned.append(elem)
            else:
                cleaned.append(item)
    return cleaned


def find_phi(text_data: str, patterns: List[Filter]) -> List[CoordinateEntry]:
    """ Runs the set, or regex on the input data
        generating a coordinate map of hits given
        (this performs a dry run on the data and doesn't transform)
    """
    result = []

    context_patters = []

    for pat in patterns:
        if pat.type == "regex" and isinstance(pat, RegexFilter):
            hits = map_regex(text=text_data, pattern=pat)
        elif pat.type == "set" and isinstance(pat, SetFilter):
            hits = map_set(text=text_data, pattern=pat)
        elif pat.type == "regex_context" and isinstance(pat, RegexContextFilter):
            context_patters.append(pat)
            hits = []
        elif pat.type == "stanford_ner" and isinstance(pat, NerFilter):
            hits = map_ner(text=text_data, pattern=pat)
        elif pat.type == "pos_matcher" and isinstance(pat, PosFilter):
            hits = map_pos(text=text_data, pattern=pat)
        elif pat.type == "match_all":
            hits = match_all(text=text_data, pattern=pat)
        else:
            raise Exception("Error, pattern type not supported: ", pat.type)
        result.extend(hits)

    context_results = []
    for context_pattern in context_patters:
        # context_hits = map_regex_context(
        #    text=text_data,
        #    all_patterns=patterns,
        #    include_map=result,
        #    pattern=context_pattern,
        # )
        # context_results.extend(context_hits)
        pass

    result.extend(context_results)

    return result


def map_regex(
    text, pattern: RegexFilter, pre_process=r"[^a-zA-Z0-9]",
) -> List[CoordinateEntry]:
    """ Creates a coordinate map from the pattern on this data
        generating a coordinate map of hits given (dry run doesn't transform)
    """
    regex = pattern.data
    result = []

    # All regexes except matchall
    matches = regex.finditer(text)

    for m in matches:
        result.append(CoordinateEntry(m.start(), m.start() + len(m.group()), pattern))

    return result


def map_regex_context(
    text,
    pattern: RegexContextFilter,
    all_patterns: List[Filter],
    include_map: List[CoordinateEntry],
    pre_process=r"[^a-zA-Z0-9]",
) -> List[CoordinateEntry]:
    """ map_regex_context creates a coordinate map from combined regex + PHI coordinates
    of all previously mapped patterns
    """
    results = []
    regex = pattern.data
    context = pattern.context
    context_filter = pattern.context_filter

    # Get PHI coordinates
    if context_filter == "all":
        current_include_map = include_map
        # Create complement exclude map (also excludes punctuation)
        full_exclude_map = current_include_map.get_complement(text)

    else:
        full_exclude_map_coordinates = all_patterns[context_filter]
        full_exclude_map = {}
        for start, stop in full_exclude_map_coordinates.filecoords():
            full_exclude_map[start] = stop

    # 1. Get coordinates of all include and exclude mathches

    punctuation_matcher = re.compile(r"[^a-zA-Z0-9*]")
    # 2. Find all patterns expressions that match regular expression
    matches = regex.finditer(text)
    for m in matches:

        # initialize phi_left and phi_right
        phi_left = False
        phi_right = False

        match_start = m.span()[0]
        match_end = m.span()[1]

        # PHI context left and right
        phi_starts = []
        phi_ends = []
        for start in full_exclude_map:
            phi_starts.append(start)
            phi_ends.append(full_exclude_map[start])

        if match_start in phi_ends:
            phi_left = True

        if match_end in phi_starts:
            phi_right = True

        # Get index of m.group()first alphanumeric character in match
        tokenized_matches = []
        match_text = m.group()
        split_match = re.split(r"(\s+)", re.sub(pre_process, " ", match_text))

        # Get all spans of tokenized match (because remove() function requires tokenized start coordinates)
        coord_tracker = 0
        for element in split_match:
            if element != "":
                if not punctuation_matcher.match(element[0]):
                    current_start = match_start + coord_tracker
                    current_end = current_start + len(element)
                    tokenized_matches.append((current_start, current_end))

                    coord_tracker += len(element)
                else:
                    coord_tracker += len(element)

        # Check for context, and add to coordinate map
        if (
            (context == "left" and phi_left is True)
            or (context == "right" and phi_right)
            or (context == "left_or_right" and (phi_right or phi_left))
            or (context == "left_and_right" and (phi_right and phi_left))
        ):
            for item in tokenized_matches:
                results.append(CoordinateEntry(item[0], item[1], pattern))

    return results


def match_all(text, pattern: Filter) -> List[CoordinateEntry]:
    """ Simply maps to the entirety of the file """
    # add the entire length of the file
    return [CoordinateEntry(0, len(text), pattern)]


def map_set(text, pattern: SetFilter) -> List[CoordinateEntry]:
    """ Creates a coordinate mapping of words any words in this set"""

    set_data = pattern.data

    result = []

    # get part of speech we will be sending through this set
    # note, if this is empty we will put all parts of speech through the set
    check_pos = False
    pos_set = set(pattern.pos)
    if len(pos_set) > 0:
        check_pos = True

    cleaned = get_clean(text)
    pos_list = nltk.pos_tag(cleaned)

    start_coordinate = 0
    for tup in pos_list:
        word = tup[0]
        pos = tup[1]
        start = start_coordinate
        stop = start_coordinate + len(word)

        # This converts spaces into empty strings, so we know to skip forward to the next real word
        word_clean = re.sub(r"[^a-zA-Z0-9]+", "", word.lower().strip())
        if len(word_clean) == 0:
            # got a blank space or something without any characters or digits, move forward
            start_coordinate += len(word)
            continue

        if not check_pos or (check_pos and pos in pos_set):
            if word_clean in set_data or word in set_data:
                result.append(CoordinateEntry(start, stop, pattern))
            else:
                pass

        # advance our start coordinate
        start_coordinate += len(word)

    return result


def map_pos(text, pattern: PosFilter) -> List[CoordinateEntry]:
    """ Creates a coordinate mapping of words which match this part of speech (POS)"""

    pos_set = set(pattern.pos)

    result = []

    # Use pre-process to split sentence by spaces AND symbols, while preserving spaces in the split list

    cleaned = get_clean(text)

    pos_list = get_pos(cleaned)  # pos_list = nltk.pos_tag(cleaned)
    start_coordinate = 0
    for tup in pos_list:
        word = tup[0]
        pos = tup[1]
        start = start_coordinate
        stop = start_coordinate + len(word)
        word_clean = re.sub(r"[^a-zA-Z0-9]+", "", word.lower().strip())
        if len(word_clean) == 0:
            # got a blank space or something without any characters or digits, move forward
            start_coordinate += len(word)
            continue

        if pos in pos_set:
            result.append(CoordinateEntry(start, stop, pattern))

        # advance our start coordinate
        start_coordinate += len(word)

    return result


def map_ner(
    text,
    pattern: NerFilter,
    stanford_ner_tagger: StanfordNERTagger,
    pre_process=r"[^a-zA-Z0-9]+",
) -> List[CoordinateEntry]:
    """ map NER tagging"""
    pos_set = set()
    result = []
    if pattern.pos:
        pos_set = set(pattern.pos)

    lst = re.split(r"(\s+)", text)
    cleaned = []
    for item in lst:
        if len(item) > 0:
            cleaned.append(item)

    ner_no_spaces = stanford_ner_tagger.tag(cleaned)
    # get our ner tags
    ner_set = {}
    for tup in ner_no_spaces:
        ner_set[tup[0]] = tup[1]
    ner_set_with_locations = {}
    start_coordinate = 0
    for w in cleaned:
        if w in ner_set:
            ner_set_with_locations[w] = (ner_set[w], start_coordinate)
        start_coordinate += len(w)

    # for the text, break into words and mark POS
    # with the parts of speech labeled, match any of these to our coordinate
    # add these coordinates to our coordinate map
    start_coordinate = 0
    for word in cleaned:

        word_clean = re.sub(pre_process, "", word.lower().strip())
        if len(word_clean) == 0:
            # got a blank space or something without any characters or digits, move forward
            start_coordinate += len(word)
            continue

        if word in ner_set_with_locations:
            ner_tag = ner_set_with_locations[word][0]
            start = ner_set_with_locations[word][1]
            if ner_tag in pos_set:
                stop = start + len(word)
                result.append(CoordinateEntry(start, stop, pattern))
                print("FOUND: ", word, "NER: ", ner_tag, start, stop)

        # advance our start coordinate
        start_coordinate += len(word)

    return result


def get_exclude_include_maps(
    pattern: Filter,
    txt,
    coord_map: CoordinateMap,
    include_map: CoordinateMap,
    exclude_map: CoordinateMap,
    phi_type_dict: Dict[str, CoordinateMap],
    data_tracker: DataTracker,
):
    exclude = pattern.exclude
    if hasattr(pattern, "filepath"):
        filter_path = pattern.filepath
    else:
        filter_path = pattern.title
    if pattern.phi_type:
        phi_type = pattern.phi_type
    else:
        phi_type = "OTHER"

    for start, stop in coord_map.filecoords():
        if pattern.type != "regex_context":
            if exclude:
                if not include_map.does_overlap(start, stop):
                    exclude_map.add_extend(start, stop)
                    phi_type_dict[phi_type].add_extend(start, stop)

            else:
                if not exclude_map.does_overlap(start, stop):
                    include_map.add_extend(start, stop)
                    data_tracker.non_phi.append(
                        NonPhiEntry(
                            start=start,
                            stop=stop,
                            word=txt[start:stop],
                            filepath=filter_path,
                        )
                    )

        # Add regex_context to map separately
        else:
            if exclude:
                exclude_map.add_extend(start, stop)
                include_map.remove(start, stop)
                phi_type_dict[phi_type].add_extend(start, stop)
            else:
                include_map.add_extend(start, stop)
                exclude_map.remove(start, stop)
                data_tracker.non_phi.append(
                    NonPhiEntry(
                        start=start,
                        stop=stop,
                        word=txt[start:stop],
                        filepath=filter_path,
                    )
                )


def save_to_asterisk(contents, output_file):
    with open(output_file, "w", encoding="utf-8", errors="surrogateescape") as f:
        f.write(contents)


def save_to_i2b2(contents, output_file):
    with open(output_file, "w", errors="xmlcharrefreplace") as f:
        f.write(contents)


def transform_text_asterisk(txt, include_map: CoordinateMap):
    last_marker = 0
    punctuation_matcher = re.compile(r"[^a-zA-Z0-9*]")
    # read the text by character, any non-punc non-overlaps will be replaced
    contents = []
    for i in range(0, len(txt)):

        if i < last_marker:
            continue

        if include_map.does_exist(i):
            # add our preserved text
            start, stop = include_map.get_coords(i)
            contents.append(txt[start:stop])
            last_marker = stop
        elif punctuation_matcher.match(txt[i]):
            contents.append(txt[i])
        else:
            contents.append("*")

    return "".join(contents)


def transform_text_i2b2(tagdata: DataTracker):
    """creates a string in i2b2-XML format"""
    root = "Philter"
    contents = [
        '<?xml version="1.0" ?>\n',
        "<" + root + ">\n",
        "<TEXT><![CDATA[",
        tagdata.text,
        "]]></TEXT>\n",
        "<TAGS>\n",
    ]
    for i, phi in enumerate(tagdata.phi):
        phi_type = phi.phi_type
        contents.append("<")
        contents.append(phi_type)
        contents.append(' id="P')
        contents.append(str(i))
        contents.append('" start="')
        contents.append(str(phi.start))
        contents.append('" end="')
        contents.append(str(phi.stop))
        contents.append('" text="')
        contents.append(phi.word)
        contents.append('" TYPE="')
        contents.append(phi_type)
        contents.append('" comment="" />\n')

    # for loop over complement - PHI, create additional tags (UNKNOWN)
    contents.append("</TAGS>\n")
    contents.append("</" + root + ">\n")

    return "".join(contents)


def detect_encoding(fp):
    if not os.path.exists(fp):
        raise Exception("Filepath does not exist", fp)

    detector = UniversalDetector()
    with open(fp, "rb") as f:
        for line in f:
            detector.feed(line)
            if detector.done:
                break
        detector.close()
    return detector.result


def phi_context(filename, word, word_index, words, context_window=10):
    """ helper function, creates our phi data type with source file, and context window"""
    if not os.path.exists(filename):
        raise Exception("Filepath does not exist", filename)

    left_index = word_index - context_window
    if left_index < 0:
        left_index = 0

    right_index = word_index + context_window
    if right_index >= len(words):
        right_index = len(words) - 1
    window = words[left_index:right_index]

    return {"filename": filename, "phi": word, "context": window}


def _nested_get(a_dict, keys):
    for key in keys:
        a_dict = a_dict[key]
    return a_dict
