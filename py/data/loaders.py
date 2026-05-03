from concurrent.futures import ThreadPoolExecutor
from pathlib import Path 
import yaml 

from .definitions import BIBLE_BOOKS, Bible, BibleBook


def _get_bible_version_from_path(
        directory: Path) -> str:
    return directory.stem


def load_bible_book_from_file (
        file :Path, 
        encoding :str = "utf-8"
) -> BibleBook: 
    """
    Load a BibleBook object from a YAML file
    """
    assert isinstance(file, Path) and file.is_file()
    with open(file, "r", encoding=encoding) as f: 
        bible_book = BibleBook(**yaml.safe_load(f))
        bible_book.verses = sorted(
            bible_book.verses,
            key=lambda v: (v.chapter, v.verse))
        return bible_book


def load_bible_from_dir (
        directory :Path
) -> Bible: 
    """
    Load a Bible, a list of bible books, from a directory
    """
    assert isinstance(directory, Path) and directory.is_dir(), f"{directory} is not a valid directory"

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(load_bible_book_from_file, book_path)
                   for book_path in directory.glob("*.yaml")]
        books = [
            future.result()
            for future in futures]
    
    # Sort the bible books
    books = [bb for bb in books if bb.book in BIBLE_BOOKS]
    books = sorted(
        books, key=lambda bb: BIBLE_BOOKS.index(bb.book))
    
    bible = Bible(
        version=_get_bible_version_from_path(directory),
        books=books)

    return bible


def load_verse_context (
        bible_book :BibleBook, 
        chapter :int, 
        verse :int, 
        context_scope :str, # "book", "chapter", or "verse"
        n_prev_context_verses :int = 2, 
        n_next_context_verses :int = 2
) -> str: 
    """
    Given a BibleBook object, bible_book, loads a string as the "context" of the verse. 
    Here is how the context being generated: 
    1. Load n_prev_context_verses verses before the targeted verse (chapter, verse), the verse itself, and n_next_context_verses after the targeted verse. 
    2. Chop the verses loaded from the previous step by context_scope. If context_scope is "book", verses of the same book of the targeted verse will be include. Similarly, if context_scope is "chapter", verses of the same chapter will be included. If context_scope is "verse", the targeted verse itself is included.
    """
    # Check parameters 
    assert(n_prev_context_verses >= 0)
    assert(n_next_context_verses >= 0)

    # Located the targeted verse 
    i_targeted_verse = 0 
    while (i_targeted_verse < len(bible_book.verses)): 
        targeted_verse = bible_book.verses[i_targeted_verse]
        if (targeted_verse.metadata["chapter"] == chapter and targeted_verse.metadata["verse"] == verse): 
            break # the targeted verse located 
        i_targeted_verse += 1

    assert(i_targeted_verse < len(bible_book.verses)), f"The targeted verse {bible_book.metadata['book']}{chapter}:{verse} not found"

    # Load the surrounding verses 
    i_from = max(0, i_targeted_verse - n_prev_context_verses) 
    i_to = min(len(bible_book.verses), i_targeted_verse + 1 + n_next_context_verses)

    context_verses = bible_book.verses[i_from : i_to]

    # chop the verses based on context_scope 
    if (context_scope == "book"): 
        pass 

    elif (context_scope == "chapter"): 
        context_verses = list(filter(
            lambda cv: cv.metadata["chapter"] == chapter
        ))

    elif (context_scope == "verse"): 
        context_verses = bible_book.verses[i_targeted_verse:i_targeted_verse+1]

    else: 
        assert(False), f"Invalid {context_scope = }"

    # return 
    return " ".join(list(map(
        lambda cv: cv.text, 
        context_verses 
    )))
