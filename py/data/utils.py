
from .definitions import Bible, BibleBook, BibleVerse, TextChunk


def _to_bible_verses(
        verses: list[dict]) -> list[BibleVerse]:
    assert all(isinstance(v, dict) or isinstance(v, BibleVerse)
               for v in verses)
    verses = [
        v if isinstance(v, BibleVerse) else BibleVerse(**v)
        for v in verses]
    return verses


def create_markdown_from_TextChunk(tc: TextChunk) -> str:
    if isinstance(tc, BibleVerse):
        return tc.text
    elif isinstance(tc, BibleBook):
        chapters = ["### Chapter 1\n"]
        for v in tc.verses:
            ch = v.chapter
            if ch > len(chapters):
                chapters.append(f"### Chapter {ch}\n")
            chapters[-1] = f"{chapters[-1]}\n{v.text}".strip()
        md_text = f"## Book of {tc.book}\n\n" + "\n\n".join(chapters)
        return md_text
    elif isinstance(tc, Bible):
        books = [
            create_markdown_from_TextChunk(b) for b in tc.books]
        md_text = f"# Bible version {tc.version}\n\n" + "\n\n".join(books)
        return md_text
    else:
        return tc.text


def encode_verse_range(
        book: str,
        from_verse: BibleVerse,
        to_verse: BibleVerse | None) -> str:
    """
    Given a from verse and an optional to verse, create the string that represents the range
    """
    to_verse = to_verse or from_verse
    from_chapter = from_verse.chapter
    to_chapter = to_verse.chapter
    assert from_verse.chapter <= to_verse.chapter
    from_v = from_verse.verse
    to_v = to_verse.verse
    if from_chapter == to_chapter:
        assert from_v <= to_v
        if from_v == to_v:
            return f"{book} {from_chapter}:{from_v}"
        else:
            return f"{book} {from_chapter}:{from_v}-{to_v}"
    else:
        return f"{book} {from_chapter}:{from_v}-{to_chapter}:{to_v}"


def extract_verses_text(
        verses: list[dict]) -> str:
    verses: list[BibleVerse] = _to_bible_verses(verses)
    return " ".join(v.text for v in verses)


def make_bible_quote(
        book: str, verses: list[dict]) -> TextChunk:
    verses: list[BibleVerse] = _to_bible_verses(verses)
    return TextChunk(
        text=extract_verses_text(verses),
        metadata={
            "range": encode_verse_range(
                book=book, from_verse=verses[0], to_verse=verses[-1])
        }
    )
