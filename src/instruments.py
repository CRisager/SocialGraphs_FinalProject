"""
Group MusicBrainz instrument names into instrument families.

MusicBrainz credits name instruments at very different levels of detail: "bass",
"bass guitar", "electric bass guitar" and "acoustic bass guitar" are the same role in a
song, and some credits are MusicBrainz's own umbrella terms ("membranophone",
"guitar family", "other instruments"). 1970-1975 alone has 122 names, 28 of them used
by a single recording. Linking songs on exact names would split one instrument into
several feature nodes and give rare spellings far too much weight, so songs are linked
on families instead (src/clean.py adds an `instrument_families` column).

Saxophone is kept apart from the other woodwinds, and the steel/folk strings (pedal
steel, banjo, mandolin, ...) apart from guitar, because they are typical of particular
styles (soul/funk horn sections, country) and so carry information that a broader
family would hide.
"""

INSTRUMENT_FAMILIES = {
    "drums": ["drums (drum set)", "snare drum", "bass drum", "tom-tom", "hi-hat",
              "cymbal", "brushes", "rototom", "electronic drum set"],
    "percussion": ["percussion", "membranophone", "congas", "bongos", "timbales",
                   "maracas", "tambourine", "shakers", "cowbell", "cabasa", "wood block",
                   "triangle", "chimes", "wind chime", "handbell", "gong",
                   "finger cymbals", "frame drum", "tabla", "timpani", "vibraphone",
                   "marimba", "xylophone", "celesta", "handclaps", "foot stomps",
                   "washboard", "agogô", "bell", "bell tree", "bicycle bell", "bodhrán",
                   "claves", "crotales", "glockenspiel", "shaken idiophone",
                   "struck idiophone", "slapstick", "temple blocks", "vibraslap",
                   "tubular bells", "steelpan"],
    "guitar": ["guitar", "electric guitar", "acoustic guitar", "12 string guitar",
               "slide guitar", "classical guitar", "guitar family"],
    "bass": ["bass", "bass guitar", "electric bass guitar", "acoustic bass guitar",
             "double bass", "fretless bass", "bass pedals"],
    "keys": ["piano", "grand piano", "tack piano", "electric piano", "Rhodes piano",
             "Wurlitzer electric piano", "Pianet", "keyboard", "clavinet", "harpsichord",
             "clavichord", "spinet", "electric grand piano"],
    "organ": ["organ", "Hammond organ", "electronic organ", "farfisa", "harmonium", "pipe organ",
              "calliope"],
    # electronic sound sources: synths, tape keyboards (the mellotron and chamberlin
    # supplied string and flute sounds), drum machines and voice effects
    "synth": ["synthesizer", "Moog", "Minimoog", "analog synthesizer",
              "string synthesizer", "synclavier", "mellotron", "electronic instruments",
              "effects", "chamberlin", "Dubreq Stylophone", "Lyricon", "bass synthesizer",
              "drum machine", "talkbox", "vocoder", "tape"],
    "brass": ["trumpet", "trombone", "bass trombone", "flugelhorn", "horn", "tuba",
              "cornet", "brass", "French horn", "euphonium", "slide brass instruments"],
    "saxophone": ["saxophone", "tenor saxophone", "alto saxophone", "baritone saxophone",
                  "soprano saxophone"],
    "woodwind": ["flute", "concert flute", "piccolo", "clarinet", "bass clarinet", "oboe",
                 "bassoon", "recorder", "whistle", "reeds", "woodwind", "wind instruments", "alto flute", "bass flute",
                 "cor anglais", "kortholt", "sopranino recorder", "soprano recorder",
                 "tenor recorder", "bagpipe"],
    "strings": ["violin", "fiddle", "viola", "cello", "strings", "harp", "electric violin", "hardingfele"],
    "folk strings": ["pedal steel guitar", "steel guitar", "resonator guitar", "banjo",
                     "five-string banjo", "mandolin", "Appalachian dulcimer", "autoharp",
                     "zither", "balalaika", "sitar", "electric sitar", "tambura", "Portuguese guitar", "bouzouki",
                     "mandocello", "tenor banjo"],
    "harmonica": ["harmonica", "mouth harp", "mouth organ", "bass harmonica"],
    "other": ["accordion", "piano accordion", "diatonic button accordion", "melodica", "duck call", "other instruments"],
}

FAMILY_OF = {name.casefold(): family
             for family, names in INSTRUMENT_FAMILIES.items() for name in names}
assert len(FAMILY_OF) == sum(map(len, INSTRUMENT_FAMILIES.values())), "name in two families"


def family(name):
    """Family of one instrument name; 'other' for names not in the table."""
    return FAMILY_OF.get(name.strip().casefold(), "other")


def families(instruments):
    """';'-joined instrument names -> ';'-joined families, sorted (None stays None)."""
    if not isinstance(instruments, str) or not instruments:
        return None
    return ";".join(sorted({family(n) for n in instruments.split(";") if n.strip()}))


def unknown(names):
    """Instrument names that are not in the table (they fall into 'other')."""
    return sorted({n for v in names.dropna() for n in v.split(";")
                   if n.strip() and n.strip().casefold() not in FAMILY_OF})


assert families("bass;electric bass guitar;drums (drum set);Hammond organ") == "bass;drums;organ"
assert families(None) is None
