import argparse
import json


def parse_tab_lexicon(path):
    lexicon = {}
    with open(path, 'r', encoding='utf-8') as fin:
        for line_number, line in enumerate(fin, start=1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            fields = line.split('\t', 1)
            if len(fields) != 2:
                raise ValueError("Line {0} must contain word<TAB>phones".format(line_number))

            word, phones = fields
            word = word.strip().lower()
            phones = ' '.join(phones.strip().split())

            if not word:
                raise ValueError("Line {0} has an empty word".format(line_number))
            if not phones:
                raise ValueError("Line {0} has empty phones for {1}".format(line_number, word))

            lexicon[word] = phones
    return lexicon


def main():
    parser = argparse.ArgumentParser(
        description="Convert a word<TAB>phones lexicon to POWER JSON format.")
    parser.add_argument("input", help="Input tab-separated lexicon")
    parser.add_argument("output", help="Output JSON lexicon")
    args = parser.parse_args()

    lexicon = parse_tab_lexicon(args.input)
    with open(args.output, 'w', encoding='utf-8') as fout:
        json.dump(lexicon, fout, ensure_ascii=False, indent=2, sort_keys=True)
        fout.write('\n')


if __name__ == "__main__":
    main()
