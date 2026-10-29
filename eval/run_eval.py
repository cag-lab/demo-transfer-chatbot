"""Run the test questions and report how the bot did.

    python eval/run_eval.py               # full test (search + answer)
    python eval/run_eval.py --search-only # only check that search finds the right document (cheaper)
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag import Index, answer  # noqa: E402

REFUSAL_PHRASE = "couldn't find"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-only", action="store_true")
    parser.add_argument("--file", default=str(Path(__file__).parent / "questions.yaml"))
    args = parser.parse_args()

    cases = yaml.safe_load(Path(args.file).read_text(encoding="utf-8"))
    index = Index()
    passed = 0

    for i, case in enumerate(cases, start=1):
        q = case["question"]
        problems = []

        hits = index.search(q)
        sources = [c.source for c, _ in hits]
        if case.get("expect_source") and case["expect_source"] not in sources[:3]:
            problems.append(f"expected {case['expect_source']} in top 3 results, got {sources[:3]}")

        reply = ""
        if not args.search_only:
            reply, _ = answer(q, index)
            low = reply.lower()
            for phrase in case.get("expect_in_answer", []):
                if phrase.lower() not in low:
                    problems.append(f"answer missing '{phrase}'")
            if case.get("expect_refusal") and REFUSAL_PHRASE not in low:
                problems.append("expected a 'couldn't find' answer")

        status = "PASS" if not problems else "FAIL"
        passed += not problems
        print(f"{status}  {i:>2}. {q}")
        for p in problems:
            print(f"        - {p}")
        if problems and reply:
            print("        answer: " + reply.replace("\n", " ")[:300])

    print(f"\n{passed}/{len(cases)} passed")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
