import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from util.identity_memory import append_identity_episode, load_identity_episodes, save_identity_episodes


def main():
    parser = argparse.ArgumentParser(description="Create or extend a ConsisID episodic identity memory bank.")
    parser.add_argument(
        "--input_identity_memory",
        type=str,
        nargs="+",
        required=True,
        help="One or more ConsisID identity memory .pt files. Use multiple realistic references to build a meaningful bank.",
    )
    parser.add_argument("--bank_path", type=str, required=True, help="Output episodic memory bank .pt file.")
    parser.add_argument("--source", type=str, default=None, help="Optional source label stored with each episode.")
    parser.add_argument("--append", action="store_true", help="Append to an existing bank instead of replacing it.")
    parser.add_argument("--max_episodes", type=int, default=64)
    args = parser.parse_args()

    episodes = []
    for input_memory in args.input_identity_memory:
        input_path = Path(input_memory)
        if not input_path.exists():
            raise FileNotFoundError(f"Identity memory not found: {input_path}")

        input_episodes = load_identity_episodes(str(input_path))
        for episode in input_episodes:
            if args.source:
                episode["source"] = args.source
            elif not episode.get("source"):
                episode["source"] = str(input_path)
        episodes.extend(input_episodes)

    if args.append:
        count = 0
        for episode in episodes:
            count = append_identity_episode(args.bank_path, episode, max_episodes=args.max_episodes)
    else:
        if args.max_episodes > 0:
            episodes = episodes[-args.max_episodes :]
        save_identity_episodes(
            args.bank_path,
            episodes,
            metadata={"note": "ConsisID episodic identity memory bank"},
        )
        count = len(episodes)

    print(f"Identity memory bank ready: {args.bank_path} ({count} episodes)")
    if count < 2:
        print(
            "Warning: this bank has fewer than 2 episodes. It is useful for smoke tests, "
            "but not for proving episodic identity persistence."
        )


if __name__ == "__main__":
    main()
