import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from util.identity_memory import append_identity_episode, load_identity_episodes, save_identity_episodes


def main():
    parser = argparse.ArgumentParser(description="Create or extend a ConsisID episodic identity memory bank.")
    parser.add_argument("--input_identity_memory", type=str, required=True, help="Single ConsisID identity memory .pt file.")
    parser.add_argument("--bank_path", type=str, required=True, help="Output episodic memory bank .pt file.")
    parser.add_argument("--source", type=str, default=None, help="Optional source label stored with each episode.")
    parser.add_argument("--append", action="store_true", help="Append to an existing bank instead of replacing it.")
    parser.add_argument("--max_episodes", type=int, default=64)
    args = parser.parse_args()

    input_path = Path(args.input_identity_memory)
    if not input_path.exists():
        raise FileNotFoundError(f"Identity memory not found: {input_path}")

    episodes = load_identity_episodes(str(input_path))
    if args.source:
        for episode in episodes:
            episode["source"] = args.source

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


if __name__ == "__main__":
    main()
