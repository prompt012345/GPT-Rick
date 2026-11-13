#!/usr/bin/env python3
"""Chat Rick Sanchez avec un GPT-2 au format GGUF (llama-cpp-python, sans TensorFlow).

Installation (Windows, sans compilateur) :
    pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

Usage :
    python src/rick_chat_gguf.py C:\\gpt2-medium-Q8_0.gguf --drunk 0.8
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rick_style  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="chemin du fichier .gguf")
    ap.add_argument("--drunk", type=float, default=0.6, help="0 sobre .. 1 bourré")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--length", type=int, default=70, help="tokens max par réplique")
    ap.add_argument("--memory", type=int, default=3, help="échanges gardés en mémoire")
    ap.add_argument("--seed", type=int, default=-1)
    args = ap.parse_args()

    from llama_cpp import Llama
    llm = Llama(model_path=args.model, n_ctx=1024, seed=args.seed, verbose=False)

    history = []
    print("*burp* Rick est dans la place. 'reset' pour recommencer, Ctrl+C pour quitter.\n")
    while True:
        try:
            msg = input("Morty >>> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nRick: Wubba lubba dub dub.")
            return
        if not msg:
            continue
        if msg.lower() == "reset":
            history = []
            print("(mémoire effacée)")
            continue

        prompt = rick_style.build_prompt(history[-args.memory:], msg)
        out = llm.create_completion(
            prompt,
            max_tokens=args.length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            stop=["\nMorty:", "\nRick:", "\n\n"],
        )
        reply = rick_style.cut_reply(out["choices"][0]["text"])
        if not reply:
            reply = "*burp* Ugh. Whatever, Morty."
        reply = rick_style.drunkify(reply, level=args.drunk)
        history.append((msg, reply))
        print("Rick: " + reply + "\n")


if __name__ == "__main__":
    main()
