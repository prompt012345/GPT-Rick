#!/usr/bin/env python3
"""Chat avec un GPT-2 qui joue Rick Sanchez (Rick et Morty).

Usage :
    python3 src/rick_chat.py --model_name=124M
    python3 src/rick_chat.py --model_name=355M --drunk=0.8 --temperature=0.9

Tu joues Morty. Tape "reset" pour effacer l'historique, Ctrl+C pour quitter.
"""
import json
import os

import fire
import numpy as np
import tensorflow as tf

import encoder
import model
import rick_style
import sample


def rick_chat(
    model_name='124M',
    seed=None,
    length=70,          # tokens max par réplique (court = plus punchy)
    temperature=0.9,    # >0.8 : plus délirant, comme Rick bourré
    top_k=40,
    top_p=0.95,
    drunk=0.6,          # 0 = sobre, 1 = complètement bourré (rots, bégaiements)
    memory=3,           # nb d'échanges précédents gardés dans le contexte
    models_dir='models',
):
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [1, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length, context=context, batch_size=1,
            temperature=temperature, top_k=top_k, top_p=top_p,
        )
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(os.path.join(models_dir, model_name)))

        history = []
        print("*burp* Rick est dans la place. Tape 'reset' pour recommencer.\n")
        while True:
            try:
                msg = input("Morty >>> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nRick: Wubba lubba dub dub.")
                return
            if not msg:
                continue
            if msg.lower() == 'reset':
                history = []
                print("(mémoire effacée)")
                continue

            prompt = rick_style.build_prompt(history[-memory:], msg)
            tokens = enc.encode(prompt)
            # le contexte doit laisser la place de générer
            max_ctx = hparams.n_ctx - length
            if len(tokens) > max_ctx:
                tokens = tokens[-max_ctx:]
            out = sess.run(output, feed_dict={context: [tokens]})[0][len(tokens):]
            reply = rick_style.cut_reply(enc.decode(out))
            if not reply:
                reply = "*burp* Ugh. Whatever, Morty."
            reply = rick_style.drunkify(reply, level=drunk)
            history.append((msg, reply))
            print("Rick: " + reply + "\n")


if __name__ == '__main__':
    fire.Fire(rick_chat)
