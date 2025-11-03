# rl_train.py
# Reinforcement Learning agent with HMM-based advisor

import pickle
import numpy as np
import random
from collections import defaultdict
from hmm_train import DiscreteHMM, load_wordlist

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# -------------------------
# Hangman Environment
# -------------------------
class HangmanEnv:
    def __init__(self, word_list, max_wrong=6):
        self.words = [w.strip().lower() for w in word_list if w.strip()]
        self.max_wrong = max_wrong
        self.alphabet = "".join(sorted({c for w in self.words for c in w}))
        self.reset()

    def reset(self, word=None):
        self.target = word or random.choice(self.words)
        self.guessed = set()
        self.wrong = 0
        self.done = False
        self.won = False
        return self._state()

    def _state(self):
        masked = "".join(c if c in self.guessed else "_" for c in self.target)
        return {
            "masked_word": masked,
            "guessed_letters": set(self.guessed),
            "wrong_guesses": self.wrong,
            "lives_remaining": self.max_wrong - self.wrong,
            "won": self.won,
            "game_over": self.done,
        }

    def step(self, letter):
        if self.done:
            return self._state(), 0, True
        if letter in self.guessed:
            return self._state(), -2, False
        self.guessed.add(letter)
        if letter in self.target:
            reward = 4.0
            if all(c in self.guessed for c in self.target):
                self.won = True
                self.done = True
                reward = 30.0
        else:
            self.wrong += 1
            reward = -3.0
            if self.wrong >= self.max_wrong:
                self.done = True
                reward = -20.0
        return self._state(), reward, self.done

    def get_valid_actions(self):
        return [c for c in self.alphabet if c not in self.guessed]


# -------------------------
# Hybrid RL Agent
# -------------------------
class HybridAdvisor:
    def __init__(self, hmms, alpha=0.1, gamma=0.95, epsilon=0.8):
        self.hmms = hmms
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.q = defaultdict(lambda: defaultdict(float))
        self.rng = random.Random(SEED)

    def _key(self, s):
        return f"{s['masked_word']}|{s['lives_remaining']}"

    def _hmm_scores(self, masked, guessed):
        L = len(masked)
        if L not in self.hmms:
            return {}
        pos_probs = self.hmms[L].score_candidate_completions(masked, guessed)
        agg = defaultdict(float)
        for pmap in pos_probs.values():
            for c, p in pmap.items():
                agg[c] += p
        s = sum(agg.values())
        return {c: p / s for c, p in agg.items()} if s else {}

    def choose_action(self, s, valid, training=True):
        masked = s["masked_word"]
        guessed = s["guessed_letters"]
        hmm_scores = self._hmm_scores(masked, guessed)
        if training and self.rng.random() < self.epsilon:
            if hmm_scores:
                letters, probs = zip(*hmm_scores.items())
                return self.rng.choices(letters, weights=probs, k=1)[0]
            return self.rng.choice(valid)
        key = self._key(s)
        q_vals = self.q[key]
        scores = {a: q_vals[a] + 1.5 * hmm_scores.get(a, 0) for a in valid}
        if not scores:
            return None
        maxv = max(scores.values())
        best = [a for a, v in scores.items() if abs(v - maxv) < 1e-8]
        return self.rng.choice(best)

    def update(self, s, a, r, ns, done):
        k, nk = self._key(s), self._key(ns)
        cur = self.q[k][a]
        nxt = max(self.q[nk].values()) if not done and self.q[nk] else 0.0
        self.q[k][a] = cur + self.alpha * (r + self.gamma * nxt - cur)

    def train(self, env, episodes=5000):
        print(f"🚀 Training for {episodes} episodes...")
        win_hist = []
        for ep in range(episodes):
            s = env.reset()
            done = False
            while not done:
                valid_actions = env.get_valid_actions()
                if not valid_actions:
                    break
                a = self.choose_action(s, valid_actions)
                if a is None:
                    break
                ns, r, done = env.step(a)
                self.update(s, a, r, ns, done)
                s = ns
            win_hist.append(1 if env.won else 0)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            if (ep + 1) % 200 == 0:
                print(f"Episode {ep+1}/{episodes} | WinRate={np.mean(win_hist[-200:])*100:.2f}% | Eps={self.epsilon:.3f}")


# -------------------------
# Evaluation
# -------------------------
def evaluate(agent, test_words):
    env = HangmanEnv(test_words)
    wins = 0
    wrong_total = 0
    repeated = 0

    for w in test_words:
        s = env.reset(w)
        done = False
        seen = set()
        while not done:
            valid = env.get_valid_actions()
            if not valid:
                break
            a = agent.choose_action(s, valid, training=False)
            if a is None:
                break
            if a in seen:
                repeated += 1
            seen.add(a)
            ns, _, done = env.step(a)
            s = ns
        if env.won:
            wins += 1
        wrong_total += env.wrong

    rate = wins / len(test_words)
    final_score = (rate * len(test_words)) - (wrong_total * 5) - (repeated * 2)
    print(f"\n✅ Success Rate: {rate*100:.2f}% | ❌ Wrong guesses: {wrong_total} | 🔁 Repeated: {repeated}")
    print(f"🏆 Final Score: {final_score:.2f}")
    return rate


if __name__ == "__main__":
    corpus = load_wordlist(input("📂 Enter path to corpus.txt: ").strip())
    test = load_wordlist(input("📂 Enter path to test.txt: ").strip())
    hmms = pickle.load(open("hmms.pkl", "rb"))
    env = HangmanEnv(corpus)
    agent = HybridAdvisor(hmms)
    agent.train(env, episodes=5000)
    print("\n🔍 Evaluating on test set...")
    evaluate(agent, test)
