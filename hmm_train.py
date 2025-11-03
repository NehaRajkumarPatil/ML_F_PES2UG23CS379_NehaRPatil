# hmm_train.py
# Hidden Markov Model training for Hangman

import os
import pickle
import numpy as np
from collections import defaultdict

SEED = 42
np.random.seed(SEED)

# -------------------------
# Discrete HMM Class
# -------------------------
class DiscreteHMM:
    def __init__(self, n_states, alphabet):
        self.N = n_states
        self.alphabet = alphabet
        self.M = len(alphabet)
        self.sym2idx = {s: i for i, s in enumerate(alphabet)}
        rng = np.random.RandomState(SEED)
        self.pi = rng.dirichlet(np.ones(self.N))
        self.A = rng.dirichlet(np.ones(self.N), size=self.N)
        self.B = rng.dirichlet(np.ones(self.M), size=self.N)

    def _obs_to_idx(self, s):
        return np.array([self.sym2idx.get(c, 0) for c in s])

    def _forward_scaled(self, obs):
        T = len(obs)
        alpha = np.zeros((T, self.N))
        scale = np.zeros(T)
        alpha[0] = self.pi * self.B[:, obs[0]]
        scale[0] = alpha[0].sum() or 1e-300
        alpha[0] /= scale[0]
        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self.A) * self.B[:, obs[t]]
            scale[t] = alpha[t].sum() or 1e-300
            alpha[t] /= scale[t]
        return alpha, scale, np.sum(np.log(scale))

    def _backward_scaled(self, obs, scale):
        T = len(obs)
        beta = np.zeros((T, self.N))
        beta[-1] = 1.0 / (scale[-1] or 1e-300)
        for t in range(T - 2, -1, -1):
            beta[t] = (self.A @ (self.B[:, obs[t + 1]] * beta[t + 1])) / (scale[t] or 1e-300)
        return beta

    def fit(self, seqs, n_iter=12, tol=1e-4, verbose=False):
        eps = 1e-8
        prev_ll = -1e300
        for it in range(n_iter):
            A_num = np.zeros_like(self.A)
            A_den = np.zeros((self.N, 1))
            B_num = np.zeros_like(self.B)
            B_den = np.zeros((self.N, 1))
            pi_acc = np.zeros(self.N)
            total_ll = 0
            for s in seqs:
                obs = self._obs_to_idx(s)
                alpha, scale, ll = self._forward_scaled(obs)
                beta = self._backward_scaled(obs, scale)
                total_ll += ll
                gamma = alpha * beta
                gamma /= gamma.sum(axis=1, keepdims=True) + eps
                pi_acc += gamma[0]
                for t in range(len(obs) - 1):
                    xi = np.outer(alpha[t], self.A @ (self.B[:, obs[t + 1]] * beta[t + 1]))
                    xi /= xi.sum() + eps
                    A_num += xi
                    A_den += gamma[t][:, None]
                for t, o in enumerate(obs):
                    B_num[:, o] += gamma[t]
                B_den += gamma.sum(axis=0)[:, None]
            self.pi = pi_acc / (pi_acc.sum() + eps)
            self.A = (A_num + eps) / (A_den + eps)
            self.B = (B_num + eps) / (B_den + eps)
            if verbose:
                print(f"HMM iter {it+1}/12: loglik={total_ll:.3f}")
            if abs(total_ll - prev_ll) < tol:
                break
            prev_ll = total_ll

    # NEW: scoring method required by RL
    def score_candidate_completions(self, masked, guessed):
        results = {}
        for i, c in enumerate(masked):
            if c != "_":
                continue
            scores = {}
            for l in self.alphabet:
                if l in guessed:
                    continue
                candidate = list(masked)
                candidate[i] = l
                try:
                    obs = self._obs_to_idx(candidate)
                    _, _, logp = self._forward_scaled(obs)
                    score = np.exp(logp / len(candidate))
                except Exception:
                    score = 0
                scores[l] = score
            total = sum(scores.values())
            if total == 0:
                valid = [ch for ch in self.alphabet if ch not in guessed]
                scores = {ch: 1 / len(valid) for ch in valid} if valid else {}
            else:
                scores = {k: v / total for k, v in scores.items()}
            results[i] = scores
        return results


# -------------------------
# Utilities
# -------------------------
def load_wordlist(path):
    with open(path, "r", encoding="utf-8") as f:
        return [w.strip().lower() for w in f if w.strip()]

def train_hmms_by_length(corpus, n_iter=12, verbose=False):
    letters = sorted({c for w in corpus for c in w})
    grouped = defaultdict(list)
    for w in corpus:
        grouped[len(w)].append(w)
    hmms = {}
    for L, words in grouped.items():
        n_states = min(max(2, L // 2), 8)
        model = DiscreteHMM(n_states, letters)
        model.fit(words, n_iter=n_iter, verbose=verbose)
        hmms[L] = model
    return hmms


if __name__ == "__main__":
    path = input("📂 Enter path to corpus.txt: ").strip()
    corpus = load_wordlist(path)
    print(f"✅ Loaded {len(corpus)} words.")
    hmms = train_hmms_by_length(corpus, verbose=True)
    with open("hmms.pkl", "wb") as f:
        pickle.dump(hmms, f)
    print("✅ Saved HMM models to hmms.pkl")
